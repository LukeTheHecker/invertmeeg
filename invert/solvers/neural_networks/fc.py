from __future__ import annotations

import logging
from copy import deepcopy

import mne
import numpy as np

_TORCH_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    import torch  # type: ignore[import-not-found]
    import torch.nn as nn  # type: ignore[import-not-found]
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]

    class _NNModule:  # pragma: no cover
        pass

    class _NN:  # pragma: no cover
        Module = _NNModule

    nn = _NN()  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc

from ...simulate import generator
from ..base import BaseSolver, SolverMeta
from .torch_utils import (
    activation_from_name,
    count_trainable_parameters,
    get_torch_device,
    loss_from_name,
)
from .utils import rescale_sources

logger = logging.getLogger(__name__)


class _FCNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_dipoles: int,
        *,
        n_dense_units: int,
        activation_function: str,
    ) -> None:
        super().__init__()
        act = activation_from_name(activation_function)
        self.net = nn.Sequential(
            nn.Linear(int(n_channels), int(n_dense_units)),
            act,
            nn.Linear(int(n_dense_units), int(n_dense_units)),
            act,
            nn.Linear(int(n_dense_units), int(n_dipoles)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, channels)
        return self.net(x)


class SolverFC(BaseSolver):
    """Class for the Fully-Connected Neural Network (FC) for
    EEG inverse solutions.

    """

    meta = SolverMeta(
        acronym="FC",
        full_name="Fully-Connected Neural Network",
        category="Neural Networks",
        description=(
            "Supervised fully-connected network trained on simulated data to map "
            "sensor time series to source activity."
        ),
        references=["Lukas Hecker 2025, unpublished"],
    )

    def __init__(self, name="Fully-Connected", **kwargs):
        self.name = name
        self.model = None
        self.optimizer = None
        self.device = None
        return super().__init__(**kwargs)

    def make_inverse_operator(
        self,
        forward,
        simulation_config,
        *args,
        n_dense_units=300,
        activation_function="tanh",
        epochs=300,
        learning_rate=1e-3,
        loss="cosine_similarity",
        size_validation_set=256,
        patience=100,
        alpha="auto",
        **kwargs,
    ):
        """Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        activation_function : str
            The activation function of the hidden layers.
        batch_size : ["auto", int]
            The batch_size used during training. If "auto", the batch_size
            defaults to the number of dipoles in the source/ forward model.
            Choose a smaller batch_size (e.g., 1000) if you run into memory
            problems (RAM or GPU memory).
        n_timepoints : int
            The number of time points to simulate and ultimately train the
            neural network on.
        batch_repetitions : int
            The number of learning repetitions on the same batch of training
            data until a new batch is simulated.
        epochs : int
            The number of epochs to train.
        learning_rate : float
            The learning rate of the optimizer that trains the neural network.
        loss : str
            The loss function of the neural network.
        n_sources : int
            The maximum number of sources to simulate for the training data.
        n_orders : int
            Controls the maximum smoothness of the sources.
        size_validation_set : int
            The size of validation data set.
        snr_range : tuple
            The range of signal to noise ratios (SNRs) in the training data (in dB).
        patience : int
            Stopping criterion for the training.
        alpha : float
            The regularization parameter.
        correlation_mode : None/str
            None implies no correlation between the noise in different channels.
            'bounded' : Colored bounded noise, where channels closer to each other will be more correlated.
            'diagonal' : Some channels have varying degrees of noise.
        noise_color_coeff : float
            The magnitude of spatial coloring of the noise.

        Return
        ------
        self : object returns itself for convenience
        """
        super().make_inverse_operator(
            forward, *args, alpha=alpha, verbose=self.verbose, **kwargs
        )
        n_channels, n_dipoles = self.leadfield.shape

        # Store simulation config
        self.simulation_config = simulation_config

        # Store Parameters
        # Architecture
        self.n_dense_units = n_dense_units
        self.activation_function = activation_function
        # Training
        self.batch_size = self.simulation_config.batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss = loss
        self.size_validation_set = size_validation_set
        self.patience = patience
        # Training Data (from simulation_config)
        self.n_timepoints = self.simulation_config.n_timepoints
        self.n_sources = self.simulation_config.n_sources
        self.n_orders = self.simulation_config.n_orders
        self.batch_repetitions = self.simulation_config.batch_repetitions
        self.snr_range = self.simulation_config.snr_range
        self.add_forward_error = self.simulation_config.add_forward_error
        self.forward_error = self.simulation_config.forward_error
        self.correlation_mode = self.simulation_config.correlation_mode
        self.noise_color_coeff = self.simulation_config.noise_color_coeff
        # Inference
        logger.info("Create Generator:..")
        self.create_generator()
        logger.info("Build Model:..")
        self.build_model()
        logger.info("Train Model:..")
        self.train_model()

        self.inverse_operators = []
        return self

    def apply_inverse_operator(self, mne_obj) -> mne.SourceEstimate:
        """Apply the inverse operator.

        Parameters
        ----------
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.

        Return
        ------
        stc : mne.SourceEstimate
            The mne Source Estimate object.
        """
        data = self.unpack_data_obj(mne_obj)

        source_mat = self.apply_model(data)
        stc = self.source_to_object(source_mat)

        return stc

    def apply_model(self, data) -> np.ndarray:
        """Compute the inverse solution of the M/EEG data.

        Parameters
        ----------
        data : numpy.ndarray
            The M/EEG data matrix.

        Return
        ------
        x_hat : numpy.ndarray
            The source esimate.

        """

        y = deepcopy(data)
        y /= np.linalg.norm(y, axis=0)
        y /= abs(y).max()

        n_channels, n_times = y.shape

        # Add empty batch and (color-) channel dimension
        y = y.T[np.newaxis]
        # Predict source(s)
        assert self.model is not None
        self.model.eval()
        device = self.device or get_torch_device()
        with torch.no_grad():
            source_pred = (
                self.model(torch.as_tensor(y, dtype=torch.float32, device=device))
                .detach()
                .cpu()
                .numpy()
            )
        source_pred = np.swapaxes(source_pred, 1, 2)  # (batch, dipoles, time)
        logger.debug(source_pred.shape)

        # Rescale sources
        y_original = deepcopy(data)
        y_original = y_original[np.newaxis]
        source_pred_scaled = rescale_sources(self.leadfield, source_pred[0], y_original)

        return source_pred_scaled

    def train_model(
        self,
    ):
        """Train the neural network model."""
        if self.model is None:
            raise RuntimeError("Model not initialized. Call build_model() first.")
        if self.optimizer is None:
            raise RuntimeError("Optimizer not initialized. Call build_model() first.")

        loss_fn = loss_from_name(self.loss)
        device = self.device or get_torch_device()

        # Get Validation data from generator
        x_val, y_val = next(self.generator)
        x_val = x_val[: self.size_validation_set]
        y_val = y_val[: self.size_validation_set]
        x_val_t = torch.as_tensor(x_val, dtype=torch.float32, device=device)
        y_val_t = torch.as_tensor(y_val, dtype=torch.float32, device=device)

        history: dict[str, list[float]] = {"loss": [], "val_loss": []}
        best_val = float("inf")
        best_state = None
        patience_left = int(self.patience)

        for _epoch in range(int(self.epochs)):
            self.model.train()
            running = 0.0
            for _step in range(int(self.batch_repetitions)):
                x_batch, y_batch = next(self.generator)
                x_t = torch.as_tensor(x_batch, dtype=torch.float32, device=device)
                y_t = torch.as_tensor(y_batch, dtype=torch.float32, device=device)

                self.optimizer.zero_grad(set_to_none=True)
                y_pred = self.model(x_t)
                loss = loss_fn(y_pred, y_t)
                loss.backward()
                self.optimizer.step()
                running += float(loss.detach().cpu().item())

            train_loss = running / float(self.batch_repetitions)

            self.model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.model(x_val_t), y_val_t).cpu().item())

            history["loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = deepcopy(self.model.state_dict())
                patience_left = int(self.patience)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        self.history = history

    def build_model(
        self,
    ):
        """Build the neural network model."""
        n_channels, n_dipoles = self.leadfield.shape
        self.device = get_torch_device()
        self.model = _FCNet(
            n_channels,
            n_dipoles,
            n_dense_units=int(self.n_dense_units),
            activation_function=str(self.activation_function),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(self.learning_rate)
        )
        logger.info(
            "Total number of trainable parameters: %d",
            count_trainable_parameters(self.model),
        )

    def create_generator(
        self,
    ):
        """Creat the data generator used for the simulations."""
        gen_args = dict(
            use_cov=False,
            return_mask=False,
            batch_size=self.batch_size,
            batch_repetitions=self.batch_repetitions,
            n_sources=self.n_sources,
            n_orders=self.n_orders,
            n_timepoints=self.n_timepoints,
            snr_range=self.snr_range,
            add_forward_error=self.add_forward_error,
            forward_error=self.forward_error,
            correlation_mode=self.correlation_mode,
            noise_color_coeff=self.noise_color_coeff,
            scale_data=True,
        )
        self.generator = generator(self.forward, **gen_args)
        self.generator.__next__()
