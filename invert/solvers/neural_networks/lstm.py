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

from ...simulate import SimulationGenerator
from ..base import BaseSolver, SolverMeta
from .torch_utils import (
    activation_from_name,
    count_trainable_parameters,
    get_torch_device,
    loss_from_name,
)
from .utils import rescale_sources

logger = logging.getLogger(__name__)


class _LSTMNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_dipoles: int,
        *,
        n_dense_units: int,
        n_lstm_units: int,
        activation_function: str,
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(int(n_channels), int(n_dense_units))
        self.activation = activation_from_name(activation_function)
        self.direct_out = nn.Linear(int(n_dense_units), int(n_dipoles))
        self.lstm = nn.LSTM(
            input_size=int(n_dense_units),
            hidden_size=int(n_lstm_units),
            batch_first=True,
            bidirectional=True,
        )
        self.mask_out = nn.Linear(int(n_lstm_units) * 2, int(n_dipoles))
        self.mask_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, channels)
        x = self.activation(self.fc(x))
        direct = self.direct_out(x)
        lstm_out, _state = self.lstm(x)
        mask = self.mask_activation(self.mask_out(lstm_out))
        return direct * mask


class SolverLSTM(BaseSolver):
    """Class for the Long-Short Term Memory Neural Network (LSTM) for
    EEG inverse solutions.

    """

    meta = SolverMeta(
        acronym="LSTM",
        full_name="Long Short-Term Memory Network",
        category="Neural Networks",
        description=(
            "Supervised recurrent (LSTM) network trained on simulated data to map "
            "sensor time series to source activity."
        ),
        references=["Lukas Hecker 2025, unpublished"],
    )

    def __init__(self, name="LSTM", **kwargs):
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
        n_lstm_units=75,
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
        simulation_config : SimulationConfig
            A SimulationConfig object for data generation containing all
            simulation parameters (batch_size, n_sources, n_orders, snr_range, etc.).
        n_dense_units : int
            The number of neurons in the fully-connected hidden layers.
            Default 300.
        n_lstm_units : int
            The number of neurons in the LSTM hidden layers.
            Default 75.
        activation_function : str
            The activation function of the hidden layers.
            Default "tanh".
        epochs : int
            The number of epochs to train.
            Default 300.
        learning_rate : float
            The learning rate of the optimizer that trains the neural network.
            Default 1e-3.
        loss : str
            The loss function of the neural network.
            Default "cosine_similarity".
        size_validation_set : int
            The size of validation data set.
            Default 256.
        patience : int
            Stopping criterion for the training.
            Default 100.
        alpha : float
            The regularization parameter.
            Default "auto".

        Return
        ------
        self : object returns itself for convenience
        """
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        n_channels, n_dipoles = self.leadfield.shape

        # Store simulation config
        self.simulation_config = simulation_config

        # Store Parameters
        # Architecture
        self.n_lstm_units = n_lstm_units
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
        self.inter_source_correlation = self.simulation_config.inter_source_correlation
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
        y -= y.mean(axis=0)
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
        self.model = _LSTMNet(
            n_channels,
            n_dipoles,
            n_dense_units=int(self.n_dense_units),
            n_lstm_units=int(self.n_lstm_units),
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
        """Create the data generator used for the simulations."""
        # Create SimulationGenerator using the config
        sim_gen = SimulationGenerator(self.forward, config=self.simulation_config)

        # Wrap the generator to transpose and return (x, y) in the right format for LSTM
        def wrapped_generator():
            for x, y, _info in sim_gen.generate():
                # Transpose from (batch, channels, timepoints) to (batch, timepoints, channels)
                x = np.swapaxes(x, 1, 2)
                # Transpose from (batch, dipoles, timepoints) to (batch, timepoints, dipoles)
                y = np.swapaxes(y, 1, 2)

                yield x, y

        self.generator = wrapped_generator()
