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
    nn = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc

from ...simulate import SimulationGenerator
from ..base import BaseSolver, SolverMeta

logger = logging.getLogger(__name__)

_META = SolverMeta(
    acronym="CNN",
    full_name="Convolutional Neural Network",
    category="Neural Networks",
    description=(
        "Supervised CNN that maps sensor time series to source activity using "
        "simulated training data."
    ),
    references=["Lukas Hecker 2025, unpublished"],
)


if _TORCH_IMPORT_ERROR is not None:  # pragma: no cover

    class SolverCNN(BaseSolver):
        """Convolutional Neural Network solver (requires PyTorch)."""

        meta = _META

        def __init__(self, name="CNN", **kwargs):
            self.name = name
            super().__init__(**kwargs)

        def make_inverse_operator(self, *args, **kwargs):
            raise ImportError(
                "SolverCNN requires PyTorch. "
                'Install it via `pip install \"invertmeeg[ann]\"` (or install `torch` directly).'
            ) from _TORCH_IMPORT_ERROR

        def apply_inverse_operator(self, *args, **kwargs):
            raise ImportError(
                "SolverCNN requires PyTorch. "
                'Install it via `pip install \"invertmeeg[ann]\"` (or install `torch` directly).'
            ) from _TORCH_IMPORT_ERROR

else:
    from .torch_utils import (
        activation_from_name,
        count_trainable_parameters,
        get_torch_device,
        loss_from_name,
    )

    class _CNNNet(nn.Module):
        def __init__(
            self,
            n_channels: int,
            n_dipoles: int,
            *,
            n_filters: int,
            activation_function: str,
        ) -> None:
            super().__init__()
            self.encoder = nn.Linear(int(n_channels), int(n_filters))
            self.activation = activation_from_name(activation_function)
            self.lstm = nn.LSTM(
                input_size=int(n_filters),
                hidden_size=128,
                batch_first=True,
                bidirectional=True,
            )
            self.out = nn.Linear(128 * 2, int(n_dipoles))
            self.out_activation = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, time, channels)
            x = self.activation(self.encoder(x))
            _out, (h_n, _c_n) = self.lstm(x)
            h_fwd = h_n[-2]
            h_bwd = h_n[-1]
            features = torch.cat([h_fwd, h_bwd], dim=1)
            return self.out_activation(self.out(features))

    class SolverCNN(BaseSolver):  # type: ignore[no-redef]
        """Class for the Convolutional Neural Network (CNN) for EEG inverse solutions.

"""

        meta = _META

        def __init__(self, name="CNN", **kwargs):
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
        n_filters="auto",
        activation_function="tanh",
        epochs=300,
        learning_rate=1e-3,
        loss="cosine_similarity",
        size_validation_set=256,
        epsilon=0.25,
        patience=300,
        alpha="auto",
        **kwargs,
    ):
        """Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        simulation_config : SimulationConfig
            A SimulationConfig object for data generation.
        n_filters : int
            Number of filters in the convolution layer.
        activation_function : str
            The activation function of the hidden layers.
        epochs : int
            The number of epochs to train.
        learning_rate : float
            The learning rate of the optimizer that trains the neural network.
        loss : str
            The loss function of the neural network.
        size_validation_set : int
            The size of validation data set.
        epsilon : float
            The threshold at which to select sources as "active". 0.25 -> select
            all sources that are active at least 25 % of the maximum dipoles.
        patience : int
            Stopping criterion for the training.
        alpha : float
            The regularization parameter.

        Return
        ------
        self : object returns itself for convenience
        """
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        n_channels, n_dipoles = self.leadfield.shape

        if n_filters == "auto":
            n_filters = int(n_channels * 4)

        # Store simulation config
        self.simulation_config = simulation_config

        # Store Parameters
        # Architecture
        self.n_filters = n_filters
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
        # Inference
        self.epsilon = epsilon

        self.create_generator()
        self.build_model()
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
        n_channels, n_times = y.shape

        # Scaling
        y /= np.linalg.norm(y, axis=0)
        y /= np.max(abs(y))
        # Reshape for keras model
        y = y.T[np.newaxis, :, :, np.newaxis]

        # Add empty batch and (color-) channel dimension
        assert self.model is not None
        self.model.eval()
        device = self.device or get_torch_device()
        with torch.no_grad():
            gammas = (
                self.model(torch.as_tensor(y[..., 0], dtype=torch.float32, device=device))
                .detach()
                .cpu()
                .numpy()[0]
            )
        gammas /= gammas.max()

        # Select dipole indices
        gammas[gammas < self.epsilon] = 0
        dipole_idc = np.where(gammas != 0)[0]
        logger.info("Active dipoles: %d", len(dipole_idc))

        # 1) Calculate weighted minimum norm solution at active dipoles
        n_dipoles = len(gammas)
        y = deepcopy(data)
        y -= y.mean(axis=0)
        x_hat = np.zeros((n_dipoles, n_times))
        L = self.leadfield[:, dipole_idc]
        W = np.diag(np.linalg.norm(L, axis=0))
        x_hat[dipole_idc, :] = np.linalg.inv(L.T @ L + W.T @ W) @ L.T @ y

        return x_hat

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
        self.model = _CNNNet(
            n_channels,
            n_dipoles,
            n_filters=int(self.n_filters),
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
                # Match apply_model() preprocessing (CAR + per-time norm + per-sample max abs)
                x = x - x.mean(axis=1, keepdims=True)
                norms = np.linalg.norm(x, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                x = x / norms
                max_abs = np.abs(x).max(axis=(1, 2), keepdims=True)
                max_abs = np.where(max_abs == 0, 1.0, max_abs)
                x = (x / max_abs).astype(np.float32, copy=False)

                # (batch, channels, time) -> (batch, time, channels)
                x = np.swapaxes(x, 1, 2)

                # Target is a per-dipole activity summary (batch, dipoles)
                y_abs_mean = np.abs(y).mean(axis=2)
                y_scale = y_abs_mean.max(axis=1, keepdims=True)
                y_scale = np.where(y_scale == 0, 1.0, y_scale)
                y_continuous = (y_abs_mean / y_scale).astype(np.float32, copy=False)

                yield x, y_continuous

        self.generator = wrapped_generator()
