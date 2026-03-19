"""DeepSIF solver — spatiotemporal source imaging via ResNet + LSTM.

Implements the TemporalInverseNet architecture from:
  Sun et al., "Deep Neural Networks Constrained by Neural Mass Models
  Improve Electrophysiological Source Imaging of Spatiotemporal Brain
  Dynamics", PNAS 119(31), 2022.
  https://github.com/bfinl/DeepSIF
"""

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
from .torch_utils import count_trainable_parameters, get_torch_device

logger = logging.getLogger(__name__)

_META = SolverMeta(
    acronym="DeepSIF",
    full_name="Deep Learning Source Imaging Framework",
    category="Neural Networks",
    description=(
        "Spatiotemporal source imaging via a spatial ResNet (FC) module "
        "followed by a multi-layer LSTM temporal module. Trained with MSE "
        "on simulated data."
    ),
    references=[
        "Sun R, Sohrabpour A, Bhatt Worber T, He B. Deep neural networks "
        "constrained by neural mass models improve electrophysiological "
        "source imaging of spatiotemporal brain dynamics. PNAS 119(31), "
        "e2201128119, 2022.",
    ],
)


if _TORCH_IMPORT_ERROR is not None:  # pragma: no cover

    class SolverDeepSIF(BaseSolver):
        """DeepSIF solver (requires PyTorch)."""

        meta = _META

        def __init__(self, name="DeepSIF", **kwargs):
            self.name = name
            super().__init__(**kwargs)

        def make_inverse_operator(self, *args, **kwargs):
            raise ImportError(
                "SolverDeepSIF requires PyTorch. "
                'Install it via `pip install "invertmeeg[ann]"` (or install `torch` directly).'
            ) from _TORCH_IMPORT_ERROR

        def apply_inverse_operator(self, *args, **kwargs):
            raise ImportError(
                "SolverDeepSIF requires PyTorch. "
                'Install it via `pip install "invertmeeg[ann]"` (or install `torch` directly).'
            ) from _TORCH_IMPORT_ERROR

else:

    class _MLPSpatialFilter(nn.Module):
        """Spatial module: two residual FC blocks.

        Block 1: n_channels → n_channels (with skip connection)
        Block 2: n_channels → n_hidden  (with learned skip projection)
        Output:  n_hidden via a value layer + activation
        """

        def __init__(self, n_channels: int, n_hidden: int) -> None:
            super().__init__()
            self.fc11 = nn.Linear(n_channels, n_channels)
            self.fc12 = nn.Linear(n_channels, n_channels)
            self.fc21 = nn.Linear(n_channels, n_hidden)
            self.fc22 = nn.Linear(n_hidden, n_hidden)
            self.fc23 = nn.Linear(n_channels, n_hidden)  # skip projection
            self.value = nn.Linear(n_hidden, n_hidden)
            self.activation = nn.ELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, time, n_channels)
            x = self.activation(self.fc12(self.activation(self.fc11(x))) + x)
            x = self.activation(
                self.fc22(self.activation(self.fc21(x))) + self.fc23(x)
            )
            return self.activation(self.value(x))

    class _TemporalFilter(nn.Module):
        """Temporal module: multi-layer LSTM."""

        def __init__(
            self, input_size: int, n_sources: int, num_layers: int
        ) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size,
                n_sources,
                num_layers=num_layers,
                batch_first=True,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, time, input_size)
            self.lstm.flatten_parameters()
            out, _ = self.lstm(x)
            return out  # (batch, time, n_sources)

    class _TemporalInverseNet(nn.Module):
        """DeepSIF: spatial ResNet → LSTM temporal module."""

        def __init__(
            self,
            n_channels: int,
            n_sources: int,
            *,
            n_hidden: int = 500,
            rnn_layers: int = 3,
        ) -> None:
            super().__init__()
            self.spatial = _MLPSpatialFilter(n_channels, n_hidden)
            self.temporal = _TemporalFilter(n_hidden, n_sources, rnn_layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, time, n_channels)
            x = self.spatial(x)  # (batch, time, n_hidden)
            return self.temporal(x)  # (batch, time, n_sources)

    class SolverDeepSIF(BaseSolver):  # type: ignore[no-redef]
        """DeepSIF: spatiotemporal source imaging via ResNet + LSTM."""

        meta = _META

        def __init__(self, name="DeepSIF", **kwargs):
            self.name = name
            self.model = None
            self.optimizer = None
            self.device = None
            self.generator = None
            super().__init__(**kwargs)

        def make_inverse_operator(
            self,
            forward,
            simulation_config,
            *args,
            n_hidden: int = 500,
            rnn_layers: int = 3,
            epochs: int = 300,
            learning_rate: float = 3e-4,
            weight_decay: float = 1e-6,
            patience: int = 100,
            noise_cov: mne.Covariance | None = None,
            alpha: str | float = "auto",
            **kwargs,
        ):
            super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
            self.prepare_whitened_forward(noise_cov)
            self.simulation_config = simulation_config

            self.n_hidden = int(n_hidden)
            self.rnn_layers = int(rnn_layers)
            self.epochs = int(epochs)
            self.learning_rate = float(learning_rate)
            self.weight_decay = float(weight_decay)
            self.patience = int(patience)

            logger.info("Create generator…")
            self.create_generator()
            logger.info("Build model…")
            self.build_model()
            logger.info("Train model…")
            self.train_model()

            self.inverse_operators = []
            return self

        def apply_inverse_operator(self, mne_obj) -> mne.SourceEstimate:
            data = self.unpack_data_obj(mne_obj)
            self.validate_operator_data_compatibility(data)
            data = self._sensor_transform @ data
            source_mat = self.apply_model(data)
            return self.source_to_object(source_mat)

        def apply_model(self, data: np.ndarray) -> np.ndarray:
            y = deepcopy(data)
            # DeepSIF normalization: subtract temporal mean, channel mean,
            # divide by max abs
            y = y - y.mean(axis=1, keepdims=True)  # temporal mean per channel
            y = y - y.mean(axis=0, keepdims=True)  # channel mean per time
            max_abs = float(np.abs(y).max())
            if max_abs > 0:
                y = y / max_abs

            # (n_channels, n_times) → (1, n_times, n_channels)
            x_in = y.T[np.newaxis, :, :].astype(np.float32)

            assert self.model is not None
            self.model.eval()
            device = self.device or get_torch_device()
            with torch.no_grad():
                out = (
                    self.model(
                        torch.as_tensor(x_in, dtype=torch.float32, device=device)
                    )
                    .detach()
                    .cpu()
                    .numpy()[0]
                )
            # out: (n_times, n_sources) → (n_sources, n_times)
            return out.T

        def create_generator(self) -> None:
            sim_gen = SimulationGenerator(
                self.forward, config=self.simulation_config
            )
            W = self._sensor_transform  # (n_whitened, n_original) or None

            def wrapped_generator():
                for batch in sim_gen.generate():
                    x, y = batch.x, batch.y
                    # x: (batch, n_channels_orig, n_timepoints)
                    # y: (batch, n_sources, n_timepoints)

                    # Apply whitening to match inference path
                    if W is not None:
                        x = np.einsum("ij,bjt->bit", W, x)

                    # DeepSIF normalization per sample
                    x = x - x.mean(axis=2, keepdims=True)  # temporal mean
                    x = x - x.mean(axis=1, keepdims=True)  # channel mean
                    max_abs = np.abs(x).max(axis=(1, 2), keepdims=True)
                    max_abs = np.where(max_abs == 0, 1.0, max_abs)
                    x = (x / max_abs).astype(np.float32, copy=False)

                    # Normalize source targets per sample
                    y_max = np.abs(y).max(axis=(1, 2), keepdims=True)
                    y_max = np.where(y_max == 0, 1.0, y_max)
                    y = (y / y_max).astype(np.float32, copy=False)

                    # Transpose to (batch, time, channels/sources)
                    x = np.swapaxes(x, 1, 2)
                    y = np.swapaxes(y, 1, 2)

                    yield x, y

            self.generator = wrapped_generator()

        def build_model(self) -> None:
            n_channels, n_dipoles = self.leadfield.shape
            self.device = get_torch_device()
            self.model = _TemporalInverseNet(
                n_channels,
                n_dipoles,
                n_hidden=self.n_hidden,
                rnn_layers=self.rnn_layers,
            ).to(self.device)
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            logger.info(
                "Total trainable parameters: %d",
                count_trainable_parameters(self.model),
            )

        def train_model(self) -> None:
            if self.model is None:
                raise RuntimeError(
                    "Model not initialized. Call build_model() first."
                )

            self.create_generator()
            criterion = torch.nn.MSELoss(reduction="sum")
            device = self.device or get_torch_device()

            # Validation batch
            x_val, y_val = next(self.generator)
            x_val_t = torch.as_tensor(x_val, dtype=torch.float32, device=device)
            y_val_t = torch.as_tensor(y_val, dtype=torch.float32, device=device)

            best_val = float("inf")
            best_state = None
            patience_left = int(self.patience)
            log_every = 10

            for epoch in range(self.epochs):
                self.model.train()
                x_batch, y_batch = next(self.generator)
                x_t = torch.as_tensor(
                    x_batch, dtype=torch.float32, device=device
                )
                y_t = torch.as_tensor(
                    y_batch, dtype=torch.float32, device=device
                )

                self.optimizer.zero_grad(set_to_none=True)
                out = self.model(x_t)
                loss = criterion(out, y_t)
                loss.backward()
                self.optimizer.step()

                self.model.eval()
                with torch.no_grad():
                    val_out = self.model(x_val_t)
                    val_loss = float(criterion(val_out, y_val_t).cpu().item())

                train_loss = float(loss.detach().cpu().item())

                if val_loss < best_val:
                    best_val = val_loss
                    best_state = deepcopy(self.model.state_dict())
                    patience_left = int(self.patience)
                    logger.info(
                        "Epoch %d/%d - loss=%.4f val_loss=%.4f (new best)",
                        epoch + 1,
                        self.epochs,
                        train_loss,
                        val_loss,
                    )
                else:
                    patience_left -= 1
                    if (epoch == 0) or ((epoch + 1) % log_every == 0):
                        logger.info(
                            "Epoch %d/%d - loss=%.4f val_loss=%.4f (patience=%d)",
                            epoch + 1,
                            self.epochs,
                            train_loss,
                            val_loss,
                            patience_left,
                        )
                    if patience_left <= 0:
                        logger.info(
                            "Early stopping at epoch %d (best_val=%.4f)",
                            epoch + 1,
                            best_val,
                        )
                        break

            if best_state is not None:
                self.model.load_state_dict(best_state)
            self.model.eval()
