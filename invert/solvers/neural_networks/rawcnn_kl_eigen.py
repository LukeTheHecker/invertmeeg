from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

import mne
import numpy as np

_TORCH_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    import torch  # type: ignore[import-not-found]
    import torch.nn as nn  # type: ignore[import-not-found]
    import torch.nn.functional as F  # type: ignore[import-not-found]
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc

from ...simulate import SimulationGenerator
from ..base import BaseSolver, InverseOperator, SolverMeta
from .torch_utils import (
    activation_from_name,
    count_trainable_parameters,
    get_torch_device,
)

logger = logging.getLogger(__name__)


def _svd_reduce(Y: np.ndarray, n_components: int) -> np.ndarray:
    """Reduce Y (n_channels, n_timepoints) to top-K SVD components.

    Returns Y_reduced: (n_channels, n_components) — the left singular
    vectors scaled by singular values.
    """
    U, s, _Vt = np.linalg.svd(Y, full_matrices=False)
    k = min(n_components, len(s))
    # U_k * S_k: (n_channels, k)
    return (U[:, :k] * s[:k]).astype(np.float32, copy=False)


class _RawCNNEigenNet(nn.Module):
    """Network operating on SVD-reduced EEG components.

    Architecture:
        Input: (batch, 1, n_channels, n_components)
        → Frozen conv2d: L.T @ input → (batch, n_dipoles, 1, n_components)
        → Flatten → (batch, n_dipoles * n_components)
        → MLP → logits
    """

    def __init__(
        self,
        leadfield: np.ndarray,
        n_components: int,
        *,
        n_outputs: int | None = None,
        n_dense_layers: int = 2,
        n_dense_units: int = 300,
        activation_function: str = "tanh",
        output_activation: str = "linear",
    ) -> None:
        super().__init__()
        n_channels, n_dipoles = leadfield.shape
        if n_outputs is None:
            n_outputs = int(n_dipoles)

        # Frozen conv: L.T applied per component
        leadfield_processed = leadfield.astype(np.float32, copy=True)
        leadfield_processed -= leadfield_processed.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(leadfield_processed, axis=0, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        leadfield_processed /= norms

        kernel = torch.from_numpy(leadfield_processed.T).unsqueeze(1).unsqueeze(3)
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=n_dipoles,
            kernel_size=(n_channels, 1),
            bias=False,
        )
        with torch.no_grad():
            self.conv.weight.copy_(kernel)
        self.conv.weight.requires_grad = False

        # MLP
        mlp_layers: list[nn.Module] = []
        in_features = int(n_dipoles * n_components)
        hidden_activation = activation_from_name(activation_function)
        for _ in range(int(n_dense_layers)):
            mlp_layers.append(nn.Linear(in_features, int(n_dense_units)))
            mlp_layers.append(hidden_activation)
            in_features = int(n_dense_units)

        self.mlp = nn.Sequential(*mlp_layers)
        self.out = nn.Linear(in_features, int(n_outputs))
        self.out_activation = activation_from_name(output_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, n_channels, n_components)
        x = self.conv(x)  # (batch, n_dipoles, 1, n_components)
        x = x.flatten(start_dim=1)  # (batch, n_dipoles * n_components)
        x = self.mlp(x)
        x = self.out(x)
        return self.out_activation(x)


class SolverRawCNNKLEigen(BaseSolver):
    """Raw-data CNN with SVD eigenspace reduction.

    Applies SVD to the raw EEG data, keeps the top-K components (denoising),
    then projects through the frozen leadfield into source space. This gives
    a compact (n_dipoles x K) feature representation that preserves temporal
    structure while removing noise dimensions.
    """

    meta = SolverMeta(
        acronym="RawCNN-KL-Eigen",
        full_name="Raw-data CNN + Eigenspace (KL divergence)",
        category="Neural Networks",
        description=(
            "Supervised ANN on SVD-reduced EEG data trained with KL divergence. "
            "Denoises input via eigenspace reduction."
        ),
        references=["Lukas Hecker 2025, unpublished"],
    )

    def __init__(
        self,
        name: str = "RawCNN-KL-Eigen",
        *,
        reduce_rank: bool = False,
        **kwargs,
    ) -> None:
        self.name = name
        self.model: Any = None
        self.optimizer: Any = None
        self.device: Any = None
        self.generator: Any = None
        return super().__init__(reduce_rank=reduce_rank, **kwargs)

    def make_inverse_operator(
        self,
        forward,
        simulation_config,
        *args,
        n_dense_units: int = 300,
        n_dense_layers: int = 2,
        activation_function: str = "tanh",
        epochs: int = 500,
        learning_rate: float = 1e-3,
        patience: int = 300,
        target_power: float = 0.5,
        temperature: float = 1.0,
        gamma_power: float = 1.5,
        n_components: int = 8,
        alpha: str | float = "auto",
        **kwargs,
    ):
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        self.forward = forward
        self.simulation_config = simulation_config
        self.n_components = int(n_components)

        self.n_dense_units = int(n_dense_units)
        self.n_dense_layers = int(n_dense_layers)
        self.activation_function = str(activation_function)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.patience = int(patience)
        self.target_power = float(target_power)
        self.temperature = float(temperature)
        self.gamma_power = float(gamma_power)

        logger.info("Create generator...")
        self.create_generator()
        logger.info("Build model...")
        self.build_model()
        logger.info("Train model...")
        self.train_model()

        self.inverse_operators: list = []
        return self

    def apply_inverse_operator(self, mne_obj, prior=None) -> mne.SourceEstimate:
        data = self.unpack_data_obj(mne_obj)
        source_mat = self.apply_model(data, prior=prior)
        return self.source_to_object(source_mat)

    def create_generator(self) -> None:
        sim_gen = SimulationGenerator(self.forward, config=self.simulation_config)
        n_components = self.n_components

        def wrapped_generator():
            for x, y, _info in sim_gen.generate():
                # x: (batch, n_channels, n_timepoints)
                # SVD reduce each sample
                x_reduced = np.stack(
                    [_svd_reduce(x[i], n_components) for i in range(x.shape[0])],
                    axis=0,
                )  # (batch, n_channels, n_components)

                # Normalize per sample
                max_abs = np.abs(x_reduced).max(axis=(1, 2), keepdims=True)
                max_abs = np.where(max_abs == 0, 1.0, max_abs)
                x_norm = (x_reduced / max_abs).astype(np.float32, copy=False)
                x_norm = x_norm[:, np.newaxis, :, :]  # (batch, 1, n_ch, n_comp)

                # Target
                y_abs_mean = np.abs(y).mean(axis=2)
                if self.target_power != 1.0:
                    y_abs_mean = y_abs_mean ** float(self.target_power)
                y_sum = y_abs_mean.sum(axis=1, keepdims=True)
                y_sum = np.where(y_sum == 0, 1.0, y_sum)
                y_dist = (y_abs_mean / y_sum).astype(np.float32, copy=False)

                yield x_norm, y_dist

        self.generator = wrapped_generator()

    def build_model(self) -> None:
        if _TORCH_IMPORT_ERROR is not None:  # pragma: no cover
            raise ImportError(
                "PyTorch is required for neural-network solvers."
            ) from _TORCH_IMPORT_ERROR

        self.device = get_torch_device()
        self.model = _RawCNNEigenNet(
            self.leadfield,
            self.n_components,
            n_dense_layers=self.n_dense_layers,
            n_dense_units=self.n_dense_units,
            activation_function=self.activation_function,
            output_activation="linear",
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        logger.info(
            "Total trainable parameters: %d",
            count_trainable_parameters(self.model),
        )

    def train_model(self) -> None:
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Model not initialized.")

        self.create_generator()

        x_val, y_val = next(self.generator)
        device = self.device or get_torch_device()
        x_val_t = torch.as_tensor(x_val, dtype=torch.float32, device=device)
        y_val_t = torch.as_tensor(y_val, dtype=torch.float32, device=device)

        best_val = float("inf")
        best_state = None
        patience_left = int(self.patience)
        log_every = 10

        for epoch in range(int(self.epochs)):
            self.model.train()
            x_batch, y_batch = next(self.generator)
            x_t = torch.as_tensor(x_batch, dtype=torch.float32, device=device)
            y_t = torch.as_tensor(y_batch, dtype=torch.float32, device=device)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(x_t) / float(self.temperature)
            log_probs = F.log_softmax(logits, dim=-1)
            loss = F.kl_div(log_probs, y_t, reduction="batchmean")
            loss.backward()
            self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                v_logits = self.model(x_val_t) / float(self.temperature)
                v_log_probs = F.log_softmax(v_logits, dim=-1)
                val_loss = float(
                    F.kl_div(v_log_probs, y_val_t, reduction="batchmean").cpu().item()
                )

            if val_loss < best_val:
                best_val = val_loss
                best_state = deepcopy(self.model.state_dict())
                patience_left = int(self.patience)
                logger.info(
                    "Epoch %d/%d - loss=%.6f val_loss=%.6f (new best)",
                    epoch + 1, self.epochs,
                    float(loss.detach().cpu().item()), val_loss,
                )
            else:
                patience_left -= 1
                if (epoch == 0) or ((epoch + 1) % log_every == 0):
                    logger.info(
                        "Epoch %d/%d - loss=%.6f val_loss=%.6f (patience_left=%d)",
                        epoch + 1, self.epochs,
                        float(loss.detach().cpu().item()), val_loss,
                        patience_left,
                    )
                if patience_left <= 0:
                    logger.info(
                        "Early stopping at epoch %d/%d (best_val=%.6f)",
                        epoch + 1, self.epochs, best_val,
                    )
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()

    def apply_model(self, data: np.ndarray, prior=None) -> np.ndarray:
        y = deepcopy(data)
        y = y - y.mean(axis=1, keepdims=True)
        n_channels, _n_times = y.shape

        # SVD reduce
        y_reduced = _svd_reduce(y, self.n_components)  # (n_channels, n_components)
        max_abs = float(np.abs(y_reduced).max())
        if max_abs > 0:
            y_reduced = y_reduced / max_abs
        y_input = y_reduced[np.newaxis, np.newaxis, :, :].astype(np.float32, copy=False)

        assert self.model is not None
        self.model.eval()
        device = self.device or get_torch_device()
        with torch.no_grad():
            logits = self.model(
                torch.as_tensor(y_input, dtype=torch.float32, device=device)
            ) / float(self.temperature)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]

        max_p = float(np.max(probs))
        gammas = probs / max_p if max_p > 0 else probs
        if getattr(self, "gamma_power", 1.0) != 1.0:
            gammas = gammas ** float(self.gamma_power)
            max_gamma = float(np.max(gammas))
            if max_gamma > 0:
                gammas = gammas / max_gamma

        if prior is not None:
            prior = np.asarray(prior, dtype=float)
            prior_max = float(np.max(prior))
            if prior_max > 0:
                gammas = gammas * (prior / prior_max)

        source_covariance = np.diag(gammas.astype(np.float64, copy=False))
        Sigma_y = self.leadfield @ source_covariance @ self.leadfield.T

        if self.alpha == "auto":
            r_grid = np.asarray(self.r_values, dtype=float)
        else:
            r_grid = np.asarray([float(self.alpha)], dtype=float)
        self.alphas = list(r_grid)

        inverse_ops = []
        trace_Sy = float(np.trace(Sigma_y))
        if not np.isfinite(trace_Sy) or trace_Sy <= 0:
            trace_Sy = 1.0
        for r in r_grid:
            reg_term = float(r) * trace_Sy / float(n_channels)
            inv = np.linalg.inv(Sigma_y + reg_term * np.eye(n_channels))
            W = source_covariance @ self.leadfield.T @ inv
            inverse_ops.append(W)

        self.inverse_operators = [InverseOperator(op, self.name) for op in inverse_ops]
        x_hat, _ = self.regularise_gcv(y)
        return x_hat
