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

from sklearn.covariance import OAS

from ...simulate import SimulationGenerator
from ..base import BaseSolver, InverseOperator, SolverMeta
from .torch_utils import (
    activation_from_name,
    count_trainable_parameters,
    get_torch_device,
)

logger = logging.getLogger(__name__)


class _IterCNNNet(nn.Module):
    """CNN with learned iterative refinement on covariance input.

    Architecture:
        Step 1: CovCNN backbone → gamma_1
        Step 2..N: [cov_features, residual_features, gamma_{k-1}] → MLP → delta_gamma_k
                   gamma_k = gamma_{k-1} + step_size * delta_gamma_k

    The refinement MLP shares weights across all steps (unrolled optimization).
    A frozen leadfield tensor is used to compute residual covariances in-graph.
    """

    def __init__(
        self,
        leadfield: np.ndarray,
        *,
        n_outputs: int | None = None,
        n_dense_layers: int = 2,
        n_dense_units: int = 300,
        activation_function: str = "tanh",
        n_refinement_steps: int = 2,
    ) -> None:
        super().__init__()
        n_channels, n_dipoles = leadfield.shape
        if n_outputs is None:
            n_outputs = int(n_dipoles)
        self.n_channels = n_channels
        self.n_dipoles = n_dipoles
        self.n_outputs = n_outputs
        self.n_refinement_steps = n_refinement_steps

        # --- Initial prediction backbone (same as CovCNN) ---
        leadfield_processed = leadfield.astype(np.float32, copy=True)
        leadfield_processed -= leadfield_processed.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(leadfield_processed, axis=0, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        leadfield_processed /= norms

        kernel = torch.from_numpy(leadfield_processed.T).unsqueeze(1).unsqueeze(2)
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=n_dipoles,
            kernel_size=(1, n_channels),
            bias=False,
        )
        with torch.no_grad():
            self.conv.weight.copy_(kernel)
        self.conv.weight.requires_grad = False

        # Initial MLP: conv_features → gamma_1
        hidden_activation = activation_from_name(activation_function)
        conv_feat_size = int(n_dipoles * n_channels)
        init_layers: list[nn.Module] = []
        in_f = conv_feat_size
        for _ in range(int(n_dense_layers)):
            init_layers.append(nn.Linear(in_f, int(n_dense_units)))
            init_layers.append(hidden_activation)
            in_f = int(n_dense_units)
        self.init_mlp = nn.Sequential(*init_layers)
        self.init_out = nn.Linear(in_f, n_outputs)

        # --- Refinement MLP (shared across steps) ---
        # Input: conv_features + residual_features + gamma_{k-1}
        # residual_features = flatten(L.T @ residual_cov) = n_dipoles * n_channels
        refine_in = conv_feat_size + conv_feat_size + n_outputs
        refine_layers: list[nn.Module] = []
        in_f = refine_in
        for _ in range(int(n_dense_layers)):
            refine_layers.append(nn.Linear(in_f, int(n_dense_units)))
            refine_layers.append(hidden_activation)
            in_f = int(n_dense_units)
        self.refine_mlp = nn.Sequential(*refine_layers)
        self.refine_out = nn.Linear(in_f, n_outputs)

        # Learnable step size
        self.step_size = nn.Parameter(torch.tensor(0.5))

        # Frozen leadfield for residual computation (in-graph)
        self.register_buffer(
            "leadfield_t",
            torch.from_numpy(leadfield.astype(np.float32, copy=True)),
        )
        self.register_buffer(
            "leadfield_proc_t",
            torch.from_numpy(leadfield_processed),
        )

    def _compute_conv_features(self, x_cov: torch.Tensor) -> torch.Tensor:
        """Apply frozen conv to covariance and flatten."""
        # x_cov: (batch, 1, n_channels, n_channels)
        out = self.conv(x_cov)  # (batch, n_dipoles, n_channels, 1)
        return out.flatten(start_dim=1)  # (batch, n_dipoles * n_channels)

    def _compute_residual_cov(
        self, x_cov: torch.Tensor, gamma: torch.Tensor
    ) -> torch.Tensor:
        """Compute residual covariance: C_obs - L @ diag(softmax(gamma)) @ L.T

        The residual is re-projected through the frozen conv to get features.
        """
        # gamma: (batch, n_outputs) — raw logits, apply softmax for probabilities
        probs = torch.softmax(gamma, dim=-1)  # (batch, n_dipoles)
        # L @ diag(probs) @ L.T — batched
        L = self.leadfield_t  # (n_channels, n_dipoles)
        # Scale leadfield columns by sqrt(probs) for efficient outer product
        # L_scaled: (batch, n_channels, n_dipoles)
        L_scaled = L.unsqueeze(0) * probs.unsqueeze(1)  # broadcast
        predicted_cov = torch.bmm(L_scaled, L.T.unsqueeze(0).expand(gamma.shape[0], -1, -1))
        # predicted_cov: (batch, n_channels, n_channels)

        # Residual
        observed_cov = x_cov.squeeze(1)  # (batch, n_channels, n_channels)
        residual = observed_cov - predicted_cov
        # Normalize residual
        max_abs = residual.abs().flatten(1).max(dim=1, keepdim=True).values.unsqueeze(-1)
        max_abs = torch.clamp(max_abs, min=1e-8)
        residual = residual / max_abs

        # Project residual through frozen conv
        residual = residual.unsqueeze(1)  # (batch, 1, n_ch, n_ch)
        residual_features = self.conv(residual).flatten(start_dim=1)
        return residual_features

    def forward(
        self, x: torch.Tensor, return_intermediates: bool = False
    ) -> torch.Tensor | list[torch.Tensor]:
        # x: (batch, 1, n_channels, n_channels) — normalized covariance
        conv_features = self._compute_conv_features(x)

        # Step 1: initial prediction
        h = self.init_mlp(conv_features)
        gamma = self.init_out(h)  # logits
        gammas = [gamma]

        # Steps 2..N: iterative refinement
        for _step in range(self.n_refinement_steps):
            residual_features = self._compute_residual_cov(x, gamma)
            refine_input = torch.cat([conv_features, residual_features, gamma], dim=1)
            h_r = self.refine_mlp(refine_input)
            delta = self.refine_out(h_r)
            gamma = gamma + self.step_size * delta
            gammas.append(gamma)

        if return_intermediates:
            return gammas
        return gamma


class SolverIterCNNKL(BaseSolver):
    """CovCNN-KL with learned iterative refinement.

    After an initial gamma prediction from the covariance features, the model
    computes a residual covariance (observed - predicted) and feeds it back
    through a shared refinement MLP. This is repeated for N steps, allowing
    the model to correct its initial estimate and detect secondary sources
    that may have been missed.
    """

    meta = SolverMeta(
        acronym="IterCNN-KL",
        full_name="Iterative CovCNN (KL divergence)",
        category="Neural Networks",
        description=(
            "Supervised ANN with learned iterative refinement on sensor covariance, "
            "trained with KL divergence and deep supervision."
        ),
        references=["Lukas Hecker 2025, unpublished"],
    )

    def __init__(
        self,
        name: str = "IterCNN-KL",
        *,
        reduce_rank: bool = False,
        use_shrinkage: bool = True,
        **kwargs,
    ) -> None:
        self.name = name
        self.use_shrinkage = bool(use_shrinkage)
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
        n_refinement_steps: int = 2,
        alpha: str | float = "auto",
        **kwargs,
    ):
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        self.forward = forward
        self.simulation_config = simulation_config

        self.n_dense_units = int(n_dense_units)
        self.n_dense_layers = int(n_dense_layers)
        self.activation_function = str(activation_function)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.patience = int(patience)
        self.target_power = float(target_power)
        self.temperature = float(temperature)
        self.gamma_power = float(gamma_power)
        self.n_refinement_steps = int(n_refinement_steps)

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

    def _shrinkage_covariance(self, Y: np.ndarray) -> np.ndarray:
        lw = OAS(assume_centered=False)
        return lw.fit(Y.T).covariance_

    def compute_covariance(self, Y: np.ndarray) -> np.ndarray:
        C = Y @ Y.T
        if self.use_shrinkage:
            C = self._shrinkage_covariance(Y)
        return C

    def create_generator(self) -> None:
        sim_gen = SimulationGenerator(self.forward, config=self.simulation_config)

        def wrapped_generator():
            for x, y, _info in sim_gen.generate():
                x_cov = np.stack([self.compute_covariance(xx) for xx in x], axis=0)
                max_abs = np.abs(x_cov).max(axis=(1, 2), keepdims=True)
                max_abs = np.where(max_abs == 0, 1.0, max_abs)
                x_cov = (x_cov / max_abs).astype(np.float32, copy=False)
                x_cov = x_cov[:, np.newaxis, :, :]

                y_abs_mean = np.abs(y).mean(axis=2)
                if self.target_power != 1.0:
                    y_abs_mean = y_abs_mean ** float(self.target_power)
                y_sum = y_abs_mean.sum(axis=1, keepdims=True)
                y_sum = np.where(y_sum == 0, 1.0, y_sum)
                y_dist = (y_abs_mean / y_sum).astype(np.float32, copy=False)

                yield x_cov, y_dist

        self.generator = wrapped_generator()

    def build_model(self) -> None:
        if _TORCH_IMPORT_ERROR is not None:  # pragma: no cover
            raise ImportError(
                "PyTorch is required for neural-network solvers."
            ) from _TORCH_IMPORT_ERROR

        self.device = get_torch_device()
        self.model = _IterCNNNet(
            self.leadfield,
            n_dense_layers=self.n_dense_layers,
            n_dense_units=self.n_dense_units,
            activation_function=self.activation_function,
            n_refinement_steps=self.n_refinement_steps,
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

        # Deep supervision weights: later steps weighted more
        n_steps = self.n_refinement_steps + 1  # initial + refinements
        step_weights = [float(i + 1) / n_steps for i in range(n_steps)]
        weight_sum = sum(step_weights)
        step_weights = [w / weight_sum for w in step_weights]

        for epoch in range(int(self.epochs)):
            self.model.train()
            x_batch, y_batch = next(self.generator)
            x_t = torch.as_tensor(x_batch, dtype=torch.float32, device=device)
            y_t = torch.as_tensor(y_batch, dtype=torch.float32, device=device)

            self.optimizer.zero_grad(set_to_none=True)
            gammas_list = self.model(x_t, return_intermediates=True)

            # Deep supervision: loss at each refinement step
            total_loss = torch.tensor(0.0, device=device)
            for step_idx, gamma_step in enumerate(gammas_list):
                logits = gamma_step / float(self.temperature)
                log_probs = F.log_softmax(logits, dim=-1)
                step_loss = F.kl_div(log_probs, y_t, reduction="batchmean")
                total_loss = total_loss + step_weights[step_idx] * step_loss

            total_loss.backward()
            self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                # Validation uses final output only
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
                    float(total_loss.detach().cpu().item()), val_loss,
                )
            else:
                patience_left -= 1
                if (epoch == 0) or ((epoch + 1) % log_every == 0):
                    logger.info(
                        "Epoch %d/%d - loss=%.6f val_loss=%.6f (patience_left=%d)",
                        epoch + 1, self.epochs,
                        float(total_loss.detach().cpu().item()), val_loss,
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

        C = self.compute_covariance(y)
        max_abs = float(np.abs(C).max())
        if max_abs > 0:
            C = C / max_abs
        C_input = C[np.newaxis, np.newaxis, :, :].astype(np.float32, copy=False)

        assert self.model is not None
        self.model.eval()
        device = self.device or get_torch_device()
        with torch.no_grad():
            logits = self.model(
                torch.as_tensor(C_input, dtype=torch.float32, device=device)
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
