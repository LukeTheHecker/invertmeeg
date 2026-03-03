"""Flex-Champagne: Sparse Bayesian Learning with flexible extent basis functions.

Combines Champagne's iterative sparse Bayesian framework with SSM's multi-order
leadfield dictionary. Each dipole at each smoothness order gets its own gamma,
allowing automatic selection of both location AND spatial extent.
"""

import logging
from copy import deepcopy

import mne
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian

from invert.util import build_source_adjacency

from ..base import BaseSolver, InverseOperator, SolverMeta
from .sbl_core import sbl_iterate

logger = logging.getLogger(__name__)


class SolverFlexChampagne(BaseSolver):
    """Champagne with flexible-extent leadfield dictionary.

    Uses diffusion-smoothed leadfields at multiple orders as the dictionary,
    then runs MacKay-style sparse Bayesian learning to jointly select source
    locations and their spatial extents.
    """

    meta = SolverMeta(
        slug="flex-champagne",
        full_name="Flex-Champagne",
        category="Bayesian",
        description=(
            "Flexible-extent Champagne variant using a multi-order diffusion-smoothed "
            "leadfield dictionary to jointly select source locations and spatial extent."
        ),
        references=[
            "Lukas Hecker 2025, unpublished",
        ],
    )

    def __init__(
        self,
        name="FlexChampagne",
        n_orders=3,
        diffusion_parameter=0.1,
        adjacency_type="spatial",
        adjacency_distance=3e-3,
        update_rule="MacKay",
        noise_init_mode="trace",
        noise_diag_shrinkage=0.0,
        order_pooling="argmax",
        min_active_dipoles=0,
        scale_min_active_with_channels=False,
        min_active_ref_channels=32,
        min_active_max_fraction=0.1,
        min_active_scale_power=0.25,
        min_active_scale_min=0.5,
        min_active_scale_max=1.5,
        **kwargs,
    ):
        self.name = name
        self.n_orders = n_orders
        self.diffusion_parameter = diffusion_parameter
        self.adjacency_type = adjacency_type
        self.adjacency_distance = adjacency_distance
        self.update_rule = update_rule
        self.noise_init_mode = noise_init_mode
        self.noise_diag_shrinkage = float(noise_diag_shrinkage)
        self.order_pooling = order_pooling
        self.min_active_dipoles = int(min_active_dipoles)
        self.scale_min_active_with_channels = bool(scale_min_active_with_channels)
        self.min_active_ref_channels = int(max(min_active_ref_channels, 1))
        self.min_active_max_fraction = float(np.clip(min_active_max_fraction, 0.0, 1.0))
        self.min_active_scale_power = float(min_active_scale_power)
        self.min_active_scale_min = float(min_active_scale_min)
        self.min_active_scale_max = float(max(min_active_scale_max, min_active_scale_min))
        self.is_prepared = False
        super().__init__(**kwargs)

    def make_inverse_operator(
        self,
        forward,
        mne_obj=None,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
        max_iter=2000,
        pruning_thresh=1e-3,
        convergence_criterion=1e-8,
        **kwargs,
    ):
        super().make_inverse_operator(forward, mne_obj, *args, alpha=alpha, **kwargs)
        wf = self.prepare_whitened_forward(noise_cov)
        self.is_prepared = False
        data = self.unpack_data_obj(mne_obj)
        data = wf.sensor_transform @ data

        if not self.is_prepared:
            self._prepare_flex()

        inverse_operator = self._flex_champagne(
            data, pruning_thresh, max_iter, convergence_criterion
        )
        self.inverse_operators = [
            InverseOperator(inverse_operator @ wf.sensor_transform, self.name)
        ]
        return self

    # ------------------------------------------------------------------
    # Prepare multi-order leadfield dictionary (adapted from SSM)
    # ------------------------------------------------------------------
    def _prepare_flex(self):
        n_dipoles = self.leadfield.shape[1]
        I = np.identity(n_dipoles)

        self.leadfields = [deepcopy(self.leadfield)]
        self.gradients = [csr_matrix(I)]

        if self.n_orders == 0:
            self.is_prepared = True
            return

        if self.adjacency_type == "spatial":
            adjacency = build_source_adjacency(
                self.forward["src"],
                adjacency_type="spatial",
                adjacency_distance=self.adjacency_distance,
                verbose=0,
            )
        else:
            adjacency = mne.spatial_dist_adjacency(
                self.forward["src"], self.adjacency_distance, verbose=None
            )

        LL = laplacian(adjacency)
        smoothing_operator = csr_matrix(I - self.diffusion_parameter * LL)

        for i in range(self.n_orders):
            S_i = smoothing_operator ** (i + 1)
            new_lf = self.leadfields[0] @ S_i
            new_grad = self.gradients[0] @ S_i
            self.leadfields.append(new_lf)
            self.gradients.append(new_grad)

        # Normalize gradients row-wise
        for i in range(len(self.gradients)):
            row_sums = self.gradients[i].sum(axis=1).ravel()
            scaling = 1.0 / np.maximum(np.abs(np.asarray(row_sums).ravel()), 1e-12)
            self.gradients[i] = csr_matrix(
                self.gradients[i].multiply(scaling.reshape(-1, 1))
            )

        self.is_prepared = True

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------
    def _flex_champagne(self, Y, pruning_thresh, max_iter, conv_crit):
        n_chans, n_dipoles = self.leadfield.shape
        n_orders = len(self.leadfields)

        # Build extended leadfield: (n_chans, n_orders * n_dipoles)
        L_ext = np.hstack(self.leadfields)  # (n_chans, n_orders * n_dipoles)
        n_ext = L_ext.shape[1]

        # Noise estimate from data covariance
        C_y = self.data_covariance(Y, center=True, ddof=1)
        noise_cov = self._build_noise_cov(
            C_y,
            n_chans,
            mode=self.noise_init_mode,
            diag_shrinkage=self.noise_diag_shrinkage,
        )

        result = sbl_iterate(
            L=L_ext,
            Y=Y,
            noise_cov=noise_cov,
            update_rule=self.update_rule,
            max_iter=max_iter,
            pruning_thresh=pruning_thresh,
            conv_crit=conv_crit,
        )
        active_set = result.active_set
        gammas = result.gammas

        # Reconstruct: map extended gammas back to (order, dipole) pairs
        gammas_full = np.zeros(n_ext)
        gammas_full[active_set] = gammas

        # Select dipole amplitudes from multi-order coefficients.
        gammas_per_order = gammas_full.reshape(n_orders, n_dipoles)
        best_order = np.argmax(gammas_per_order, axis=0)  # (n_dipoles,)
        if self.order_pooling == "sum":
            best_gamma = np.sum(gammas_per_order, axis=0)
        else:
            best_gamma = np.max(gammas_per_order, axis=0)

        # Build final inverse operator using the gradient approach (like SSM)
        # For active dipoles, use their best-order gradient to map back
        active_dipoles = self._select_active_dipoles(
            best_gamma,
            pruning_thresh,
            n_chans,
            n_dipoles,
            min_active_dipoles=self.min_active_dipoles,
            scale_min_active_with_channels=self.scale_min_active_with_channels,
        )

        if len(active_dipoles) == 0:
            return np.zeros((n_dipoles, n_chans))

        # Build reduced leadfield and source covariance
        L_reduced = np.stack(
            [self.leadfields[best_order[d]][:, d] for d in active_dipoles], axis=1
        )
        gamma_reduced = best_gamma[active_dipoles]
        Gamma_r = np.diag(gamma_reduced)

        # Gradient matrix: maps reduced sources back to full source space
        grad_cols = []
        for d in active_dipoles:
            g = self.gradients[best_order[d]][d].toarray().ravel()
            grad_cols.append(g)
        G = np.stack(grad_cols, axis=1)  # (n_dipoles, n_active)

        # Inverse operator: G @ Gamma_r @ L_reduced.T @ inv(L_reduced @ Gamma_r @ L_reduced.T + noise)
        Sigma_y_r = noise_cov + (L_reduced * gamma_reduced) @ L_reduced.T
        Sigma_y_r = 0.5 * (Sigma_y_r + Sigma_y_r.T)
        Sigma_y_r_inv = self._robust_inv(Sigma_y_r)
        inv_op = G @ Gamma_r @ L_reduced.T @ Sigma_y_r_inv

        return inv_op

    def _build_noise_cov(
        self,
        C_y: np.ndarray,
        n_chans: int,
        *,
        mode: str | None = None,
        diag_shrinkage: float | None = None,
    ) -> np.ndarray:
        mode = str(self.noise_init_mode if mode is None else mode).lower()
        if mode == "diag":
            diag = np.maximum(np.diag(C_y), 1e-12)
            shrink_base = self.noise_diag_shrinkage if diag_shrinkage is None else diag_shrinkage
            shrink = float(np.clip(shrink_base, 0.0, 1.0))
            mean_diag = float(np.mean(diag))
            diag = (1.0 - shrink) * diag + shrink * mean_diag
            return np.diag(diag)
        alpha_noise = float(np.trace(C_y) / n_chans)
        return alpha_noise * np.identity(n_chans)

    def _select_active_dipoles(
        self,
        best_gamma: np.ndarray,
        pruning_thresh: float,
        n_chans: int,
        n_dipoles: int,
        *,
        min_active_dipoles: int | None = None,
        scale_min_active_with_channels: bool | None = None,
    ) -> np.ndarray:
        if best_gamma.size == 0:
            return np.array([], dtype=int)
        max_gamma = float(np.max(best_gamma))
        if max_gamma <= 0:
            return np.array([], dtype=int)
        active = np.where(best_gamma > pruning_thresh * max_gamma)[0]
        min_active = (
            self.min_active_dipoles
            if min_active_dipoles is None
            else int(min_active_dipoles)
        )
        scale_with_channels = (
            self.scale_min_active_with_channels
            if scale_min_active_with_channels is None
            else bool(scale_min_active_with_channels)
        )
        if min_active > 0 and scale_with_channels:
            ratio = max(float(n_chans), 1.0) / float(self.min_active_ref_channels)
            scale = ratio**self.min_active_scale_power
            scale = float(np.clip(scale, self.min_active_scale_min, self.min_active_scale_max))
            min_active = int(np.ceil(min_active * scale))
        if self.min_active_max_fraction > 0:
            max_allowed = int(np.ceil(self.min_active_max_fraction * max(int(n_dipoles), 1)))
            if max_allowed > 0:
                min_active = min(min_active, max_allowed)

        if min_active > 0 and len(active) < min_active:
            k = min(min_active, best_gamma.size)
            active = np.argsort(best_gamma)[-k:]
        return np.asarray(np.sort(active), dtype=int)

    @staticmethod
    def _robust_inv(M):
        try:
            return np.linalg.inv(M)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(M)
