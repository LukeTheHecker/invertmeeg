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

from ..base import BaseSolver, InverseOperator, SolverMeta

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
        **kwargs,
    ):
        self.name = name
        self.n_orders = n_orders
        self.diffusion_parameter = diffusion_parameter
        self.adjacency_type = adjacency_type
        self.adjacency_distance = adjacency_distance
        self.update_rule = update_rule
        self.is_prepared = False
        super().__init__(**kwargs)

    def make_inverse_operator(
        self,
        forward,
        mne_obj,
        *args,
        alpha="auto",
        max_iter=2000,
        pruning_thresh=1e-3,
        convergence_criterion=1e-8,
        **kwargs,
    ):
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)

        if not self.is_prepared:
            self._prepare_flex()

        inverse_operator = self._flex_champagne(
            data, pruning_thresh, max_iter, convergence_criterion
        )
        self.inverse_operators = [InverseOperator(inverse_operator, self.name)]
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
            adjacency = mne.spatial_src_adjacency(self.forward["src"], verbose=0)
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
        n_times = Y.shape[1]
        n_orders = len(self.leadfields)

        # Build extended leadfield: (n_chans, n_orders * n_dipoles)
        # Column-normalise each order's leadfield for stable convergence
        L_blocks = []
        for lf in self.leadfields:
            lf_norm = lf / (np.linalg.norm(lf, axis=0, keepdims=True) + 1e-12)
            L_blocks.append(lf_norm)
        L_ext = np.hstack(L_blocks)  # (n_chans, n_orders * n_dipoles)
        n_ext = L_ext.shape[1]

        # Scale data for numerical stability (does not affect spatial metrics)
        scale = float(np.mean(np.abs(Y))) + 1e-12
        Y_scaled = Y / scale

        # Noise estimate
        C_y = self.data_covariance(Y_scaled, center=True, ddof=1)
        alpha_noise = float(np.trace(C_y) / (n_chans * 100))
        I_c = np.identity(n_chans)
        noise_cov = alpha_noise * I_c

        # Gammas: one per extended-dictionary atom
        gammas = np.ones(n_ext)
        active_set = np.arange(n_ext)
        L = deepcopy(L_ext)

        Sigma_y = noise_cov + (L * gammas) @ L.T
        Sigma_y = 0.5 * (Sigma_y + Sigma_y.T)
        Sigma_y_inv = self._robust_inv(Sigma_y)
        mu_x = (L.T @ Sigma_y_inv @ Y_scaled) * gammas[:, None]

        loss_list = []

        for _i_iter in range(max_iter):
            old_gammas = deepcopy(gammas)

            # MacKay update
            upper = np.mean(mu_x**2, axis=1)
            L_Sigma = Sigma_y_inv @ L
            z_diag = np.sum(L * L_Sigma, axis=0)

            if self.update_rule == "MacKay":
                gammas = upper / (gammas * z_diag + 1e-20)
            elif self.update_rule == "Convexity":
                gammas = np.sqrt(upper / (z_diag + 1e-20))
            else:
                gammas = upper / (gammas * z_diag + 1e-20)

            gammas[~np.isfinite(gammas)] = 0.0
            gammas = np.maximum(gammas, 0.0)

            if np.linalg.norm(gammas) == 0:
                gammas = old_gammas
                break

            # Pruning
            thresh = pruning_thresh * gammas.max()
            keep = np.where(gammas > thresh)[0]
            if len(keep) == 0:
                gammas = old_gammas
                break
            active_set = active_set[keep]
            gammas = gammas[keep]
            L = L[:, keep]

            Sigma_y = noise_cov + (L * gammas) @ L.T
            Sigma_y = 0.5 * (Sigma_y + Sigma_y.T)
            Sigma_y_inv = self._robust_inv(Sigma_y)
            mu_x = (L.T @ Sigma_y_inv @ Y_scaled) * gammas[:, None]

            # Loss
            data_fit = np.trace(Sigma_y_inv @ Y_scaled @ Y_scaled.T) / n_times
            eigvals = np.linalg.eigvalsh(Sigma_y)
            log_det = float(np.sum(np.log(np.maximum(eigvals, 1e-20))))
            loss = float(data_fit + log_det)
            loss_list.append(loss)

            if len(loss_list) > 1:
                rel_change = (loss_list[-2] - loss) / (abs(loss_list[-2]) + 1e-20)
                if rel_change > 0 and rel_change < conv_crit:
                    break

        # Reconstruct: map extended gammas back to (order, dipole) pairs
        gammas_full = np.zeros(n_ext)
        gammas_full[active_set] = gammas

        # For each dipole, pick the order with the largest gamma
        gammas_per_order = gammas_full.reshape(n_orders, n_dipoles)
        best_order = np.argmax(gammas_per_order, axis=0)  # (n_dipoles,)
        best_gamma = np.max(gammas_per_order, axis=0)  # (n_dipoles,)

        # Build final inverse operator using the gradient approach (like SSM)
        # For active dipoles, use their best-order gradient to map back
        active_dipoles = np.where(best_gamma > pruning_thresh * best_gamma.max())[0]

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

    @staticmethod
    def _robust_inv(M):
        try:
            return np.linalg.inv(M)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(M)
