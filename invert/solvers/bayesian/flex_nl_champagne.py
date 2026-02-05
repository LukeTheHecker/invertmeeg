"""FlexNLChampagne: Refined FlexChampagne with two-pass SBL and Convexity updates.

Improves on FlexChampagne by:
1. Using the Convexity (MM) update rule which has tighter bounds
2. A two-pass refinement: first pass identifies active atoms in the extended
   dictionary, second pass refines gammas on just the active atoms with
   a fresh noise estimate, improving focal accuracy.
3. Annealed pruning: starts lenient and tightens, preventing premature pruning.
"""

import logging
from copy import deepcopy

import mne
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverFlexNLChampagne(BaseSolver):
    """Two-pass refined FlexChampagne with Convexity updates."""

    meta = SolverMeta(
        slug="flex-nl-champagne",
        full_name="Flex-NL-Champagne",
        category="Bayesian",
        description=(
            "Two-pass refined flexible-extent Champagne variant using a multi-order "
            "diffusion dictionary with Convexity/MM updates and refinement."
        ),
        references=[
            "Lukas Hecker 2025, unpublished",
        ],
    )

    def __init__(
        self,
        name="FlexNLChampagne",
        n_orders=4,
        diffusion_parameter=0.1,
        adjacency_type="spatial",
        adjacency_distance=3e-3,
        **kwargs,
    ):
        self.name = name
        self.n_orders = n_orders
        self.diffusion_parameter = diffusion_parameter
        self.adjacency_type = adjacency_type
        self.adjacency_distance = adjacency_distance
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

        inverse_operator = self._two_pass_flex(
            data, pruning_thresh, max_iter, convergence_criterion
        )
        self.inverse_operators = [InverseOperator(inverse_operator, self.name)]
        return self

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

        for i in range(len(self.gradients)):
            row_sums = self.gradients[i].sum(axis=1).ravel()
            scaling = 1.0 / np.maximum(np.abs(np.asarray(row_sums).ravel()), 1e-12)
            self.gradients[i] = csr_matrix(
                self.gradients[i].multiply(scaling.reshape(-1, 1))
            )

        self.is_prepared = True

    def _run_sbl(
        self,
        L,
        Y_scaled,
        noise_cov,
        max_iter,
        pruning_thresh,
        conv_crit,
        update_rule="Convexity",
        init_gammas=None,
    ):
        """Core SBL loop. Returns (active_set, gammas)."""
        L.shape[0]
        n_times = Y_scaled.shape[1]
        n_atoms = L.shape[1]

        gammas = init_gammas if init_gammas is not None else np.ones(n_atoms)
        active_set = np.arange(n_atoms)
        L_act = deepcopy(L)

        loss_list = []

        for i_iter in range(max_iter):
            old_gammas = deepcopy(gammas)

            Sigma_y = noise_cov + (L_act * gammas) @ L_act.T
            Sigma_y = 0.5 * (Sigma_y + Sigma_y.T)
            Sigma_y_inv = self._robust_inv(Sigma_y)
            mu_x = (L_act.T @ Sigma_y_inv @ Y_scaled) * gammas[:, None]

            upper = np.mean(mu_x**2, axis=1)
            L_Sigma = Sigma_y_inv @ L_act
            z_diag = np.sum(L_act * L_Sigma, axis=0)

            if update_rule == "Convexity":
                gammas = np.sqrt(upper / (z_diag + 1e-20))
            else:  # MacKay
                gammas = upper / (gammas * z_diag + 1e-20)

            gammas[~np.isfinite(gammas)] = 0.0
            gammas = np.maximum(gammas, 0.0)

            if np.linalg.norm(gammas) == 0:
                gammas = old_gammas
                break

            # Annealed pruning: start lenient, tighten
            anneal = min(1.0, (i_iter + 1) / 50.0)
            thresh = (pruning_thresh * anneal) * gammas.max()
            keep = np.where(gammas > thresh)[0]
            if len(keep) == 0:
                gammas = old_gammas
                break
            active_set = active_set[keep]
            gammas = gammas[keep]
            L_act = L_act[:, keep]

            Sigma_y = noise_cov + (L_act * gammas) @ L_act.T
            Sigma_y = 0.5 * (Sigma_y + Sigma_y.T)
            Sigma_y_inv = self._robust_inv(Sigma_y)

            data_fit = np.trace(Sigma_y_inv @ Y_scaled @ Y_scaled.T) / n_times
            eigvals = np.linalg.eigvalsh(Sigma_y)
            log_det = float(np.sum(np.log(np.maximum(eigvals, 1e-20))))
            loss = float(data_fit + log_det)
            loss_list.append(loss)

            if len(loss_list) > 1:
                rel_change = (loss_list[-2] - loss) / (abs(loss_list[-2]) + 1e-20)
                if rel_change > 0 and rel_change < conv_crit:
                    break

        return active_set, gammas

    def _two_pass_flex(self, Y, pruning_thresh, max_iter, conv_crit):
        n_chans, n_dipoles = self.leadfield.shape
        n_times = Y.shape[1]
        n_orders = len(self.leadfields)

        # Build extended leadfield
        L_blocks = []
        for lf in self.leadfields:
            lf_norm = lf / (np.linalg.norm(lf, axis=0, keepdims=True) + 1e-12)
            L_blocks.append(lf_norm)
        L_ext = np.hstack(L_blocks)
        n_ext = L_ext.shape[1]

        # Scale data
        scale = float(np.mean(np.abs(Y))) + 1e-12
        Y_scaled = Y / scale

        # Noise estimate
        C_y = self.data_covariance(Y_scaled, center=True, ddof=1)
        alpha_noise = float(np.trace(C_y) / (n_chans * 100))
        noise_cov = alpha_noise * np.identity(n_chans)

        # === Pass 1: Identify active atoms with Convexity rule ===
        active_set_1, gammas_1 = self._run_sbl(
            L_ext,
            Y_scaled,
            noise_cov,
            max_iter=max_iter,
            pruning_thresh=pruning_thresh,
            conv_crit=conv_crit,
            update_rule="Convexity",
        )

        if len(active_set_1) == 0:
            return np.zeros((n_dipoles, n_chans))

        # === Pass 2: Refine on active atoms with fresh noise estimate ===
        # Use active atoms from pass 1 as the dictionary
        L_refined = L_ext[:, active_set_1]

        # Re-estimate noise from pass 1 residuals
        Sigma_y_1 = noise_cov + (L_refined * gammas_1) @ L_refined.T
        Sigma_y_1 = 0.5 * (Sigma_y_1 + Sigma_y_1.T)
        Sigma_y_1_inv = self._robust_inv(Sigma_y_1)
        mu_x_1 = (L_refined.T @ Sigma_y_1_inv @ Y_scaled) * gammas_1[:, None]
        residuals = Y_scaled - L_refined @ mu_x_1
        refined_noise = float(np.trace(residuals @ residuals.T) / (n_chans * n_times))
        refined_noise = max(refined_noise, 1e-10)
        noise_cov_2 = refined_noise * np.identity(n_chans)

        # Run second pass with MacKay, warm-started from pass 1 gammas
        active_set_2, gammas_2 = self._run_sbl(
            L_refined,
            Y_scaled,
            noise_cov_2,
            max_iter=max_iter,
            pruning_thresh=pruning_thresh,
            conv_crit=conv_crit,
            update_rule="MacKay",
            init_gammas=gammas_1.copy(),
        )

        # Map back to global indices
        final_active = active_set_1[active_set_2]
        final_gammas = gammas_2

        # Reconstruct
        gammas_full = np.zeros(n_ext)
        gammas_full[final_active] = final_gammas

        gammas_per_order = gammas_full.reshape(n_orders, n_dipoles)
        best_order = np.argmax(gammas_per_order, axis=0)
        best_gamma = np.max(gammas_per_order, axis=0)

        active_dipoles = np.where(best_gamma > pruning_thresh * best_gamma.max())[0]

        if len(active_dipoles) == 0:
            return np.zeros((n_dipoles, n_chans))

        L_reduced = np.stack(
            [self.leadfields[best_order[d]][:, d] for d in active_dipoles], axis=1
        )
        gamma_reduced = best_gamma[active_dipoles]

        grad_cols = []
        for d in active_dipoles:
            g = self.gradients[best_order[d]][d].toarray().ravel()
            grad_cols.append(g)
        G = np.stack(grad_cols, axis=1)

        Sigma_y_r = noise_cov_2 + (L_reduced * gamma_reduced) @ L_reduced.T
        Sigma_y_r = 0.5 * (Sigma_y_r + Sigma_y_r.T)
        Sigma_y_r_inv = self._robust_inv(Sigma_y_r)
        inv_op = G @ np.diag(gamma_reduced) @ L_reduced.T @ Sigma_y_r_inv

        return inv_op

    @staticmethod
    def _robust_inv(M):
        try:
            return np.linalg.inv(M)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(M)
