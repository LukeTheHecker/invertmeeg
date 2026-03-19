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

from invert.util import build_source_adjacency

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
            "Wipf, D. P., Owen, J. P., Attias, H. T., Sekihara, K., & Nagarajan, S. S. (2010). Robust Bayesian estimation of the location, orientation, and time course of multiple correlated neural sources using MEG. NeuroImage, 49(1), 641–655.",
            "Cai, C., Hashemi, A., Diwakar, M., Haufe, S., Sekihara, K., & Nagarajan, S. S. (2021). Robust estimation of noise for electromagnetic brain imaging with the champagne algorithm. NeuroImage, 225, 117411.",
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
        noise_init_mode="trace",
        noise_diag_shrinkage=0.0,
        refined_noise_mode="scalar",
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
        self.noise_init_mode = noise_init_mode
        self.noise_diag_shrinkage = float(noise_diag_shrinkage)
        self.refined_noise_mode = refined_noise_mode
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

        inverse_operator = self._two_pass_flex(
            data, pruning_thresh, max_iter, convergence_criterion
        )
        self.inverse_operators = [
            InverseOperator(inverse_operator @ wf.sensor_transform, self.name)
        ]
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
        L_ext = np.hstack(self.leadfields)
        n_ext = L_ext.shape[1]

        # Noise estimate from data covariance
        C_y = self.data_covariance(Y, center=True, ddof=1)
        noise_cov = self._build_noise_cov(
            C_y,
            n_chans,
            mode=self.noise_init_mode,
            diag_shrinkage=self.noise_diag_shrinkage,
        )

        Y_scaled = Y

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
        noise_cov_2 = self._build_refined_noise_cov(
            residuals, n_chans, n_times, mode=self.refined_noise_mode
        )

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
        if self.order_pooling == "sum":
            best_gamma = np.sum(gammas_per_order, axis=0)
        else:
            best_gamma = np.max(gammas_per_order, axis=0)

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

    def _build_refined_noise_cov(
        self,
        residuals: np.ndarray,
        n_chans: int,
        n_times: int,
        *,
        mode: str | None = None,
    ) -> np.ndarray:
        mode = str(self.refined_noise_mode if mode is None else mode).lower()
        if mode == "diag":
            diag = np.sum(residuals**2, axis=1) / max(n_times, 1)
            diag = np.maximum(diag, 1e-10)
            return np.diag(diag)
        refined_noise = float(np.trace(residuals @ residuals.T) / (n_chans * n_times))
        refined_noise = max(refined_noise, 1e-10)
        return refined_noise * np.identity(n_chans)

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
