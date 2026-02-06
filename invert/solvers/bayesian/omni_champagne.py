"""OmniChampagne: an adaptive sparse Bayesian solver that tries to win everywhere.

Motivation (from benchmark observations):
- Focal (single/multi dipole): Champagne-style sparse Bayesian learning excels.
- Extended patches: FLEX/SSM-style multi-order dictionaries are needed.

This solver evaluates two related SBL models per sample:
1) Dipole model (order 0 only)  -> behaves like Champagne.
2) Patch model (multi-order diffusion basis matching the simulator) -> behaves like Flex-like patch solvers.

It picks the model with the best penalized evidence (negative log-likelihood +
complexity penalty), then returns the corresponding inverse operator in the
original source space.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mne
import numpy as np
from scipy.sparse import csr_matrix, vstack
from scipy.sparse.csgraph import laplacian

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SBLFit:
    active_set: np.ndarray  # indices into atoms
    gammas: np.ndarray  # gammas for active atoms (len == len(active_set))
    sigma_y_inv: np.ndarray  # (n_chans, n_chans)
    loss: float


class SolverOmniChampagne(BaseSolver):
    """Adaptive Champagne/Flex-like solver via model selection."""

    meta = SolverMeta(
        slug="omni-champagne",
        full_name="OmniChampagne",
        category="Bayesian",
        description=(
            "Adaptive sparse Bayesian solver that selects between a dipole-only "
            "Champagne-style model and a multi-order patch-dictionary model based "
            "on penalized evidence."
        ),
        references=[
            "Lukas Hecker 2025, unpublished",
        ],
    )

    def __init__(
        self,
        name: str = "OmniChampagne",
        # Patch basis settings (match simulator defaults)
        n_orders: int = 3,
        diffusion_parameter: float = 0.1,
        adjacency_type: str = "spatial",
        adjacency_distance: float = 3e-3,
        # SBL settings
        update_rule: str = "MacKay",
        max_iter: int = 2000,
        pruning_thresh: float = 1e-3,
        convergence_criterion: float = 1e-8,
        # Model selection (penalize extra active atoms)
        complexity_penalty: float = 0.2,
        **kwargs,
    ) -> None:
        self.name = name
        self.n_orders = int(n_orders)
        self.diffusion_parameter = float(diffusion_parameter)
        self.adjacency_type = str(adjacency_type)
        self.adjacency_distance = float(adjacency_distance)

        self.update_rule = str(update_rule)
        self.max_iter = int(max_iter)
        self.pruning_thresh = float(pruning_thresh)
        self.convergence_criterion = float(convergence_criterion)

        self.complexity_penalty = float(complexity_penalty)

        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def make_inverse_operator(  # type: ignore[override]
        self,
        forward,
        mne_obj,
        *args,
        alpha: str | float = "auto",
        **kwargs,
    ):
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)

        inv_op = self._fit_and_build_inverse_operator(data)
        self.inverse_operators = [InverseOperator(inv_op, self.name)]
        return self

    # ------------------------------------------------------------------
    # Model selection + inverse construction
    # ------------------------------------------------------------------
    def _fit_and_build_inverse_operator(self, Y: np.ndarray) -> np.ndarray:
        n_chans, n_dipoles = self.leadfield.shape

        # Noise estimate from data covariance
        C_y = self.data_covariance(Y, center=True, ddof=1)
        alpha_noise = float(np.trace(C_y) / n_chans)
        noise_cov = alpha_noise * np.eye(n_chans)
        Y_scaled = Y

        # Model A: dipole-only dictionary
        L_dip = self.leadfield
        fit_dip = self._fit_sbl(
            L_dip,
            Y_scaled,
            noise_cov=noise_cov,
            max_iter=self.max_iter,
            pruning_thresh=self.pruning_thresh,
            conv_crit=self.convergence_criterion,
            update_rule=self.update_rule,
        )
        W_dip = self._inverse_from_fit(L_dip, fit_dip, noise_cov=noise_cov)

        # Model B: patch dictionary (multi-order diffusion basis)
        sources_full = self._build_simulator_basis(n_dipoles)
        # For the patch model we intentionally exclude the order-0 (identity)
        # basis because we evaluate a separate dipole-only model above.
        sources = (
            sources_full[n_dipoles:, :]
            if sources_full.shape[0] > n_dipoles
            else sources_full
        )
        L_patch = self.leadfield @ sources.T  # (n_chans, n_candidates)
        fit_patch = self._fit_sbl(
            L_patch,
            Y_scaled,
            noise_cov=noise_cov,
            max_iter=self.max_iter,
            pruning_thresh=self.pruning_thresh,
            conv_crit=self.convergence_criterion,
            update_rule=self.update_rule,
        )
        W_patch_coeff = self._inverse_from_fit(L_patch, fit_patch, noise_cov=noise_cov)
        # Map coefficients back to dipole space (sources.T is (n_dipoles, n_candidates))
        W_patch = (
            sources.T[:, fit_patch.active_set] @ W_patch_coeff[fit_patch.active_set]
        )

        # Penalized evidence for selection
        score_dip = fit_dip.loss + self.complexity_penalty * float(
            len(fit_dip.active_set)
        )
        score_patch = fit_patch.loss + self.complexity_penalty * float(
            len(fit_patch.active_set)
        )

        if score_patch < score_dip:
            return W_patch
        return W_dip

    # ------------------------------------------------------------------
    # Basis construction (match SimulationGenerator logic)
    # ------------------------------------------------------------------
    def _build_simulator_basis(self, n_dipoles: int) -> np.ndarray:
        """Return stacked multi-order basis S with shape (n_candidates, n_dipoles)."""
        if self.n_orders <= 1:
            return np.eye(n_dipoles)

        I = np.eye(n_dipoles)
        if self.adjacency_type == "spatial":
            adjacency = mne.spatial_src_adjacency(self.forward["src"], verbose=0)
        else:
            adjacency = mne.spatial_dist_adjacency(
                self.forward["src"], self.adjacency_distance, verbose=None
            )
        adjacency = csr_matrix(adjacency)
        G = csr_matrix(I - self.diffusion_parameter * laplacian(adjacency))

        sources = csr_matrix(I)
        for _ in range(1, self.n_orders):
            last_block = sources.toarray()[-n_dipoles:, -n_dipoles:]
            new_sources = csr_matrix(last_block) @ G
            col_max = new_sources.max(axis=0).toarray().ravel()
            col_max = np.maximum(col_max, 1e-12)
            new_sources = new_sources / col_max[np.newaxis]
            sources = vstack([sources, new_sources])

        return sources.toarray()

    # ------------------------------------------------------------------
    # Sparse Bayesian learning core
    # ------------------------------------------------------------------
    def _fit_sbl(
        self,
        L_orig: np.ndarray,
        Y_scaled: np.ndarray,
        *,
        noise_cov: np.ndarray,
        max_iter: int,
        pruning_thresh: float,
        conv_crit: float,
        update_rule: str,
    ) -> _SBLFit:
        n_chans, n_atoms = L_orig.shape
        n_times = Y_scaled.shape[1]

        L = L_orig

        gammas = np.ones(n_atoms)
        active_set = np.arange(n_atoms)

        # Start with full set
        L_act = L
        gam_act = gammas

        loss_prev = None
        for _ in range(max_iter):
            # Posterior for current active set
            Sigma_y = noise_cov + (L_act * gam_act) @ L_act.T
            Sigma_y = 0.5 * (Sigma_y + Sigma_y.T)
            Sigma_y_inv = self._robust_inv(Sigma_y)
            mu_x = (L_act.T @ Sigma_y_inv @ Y_scaled) * gam_act[:, None]

            # Update gammas
            upper = np.mean(mu_x**2, axis=1)
            L_Sigma = Sigma_y_inv @ L_act
            z_diag = np.sum(L_act * L_Sigma, axis=0)

            rule = update_rule.lower()
            if rule == "convexity" or rule == "mm":
                gam_new = np.sqrt(upper / (z_diag + 1e-20))
            elif rule == "em":
                diag_sigma_x = gam_act - gam_act**2 * z_diag
                gam_new = diag_sigma_x + upper
            else:  # MacKay default
                gam_new = upper / (gam_act * z_diag + 1e-20)

            gam_new[~np.isfinite(gam_new)] = 0.0
            gam_new = np.maximum(gam_new, 0.0)
            if float(np.linalg.norm(gam_new)) == 0.0:
                break

            # Prune
            thresh = pruning_thresh * float(gam_new.max())
            keep = np.where(gam_new > thresh)[0]
            if keep.size == 0:
                break

            active_set = active_set[keep]
            gam_act = gam_new[keep]
            L_act = L_act[:, keep]

            # Recompute after pruning for loss and convergence
            Sigma_y = noise_cov + (L_act * gam_act) @ L_act.T
            Sigma_y = 0.5 * (Sigma_y + Sigma_y.T)
            Sigma_y_inv = self._robust_inv(Sigma_y)

            data_fit = float(np.trace(Sigma_y_inv @ Y_scaled @ Y_scaled.T) / n_times)
            eigvals = np.linalg.eigvalsh(Sigma_y)
            log_det = float(np.sum(np.log(np.maximum(eigvals, 1e-20))))
            loss = data_fit + log_det

            if loss_prev is not None:
                rel_change = (loss_prev - loss) / (abs(loss_prev) + 1e-20)
                if rel_change > 0 and rel_change < conv_crit:
                    loss_prev = loss
                    break
            loss_prev = loss

        # Final loss on the last active set state
        if loss_prev is None:
            Sigma_y = noise_cov + (L_act * gam_act) @ L_act.T
            Sigma_y = 0.5 * (Sigma_y + Sigma_y.T)
            Sigma_y_inv = self._robust_inv(Sigma_y)
            data_fit = float(np.trace(Sigma_y_inv @ Y_scaled @ Y_scaled.T) / n_times)
            eigvals = np.linalg.eigvalsh(Sigma_y)
            log_det = float(np.sum(np.log(np.maximum(eigvals, 1e-20))))
            loss_prev = data_fit + log_det
        else:
            Sigma_y_inv = Sigma_y_inv

        return _SBLFit(
            active_set=active_set.astype(int, copy=False),
            gammas=gam_act.astype(float, copy=False),
            sigma_y_inv=Sigma_y_inv,
            loss=float(loss_prev),
        )

    @staticmethod
    def _inverse_from_fit(
        L: np.ndarray, fit: _SBLFit, *, noise_cov: np.ndarray
    ) -> np.ndarray:
        """Build full inverse operator W (n_atoms, n_chans) for a given dictionary."""
        n_chans, n_atoms = L.shape

        gam_full = np.zeros(n_atoms, dtype=float)
        gam_full[fit.active_set] = fit.gammas
        Gamma = np.diag(gam_full)

        Sigma_y = noise_cov + (L * gam_full) @ L.T
        Sigma_y = 0.5 * (Sigma_y + Sigma_y.T)
        Sigma_y_inv = SolverOmniChampagne._robust_inv(Sigma_y)
        return Gamma @ L.T @ Sigma_y_inv  # (n_atoms, n_chans)

    @staticmethod
    def _robust_inv(M: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.inv(M)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(M)
