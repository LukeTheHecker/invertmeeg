from __future__ import annotations

import logging
from typing import Any

import mne
import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverVBSBL(BaseSolver):
    """Variational Bayes Sparse Bayesian Learning (VB-SBL).

    Hierarchical model:
        Y = L X + E
        X_i,t ~ N(0, γ_i)
        γ_i governed by a Gamma hyperprior on the corresponding precision.

    The implementation uses efficient sensor-space updates (invert m×m matrices)
    and returns the posterior mean as a linear inverse operator.
    """

    meta = SolverMeta(
        acronym="VB-SBL",
        full_name="Variational Bayes Sparse Bayesian Learning",
        category="Bayesian",
        description=(
            "Variational Bayesian ARD/SBL solver with Gamma hyperpriors, implemented "
            "with efficient sensor-space updates."
        ),
        references=[
            "Tipping, M. E. (2001). Sparse Bayesian learning and the relevance vector machine. Journal of Machine Learning Research, 1, 211–244.",
            "Wipf, D., & Nagarajan, S. (2009). A unified Bayesian framework for MEG/EEG source imaging. NeuroImage, 44(3), 947–966.",
        ],
    )

    def __init__(
        self,
        name: str = "VB-SBL",
        a0: float = 1e-6,
        b0: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        self.name = name
        self.a0 = float(a0)
        self.b0 = float(b0)
        super().__init__(**kwargs)

    def make_inverse_operator(  # type: ignore[override]
        self,
        forward,
        mne_obj,
        *args: Any,
        alpha: str | float = "auto",
        max_iter: int = 300,
        noise_cov: mne.Covariance | None = None,
        prune: bool = True,
        pruning_thresh: float = 1e-4,
        convergence_criterion: float = 1e-6,
        **kwargs: Any,
    ):
        super().make_inverse_operator(forward, mne_obj, *args, alpha=alpha, **kwargs)

        Y = self.unpack_data_obj(mne_obj)
        Y = Y - Y.mean(axis=0, keepdims=True)

        wf = self.prepare_whitened_forward(noise_cov)
        Yw = wf.sensor_transform @ Y

        data_cov = self.data_covariance(Yw, center=False, ddof=1)
        self.get_alphas(reference=data_cov)

        inverse_operators = []
        for alpha_eff in self.alphas:
            K = self._vb_sbl(
                wf.G_white,
                Yw,
                wf.sensor_transform,
                float(alpha_eff),
                max_iter=max_iter,
                prune=prune,
                pruning_thresh=pruning_thresh,
                conv_crit=convergence_criterion,
            )
            inverse_operators.append(K)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        return self

    def _vb_sbl(
        self,
        G_white: np.ndarray,
        Yw: np.ndarray,
        sensor_transform: np.ndarray,
        noise_var: float,
        *,
        max_iter: int,
        prune: bool,
        pruning_thresh: float,
        conv_crit: float,
    ) -> np.ndarray:
        n_eff, n_times = Yw.shape
        L = G_white.copy()
        L_norm = np.linalg.norm(L, axis=0)
        L_norm = np.where(L_norm <= 0, 1.0, L_norm)
        L /= L_norm

        n_dipoles = L.shape[1]
        gammas = np.ones(n_dipoles, dtype=np.float64)
        denom_const = self.a0 + 0.5 * float(n_times)

        for _it in range(int(max_iter)):
            Sigma_y = noise_var * np.eye(n_eff) + (L * gammas) @ L.T
            Sigma_y_inv = self.robust_inverse(Sigma_y)

            L_Sigma = Sigma_y_inv @ L
            z_diag = np.sum(L * L_Sigma, axis=0)
            diag_Sigma_x = gammas - gammas * gammas * z_diag

            mu_x = (gammas[:, None] * (L.T @ (Sigma_y_inv @ Yw))).astype(np.float64)

            Ex2 = np.sum(mu_x * mu_x, axis=1) + float(n_times) * diag_Sigma_x
            gammas_new = (self.b0 + 0.5 * Ex2) / denom_const

            if prune:
                thr = float(pruning_thresh) * float(np.max(gammas_new))
                gammas_new = np.where(gammas_new >= thr, gammas_new, 0.0)

            rel = float(np.linalg.norm(gammas_new - gammas)) / max(
                float(np.linalg.norm(gammas)), 1e-15
            )
            gammas = gammas_new
            if rel < float(conv_crit):
                break

        Sigma_y = noise_var * np.eye(n_eff) + (L * gammas) @ L.T
        Sigma_y_inv = self.robust_inverse(Sigma_y)

        # K_white maps from whitened sensor space to column-normalized sources.
        # Compose with sensor_transform to map from raw sensor space, and
        # undo column normalization.
        K_white = gammas[:, None] * (L.T @ Sigma_y_inv)
        K = (K_white @ sensor_transform / L_norm[:, None]).astype(np.float64)
        return K
