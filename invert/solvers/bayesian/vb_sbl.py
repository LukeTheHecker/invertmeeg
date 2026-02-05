import logging
from copy import deepcopy
from typing import Any

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
        noise_cov: np.ndarray | None = None,
        prune: bool = True,
        pruning_thresh: float = 1e-4,
        convergence_criterion: float = 1e-6,
        **kwargs: Any,
    ):
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)

        Y = self.unpack_data_obj(mne_obj)
        Y = Y - Y.mean(axis=0, keepdims=True)

        n_chans = self.leadfield.shape[0]
        if noise_cov is None:
            noise_cov = np.eye(n_chans)

        data_cov = self.data_covariance(Y, center=False, ddof=1)
        self.get_alphas(reference=data_cov)

        inverse_operators = []
        for alpha_eff in self.alphas:
            K = self._vb_sbl(
                Y,
                float(alpha_eff),
                noise_cov=noise_cov,
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
        Y: np.ndarray,
        noise_var: float,
        *,
        noise_cov: np.ndarray,
        max_iter: int,
        prune: bool,
        pruning_thresh: float,
        conv_crit: float,
    ) -> np.ndarray:
        n_chans, n_times = Y.shape
        L = deepcopy(self.leadfield)
        L_norm = np.linalg.norm(L, axis=0)
        L_norm = np.where(L_norm <= 0, 1.0, L_norm)
        L /= L_norm

        noise_cov = np.asarray(noise_cov, dtype=np.float64)
        noise_cov = noise_cov + 1e-12 * np.eye(noise_cov.shape[0])
        try:
            chol = np.linalg.cholesky(noise_cov)
            Wn = np.linalg.inv(chol)
        except np.linalg.LinAlgError:
            d = np.sqrt(np.clip(np.diag(noise_cov), 1e-12, None))
            Wn = np.diag(1.0 / d)

        Yw = Wn @ Y
        Lw = Wn @ L

        n_dipoles = Lw.shape[1]
        gammas = np.ones(n_dipoles, dtype=np.float64)
        denom_const = self.a0 + 0.5 * float(n_times)

        for _it in range(int(max_iter)):
            Sigma_y = noise_var * np.eye(n_chans) + (Lw * gammas) @ Lw.T
            Sigma_y_inv = self.robust_inverse(Sigma_y)

            L_Sigma = Sigma_y_inv @ Lw
            z_diag = np.sum(Lw * L_Sigma, axis=0)
            diag_Sigma_x = gammas - gammas * gammas * z_diag

            mu_x = (gammas[:, None] * (Lw.T @ (Sigma_y_inv @ Yw))).astype(np.float64)

            Ex2 = np.sum(mu_x * mu_x, axis=1) + float(n_times) * diag_Sigma_x
            gammas_new = (self.b0 + 0.5 * Ex2) / denom_const

            if prune:
                thr = float(pruning_thresh) * float(np.max(gammas_new))
                gammas_new = np.where(gammas_new >= thr, gammas_new, 0.0)

            rel = float(np.linalg.norm(gammas_new - gammas)) / max(float(np.linalg.norm(gammas)), 1e-15)
            gammas = gammas_new
            if rel < float(conv_crit):
                break

        Sigma_y = noise_var * np.eye(n_chans) + (Lw * gammas) @ Lw.T
        Sigma_y_inv = self.robust_inverse(Sigma_y)

        temp = Sigma_y_inv @ Wn
        K_scaled = gammas[:, None] * (Lw.T @ temp)
        K = (K_scaled / L_norm[:, None]).astype(np.float64)
        return K
