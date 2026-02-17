import logging

import mne
import numpy as np

from ..base import InverseOperator, SolverMeta
from .base_beamformer import BaseBeamformer

logger = logging.getLogger(__name__)


class SolverEBB(BaseBeamformer):
    """Empirical Bayesian Beamformer (EBB) for M/EEG inverse problem.

    Iterative empirical-Bayes / ARD-style beamformer that estimates per-source
    variances (hyperparameters) via EM and forms corresponding spatial filters.

    This is equivalent to Champagne with EM updates, using the efficient
    sensor-space (n_channels x n_channels) formulation.

    References
    ----------
    [1] Wipf, D., & Nagarajan, S. (2009). A unified Bayesian framework for
        MEG/EEG source imaging. NeuroImage, 44(3), 947-966.
    """

    meta = SolverMeta(
        slug="ebb",
        full_name="Empirical Bayesian Beamformer",
        category="Beamformers",
        description=(
            "Iterative empirical-Bayes / ARD-style beamformer that updates per-source "
            "variances (hyperparameters) from the data and forms corresponding spatial "
            "filters."
        ),
        references=[
            "Wipf, D., & Nagarajan, S. (2009). A unified Bayesian framework for "
            "MEG/EEG source imaging. NeuroImage, 44(3), 947-966.",
        ],
    )

    def __init__(
        self,
        name="Empirical Bayesian Beamformer",
        reduce_rank=True,
        rank="auto",
        **kwargs,
    ):
        self.name = name
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(
        self,
        forward,
        mne_obj=None,
        *args,
        weight_norm=True,
        noise_cov: mne.Covariance | None = None,
        alpha="auto",
        max_iter=300,
        tol=1e-6,
        **kwargs,
    ):
        """Compute the EBB inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            Forward model.
        mne_obj : mne.Evoked | mne.Epochs | mne.io.Raw
            The MNE data object.
        weight_norm : bool
            Apply unit-noise-gain weight normalization.
        noise_cov : mne.Covariance | None
            Noise covariance. None uses identity.
        alpha : float | str
            Regularization parameter.
        max_iter : int
            Maximum EM iterations.
        tol : float
            Convergence tolerance for source variance updates.
        """
        super().make_inverse_operator(forward, mne_obj, *args, alpha=alpha, **kwargs)

        data = self.unpack_data_obj(mne_obj)

        wf = self.prepare_whitened_forward(noise_cov)
        L = wf.G_white.copy()
        sensor_transform = wf.sensor_transform
        n_channels = wf.n_eff
        n_sources = L.shape[1]

        # Whiten data and compute covariance
        data_w = sensor_transform @ data
        data_cov = self.data_covariance(data_w, center=True, ddof=1)

        inverse_operators = []
        self.alphas = self.get_alphas(reference=data_cov)

        I_m = np.identity(n_channels)

        for alpha in self.alphas:
            gamma = np.ones(n_sources)
            Cn = alpha * I_m

            for n_iter in range(max_iter):
                # Model covariance: Sigma_y = L @ diag(gamma) @ L^T + Cn
                # Invert in sensor space: O(m^3) where m = n_channels
                Sigma_y = L @ (gamma[:, None] * L.T) + Cn
                Sigma_y_inv = self.robust_inverse(Sigma_y)

                # Posterior source estimates: mu = gamma * L^T @ Sigma_y^{-1} @ Y
                LT_Sy_inv = L.T @ Sigma_y_inv
                mu = (gamma[:, None] * LT_Sy_inv) @ data_w

                # Posterior variance (diagonal only):
                # diag(Sigma_s) = gamma - gamma^2 * diag(L^T Sigma_y^{-1} L)
                z_diag = np.sum(L * (Sigma_y_inv @ L), axis=0)
                diag_Sigma_s = gamma - gamma**2 * z_diag

                # EM update: gamma_new = diag(Sigma_s) + mean(mu^2, axis=1)
                gamma_new = diag_Sigma_s + np.mean(mu**2, axis=1)
                gamma_new = np.maximum(gamma_new, 1e-12)

                # Convergence check
                rel_change = np.abs(gamma_new - gamma) / (np.abs(gamma) + 1e-10)
                max_change = np.max(rel_change)

                gamma = gamma_new

                if max_change < tol:
                    logger.info(
                        "EBB converged after %d iterations (max change: %.2e)",
                        n_iter + 1,
                        max_change,
                    )
                    break
            else:
                logger.warning(
                    "EBB reached max iterations (%d), max change: %.2e",
                    max_iter,
                    max_change,
                )

            # Final beamformer weights with converged hyperparameters
            Sigma_y = L @ (gamma[:, None] * L.T) + Cn
            Sigma_y_inv = self.robust_inverse(Sigma_y)
            W = (gamma[:, None] * L.T) @ Sigma_y_inv

            # Unit-noise-gain normalization
            if weight_norm:
                # noise_gain_i = W_i @ Cn @ W_i^T = alpha * ||W_i||^2
                noise_gain = alpha * np.sum(W**2, axis=1)
                W /= np.sqrt(np.maximum(noise_gain, 1e-30))[:, None]

            W_raw = W @ sensor_transform
            inverse_operators.append(W_raw)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]

        return self
