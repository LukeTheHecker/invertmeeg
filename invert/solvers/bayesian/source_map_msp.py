import mne
import numpy as np
from scipy.sparse.csgraph import laplacian

from invert.util import build_source_adjacency

from ..base import BaseSolver, InverseOperator, SolverMeta


class SolverSourceMAPMSP(BaseSolver):
    """Class for the Source Maximum A Posteriori (Source-MAP) inverse solution
    using multiple sparse priors [1]. The method is conceptually similar to [2],
    but formally not equal.

    References
    ----------
    [1] Wipf, D., & Nagarajan, S. (2009). A unified Bayesian framework for
    MEG/EEG source imaging. NeuroImage, 44(3), 947-966.

    [2] Friston, K., Harrison, L., Daunizeau, J., Kiebel, S., Phillips, C.,
    Trujillo-Barreto, N., ... & Mattout, J. (2008). Multiple sparse priors for
    the M/EEG inverse problem. NeuroImage, 39(3), 1104-1120.

    """

    meta = SolverMeta(
        acronym="Source-MAP-MSP",
        full_name="Source MAP with MSP Priors",
        category="Bayesian",
        description=(
            "Source-MAP sparse Bayesian inverse approach augmented with MSP-style "
            "spatial priors/patch smoothing (conceptually related to MSP)."
        ),
        references=[
            "Wipf, D., & Nagarajan, S. (2009). A unified Bayesian framework for MEG/EEG source imaging. NeuroImage, 44(3), 947–966.",
            "Friston, K., Harrison, L., Daunizeau, J., Kiebel, S., Phillips, C., Trujillo-Barreto, N., & Mattout, J. (2008). Multiple sparse priors for the M/EEG inverse problem. NeuroImage, 39(3), 1104–1120.",
        ],
    )

    def __init__(self, name="Source-MAP-MSP", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(
        self,
        forward,
        mne_obj=None,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
        max_iter=100,
        p=0.5,
        smoothness_order=1,
        verbose=0,
        **kwargs,
    ):
        """Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        alpha : float
            The regularization parameter.
        p : 0 < p < 2
            Hyperparameter which controls sparsity. Default: p = 0.5
        max_iter : int
            Maximum numbers of iterations to find the optimal hyperparameters.
            max_iter = 1 corresponds to sLORETA.
        smoothness_order : int
            Controls the smoothness prior. The higher this integer, the higher
            the pursued smoothness of the inverse solution.

        Return
        ------
        self : object returns itself for convenience

        """
        super().make_inverse_operator(forward, mne_obj, *args, alpha=alpha, **kwargs)
        wf = self.prepare_whitened_forward(noise_cov)
        data = self.unpack_data_obj(mne_obj)
        data = wf.sensor_transform @ data
        leadfield = self.leadfield
        data_cov = self.data_covariance(data, center=True, ddof=1)
        L_smooth, _gradient = self.get_smooth_prior_cov(leadfield, smoothness_order)
        # Scale alphas against the raw leadfield (not the column-normalised
        # L_smooth) because alpha enters the *final* inverse via raw L.
        reg_reference = leadfield @ leadfield.T
        reg_reference = 0.5 * (reg_reference + reg_reference.T)
        if not np.all(np.isfinite(reg_reference)) or np.linalg.norm(reg_reference) == 0:
            reg_reference = data_cov
        self.get_alphas(reference=reg_reference)

        inverse_operators = []
        for alpha in self.alphas:
            inverse_operator = self.make_source_map_inverse_operator(
                data, alpha, max_iter=max_iter, p=p, smoothness_order=smoothness_order
            )
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [
            InverseOperator(inverse_operator @ wf.sensor_transform, self.name)
            for inverse_operator in inverse_operators
        ]
        return self

    def make_source_map_inverse_operator(
        self, B, alpha, max_iter=100, p=0.5, smoothness_order=1
    ):
        """Computes the source MAP inverse operator based on the M/EEG data.

        Parameters
        ----------
        B : numpy.ndarray
            The M/EEG data matrix (channels, time points).
        alpha : float
            The regularization parameter.
        max_iter : int
            Maximum numbers of iterations to find the optimal hyperparameters.
            max_iter = 1 corresponds to sLORETA.
        p : 0 < p < 2
            Hyperparameter which controls sparsity. Default: p = 0.5
        smoothness_order : int
            Controls the smoothness prior. The higher this integer, the higher
            the pursued smoothness of the inverse solution.

        Return
        ------
        inverse_operator : numpy.ndarray
            The inverse operator which can be used to compute inverse solutions from new data.

        """

        L = self.leadfield.copy()
        db, n = B.shape
        ds = L.shape[1]

        # Ensure Common average reference
        B -= B.mean(axis=0)
        L -= L.mean(axis=0)

        L_smooth, gradient = self.get_smooth_prior_cov(L, smoothness_order)
        gammas = np.ones(ds, dtype=float)
        sigma_e = float(alpha) * np.identity(db)
        exponent = float((2.0 - p) / 2.0)
        exponent = float(np.clip(exponent, 1e-6, 2.0))

        for _k in range(max_iter):
            old_gammas = gammas.copy()

            # sigma_b = alpha*I + L_smooth @ diag(gammas) @ L_smooth.T
            sigma_b = sigma_e + (L_smooth * gammas) @ L_smooth.T
            sigma_b = 0.5 * (sigma_b + sigma_b.T)
            sigma_b_inv = np.linalg.inv(sigma_b)

            # Compute sigma_b_inv @ L_smooth once, reuse for both terms
            SiL = sigma_b_inv @ L_smooth  # (m, d)
            LtSiB = SiL.T @ B  # (d, t)

            term_1 = (gammas / np.sqrt(n)) * np.sqrt(
                np.sum(LtSiB ** 2, axis=1)
            )
            # diag(L.T @ sigma_b_inv @ L) via column-wise dot product
            denom = np.sum(L_smooth * SiL, axis=0)
            denom = np.maximum(denom, 1e-15)
            term_2 = 1 / np.sqrt(denom)
            gammas = np.maximum(term_1 * term_2, 1e-15) ** exponent

            if not np.all(np.isfinite(gammas)) or np.linalg.norm(gammas) == 0:
                gammas = old_gammas
                break

        # Smooth gammas according to smooth priors
        gammas_final = np.abs(gammas @ gradient)
        gamma_max = float(np.max(gammas_final))
        if not np.isfinite(gamma_max) or gamma_max <= 0:
            gammas_final = np.ones_like(gammas_final)
        else:
            gammas_final = gammas_final / gamma_max

        # Final: diag(gammas_final) @ L.T @ inv(alpha*I + L @ diag(gammas_final) @ L.T)
        sigma_b_final = sigma_e + (L * gammas_final) @ L.T
        inverse_operator = (gammas_final[:, None] * L.T) @ np.linalg.inv(sigma_b_final)
        return inverse_operator

    @staticmethod
    def frob(x):
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        return np.sqrt(np.trace(x @ x.T))

    def get_smooth_prior_cov(self, L, smoothness_order):
        adjacency = build_source_adjacency(self.forward["src"], verbose=0)
        gradient = laplacian(adjacency).toarray().astype(np.float32)

        for _i in range(smoothness_order):
            gradient = gradient @ gradient
        L = L @ abs(gradient)
        # L -= L.mean(axis=0)
        norms = np.linalg.norm(L, axis=0)
        norms = np.maximum(norms, 1e-15)
        L /= norms
        return L, gradient
