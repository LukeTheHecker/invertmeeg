import mne
import numpy as np
from scipy.sparse.csgraph import laplacian

from invert.util import build_source_adjacency

from ..base import BaseSolver, InverseOperator, SolverMeta


class SolverGammaMAP(BaseSolver):
    """Class for the Gamma Maximum A Posteriori (Gamma-MAP) inverse solution [1].

    References
    ----------
    Wipf, D., & Nagarajan, S. (2009). A unified Bayesian framework for MEG/EEG
    source imaging. NeuroImage, 44(3), 947-966.

    """

    meta = SolverMeta(
        acronym="Gamma-MAP",
        full_name="Gamma Maximum A Posteriori",
        category="Bayesian",
        description=(
            "Sparse Bayesian inverse solution using a Gamma hyperprior (ARD-style) "
            "within the unified Bayesian source imaging framework."
        ),
        references=[
            "Wipf, D., & Nagarajan, S. (2009). A unified Bayesian framework for MEG/EEG source imaging. NeuroImage, 44(3), 947–966.",
        ],
    )

    def __init__(self, name="Gamma-MAP", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(
        self,
        forward,
        mne_obj=None,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
        smoothness_prior=False,
        max_iter=100,
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
        alpha : str/ float
            The regularization parameter.
        max_iter : int
            Maximum numbers of iterations to find the optimal hyperparameters.
            max_iter = 1 corresponds to sLORETA.
        smoothness_prior : bool



        Return
        ------
        self : object returns itself for convenience

        """
        super().make_inverse_operator(forward, mne_obj, *args, alpha=alpha, **kwargs)
        wf = self.prepare_whitened_forward(noise_cov)
        leadfield = self.leadfield
        n_chans, n_dipoles = leadfield.shape
        data = self.unpack_data_obj(mne_obj)
        data = wf.sensor_transform @ data
        if smoothness_prior:
            adjacency = build_source_adjacency(self.forward["src"], verbose=0)
            self.gradient = laplacian(adjacency).toarray().astype(np.float32)
            self.sigma_s = np.identity(n_dipoles) @ abs(self.gradient)
        else:
            self.gradient = None
            self.sigma_s = np.identity(n_dipoles)

        # Scale regularization against the actual sensor-space model term
        # that alpha regularizes: Sigma_b = alpha*I + L*Sigma_s*L^T.
        reg_reference = leadfield @ self.sigma_s @ leadfield.T
        reg_reference = 0.5 * (reg_reference + reg_reference.T)
        data_cov = self.data_covariance(data, center=True, ddof=1)
        if not np.all(np.isfinite(reg_reference)) or np.linalg.norm(reg_reference) == 0:
            reg_reference = data_cov
        self.get_alphas(reference=reg_reference)

        inverse_operators = []
        for alpha in self.alphas:
            inverse_operator = self.make_gamma_map_inverse_operator(
                data, alpha, max_iter=max_iter
            )
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [
            InverseOperator(inverse_operator @ wf.sensor_transform, self.name)
            for inverse_operator in inverse_operators
        ]
        return self

    def make_gamma_map_inverse_operator(self, B, alpha, max_iter=100):
        """Computes the gamma MAP inverse operator based on the M/EEG data.

        Parameters
        ----------
        B : numpy.ndarray
            The M/EEG data matrix (channels, time points).
        alpha : float
            The regularization parameter.
        max_iter : int
            Maximum numbers of iterations to find the optimal hyperparameters.
            max_iter = 1 corresponds to sLORETA.

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

        gammas = np.ones(ds, dtype=float)
        sigma_e = float(alpha) * np.identity(db)
        sigma_s_is_identity = np.array_equal(self.sigma_s, np.identity(ds))

        for _k in range(max_iter):
            old_gammas = gammas.copy()

            # sigma_b = alpha*I + L @ diag(gammas) @ sigma_s @ L.T
            if sigma_s_is_identity:
                # (L * gammas) @ L.T avoids creating d×d diagonal matrix
                sigma_b = sigma_e + (L * gammas) @ L.T
            else:
                sigma_s_hat = gammas[:, None] * self.sigma_s
                sigma_b = sigma_e + L @ sigma_s_hat @ L.T
            sigma_b = 0.5 * (sigma_b + sigma_b.T)
            sigma_b_inv = np.linalg.inv(sigma_b)

            # Compute sigma_b_inv @ L once, reuse for both terms
            SiL = sigma_b_inv @ L  # (m, d)

            # term_1: gamma-scaled RMS of projected data
            # L.T @ sigma_b_inv @ B = SiL.T @ B (reuse SiL transposed)
            LtSiB = SiL.T @ B  # (d, t)
            term_1 = (gammas / np.sqrt(n)) * np.sqrt(
                np.sum(LtSiB ** 2, axis=1)
            )
            # diag(L.T @ sigma_b_inv @ L) = column-wise dot product
            denom = np.sum(L * SiL, axis=0)  # (d,)
            denom = np.maximum(denom, 1e-15)
            term_2 = 1 / np.sqrt(denom)

            gammas = term_1 * term_2
            if not np.all(np.isfinite(gammas)) or np.linalg.norm(gammas) == 0:
                gammas = old_gammas
                break

        gamma_max = float(np.max(gammas))
        if not np.isfinite(gamma_max) or gamma_max <= 0:
            gammas_final = np.ones_like(gammas)
        else:
            gammas_final = gammas / gamma_max

        # Final inverse: diag(gammas_final) @ sigma_s @ L.T @ inv(...)
        if sigma_s_is_identity:
            sigma_b_final = sigma_e + (L * gammas_final) @ L.T
            inverse_operator = (gammas_final[:, None] * L.T) @ np.linalg.inv(sigma_b_final)
        else:
            sigma_s_hat = gammas_final[:, None] * self.sigma_s
            inverse_operator = (
                sigma_s_hat @ L.T @ np.linalg.inv(sigma_e + L @ sigma_s_hat @ L.T)
            )

        return inverse_operator

    @staticmethod
    def frob(x):
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        return np.sqrt(np.trace(x @ x.T))
