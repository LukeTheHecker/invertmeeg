from copy import deepcopy

import mne
import numpy as np
from scipy.sparse.csgraph import laplacian

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
        mne_obj,
        *args,
        alpha="auto",
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
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)
        leadfield = self.leadfield
        n_chans, _ = leadfield.shape
        data_cov = self.data_covariance(data, center=True, ddof=1)
        self.get_alphas(reference=data_cov)

        inverse_operators = []
        for alpha in self.alphas:
            inverse_operator = self.make_source_map_inverse_operator(
                data, alpha, max_iter=max_iter, p=p, smoothness_order=smoothness_order
            )
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
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

        L = deepcopy(self.leadfield)
        db, n = B.shape
        ds = L.shape[1]

        # Ensure Common average reference
        B -= B.mean(axis=0)
        L -= L.mean(axis=0)

        # Data Covariance Matrix
        # Cb = B @ B.T
        L_smooth, gradient = self.get_smooth_prior_cov(L, smoothness_order)
        gammas = np.ones(ds)
        sigma_e = alpha * np.identity(db)
        sigma_s = np.identity(
            ds
        )  # identity leads to weighted minimum L2 Norm-type solution
        sigma_b = sigma_e + L_smooth @ sigma_s @ L_smooth.T
        sigma_b_inv = np.linalg.inv(sigma_b)

        for _k in range(max_iter):
            # print(k)
            old_gammas = deepcopy(gammas)

            gammas = (
                (1 / n)
                * np.sqrt(
                    np.sum(
                        (np.diag(gammas) @ L_smooth.T @ sigma_b_inv @ B) ** 2, axis=1
                    )
                )
            ) ** ((2 - p) / 2)

            if np.linalg.norm(gammas) == 0:
                gammas = old_gammas
                break
            # print(gammas.min(), gammas.max())
            # gammas /= np.linalg.norm(gammas)

        # Smooth gammas according to smooth priors
        gammas_final = abs(gammas @ gradient)
        gammas_final = gammas / gammas.max()

        sigma_s_hat = (
            np.diag(gammas_final) @ sigma_s
        )  #  np.array([gammas_final[i] * C[i] for i in range(ds)])
        inverse_operator = (
            sigma_s_hat @ L.T @ np.linalg.inv(sigma_e + L @ sigma_s_hat @ L.T)
        )
        # S = inverse_operator @ B
        return inverse_operator

    @staticmethod
    def frob(x):
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        return np.sqrt(np.trace(x @ x.T))

    def get_smooth_prior_cov(self, L, smoothness_order):
        adjacency = mne.spatial_src_adjacency(self.forward["src"], verbose=0)
        gradient = laplacian(adjacency).toarray().astype(np.float32)

        for _i in range(smoothness_order):
            gradient = gradient @ gradient
        L = L @ abs(gradient)
        # L -= L.mean(axis=0)
        L /= np.linalg.norm(L, axis=0)
        return L, gradient
