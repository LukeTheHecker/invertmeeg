from copy import deepcopy

import mne
import numpy as np
from scipy.sparse.csgraph import laplacian

from ..base import BaseSolver, InverseOperator, SolverMeta


class SolverGammaMAPMSP(BaseSolver):
    """Class for the Gamma Maximum A Posteriori (Gamma-MAP) inverse solution
    using multiple sparse priors (MSP).

    References
    ----------
    Wipf, D., & Nagarajan, S. (2009). A unified Bayesian framework for MEG/EEG
    source imaging. NeuroImage, 44(3), 947-966.

    """

    meta = SolverMeta(
        acronym="Gamma-MAP-MSP",
        full_name="Gamma MAP with MSP Priors",
        category="Bayesian",
        description=(
            "Gamma-MAP sparse Bayesian inverse approach combined with MSP-style "
            "spatial smoothness/patch priors."
        ),
        references=[
            "Wipf, D., & Nagarajan, S. (2009). A unified Bayesian framework for MEG/EEG source imaging. NeuroImage, 44(3), 947–966.",
        ],
    )

    def __init__(self, name="Gamma-MAP-MSP", **kwargs):
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
        max_iter : int
            Maximum numbers of iterations to find the optimal hyperparameters.
            max_iter = 1 corresponds to sLORETA-like solution.
        p : 0 < p < 2
            Hyperparameter which controls sparsity. Default: p = 0
        smoothness_order : int
            Controls the smoothness prior. The higher this integer, the higher
            the pursued smoothness of the inverse solution.

        Return
        ------
        self : object returns itself for convenience
        """
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)
        data_cov = self.data_covariance(data, center=True, ddof=1)

        inverse_operators = []
        self.get_alphas(reference=data_cov)
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

            # gammas = ((1/n) * np.sqrt(np.sum(( np.diag(gammas) @ L_smooth.T @ sigma_b_inv @ B )**2, axis=1)))**((2-p)/2)

            term_1 = (gammas / np.sqrt(n)) * np.sqrt(
                np.sum((L_smooth.T @ sigma_b_inv @ B) ** 2, axis=1)
            )
            term_2 = 1 / np.sqrt(np.diagonal(L_smooth.T @ sigma_b_inv @ L_smooth))
            gammas = term_1 * term_2

            if np.linalg.norm(gammas) == 0:
                gammas = old_gammas
                break
            # print(gammas.min(), gammas.max())
            # gammas /= np.linalg.norm(gammas)

        # Smooth gammas according to smooth priors
        gammas_final = abs(gammas @ gradient)
        gammas_final = gammas / gammas.max()

        sigma_s_hat = np.diag(gammas_final) @ sigma_s
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
        """Create a smooth prior on the covariance matrix.

        Parameters
        ----------
        L : numpy.ndarray
            Leadfield matrix (channels, dipoles)
        smoothness_order : int
            The higher the order, the smoother the prior.

        Return
        ------
        L : numpy.ndarray
            The smoothed Leadfield matrix (channels, dipoles)
        gradient : numpy.ndarray
            The smoothness gradient (laplacian matrix)

        """
        adjacency = mne.spatial_src_adjacency(self.forward["src"], verbose=0)
        gradient = laplacian(adjacency).toarray().astype(np.float32)

        for _ in range(smoothness_order):
            gradient = gradient @ gradient
        L = L @ abs(gradient)
        # L -= L.mean(axis=0)
        L /= np.linalg.norm(L, axis=0)
        return L, gradient
