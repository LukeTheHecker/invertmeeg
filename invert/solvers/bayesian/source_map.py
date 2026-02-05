from copy import deepcopy

import mne
import numpy as np
from scipy.sparse.csgraph import laplacian

from ..base import BaseSolver, InverseOperator, SolverMeta


class SolverSourceMAP(BaseSolver):
    """Class for the Source Maximum A Posteriori (Source-MAP) inverse solution [1].

    References
    ----------
    Wipf, D., & Nagarajan, S. (2009). A unified Bayesian framework for MEG/EEG
    source imaging. NeuroImage, 44(3), 947-966.

    """

    meta = SolverMeta(
        acronym="Source-MAP",
        full_name="Source Maximum A Posteriori",
        category="Bayesian",
        description=(
            "Sparse Bayesian inverse solution computing a source MAP estimate "
            "in the unified Bayesian source imaging framework."
        ),
        references=[
            "Wipf, D., & Nagarajan, S. (2009). A unified Bayesian framework for MEG/EEG source imaging. NeuroImage, 44(3), 947–966.",
        ],
    )

    def __init__(self, name="Source-MAP", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(
        self,
        forward,
        mne_obj,
        *args,
        alpha="auto",
        smoothness_prior=False,
        max_iter=100,
        p=0.5,
        verbose=0,
        **kwargs,
    ):
        """Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        alpha : float
            The regularization parameter.
        max_iter : int
            Maximum numbers of iterations to find the optimal hyperparameters.
            max_iter = 1 corresponds to sLORETA.
        p : 0 < p < 2
            Hyperparameter which controls sparsity. Default: p = 0.5

        Return
        ------
        self : object returns itself for convenience

        """
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)
        leadfield = self.leadfield
        n_chans, n_dipoles = leadfield.shape
        data_cov = self.data_covariance(data, center=True, ddof=1)
        self.get_alphas(reference=data_cov)

        if smoothness_prior:
            adjacency = mne.spatial_src_adjacency(self.forward["src"], verbose=0)
            self.gradient = laplacian(adjacency).toarray().astype(np.float32)
            self.sigma_s = np.identity(n_dipoles) @ abs(self.gradient)
        else:
            self.gradient = None
            self.sigma_s = np.identity(n_dipoles)

        inverse_operators = []
        for alpha in self.alphas:
            inverse_operator = self.make_source_map_inverse_operator(
                data, alpha, max_iter=max_iter, p=p
            )
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        return self

    def make_source_map_inverse_operator(self, B, alpha, max_iter=100, p=0.5):
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
        L /= np.linalg.norm(L, axis=0)

        # Data Covariance Matrix
        # Cb = B @ B.T
        gammas = np.ones(ds)
        sigma_e = alpha * np.identity(db)

        sigma_b = sigma_e + L @ self.sigma_s @ L.T
        sigma_b_inv = np.linalg.inv(sigma_b)

        for _k in range(max_iter):
            # print(k)
            old_gammas = deepcopy(gammas)

            gammas = (
                (1 / n)
                * np.sqrt(
                    np.sum((np.diag(gammas) @ L.T @ sigma_b_inv @ B) ** 2, axis=1)
                )
            ) ** ((2 - p) / 2)

            if np.linalg.norm(gammas) == 0:
                gammas = old_gammas
                break
            # gammas /= np.linalg.norm(gammas)

        gammas_final = gammas / gammas.max()
        sigma_s_hat = (
            np.diag(gammas_final) @ self.sigma_s
        )  #  np.array([gammas_final[i] * C[i] for i in range(ds)])
        inverse_operator = (
            sigma_s_hat @ L.T @ np.linalg.inv(sigma_e + L @ sigma_s_hat @ L.T)
        )

        # This way the inverse operator would be applied to M/EEG matrix B:
        # S = inverse_operator @ B

        return inverse_operator

    @staticmethod
    def frob(x):
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        return np.sqrt(np.trace(x @ x.T))
