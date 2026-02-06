import logging
from copy import deepcopy

import mne
import numpy as np
from scipy.sparse.csgraph import laplacian

from ..base import BaseSolver, SolverMeta
from .utils import soft_threshold

logger = logging.getLogger(__name__)


class SolverGFTMinimumL1Norm(BaseSolver):
    """Class for the Minimum Current Estimate (MCE) inverse solution using the
        FISTA solver [1].

    References
    ----------
    [1] Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding
    algorithm for linear inverse problems. SIAM journal on imaging sciences,
    2(1), 183-202.
    """

    meta = SolverMeta(
        acronym="GFT-MCE",
        full_name="Graph Fourier Minimum Current Estimate (L1)",
        category="Sparse / Mixed Norm",
        description=(
            "Minimum-current (L1) inverse solved in a truncated graph Fourier "
            "basis to encourage smoothness on the source-space graph."
        ),
        references=[
            "Uutela, K., Hämäläinen, M., & Somersalo, E. (1999). Visualization of magnetoencephalographic data using minimum current estimates. NeuroImage, 10(2), 173–180.",
            "Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. SIAM Journal on Imaging Sciences, 2(1), 183–202.",
            "Shuman, D. I., Narang, S. K., Frossard, P., Ortega, A., & Vandergheynst, P. (2013). The emerging field of signal processing on graphs: Extending high-dimensional data analysis to networks and other irregular domains. IEEE Signal Processing Magazine, 30(3), 83–98.",
        ],
        internal=True,
    )

    def __init__(self, name="Minimum Current Estimate", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(
        self,
        forward,
        *args,
        alpha="auto",
        n_modes=None,
        mode_fraction=1.0,
        high_freq_penalty=0.0,
        max_iter=1000,
        noise_cov=None,
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

        Return
        ------
        self : object returns itself for convenience
        """

        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        n_chans = self.leadfield.shape[0]
        if noise_cov is None:
            noise_cov = np.identity(n_chans)

        adjacency = mne.spatial_src_adjacency(forward["src"], verbose=0)
        graph_laplacian = laplacian(adjacency, normed=False).astype(float).toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(graph_laplacian)

        n_vertices = eigenvectors.shape[0]
        n_modes_use = self._resolve_n_modes(
            n_vertices=n_vertices, n_modes=n_modes, mode_fraction=mode_fraction
        )
        self.U = np.real(eigenvectors[:, :n_modes_use])
        self.graph_laplacian_eigenvalues = np.real(eigenvalues[:n_modes_use])

        max_eig = float(np.max(self.graph_laplacian_eigenvalues))
        if max_eig <= 0:
            normalized = np.zeros_like(self.graph_laplacian_eigenvalues)
        else:
            normalized = self.graph_laplacian_eigenvalues / max_eig
        self.mode_weights = 1.0 + float(high_freq_penalty) * normalized

        self.noise_cov = noise_cov
        self.inverse_operators = []
        return self

    @staticmethod
    def _resolve_n_modes(n_vertices: int, n_modes, mode_fraction: float) -> int:
        if n_modes is not None:
            n_modes_use = int(n_modes)
        else:
            n_modes_use = int(np.ceil(float(mode_fraction) * n_vertices))
        n_modes_use = max(2, n_modes_use)
        n_modes_use = min(n_vertices, n_modes_use)
        return n_modes_use

    def apply_inverse_operator(
        self,
        mne_obj,
        max_iter=1000,
        l1_reg=1e-2,
        l2_reg=0,
        tol=1e-4,
        center_data=True,
        scale_l1_by_lmax=True,
    ) -> mne.SourceEstimate:
        """Apply the inverse operator.

        Parameters
        ----------
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        max_iter : int
            Maximum number of iterations
        l1_reg : float
            Controls the spatial L1 regularization
        l2_reg : float
            Controls the spatial L2 regularization
        tol : float
            Tolerance at which convergence is met.
        center_data : bool
            If True, subtract the per-timepoint channel mean from data.
        scale_l1_by_lmax : bool
            If True, interpret ``l1_reg`` as a fraction of lambda_max.

        Return
        ------
        stc : mne.SourceEstimate
            The mne Source Estimate object.
        """
        data = self.unpack_data_obj(mne_obj)
        source_mat = self.fista_wrap(
            data,
            max_iter=max_iter,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            tol=tol,
            center_data=center_data,
            scale_l1_by_lmax=scale_l1_by_lmax,
        )
        stc = self.source_to_object(source_mat)
        return stc

    def fista_wrap(
        self,
        y_mat,
        max_iter=1000,
        l1_reg=1e-2,
        l2_reg=0,
        tol=1e-4,
        center_data=True,
        scale_l1_by_lmax=True,
    ):
        return self.fista(
            y_mat,
            max_iter=max_iter,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            tol=tol,
            center_data=center_data,
            scale_l1_by_lmax=scale_l1_by_lmax,
        )

    def fista(
        self,
        y,
        l1_reg=1e-2,
        l2_reg=0,
        max_iter=1000,
        tol=1e-4,
        center_data=True,
        scale_l1_by_lmax=True,
    ):
        """
        Solves the EEG inverse problem:
            min_x ||y - Ax||_2^2 + l1_reg * ||x||_1 + l2_reg * ||x||_2^2
        using the FISTA algorithm.

        Parameters
        ----------
        y : ndarray, shape (m, t)
            EEG measurements over time.
        A : ndarray, shape (m, n)
            Forward model.
        x0 : ndarray, shape (n,)
            Initial guess for the CSDs.
        l1_reg : float, optional (default: 1e-3)
            L1 regularization strength.
        l2_reg : float, optional (default: 0)
            L2 regularization strength.
        max_iter : int, optional (default: 1000)
            Maximum number of iterations to run.
        tol : float, optional (default: 1e-6)
            Tolerance for the stopping criteria.

        Returns
        -------
        x : ndarray, shape (n, t)
            Estimated CSDs.
        """

        A = deepcopy(self.leadfield) @ self.U

        y_mat = np.asarray(y, dtype=float)
        if y_mat.ndim == 1:
            y_mat = y_mat[:, np.newaxis]
        Y = y_mat.copy()
        if center_data:
            Y -= Y.mean(axis=0, keepdims=True)
        if np.allclose(Y, 0):
            return np.zeros((self.U.shape[1], Y.shape[1]), dtype=float)

        # Compute step size from Lipschitz constant
        L = np.linalg.norm(A, ord=2) ** 2 + l2_reg
        L = max(float(L), 1e-12)
        lr = 1.0 / L

        if scale_l1_by_lmax:
            weighted_proj = np.abs(A.T @ Y) / self.mode_weights[:, np.newaxis]
            lambda_max = float(np.max(weighted_proj))
            lambda_eff = float(l1_reg) * lambda_max
        else:
            lambda_eff = float(l1_reg)
        lambda_eff = max(lambda_eff, 0.0)

        # Calculate initial guess
        x0 = np.linalg.pinv(A) @ Y

        x = x0.copy()
        z = x0.copy()

        t = 1.0
        for _i in range(max_iter):
            x_prev = x.copy()
            # Gradient descent step on momentum variable
            grad = A.T @ (A @ z - Y) + l2_reg * z
            x = z - lr * grad
            # Soft thresholding step (proximal for L1)
            thresholds = (lambda_eff * lr) * self.mode_weights[:, np.newaxis]
            x = soft_threshold(x, thresholds)
            # Update z and t (FISTA momentum)
            t_prev = t
            t = (1 + (1 + 4 * t**2) ** 0.5) / 2
            z = x + (t_prev - 1) / t * (x - x_prev)

            if np.linalg.norm(z) == 0 or np.isnan(z).any():
                logger.warning("norm is zero")
                x = x_prev
                break
            # Check stopping criteria
            denom = max(np.linalg.norm(x_prev), 1e-12)
            if np.linalg.norm(x - x_prev) / denom < tol:
                logger.debug("criterion met")
                break
        return self.U @ x
