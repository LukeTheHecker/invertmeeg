import logging
from copy import deepcopy

import mne
import numpy as np

from ..base import BaseSolver, SolverMeta
from .utils import soft_threshold

logger = logging.getLogger(__name__)


class SolverMinimumL1Norm(BaseSolver):
    """Class for the Minimum Current Estimate (MCE) inverse solution using the
        FISTA solver [1].

    References
    ----------
    [1] Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding
    algorithm for linear inverse problems. SIAM journal on imaging sciences,
    2(1), 183-202.
    """

    meta = SolverMeta(
        acronym="MCE",
        full_name="Minimum Current Estimate (L1)",
        category="Sparse / Mixed Norm",
        description=(
            "Sparse (L1) distributed inverse, typically solved via iterative "
            "shrinkage/thresholding to promote focal source estimates."
        ),
        references=[
            "Uutela, K., Hämäläinen, M., & Somersalo, E. (1999). Visualization of magnetoencephalographic data using minimum current estimates. NeuroImage, 10(2), 173–180.",
            "Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. SIAM Journal on Imaging Sciences, 2(1), 183–202.",
        ],
    )

    def __init__(self, name="Minimum Current Estimate", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(
        self,
        forward,
        *args,
        alpha="auto",
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

        self.noise_cov = noise_cov
        self.inverse_operators = []
        return self

    def apply_inverse_operator(
        self,
        mne_obj,
        max_iter=1000,
        l1_reg=1e-3,
        l2_reg=0,
        tol=1e-2,
        depth_weighting=0.5,
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
        depth_weighting : float
            Exponent for depth weighting compensation (0 = no weighting, 1 = full compensation).
            Default 0.5 balances depth bias reduction with noise sensitivity.

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
            depth_weighting=depth_weighting,
        )
        stc = self.source_to_object(source_mat)
        return stc

    def fista_wrap(
        self, y_mat, max_iter=1000, l1_reg=1e-3, l2_reg=0, tol=1e-2, depth_weighting=0.5
    ):
        srcs = []
        for y in y_mat.T:
            srcs.append(
                self.fista(
                    y,
                    max_iter=max_iter,
                    l1_reg=l1_reg,
                    l2_reg=l2_reg,
                    tol=tol,
                    depth_weighting=depth_weighting,
                )
            )
        return np.stack(srcs, axis=1)

    def fista(
        self, y, l1_reg=1e-3, l2_reg=0, max_iter=1000, tol=1e-2, depth_weighting=0.5
    ):
        """
        Solves the EEG inverse problem:
            min_x ||y - Ax||_2^2 + l1_reg * ||x||_1 + l2_reg * ||x||_2^2
        using the FISTA algorithm.

        Parameters
        ----------
        y : ndarray, shape (m,)
            EEG measurements.
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
        depth_weighting : float, optional (default: 0.5)
            Exponent for depth weighting to compensate for superficial source bias.

        Returns
        -------
        x : ndarray, shape (n,)
            Estimated CSDs.
        """

        A = deepcopy(self.leadfield)
        leadfield_norms = np.linalg.norm(A, axis=0)
        depth_weights = leadfield_norms**depth_weighting
        A /= leadfield_norms
        A *= depth_weights

        y_scaled = y.copy()
        # Rereference
        y_scaled -= y_scaled.mean()
        # Scale to unit norm
        norm_y = np.linalg.norm(y_scaled)
        y_scaled /= norm_y

        # Compute step size from Lipschitz constant
        L = np.linalg.norm(A, ord=2) ** 2 + l2_reg
        lr = 1.0 / L

        def grad_f(x):
            """Gradient of the smooth part: A.T @ (A @ x - y) + l2_reg * x"""
            return A.T @ (A @ x - y_scaled) + l2_reg * x

        # Calculate initial guess
        x0 = np.linalg.pinv(A) @ y_scaled
        # Scale to unit norm
        x0 /= np.linalg.norm(x0)

        x = x0.copy()
        z = x0.copy()

        t = 1.0
        for _i in range(max_iter):
            x_prev = x.copy()
            # Gradient descent step on momentum variable
            x = z - lr * grad_f(z)
            # Soft thresholding step (proximal for L1)
            x = soft_threshold(x, l1_reg * lr)
            # Update z and t (FISTA momentum)
            t_prev = t
            t = (1 + (1 + 4 * t**2) ** 0.5) / 2
            z = x + (t_prev - 1) / t * (x - x_prev)

            if np.linalg.norm(z) == 0:
                break
            # Check stopping criteria
            if np.linalg.norm(x - x_prev) < tol:
                break
        # Rescale source
        x = x * norm_y

        return x
