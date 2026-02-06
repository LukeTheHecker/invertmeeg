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
        l1_reg=5e-2,
        l2_reg=0,
        tol=1e-4,
        depth_weighting=0.0,
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
        depth_weighting : float
            Exponent for depth weighting compensation (0 = no weighting, 1 = full compensation).
            Default 0.0 disables depth weighting.
        center_data : bool
            If True, subtract the per-timepoint channel mean from data.
        scale_l1_by_lmax : bool
            If True, interpret ``l1_reg`` as a fraction of the data-dependent
            maximum L1 penalty (lambda_max = max(abs(A.T @ Y))).

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
            center_data=center_data,
            scale_l1_by_lmax=scale_l1_by_lmax,
        )
        stc = self.source_to_object(source_mat)
        return stc

    def fista_wrap(
        self,
        y_mat,
        max_iter=1000,
        l1_reg=5e-2,
        l2_reg=0,
        tol=1e-4,
        depth_weighting=0.0,
        center_data=True,
        scale_l1_by_lmax=True,
    ):
        return self.fista(
            y_mat,
            max_iter=max_iter,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            tol=tol,
            depth_weighting=depth_weighting,
            center_data=center_data,
            scale_l1_by_lmax=scale_l1_by_lmax,
        )

    def fista(
        self,
        y,
        l1_reg=5e-2,
        l2_reg=0,
        max_iter=1000,
        tol=1e-4,
        depth_weighting=0.0,
        center_data=True,
        scale_l1_by_lmax=True,
    ):
        """
        Solves the EEG inverse problem:
            min_X 0.5 * ||Y - A X||_F^2 + lambda * ||X||_1 + 0.5 * l2_reg * ||X||_F^2
        using FISTA over all timepoints jointly.

        Parameters
        ----------
        y : ndarray, shape (m, t)
            EEG measurements over time.
        A : ndarray, shape (m, n)
            Forward model.
        x0 : ndarray, shape (n, t)
            Initial guess for source currents.
        l1_reg : float, optional (default: 1e-3)
            L1 regularization strength. If ``scale_l1_by_lmax=True``, this is
            interpreted as a fraction of lambda_max.
        l2_reg : float, optional (default: 0)
            L2 regularization strength.
        max_iter : int, optional (default: 1000)
            Maximum number of iterations to run.
        tol : float, optional (default: 1e-6)
            Tolerance for the stopping criteria.
        depth_weighting : float, optional (default: 0.0)
            Exponent for depth weighting to compensate for superficial source bias.
            This is applied only if the base class did not already prepare/depth-weight
            the leadfield.

        Returns
        -------
        x : ndarray, shape (n, t)
            Estimated CSDs for all timepoints.
        """
        y_mat = np.asarray(y, dtype=float)
        if y_mat.ndim == 1:
            y_mat = y_mat[:, np.newaxis]

        A = deepcopy(self.leadfield)
        if not self.prep_leadfield and depth_weighting > 0:
            leadfield_norms = np.linalg.norm(A, axis=0)
            leadfield_norms = np.maximum(leadfield_norms, 1e-12)
            depth_weights = leadfield_norms**depth_weighting
            A /= leadfield_norms
            A *= depth_weights

        Y = y_mat.copy()
        if center_data:
            Y -= Y.mean(axis=0, keepdims=True)
        if np.allclose(Y, 0):
            return np.zeros((A.shape[1], Y.shape[1]), dtype=float)

        # Compute step size from Lipschitz constant
        L = np.linalg.norm(A, ord=2) ** 2 + l2_reg
        L = max(float(L), 1e-12)
        lr = 1.0 / L

        if scale_l1_by_lmax:
            lambda_max = float(np.max(np.abs(A.T @ Y)))
            lambda_eff = float(l1_reg) * lambda_max
        else:
            lambda_eff = float(l1_reg)
        lambda_eff = max(lambda_eff, 0.0)

        # Least-squares warm start keeps early iterations stable.
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
            x = soft_threshold(x, lambda_eff * lr)
            # Update z and t (FISTA momentum)
            t_prev = t
            t = (1 + (1 + 4 * t**2) ** 0.5) / 2
            z = x + (t_prev - 1) / t * (x - x_prev)

            if np.linalg.norm(z) == 0 or np.isnan(z).any():
                break
            # Check stopping criteria
            denom = max(np.linalg.norm(x_prev), 1e-12)
            if np.linalg.norm(x - x_prev) / denom < tol:
                break

        return x
