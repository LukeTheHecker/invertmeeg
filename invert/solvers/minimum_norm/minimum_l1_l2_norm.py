import logging

import mne
import numpy as np

from ..base import BaseSolver, SolverMeta

logger = logging.getLogger(__name__)


class SolverMinimumL1L2Norm(BaseSolver):
    """Class for the Minimum L1-L2 Norm solution (MCE) inverse solution. It
        imposes a L1 norm on the source and L2 on the source time courses.

    References
    ----------
    [!] Missing reference - please contact developers if you have it!

    """

    meta = SolverMeta(
        acronym="MxNE",
        full_name="Mixed-Norm Estimate (L1/L2)",
        category="Sparse / Mixed Norm",
        description=(
            "Mixed-norm (L1/L2) inverse that promotes group sparsity across time "
            "by applying an L1 penalty over sources and an L2 norm over each "
            "source time course."
        ),
        references=[
            "Gramfort, A., Kowalski, M., & Hämäläinen, M. (2012). Mixed-norm estimates for the M/EEG inverse problem using accelerated gradient methods. Physics in Medicine and Biology, 57(7), 1937–1961.",
        ],
    )

    def __init__(self, name="Minimum L1-L2 Norm", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, *args, alpha=0.01, **kwargs):
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

        return self

    def apply_inverse_operator(
        self,
        mne_obj,
        alpha="auto",
        max_iter=1000,
        l1_spatial=5e-2,
        l2_temporal=0,
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
        l1_spatial : float
            Controls the spatial L1 regularization
        l2_temporal : float
            Controls additional temporal ridge regularization.
        tol : float
            Tolerance at which convergence is met.
        depth_weighting : float
            Exponent for depth weighting compensation (0 = no weighting, 1 = full compensation).
            Default 0.0 disables depth weighting.
            Use 0 to disable depth weighting entirely.
        center_data : bool
            If True, subtract the per-timepoint channel mean from data.
        scale_l1_by_lmax : bool
            If True, interpret ``l1_spatial`` as a fraction of
            ``lambda_max = max_i ||(A^T Y)_i||_2``.

        Return
        ------
        stc : mne.SourceEstimate
            The mne Source Estimate object.
        """

        data = self.unpack_data_obj(mne_obj)

        source_mat = self.fista_eeg(
            data,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            l1_spatial=l1_spatial,
            l2_temporal=l2_temporal,
            depth_weighting=depth_weighting,
            center_data=center_data,
            scale_l1_by_lmax=scale_l1_by_lmax,
        )
        stc = self.source_to_object(source_mat)
        return stc

    @staticmethod
    def _group_soft_threshold(X, threshold):
        """Row-wise proximal operator for the L21 norm."""
        row_norms = np.linalg.norm(X, axis=1, keepdims=True)
        row_norms = np.maximum(row_norms, 1e-12)
        scale = np.maximum(1.0 - threshold / row_norms, 0.0)
        return X * scale

    def fista_eeg(
        self,
        y,
        alpha="auto",
        l1_spatial=5e-2,
        l2_temporal=0,
        max_iter=1000,
        tol=1e-4,
        depth_weighting=0.0,
        center_data=True,
        scale_l1_by_lmax=True,
    ):
        """
        Solve a mixed-norm inverse problem over all timepoints jointly:
            min_X 0.5 ||Y - A X||_F^2 + lambda * sum_i ||X_i||_2
                  + 0.5 * l2_temporal * ||X||_F^2

        Parameters:
        - A: array of shape (n_sensors, n_sources)
        - y: array of shape (n_sensors, n_timepoints)
        - l1_spatial: float, mixed-norm regularization strength
        - l2_temporal: float, additional ridge regularization on X
        - max_iter: int, maximum number of iterations
        - tol: float, tolerance for convergence
        - depth_weighting: float, exponent for depth weighting (0=no weighting, 1=full compensation)
                          Default 0.0 disables depth weighting

        Returns:
        - x: array of shape (n_sources, n_timepoints), the solution to the EEG inverse problem
        """
        y_mat = np.asarray(y, dtype=float)
        if y_mat.ndim == 1:
            y_mat = y_mat[:, np.newaxis]

        A = self.leadfield.copy()
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

        if alpha != "auto":
            l1_spatial = float(alpha)

        if scale_l1_by_lmax:
            gram_proj = A.T @ Y
            lambda_max = float(np.max(np.linalg.norm(gram_proj, axis=1)))
            lambda_eff = float(l1_spatial) * lambda_max
        else:
            lambda_eff = float(l1_spatial)
        lambda_eff = max(lambda_eff, 0.0)

        L = np.linalg.norm(A, ord=2) ** 2 + float(l2_temporal)
        L = max(float(L), 1e-12)

        x = np.linalg.pinv(A) @ Y
        z = x.copy()
        t = 1.0

        for _i in range(max_iter):
            grad = A.T @ (A @ z - Y) + float(l2_temporal) * z
            v = z - grad / L
            x_new = self._group_soft_threshold(v, lambda_eff / L)
            t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
            z_new = x_new + (t - 1) / t_new * (x_new - x)
            denom = max(np.linalg.norm(x), 1e-12)
            if np.linalg.norm(x_new - x) / denom < tol:
                break
            x = x_new
            t = t_new
            z = z_new

        return x

    @staticmethod
    def calc_norm(x, n_time):
        return np.sqrt((x**2).sum() / n_time)
