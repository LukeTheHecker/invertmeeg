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
        max_iter=100,
        l1_spatial=1e-3,
        l2_temporal=1e-3,
        tol=1e-6,
        depth_weighting=0.5,
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
            Controls the temporal L2 regularization
        tol : float
            Tolerance at which convergence is met.
        depth_weighting : float
            Exponent for depth weighting compensation (0 = no weighting, 1 = full compensation).
            Default 0.5 balances depth bias reduction with noise sensitivity.
            Use 0 to disable depth weighting entirely.

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
        )
        stc = self.source_to_object(source_mat)
        return stc

    def fista_eeg(
        self,
        y,
        alpha="auto",
        l1_spatial=1e-3,
        l2_temporal=1e-3,
        max_iter=1000,
        tol=1e-6,
        depth_weighting=0.5,
    ):
        """
        Solves the EEG inverse problem using FISTA with L1 regularization on the spatial
        dimension and L2 regularization on the temporal dimension.

        Parameters:
        - A: array of shape (n_sensors, n_sources)
        - y: array of shape (n_sensors, n_timepoints)
        - l1_spatial: float, strength of L1 regularization on the spatial dimension
        - l2_temporal: float, strength of L2 regularization on the temporal dimension
        - max_iter: int, maximum number of iterations
        - tol: float, tolerance for convergence
        - depth_weighting: float, exponent for depth weighting (0=no weighting, 1=full compensation)
                          Default 0.5 provides a balance between depth bias and noise sensitivity

        Returns:
        - x: array of shape (n_sources, n_timepoints), the solution to the EEG inverse problem
        """
        A = self.leadfield.copy()

        # Compute depth weights to compensate for superficial bias
        # Larger column norms = superficial sources
        leadfield_norms = np.linalg.norm(A, axis=0)
        depth_weights = leadfield_norms**depth_weighting

        # Normalize leadfield columns
        A /= leadfield_norms

        # Apply depth weighting to compensate for the bias
        A *= depth_weights

        norm_y = np.linalg.norm(y)
        y -= y.mean(axis=0)
        y_scaled = y / norm_y

        # Regularization
        if alpha == "auto":
            alpha = l1_spatial

        # Initialize x and z to be the same, and set t to 1
        W = np.diag(np.linalg.norm(A, axis=0))
        WTW = np.linalg.inv(W.T @ W)
        LWTWL = A @ WTW @ A.T
        inverse_operator = (
            WTW @ A.T @ np.linalg.inv(LWTWL + alpha * np.identity(A.shape[0]))
        )
        x = z = inverse_operator @ y_scaled

        # x = z = np.linalg.pinv(A) @ y_scaled

        x /= np.linalg.norm(x)

        t = 1

        # Compute the Lipschitz constant
        L = np.linalg.norm(A, ord=2) ** 2

        for _i in range(max_iter):
            # Compute the gradient of the smooth part using momentum variable z
            grad = A.T @ (A @ z - y_scaled)

            # Compute the proximal operator of the L1 regularization
            x_new = np.sign(z - grad / L) * np.maximum(
                np.abs(z - grad / L) - l1_spatial / L, 0
            )

            # Compute the proximal operator of the L2 temporal regularization (per-source temporal norm)
            norm_x = np.linalg.norm(x_new, ord=2, axis=1)
            # Avoid division by zero
            norm_x = np.maximum(norm_x, 1e-10)
            scale = np.maximum(norm_x - l2_temporal / L, 0) / norm_x
            x_new = x_new * scale[:, np.newaxis]

            # Update t and z
            t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
            z_new = x_new + (t - 1) / t_new * (x_new - x)

            # Check for convergence
            diff = np.linalg.norm(x_new - x)
            logger.debug(diff)
            if diff < tol or np.any(abs(x).max(axis=0) < 1e-10):
                break

            # Update x, t, and z
            x = x_new
            t = t_new
            z = z_new
        logger.info("convergence after %d", _i)
        # Rescale Sources
        x = x * norm_y

        return x

    @staticmethod
    def calc_norm(x, n_time):
        return np.sqrt((x**2).sum() / n_time)
