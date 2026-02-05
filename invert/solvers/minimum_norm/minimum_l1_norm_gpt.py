import logging
from copy import deepcopy

import mne
import numpy as np

from ..base import BaseSolver, SolverMeta
from .utils import soft_threshold

logger = logging.getLogger(__name__)


class SolverMinimumL1NormGPT(BaseSolver):
    """Class for the Minimum Current Estimate inverse solution using
        interesting code from the Chat GPT AI by openai.com (GPT-solver).

        I (Lukas Hecker) prompted the task to write a sparsified eLORETA-type
        inverse solution and this came up with little adjustments required.

        I can't express how weird it is for me, too.

    References
    ----------
    Open AI chat GPT (openai.com)

    """

    meta = SolverMeta(
        acronym="GPT",
        full_name="GPT-derived Sparse Solver",
        category="Experimental",
        description=(
            "Experimental sparse inverse solver derived from an LLM-generated "
            "prototype and lightly adapted; not a published algorithm."
        ),
        references=["tbd"],
        internal=True,
    )

    def __init__(self, name="GPT Solver", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, *args, alpha="auto", **kwargs):
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
        self.inverse_operators = []
        return self

    def apply_inverse_operator(
        self, mne_obj, max_iter=1000, l1_reg=1e-3, tol=1e-2, depth_weighting=0.5
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

        source_mat = self.solver_wrap(
            data,
            max_iter=max_iter,
            l1_reg=l1_reg,
            tol=tol,
            depth_weighting=depth_weighting,
        )
        stc = self.source_to_object(source_mat)
        return stc

    def solver_wrap(
        self, y_mat, max_iter=1000, l1_reg=1e-3, tol=1e-2, depth_weighting=0.5
    ):
        srcs = []
        for y in y_mat.T:
            srcs.append(
                self.solve(
                    y,
                    max_iter=max_iter,
                    l1_reg=l1_reg,
                    tol=tol,
                    depth_weighting=depth_weighting,
                )
            )
        return np.stack(srcs, axis=1)

    def solve(self, y, l1_reg=1e-3, max_iter=1000, tol=1e-2, depth_weighting=0.5):
        """
        Solves the EEG inverse problem:
            min_x ||y - Ax||_2^2 + l1_reg * ||x||_1 + l2_reg * ||x||_2^2
        using the FISTA algorithm.

        Parameters
        ----------
        y : ndarray, shape (m,)
            EEG measurements.
        l1_reg : float, optional (default: 1e-3)
            L1 regularization strength.
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
        L = np.linalg.norm(A, ord=2) ** 2
        lr = 1.0 / L

        # Calculate initial guess
        x0 = np.linalg.pinv(A) @ y_scaled
        # Scale to unit norm
        x0 /= np.linalg.norm(x0)

        x = x0.copy()

        for _i in range(max_iter):
            x_prev = x.copy()
            # Gradient descent step (ISTA, no momentum)
            x = x - lr * self.grad_f(x, A, y_scaled)
            # Soft thresholding step
            x = soft_threshold(x, l1_reg * lr)

            # Check stopping criteria
            if np.linalg.norm(x) == 0:
                x = x_prev
                break
            if np.linalg.norm(x - x_prev) < tol:
                break
        # Rescale source
        x = x * norm_y

        return x

    @staticmethod
    def grad_f(x, A, y_scaled):
        """Gradient of the smooth part: A.T @ (A @ x - y)"""
        return A.T.dot(A.dot(x) - y_scaled)
