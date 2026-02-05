import logging
from copy import deepcopy

import mne
import numpy as np

from ...util import (
    best_index_residual,
    thresholding,
)
from ..base import BaseSolver, SolverMeta

logger = logging.getLogger(__name__)


class SolverSOMP(BaseSolver):
    """Class for the Simultaneous Orthogonal Matching Pursuit (S-OMP) inverse
        solution.

    References
    ----------
    [1] Duarte, M. F., & Eldar, Y. C. (2011). Structured compressed sensing:
    From theory to applications. IEEE Transactions on signal processing, 59(9),
    4053-4085.

    [2] Donoho, D. L. (2006). Compressed sensing. IEEE Transactions on
    information theory, 52(4), 1289-1306.

    """

    meta = SolverMeta(
        acronym="SOMP",
        full_name="Simultaneous Orthogonal Matching Pursuit",
        category="Matching Pursuit",
        description=(
            "Multi-measurement-vector extension of OMP that selects atoms jointly across "
            "time/conditions by aggregating per-atom correlation norms."
        ),
        references=[
            "Tropp, J. A., Gilbert, A. C., & Strauss, M. J. (2006). Algorithms for simultaneous sparse approximation. Part I: Greedy pursuit. Signal Processing, 86(3), 572–588.",
        ],
    )

    def __init__(self, name="Simultaneous Orthogonal Matching Pursuit", **kwargs):
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
        # Store original leadfield for coefficient estimation
        self.leadfield_original = self.leadfield.copy()
        # Use robust normalization from base class for atom selection
        self.leadfield_normed = self.robust_normalize_leadfield(self.leadfield)

        self.inverse_operators = []
        return self

    def apply_inverse_operator(
        self, mne_obj, K="auto", max_iter=None
    ) -> mne.SourceEstimate:  # type: ignore
        """Apply the inverse operator.
        Parameters
        ----------
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        K : int
            The number of atoms to select per iteration.
        max_iter : int
            The maximum number of iterations.

        Return
        ------
        stc : mne.SourceEstimate
            The mne Source Estimate object
        """
        data = self.unpack_data_obj(mne_obj)
        source_mat = self.calc_somp_solution(data, K=K, max_iter=max_iter)
        stc = self.source_to_object(source_mat)
        return stc

    def calc_somp_solution(self, y, K="auto", max_iter=None):
        """Calculates the S-OMP inverse solution.

        Parameters
        ----------
        y : numpy.ndarray
            The data matrix (channels, time).
        K : ["auto", int]
            The number of atoms to select per iteration.
        max_iter : int
            The maximum number of iterations.
        Return
        ------
        x_hat : numpy.ndarray
            The inverse solution (dipoles, time)

        """
        n_chans, n_time = y.shape
        if K == "auto":
            K = int(n_chans / 2)
        if max_iter is None:
            max_iter = int(n_chans / 2)
        _, n_dipoles = self.leadfield.shape

        x_hat = np.zeros((n_dipoles, n_time))
        x_hats = [deepcopy(x_hat)]
        residuals = np.array(
            [
                np.linalg.norm(y - self.leadfield_original @ x_hat),
            ]
        )
        source_norms = np.array(
            [
                0,
            ]
        )

        R = deepcopy(y)
        omega = np.array([])
        q = 2
        for _i in range(max_iter):
            # Use normalized leadfield for atom selection
            b_n = np.linalg.norm(self.leadfield_normed.T @ R, axis=1, ord=q)

            b_thresh = thresholding(b_n, K)
            new_atoms = np.where(b_thresh != 0)[0]
            omega = np.append(omega, new_atoms)
            omega = np.unique(omega.astype(int))  # Keep unique for SOMP

            # Use robust inverse from base class for coefficient estimation
            if len(omega) > 0:
                L_omega = self.leadfield_original[:, omega]
                x_hat[omega] = self.robust_inverse_solution(L_omega, y)

            y_hat = self.leadfield_original @ x_hat
            R = y - y_hat

            residuals = np.append(residuals, np.linalg.norm(R))
            source_norms = np.append(source_norms, np.sum(x_hat**2))
            x_hats.append(deepcopy(x_hat))

            # Early stopping if residual starts increasing
            if len(residuals) > 1 and residuals[-1] > residuals[-2]:
                break

        x_hat = best_index_residual(residuals, x_hats, plot=False)

        return x_hat
