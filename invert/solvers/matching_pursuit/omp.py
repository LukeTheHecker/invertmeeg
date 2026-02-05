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


class SolverOMP(BaseSolver):
    """Class for the Orthogonal Matching Pursuit (OMP) inverse solution [1].
        The algorithm as described by [2] was implemented.

    References
    ----------
    [1] Tropp, J. A., & Gilbert, A. C. (2007). Signal recovery from random
    measurements via orthogonal matching pursuit. IEEE Transactions on
    information theory, 53(12), 4655-4666.

    [2] Duarte, M. F., & Eldar, Y. C. (2011). Structured compressed sensing:
    From theory to applications. IEEE Transactions on signal processing, 59(9),
    4053-4085.

    """

    meta = SolverMeta(
        acronym="OMP",
        full_name="Orthogonal Matching Pursuit",
        category="Matching Pursuit",
        description=(
            "Greedy sparse recovery that iteratively selects the best-correlating atom and "
            "re-solves a least-squares fit on the selected support."
        ),
        references=[
            "Rezaiifar, R., & Krishnaprasad, P. S. (1995). Orthogonal matching pursuit: Recursive function approximation with applications to wavelet decomposition. Proceedings of the 27th Annual Asilomar Conference on Signals, Systems, and Computers.",
            "Tropp, J. A., & Gilbert, A. C. (2007). Signal recovery from random measurements via orthogonal matching pursuit. IEEE Transactions on Information Theory, 53(12), 4655–4666.",
        ],
    )

    def __init__(self, name="Orthogonal Matching Pursuit", **kwargs):
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
        Return
        ------
        stc : mne.SourceEstimate
            The mne Source Estimate object
        """
        data = self.unpack_data_obj(mne_obj)
        source_mat = np.stack(
            [self.calc_omp_solution(y, K=K, max_iter=max_iter) for y in data.T], axis=1
        )
        stc = self.source_to_object(source_mat)
        return stc

    def calc_omp_solution(self, y, K="auto", max_iter=None):
        """Calculates the Orthogonal Matching Pursuit (OMP) inverse solution.

        Parameters
        ----------
        y : numpy.ndarray
            The data matrix (channels,).
        K : ["auto", int]
            The number of atoms to select per iteration.

        Return
        ------
        x_hat : numpy.ndarray
            The inverse solution (dipoles,)
        """
        if K == "auto":
            K = int(len(y) / 2)

        if max_iter is None:
            max_iter = len(y)
        _, n_dipoles = self.leadfield.shape

        x_hat = np.zeros(n_dipoles)
        x_hats = [deepcopy(x_hat)]

        omega = np.array([])
        r = deepcopy(y)
        y_hat = self.leadfield_original @ x_hat
        residuals = np.array(
            [
                np.linalg.norm(y - y_hat),
            ]
        )

        for _ in range(max_iter):
            # Use normalized leadfield for atom selection
            b = self.leadfield_normed.T @ r

            b_thresh = thresholding(b, K)
            new_atoms = np.where(b_thresh != 0)[0]
            omega = np.append(omega, new_atoms)
            omega = np.unique(omega.astype(int))  # Remove duplicates

            # Use original leadfield for coefficient estimation with numerical stability
            L_omega = self.leadfield_original[:, omega]
            # Check condition number for numerical stability
            if len(omega) > 0:
                cond_num = np.linalg.cond(L_omega)
                if cond_num > 1e12:  # Add small regularization if ill-conditioned
                    regularization = 1e-12 * np.eye(L_omega.shape[1])
                    x_hat[omega] = np.linalg.solve(
                        L_omega.T @ L_omega + regularization, L_omega.T @ y
                    )
                else:
                    x_hat[omega] = np.linalg.pinv(L_omega) @ y

            y_hat = self.leadfield_original @ x_hat
            r = y - y_hat

            residuals = np.append(residuals, np.linalg.norm(r))
            x_hats.append(deepcopy(x_hat))

            # Early stopping if residual starts increasing
            if len(residuals) > 1 and residuals[-1] > residuals[-2]:
                break

        x_hat = best_index_residual(residuals, x_hats, plot=False)

        return x_hat
