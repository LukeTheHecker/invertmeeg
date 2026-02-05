import logging

import mne
import numpy as np

from ...util import (
    thresholding,
)
from ..base import BaseSolver, SolverMeta

logger = logging.getLogger(__name__)


class SolverSSP(BaseSolver):
    """Class for the Simultaneous Subspace Pursuit (SSP) inverse solution.

    References
    ----------
    [1] Dai, W., & Milenkovic, O. (2009). Subspace pursuit for
    compressive sensing signal reconstruction. IEEE transactions on Information
    Theory, 55(5), 2230-2249.

    [2] Duarte, M. F., & Eldar, Y. C. (2011). Structured compressed sensing:
    From theory to applications. IEEE Transactions on signal processing, 59(9),
    4053-4085.
    """

    meta = SolverMeta(
        acronym="SSP",
        full_name="Simultaneous Subspace Pursuit",
        category="Matching Pursuit",
        description=(
            "Multi-measurement-vector extension of Subspace Pursuit that selects and prunes "
            "a joint support using aggregated (across time) correlation and coefficient norms."
        ),
        references=[
            "Feng, J.-M., & Lee, C.-H. (2013). Generalized subspace pursuit for signal recovery from multiple-measurement vectors. Proceedings of IEEE WCNC 2013, 4347–4352.",
            "Dai, W., & Milenkovic, O. (2009). Subspace pursuit for compressive sensing signal reconstruction. IEEE Transactions on Information Theory, 55(5), 2230–2249.",
        ],
    )

    def __init__(self, name="Simultaneous Subspace Pursuit", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, *args, alpha="auto", verbose=0, **kwargs):
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

    def apply_inverse_operator(self, mne_obj, K="auto") -> mne.SourceEstimate:
        """Apply the SSP inverse solution.

        Parameters
        ----------
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        K : int
            The number of atoms to select per iteration.

        Return
        ------
        stc : mne.SourceEstimate
            The source estimate containing the inverse solution.
        """
        data = self.unpack_data_obj(mne_obj)
        source_mat = self.calc_ssp_solution(data, K=K)
        stc = self.source_to_object(source_mat)
        return stc

    def calc_ssp_solution(self, y, K="auto"):
        """Calculates the Simultaneous Subspace Pursuit (SSP) inverse solution.

        Parameters
        ----------
        y : numpy.ndarray
            The data matrix (channels, time).
        K : ["auto", int]
            The number of atoms to select per iteration.

        Return
        ------
        x_hat : numpy.ndarray
            The inverse solution (dipoles, time)
        """
        n_chans, n_time = y.shape
        _, n_dipoles = self.leadfield.shape

        if K == "auto":
            K = int(n_chans / 2)

        # Use robust residual from base class
        def resid(y, phi):
            return self.robust_residual(y, phi)

        # Use normalized leadfield for initial atom selection
        b = np.linalg.norm(self.leadfield_normed.T @ y, axis=1)

        T0 = np.where(thresholding(b, K) != 0)[0]
        R = resid(y, self.leadfield_original[:, T0])

        T_list = [
            T0,
        ]
        R_list = [
            R,
        ]

        for i in range(1, n_chans + 1):
            # Use normalized leadfield for atom selection
            b = np.linalg.norm(self.leadfield_normed.T @ R, axis=1)

            new_T = np.where(thresholding(b, K) != 0)[0]
            T_tilde = np.unique(np.concatenate([T_list[i - 1], new_T]))

            # Use robust inverse from base class for coefficient estimation
            if len(T_tilde) > 0:
                L_tilde = self.leadfield_original[:, T_tilde]
                xp = self.robust_inverse_solution(L_tilde, y)
                xp_norm = np.linalg.norm(xp, axis=1)
                T_l = T_tilde[np.where(thresholding(xp_norm, K) != 0)[0]]
            else:
                T_l = T_tilde

            T_list.append(T_l)

            # Calculate residual using robust methods from base class
            if len(T_l) > 0:
                L_final = self.leadfield_original[:, T_l]
                xp_final = self.robust_inverse_solution(L_final, y)
                R = y - L_final @ xp_final
            else:
                R = y

            R_list.append(R)

            # Early stopping with improved criterion
            if len(R_list) > 1 and (
                np.linalg.norm(R_list[-1]) >= np.linalg.norm(R_list[-2]) or i == n_chans
            ):
                T_l = T_list[-2]
                logger.debug(
                    f"Early stopping at iteration {i} because residual is increasing"
                )
                break
            else:
                T_l = T_list[-1]

        x_hat = np.zeros((n_dipoles, n_time))
        if len(T_l) > 0:
            L_final = self.leadfield_original[:, T_l]
            x_hat[T_l] = self.robust_inverse_solution(L_final, y)
        return x_hat
