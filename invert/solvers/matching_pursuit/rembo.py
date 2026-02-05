import logging
from copy import deepcopy

import mne
import numpy as np

from ...util import (
    calc_residual_variance,
    find_corner,
    thresholding,
)
from ..base import BaseSolver, SolverMeta

logger = logging.getLogger(__name__)


class SolverREMBO(BaseSolver):
    """Class for the Reduce Multi-Measurement-Vector and Boost (ReMBo) inverse
        solution [1]. The algorithm as describe in [2] was used for the
        imlpementation.

    References
    ----------
    [1] Mishali, M., & Eldar, Y. C. (2008). Reduce and boost: Recovering
    arbitrary sets of jointly sparse vectors. IEEE Transactions on Signal
    Processing, 56(10), 4692-4702.
    [2] Duarte, M. F., & Eldar, Y. C. (2011).
    Structured compressed sensing: From theory to applications. IEEE
    Transactions on signal processing, 59(9), 4053-4085.
    """

    meta = SolverMeta(
        acronym="ReMBo",
        full_name="Reduce Multi-Measurement-Vector and Boost",
        category="Matching Pursuit",
        description=(
            "Randomly reduces a multi-measurement problem to repeated single-measurement "
            "OMP-style recovery, then boosts by re-fitting on the recovered joint support."
        ),
        references=[
            "Mishali, M., & Eldar, Y. C. (2008). Reduce and boost: Recovering arbitrary sets of jointly sparse vectors. IEEE Transactions on Signal Processing, 56(10), 4692–4702.",
            "Duarte, M. F., & Eldar, Y. C. (2011). Structured compressed sensing: From theory to applications. IEEE Transactions on Signal Processing, 59(9), 4053–4085.",
        ],
    )

    def __init__(self, name="Reduce Multi-Measurement-Vector and Boost", **kwargs):
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
        self.leadfield_original = self.leadfield.copy()
        self.leadfield_normed = self.robust_normalize_leadfield(self.leadfield)

        self.inverse_operators = []
        return self

    def apply_inverse_operator(self, mne_obj, K="auto") -> mne.SourceEstimate:  # type: ignore
        """Apply the REMBO inverse solution.

        Parameters
        ----------
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        K : ["auto", int]
            The number of atoms to select per iteration.

        Return
        ------
        stc : mne.SourceEstimate
            The source estimate containing the inverse solution.
        """
        data = self.unpack_data_obj(mne_obj)
        source_mat = self.calc_rembo_solution(data, K=K)
        stc = self.source_to_object(source_mat)
        return stc

    def calc_rembo_solution(self, y, K="auto"):
        """Calculate the REMBO inverse solution based on the measurement vector
            y.

        Parameters
        ----------
        y : numpy.ndarray
            The EEG matrix (channels, time)
        K : ["auto", int]
            The number of atoms to select per iteration.

        Return
        ------
        x_hat : numpy.ndarray
            The source matrix (dipoles, time)
        """

        rand = np.random.rand
        n_chans, n_time = y.shape
        if K == "auto":
            K = int(n_chans / 2)
        max_iter = int(n_chans / 2)
        n_dipoles = self.leadfield.shape[1]

        unexplained_variance = np.array(
            [
                calc_residual_variance(np.zeros(y.shape), y),
            ]
        )
        n_sources = np.array(
            [
                0,
            ]
        )
        x_hats = [np.zeros((n_dipoles, n_time))]

        for _i in range(max_iter):
            a = rand(n_time)
            y_vec = y @ a  # sample randomly from the measurement matrix
            x_hat = self.calc_omp_solution(y_vec, K=K)
            S_hat = np.where(x_hat != 0)[0].astype(int)
            As_pinv = np.linalg.pinv(self.leadfield[:, S_hat])

            x_hat = np.zeros((n_dipoles, n_time))
            x_hat[S_hat, :] = As_pinv @ y
            x_hats.append(x_hat)

            unexplained_variance = np.append(
                unexplained_variance, calc_residual_variance(self.leadfield @ x_hat, y)
            )
            n_sources = np.append(n_sources, len(S_hat)).astype(float)

        idc = np.argsort(n_sources)
        n_sources = n_sources[idc]
        unexplained_variance = unexplained_variance[idc]

        # REFACTOR THIS:
        unique_sources = np.sort(np.unique(n_sources))
        n_sources_new = []
        unexp_new = []
        x_hats_new = []
        for k in unique_sources:
            idc = np.where(n_sources == k)[0]
            values = unexplained_variance[idc]
            min_sidx = np.argmin(values)
            good_index = idc[min_sidx]

            n_sources_new.append(k)
            unexp_new.append(values.min())
            x_hats_new.append(x_hats[good_index])
        n_sources = np.array(n_sources_new)[1:]
        unexplained_variance = np.array(unexp_new)[1:]
        x_hats = x_hats_new[1:]
        corner_idx = find_corner(n_sources, unexplained_variance)

        # corner_idx = find_corner(n_sources[1:], unexplained_variance[1:])+1
        # corner_idx = idc[corner_idx]
        x_hat = x_hats[corner_idx]
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(n_sources, unexplained_variance, 'k*')
        # plt.plot(n_sources[corner_idx], unexplained_variance[corner_idx], 'ro')

        return x_hat

    def calc_omp_solution(self, y, K="auto"):
        """Calculates the Orthogonal Matching Pursuit (OMP) inverse solution.
        (Used by REMBO algorithm)

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
        n_chans = len(y)
        _, n_dipoles = self.leadfield.shape

        if K == "auto":
            K = int(n_chans / 2)

        x_hat = np.zeros(n_dipoles)
        x_hats = [deepcopy(x_hat)]
        source_norms = np.array(
            [
                0,
            ]
        )

        omega = np.array([])
        r = deepcopy(y)
        residuals = np.array(
            [
                np.linalg.norm(y - self.leadfield_original @ x_hat),
            ]
        )
        x_hats = [
            deepcopy(x_hat),
        ]

        for _i in range(n_chans):
            # Use original leadfield for both selection and estimation (REMBO variant)
            b = self.leadfield_original.T @ r
            b_thresh = thresholding(b, K)
            new_atoms = np.where(b_thresh != 0)[0]
            omega = np.append(omega, new_atoms)
            omega = np.unique(omega.astype(int))

            # Use robust inverse from base class for coefficient estimation
            if len(omega) > 0:
                L_omega = self.leadfield_original[:, omega]
                x_hat[omega] = self.robust_inverse_solution(L_omega, y)

            r = y - self.leadfield_original @ x_hat

            residuals = np.append(residuals, np.linalg.norm(r))
            source_norms = np.append(source_norms, np.sum(x_hat**2))
            x_hats.append(deepcopy(x_hat))

            # Early stopping if residual starts increasing
            if len(residuals) > 1 and residuals[-1] > residuals[-2]:
                break

        iters = np.arange(len(residuals)).astype(float)
        corner_idx = find_corner(iters, residuals)
        x_hat = x_hats[corner_idx]
        return x_hat
