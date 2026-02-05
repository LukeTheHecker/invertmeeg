import logging
from copy import deepcopy

import mne
import numpy as np

from ...util import (
    best_index_residual,
    calc_residual_variance,
    thresholding,
)
from ..base import BaseSolver, SolverMeta

logger = logging.getLogger(__name__)


class SolverCOSAMP(BaseSolver):
    """Class for the Compressed Sampling Matching Pursuit (CoSaMP) inverse
        solution [1]. The algorithm as described in [2] was used for this
        imlementation.

    References
    ----------
    [1] Needell, D., & Tropp, J. A. (2009). CoSaMP: Iterative signal recovery
    from incomplete and inaccurate samples. Applied and computational harmonic
    analysis, 26(3), 301-321.

    [2] Duarte, M. F., & Eldar, Y. C. (2011). Structured compressed sensing:
    From theory to applications. IEEE Transactions on signal processing, 59(9),
    4053-4085.
    """

    meta = SolverMeta(
        acronym="CoSaMP",
        full_name="Compressive Sampling Matching Pursuit",
        category="Matching Pursuit",
        description=(
            "Greedy sparse recovery algorithm for single-measurement vectors. Iteratively "
            "identifies a support set and solves a least-squares fit on that support."
        ),
        references=[
            "Needell, D., & Tropp, J. A. (2009). CoSaMP: Iterative signal recovery from incomplete and inaccurate samples. Applied and Computational Harmonic Analysis, 26(3), 301–321.",
            "Duarte, M. F., & Eldar, Y. C. (2011). Structured compressed sensing: From theory to applications. IEEE Transactions on Signal Processing, 59(9), 4053–4085.",
        ],
    )

    def __init__(self, name="Compressed Sampling Matching Pursuit", **kwargs):
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
        # Create normalized leadfield for atom selection
        leadfield_norms = np.linalg.norm(self.leadfield, axis=0)
        # Avoid division by zero
        leadfield_norms[leadfield_norms == 0] = 1
        self.leadfield_normed = self.leadfield / leadfield_norms

        self.inverse_operators = []
        return self

    def apply_inverse_operator(
        self, mne_obj, K="auto", rv_thresh=1
    ) -> mne.SourceEstimate:  # type: ignore
        """Apply the inverse operator.

        Parameters
        ----------
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.

        Return
        ------
        stc : mne.SourceEstimate
            The mne Source Estimate object
        """
        data = self.unpack_data_obj(mne_obj)

        source_mat = np.stack(
            [self.calc_cosamp_solution(y, K=K, rv_thresh=rv_thresh) for y in data.T],
            axis=1,
        )
        stc = self.source_to_object(source_mat)
        return stc

    def calc_cosamp_solution(self, y, K="auto", rv_thresh=1):
        """Calculates the CoSaMP inverse solution.

        Parameters
        ----------
        y : numpy.ndarray
            The data matrix (channels, time).
        K : ["auto", int]
            Positive integer determining the sparsity of the reconstructed
            signal.
        rv_thresh : float
            The residual variance threshold as a stopping criterion. The
            smaller, the sooner the atom search is considered complete, i.e.,
            the less of the data is fitted. It can therefore be used for
            regularization.

        Return
        ------
        x_hat : numpy.ndarray
            The inverse solution (dipoles, time)
        """
        n_chans = len(y)
        _, n_dipoles = self.leadfield.shape

        if K == "auto":
            K = int(n_chans / 2)

        x_hat = np.zeros(n_dipoles)
        x_hats = [deepcopy(x_hat)]
        r = deepcopy(y)
        y_hat = self.leadfield_original @ x_hat
        residuals = np.array(
            [
                np.linalg.norm(y - y_hat),
            ]
        )
        source_norms = np.array(
            [
                0,
            ]
        )
        unexplained_variance = np.array(
            [
                calc_residual_variance(self.leadfield_original @ x_hat, y),
            ]
        )

        for i in range(1, n_chans + 1):
            # Use normalized leadfield for atom selection
            e = self.leadfield_normed.T @ r
            e_thresh = thresholding(e, 2 * K)
            omega = np.where(e_thresh != 0)[0]
            old_activations = np.where(x_hats[i - 1] != 0)[0]
            T = np.unique(np.concatenate([omega, old_activations]))

            # Use robust inverse from base class for coefficient estimation
            b = np.zeros(n_dipoles)
            if len(T) > 0:
                L_T = self.leadfield_original[:, T]
                b[T] = self.robust_inverse_solution(L_T, y)

            x_hat = thresholding(b, K)
            y_hat = self.leadfield_original @ x_hat
            r = y - y_hat

            residuals = np.append(residuals, np.linalg.norm(r))
            source_norms = np.append(source_norms, np.sum(x_hat**2))
            unexplained_variance = np.append(
                unexplained_variance,
                calc_residual_variance(self.leadfield_original @ x_hat, y),
            )
            x_hats.append(deepcopy(x_hat))

            # Improved stopping criterion
            if len(residuals) > 2 and (
                residuals[-1] >= residuals[-2] or unexplained_variance[-1] < rv_thresh
            ):
                break

        x_hat = best_index_residual(residuals, x_hats)

        return x_hat
