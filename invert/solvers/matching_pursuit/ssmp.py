import logging
from copy import deepcopy

import mne
import numpy as np
from scipy.sparse.csgraph import laplacian

from ...util import (
    best_index_residual,
    thresholding,
)
from ..base import BaseSolver, SolverMeta

logger = logging.getLogger(__name__)


class SolverSSMP(BaseSolver):
    """Class for the Smooth Simultaneous Matching Pursuit (SSMP) inverse
        solution. Developed by Lukas Hecker as a smooth extension of the
        orthogonal matching pursuit algorithm [1,2], 19.10.2022.


    References
    ----------
    [1] Duarte, M. F., & Eldar, Y. C. (2011). Structured compressed sensing:
    From theory to applications. IEEE Transactions on signal processing, 59(9),
    4053-4085.

    [2] Donoho, D. L. (2006). Compressed sensing. IEEE Transactions on
    information theory, 52(4), 1289-1306.

    """

    meta = SolverMeta(
        acronym="SSMP",
        full_name="Smooth Simultaneous Matching Pursuit",
        category="Matching Pursuit",
        description=(
            "Multi-measurement smooth matching pursuit that selects spatial patches using a "
            "smoothed (adjacency-based) dictionary and aggregates evidence across time."
        ),
        references=[
            "Lukas Hecker 2025, unpublished",
        ],
    )

    def __init__(self, name="Smooth Simultaneous Matching Pursuit", **kwargs):
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

        # Prepare spatial adjacency and patch dictionary
        adjacency = mne.spatial_src_adjacency(self.forward["src"], verbose=0).toarray()
        self.adjacency = adjacency
        laplace_operator = laplacian(adjacency)
        self.laplace_operator = laplace_operator

        patch_operator = adjacency + np.eye(adjacency.shape[0])
        leadfield_smooth = self.leadfield @ patch_operator

        self.leadfield_smooth = leadfield_smooth
        norms = np.linalg.norm(self.leadfield_smooth, axis=0)
        norms[norms == 0] = 1
        self.leadfield_smooth_normed = self.leadfield_smooth / norms
        norms = np.linalg.norm(self.leadfield, axis=0)
        norms[norms == 0] = 1
        self.leadfield_normed = self.leadfield / norms

        return self

    def apply_inverse_operator(
        self, mne_obj, include_singletons=True
    ) -> mne.SourceEstimate:
        """Apply the inverse operator.
        Parameters
        ----------
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        include_singletons : bool
            If True -> Include not only smooth patches but also single dipoles.

        Return
        ------
        stc : mne.SourceEstimate
            The mne Source Estimate object
        """
        data = self.unpack_data_obj(mne_obj)
        source_mat = self.calc_ssmp_solution(
            data, include_singletons=include_singletons
        )
        stc = self.source_to_object(source_mat)
        return stc

    def calc_ssmp_solution(self, y, include_singletons=True):
        """Calculates the Smooth Simultaneous Orthogonal Matching Pursuit (SSMP) inverse solution.

        Parameters
        ----------
        y : numpy.ndarray
            The data matrix (channels,).
        include_singletons : bool
            If True -> Include not only smooth patches but also single dipoles.

        Return
        ------
        x_hat : numpy.ndarray
            The inverse solution (dipoles,)
        """

        n_chans, n_time = y.shape
        max_iter = int(n_chans / 2)
        _, n_dipoles = self.leadfield.shape

        x_hat = np.zeros((n_dipoles, n_time))
        x_hats = [deepcopy(x_hat)]

        R = deepcopy(y)
        omega = np.array([], dtype=int)

        y_hat = self.leadfield @ x_hat
        residuals = np.array(
            [
                np.linalg.norm(y - y_hat),
            ]
        )
        x_hats = [
            deepcopy(x_hat),
        ]
        q = 2

        for _ in range(max_iter):
            b_n_smooth = np.linalg.norm(
                self.leadfield_smooth_normed.T @ R, axis=1, ord=q
            )
            b_n_sparse = np.linalg.norm(self.leadfield_normed.T @ R, axis=1, ord=q)

            if include_singletons and (abs(b_n_sparse).max() > abs(b_n_smooth).max()):
                b_n_sparse_thresh = thresholding(b_n_sparse, 1)
                new_patch = np.where(b_n_sparse_thresh != 0)[0]
            else:
                b_n_smooth_thresh = thresholding(b_n_smooth, 1)
                best_idx = np.where(b_n_smooth_thresh != 0)[0]
                new_patch = np.where(self.adjacency[best_idx[0]] != 0)[0]
                new_patch = np.append(new_patch, best_idx)

            omega = np.unique(np.append(omega, new_patch)).astype(int)
            x_hat = np.zeros((n_dipoles, n_time))
            x_hat[omega], _, _, _ = np.linalg.lstsq(
                self.leadfield[:, omega], y, rcond=None
            )

            y_hat = self.leadfield @ x_hat
            R = y - y_hat

            residuals = np.append(residuals, np.linalg.norm(y - y_hat))
            x_hats.append(deepcopy(x_hat))

            if residuals[-1] > residuals[-2]:
                break

        # Model selection (Regularisation)
        x_hat = best_index_residual(residuals, x_hats, plot=False)

        return x_hat
