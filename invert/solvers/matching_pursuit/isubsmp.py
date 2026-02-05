import logging
from copy import deepcopy

import mne
import numpy as np
from scipy.sparse.csgraph import laplacian

from ...util import (
    calc_residual_variance,
    thresholding,
)
from ..base import BaseSolver, SolverMeta

logger = logging.getLogger(__name__)


class SolverISubSMP(BaseSolver):
    """Class for the Subspace Smooth Matching Pursuit (SubSMP) inverse solution. Developed
        by Lukas Hecker as a smooth extension of the orthogonal matching pursuit
        algorithm [1], 19.10.2022.


    References
    ----------
    [1] Duarte, M. F., & Eldar, Y. C. (2011). Structured compressed sensing:
    From theory to applications. IEEE Transactions on signal processing, 59(9),
    4053-4085.

    """

    meta = SolverMeta(
        acronym="ISubSMP",
        full_name="Iterative Subspace Smooth Matching Pursuit",
        category="Matching Pursuit",
        description=(
            "Smooth, subspace-based matching pursuit variant that operates on a spatially "
            "smoothed (patch) dictionary and iterates component-wise for multi-source data."
        ),
        references=[
            "Lukas Hecker 2025, unpublished",
        ],
    )

    def __init__(self, name="Iterative Subspace Smooth Matching Pursuit", **kwargs):
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
        self.inverse_operators = []

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

        Return
        ------
        stc : mne.SourceEstimate
            The mne Source Estimate object
        """
        data = self.unpack_data_obj(mne_obj)
        source_mat = self.calc_subsmp_solution(
            data, include_singletons=include_singletons
        )
        stc = self.source_to_object(source_mat)
        return stc

    def calc_subsmp_solution(self, y, include_singletons=True, var_thresh=0.1):
        """Calculate the Subspace Smooth Matching Pursuit (SubSMP) solution."""
        x_hat = self.calc_isubsmp_solution(
            y, include_singletons=include_singletons, var_thresh=var_thresh
        )
        return x_hat

    def calc_isubsmp_solution(self, y, include_singletons=True, var_thresh=1):
        """Calculates the Iterative Subspace Smooth Matching Pursuit inverse solution.

        Parameters
        ----------
        y : numpy.ndarray
            The data matrix (channels, n_time).
        include_singletons : bool
            If True -> Include not only smooth patches but also single dipoles.

        Return
        ------
        x_hat : numpy.ndarray
            The inverse solution (dipoles, n_time)
        """
        n_chans, n_time = y.shape
        _, n_dipoles = self.leadfield.shape

        x_hat = np.zeros((n_dipoles, n_time))
        omega = np.array([], dtype=int)
        r = deepcopy(y)
        r_component = np.linalg.svd(r, full_matrices=False)[0][:, 0]
        y_hat = self.leadfield @ x_hat
        residuals = np.array(
            [
                np.linalg.norm(y - y_hat),
            ]
        )
        unexplained_variance = np.array(
            [
                calc_residual_variance(y_hat, y),
            ]
        )
        x_hats = [
            deepcopy(x_hat),
        ]

        for _ in range(n_chans):
            b_smooth = self.leadfield_smooth_normed.T @ r_component
            b_sparse = self.leadfield_normed.T @ r_component

            if include_singletons and (abs(b_sparse).max() > abs(b_smooth).max()):
                b_sparse_thresh = thresholding(b_sparse, 1)
                new_patch = np.where(b_sparse_thresh != 0)[0]
            else:
                b_smooth_thresh = thresholding(b_smooth, 1)
                best_idx = np.where(b_smooth_thresh != 0)[0]
                new_patch = np.where(self.adjacency[best_idx[0]] != 0)[0]
                new_patch = np.append(new_patch, best_idx)

            omega = np.unique(np.append(omega, new_patch)).astype(int)
            x_hat = np.zeros((n_dipoles, n_time))
            x_hat[omega], _, _, _ = np.linalg.lstsq(
                self.leadfield[:, omega], y, rcond=None
            )
            y_hat = self.leadfield @ x_hat
            r = y - y_hat
            r_component = np.linalg.svd(r, full_matrices=False)[0][:, 0]

            residuals = np.append(residuals, np.linalg.norm(r))
            unexplained_variance = np.append(
                unexplained_variance, calc_residual_variance(y_hat, y)
            )
            x_hats.append(deepcopy(x_hat))
            if residuals[-1] > residuals[-2] or unexplained_variance[-1] < var_thresh:
                break

        return x_hat
