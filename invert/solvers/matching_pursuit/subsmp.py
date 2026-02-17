import logging
from copy import deepcopy

import mne
import numpy as np
from scipy.sparse.csgraph import laplacian

from ...util import (
    best_index_residual,
    build_source_adjacency,
    calc_residual_variance,
    thresholding,
)
from ..base import BaseSolver, SolverMeta

logger = logging.getLogger(__name__)


class SolverSubSMP(BaseSolver):
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
        acronym="SubSMP",
        full_name="Subspace Smooth Matching Pursuit",
        category="Matching Pursuit",
        description=(
            "Smooth matching pursuit variant that projects data into a low-dimensional "
            "subspace and merges patch supports across components."
        ),
        references=[
            "Lukas Hecker 2025, unpublished",
        ],
    )

    def __init__(self, name="Subspace Smooth Matching Pursuit", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    @staticmethod
    def _adjacency_from_discrete_src(src):
        """Fallback adjacency for discrete source spaces."""
        if len(src) != 1 or src[0].get("type") != "discrete":
            msg = "Fallback adjacency only supports single discrete source spaces."
            raise RuntimeError(msg)

        space = src[0]
        vertno = np.asarray(space["vertno"], dtype=int)
        neighbor_vert = space.get("neighbor_vert")
        if neighbor_vert is None:
            msg = "Discrete source space is missing 'neighbor_vert'."
            raise RuntimeError(msg)

        n_vertices = len(vertno)
        vertex_to_local = {int(v): idx for idx, v in enumerate(vertno)}
        adjacency = np.zeros((n_vertices, n_vertices), dtype=float)

        for local_i, vertex in enumerate(vertno):
            neighbors = neighbor_vert[int(vertex)]
            if neighbors is None:
                continue
            for neighbor in np.asarray(neighbors, dtype=int):
                local_j = vertex_to_local.get(int(neighbor))
                if local_j is not None:
                    adjacency[local_i, local_j] = 1.0

        adjacency = np.maximum(adjacency, adjacency.T)
        np.fill_diagonal(adjacency, 0.0)
        return adjacency

    def make_inverse_operator(
        self,
        forward,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
        **kwargs,
    ):
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
        self.prepare_whitened_forward(noise_cov)
        self.inverse_operators = []

        adjacency = build_source_adjacency(self.forward["src"], verbose=0).toarray()
        self.adjacency = adjacency
        laplace_operator = laplacian(adjacency)
        self.laplace_operator = laplace_operator

        patch_operator = adjacency + np.eye(adjacency.shape[0])
        self.leadfield_original = self.leadfield.copy()
        leadfield_smooth = self.leadfield_original @ patch_operator

        self.leadfield_smooth = leadfield_smooth
        self.leadfield_smooth_normed = self.robust_normalize_leadfield(
            self.leadfield_smooth
        )
        self.leadfield_normed = self.robust_normalize_leadfield(self.leadfield_original)

        return self

    def apply_inverse_operator(
        self,
        mne_obj,
        include_singletons=True,
        include_patches=False,
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
        self.validate_operator_data_compatibility(data)
        data = self._sensor_transform @ data
        source_mat = self.calc_subsmp_solution(
            data, include_singletons=include_singletons
        )
        stc = self.source_to_object(source_mat)
        return stc

    def calc_subsmp_solution(self, y, include_singletons=True, var_thresh=1):
        """Calculate the Subspace Smooth Matching Pursuit (SubSMP) solution.

        Parameters
        ----------
        y : numpy.ndarray
            The M/EEG data Matrix
        include_singletons : bool
            If True -> include single dipoles as candidates,
            else include only smooth patches
        var_thresh : float
            Threshold how much variance will be explained in the data

        """
        _, n_time = y.shape
        n_dipoles = self.leadfield.shape[1]

        u, s, _ = np.linalg.svd(y, full_matrices=False)

        # Select components using base class method (find_corner returns an index, need count)
        n_comps = max(self.get_comps_L(s) + 1, 1)
        logger.info(f"Selected {n_comps} components")

        # Find matching sources to each of the selected eigenvectors (components)
        topos = u[:, :n_comps]
        x_hats = []
        for topo in topos.T:
            x_hats.append(
                self.calc_smp_solution(
                    topo, include_singletons=include_singletons, var_thresh=var_thresh
                )
            )

        # Extract all active dipoles and patches:
        omegas = np.unique(
            np.where((np.stack(x_hats, axis=0) ** 2).sum(axis=0) != 0)[0]
        )

        # Calculate minimum-norm solution at active dipoles and patches
        x_hat = np.zeros((n_dipoles, n_time))

        if len(omegas) > 0:
            x_hat[omegas, :] = self.robust_inverse_solution(
                self.leadfield_original[:, omegas], y
            )

        return x_hat

    def calc_smp_solution(self, y, include_singletons=True, var_thresh=1):
        """Calculates the Smooth Matching Pursuit (SMP) inverse solution.

        Parameters
        ----------
        y : numpy.ndarray
            The data matrix (channels,).
        include_singletons : bool
            If True -> Include not only smooth patches but also single dipoles.
        var_thresh : float
            Threshold how much variance will be explained in the data

        Return
        ------
        x_hat : numpy.ndarray
            The inverse solution (dipoles,)
        """
        n_chans = len(y)
        _, n_dipoles = self.leadfield.shape

        x_hat = np.zeros(n_dipoles)
        omega = np.array([], dtype=int)
        r = deepcopy(y)
        y_hat = self.leadfield_original @ x_hat
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
            b_smooth = self.leadfield_smooth_normed.T @ r
            b_sparse = self.leadfield_normed.T @ r

            if include_singletons and (abs(b_sparse).max() > abs(b_smooth).max()):
                b_sparse_thresh = thresholding(b_sparse, 1)
                new_patch = np.where(b_sparse_thresh != 0)[0]
            else:
                b_smooth_thresh = thresholding(b_smooth, 1)
                best_idx = np.where(b_smooth_thresh != 0)[0]
                new_patch = np.where(self.adjacency[best_idx[0]] != 0)[0]
                new_patch = np.append(new_patch, best_idx)

            omega = np.unique(np.append(omega, new_patch)).astype(int)
            x_hat = np.zeros(n_dipoles)
            if len(omega) > 0:
                x_hat[omega] = self.robust_inverse_solution(
                    self.leadfield_original[:, omega], y
                )
            y_hat = self.leadfield_original @ x_hat
            r = y - y_hat

            residuals = np.append(residuals, np.linalg.norm(r))
            unexplained_variance = np.append(
                unexplained_variance, calc_residual_variance(y_hat, y)
            )
            x_hats.append(deepcopy(x_hat))
            if residuals[-1] > residuals[-2]:
                break

        if unexplained_variance[1] > var_thresh:
            x_hat = best_index_residual(unexplained_variance, x_hats, plot=False)
        else:
            x_hat = x_hats[1]

        return x_hat
