import logging
from copy import deepcopy

import mne
import numpy as np

from ...util import (
    best_index_residual,
    build_source_adjacency,
    thresholding,
)
from ..base import BaseSolver, SolverMeta

logger = logging.getLogger(__name__)


class SolverOMP(BaseSolver):
    """Orthogonal Matching Pursuit (OMP) with MMV and patch-dictionary support.

    Greedy sparse recovery that iteratively selects the best-correlating atom
    (singleton dipole or spatial patch) and re-solves a least-squares fit on the
    selected support.  All time points are used jointly (MMV) for atom selection.

    References
    ----------
    [1] Tropp, J. A., & Gilbert, A. C. (2007). Signal recovery from random
    measurements via orthogonal matching pursuit. IEEE Transactions on
    Information Theory, 53(12), 4655-4666.

    [2] Duarte, M. F., & Eldar, Y. C. (2011). Structured compressed sensing:
    From theory to applications. IEEE Transactions on Signal Processing, 59(9),
    4053-4085.
    """

    meta = SolverMeta(
        acronym="OMP",
        full_name="Orthogonal Matching Pursuit",
        category="Matching Pursuit",
        description=(
            "Greedy sparse recovery that iteratively selects the best-correlating atom "
            "(singleton or spatial patch) using all time points jointly (MMV) and "
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

        # Adjacency and patch dictionary
        adjacency = build_source_adjacency(self.forward["src"], verbose=0).toarray()
        self.adjacency = adjacency
        patch_operator = adjacency + np.eye(adjacency.shape[0])

        self.leadfield_original = self.leadfield.copy()
        leadfield_smooth = self.leadfield_original @ patch_operator

        self.leadfield_smooth_normed = self.robust_normalize_leadfield(leadfield_smooth)
        self.leadfield_normed = self.robust_normalize_leadfield(self.leadfield_original)

        self.inverse_operators = []
        return self

    def apply_inverse_operator(
        self, mne_obj, K="auto", max_iter=None,
        include_singletons=True, include_patches=False,
    ) -> mne.SourceEstimate:
        """Apply the inverse operator.

        Parameters
        ----------
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        K : int
            The number of atoms to select per iteration.
        max_iter : int
            The maximum number of iterations.
        include_singletons : bool
            Include individual dipoles as candidate atoms.
        include_patches : bool
            Include spatial-patch atoms built from source adjacency.

        Return
        ------
        stc : mne.SourceEstimate
            The mne Source Estimate object
        """
        data = self.unpack_data_obj(mne_obj)
        self.validate_operator_data_compatibility(data)
        data = self._sensor_transform @ data
        source_mat = self.calc_omp_solution(
            data, K=K, max_iter=max_iter,
            include_singletons=include_singletons,
            include_patches=include_patches,
        )
        stc = self.source_to_object(source_mat)
        return stc

    def calc_omp_solution(self, y, K="auto", max_iter=None,
                          include_singletons=True, include_patches=False):
        """Calculates the OMP inverse solution (MMV, patch-aware).

        Parameters
        ----------
        y : numpy.ndarray
            The data matrix, shape ``(n_chans,)`` or ``(n_chans, n_time)``.
        K : int | "auto"
            The number of atoms to select per iteration.
        max_iter : int | None
            Maximum number of greedy iterations.
        include_singletons : bool
            Include individual-dipole atoms.
        include_patches : bool
            Include spatial-patch atoms.

        Return
        ------
        x_hat : numpy.ndarray
            The inverse solution, shape ``(n_dipoles,)`` or ``(n_dipoles, n_time)``.
        """
        if not include_singletons and not include_patches:
            raise ValueError("At least one of include_patches/include_singletons must be True")

        # Handle both SMV and MMV input
        squeeze = False
        if y.ndim == 1:
            y = y[:, np.newaxis]
            squeeze = True

        n_chans, n_time = y.shape
        if K == "auto":
            K = 1
        K = int(K)
        if K <= 0:
            raise ValueError("K must be positive")

        if max_iter is None:
            max_iter = int(n_chans / 2)
        _, n_dipoles = self.leadfield_original.shape

        x_hat = np.zeros((n_dipoles, n_time))
        x_hats = [deepcopy(x_hat)]

        R = y.copy()
        omega = np.array([], dtype=int)

        y_hat = self.leadfield_original @ x_hat
        residuals = np.array([np.linalg.norm(y - y_hat)])

        for _ in range(max_iter):
            # Aggregated correlation across time (L2 norm over time dimension)
            if include_patches and include_singletons:
                b_smooth = np.linalg.norm(self.leadfield_smooth_normed.T @ R, axis=1, ord=2)
                b_sparse = np.linalg.norm(self.leadfield_normed.T @ R, axis=1, ord=2)
                if b_sparse.max() > b_smooth.max():
                    b_thresh = thresholding(b_sparse, K)
                    new_atoms = np.where(b_thresh != 0)[0]
                else:
                    b_thresh = thresholding(b_smooth, K)
                    best_idx = np.where(b_thresh != 0)[0]
                    new_atoms = []
                    for idx in best_idx:
                        patch = np.where(self.adjacency[idx] != 0)[0]
                        patch = np.append(patch, idx)
                        new_atoms.append(patch)
                    new_atoms = np.unique(np.concatenate(new_atoms)) if new_atoms else np.array([], dtype=int)
            elif include_patches:
                b_smooth = np.linalg.norm(self.leadfield_smooth_normed.T @ R, axis=1, ord=2)
                b_thresh = thresholding(b_smooth, K)
                best_idx = np.where(b_thresh != 0)[0]
                new_atoms = []
                for idx in best_idx:
                    patch = np.where(self.adjacency[idx] != 0)[0]
                    patch = np.append(patch, idx)
                    new_atoms.append(patch)
                new_atoms = np.unique(np.concatenate(new_atoms)) if new_atoms else np.array([], dtype=int)
            else:  # include_singletons only
                b_sparse = np.linalg.norm(self.leadfield_normed.T @ R, axis=1, ord=2)
                b_thresh = thresholding(b_sparse, K)
                new_atoms = np.where(b_thresh != 0)[0]

            omega = np.unique(np.append(omega, new_atoms)).astype(int)

            x_hat = np.zeros((n_dipoles, n_time))
            if len(omega) > 0:
                x_hat[omega] = self.robust_inverse_solution(
                    self.leadfield_original[:, omega], y
                )

            y_hat = self.leadfield_original @ x_hat
            R = y - y_hat

            residuals = np.append(residuals, np.linalg.norm(R))
            x_hats.append(deepcopy(x_hat))

            if len(residuals) > 1 and residuals[-1] > residuals[-2]:
                break

        x_hat = best_index_residual(residuals, x_hats, plot=False)

        if squeeze:
            x_hat = x_hat[:, 0]
        return x_hat
