import logging
from copy import deepcopy

import mne
import numpy as np

from ...util import (
    best_index_residual,
    build_source_adjacency,
    calc_residual_variance,
    thresholding,
)
from ..base import BaseSolver, SolverMeta

logger = logging.getLogger(__name__)


class SolverCOSAMP(BaseSolver):
    """Compressive Sampling Matching Pursuit (CoSaMP) with MMV and patch support.

    Greedy sparse recovery that iteratively identifies a candidate support set,
    solves least-squares on that support, and prunes to K atoms.  All time
    points are used jointly (MMV) and spatial patches can be selected via an
    adjacency-based smooth dictionary.

    References
    ----------
    [1] Needell, D., & Tropp, J. A. (2009). CoSaMP: Iterative signal recovery
    from incomplete and inaccurate samples. Applied and Computational Harmonic
    Analysis, 26(3), 301-321.

    [2] Duarte, M. F., & Eldar, Y. C. (2011). Structured compressed sensing:
    From theory to applications. IEEE Transactions on Signal Processing, 59(9),
    4053-4085.
    """

    meta = SolverMeta(
        acronym="CoSaMP",
        full_name="Compressive Sampling Matching Pursuit",
        category="Matching Pursuit",
        description=(
            "Greedy sparse recovery that identifies 2K candidate atoms (singletons or "
            "patches) using all time points jointly (MMV), solves least-squares on the "
            "merged support, and prunes to K atoms each iteration."
        ),
        references=[
            "Needell, D., & Tropp, J. A. (2009). CoSaMP: Iterative signal recovery from incomplete and inaccurate samples. Applied and Computational Harmonic Analysis, 26(3), 301–321.",
            "Duarte, M. F., & Eldar, Y. C. (2011). Structured compressed sensing: From theory to applications. IEEE Transactions on Signal Processing, 59(9), 4053–4085.",
        ],
    )

    def __init__(self, name="Compressed Sampling Matching Pursuit", **kwargs):
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
        self,
        mne_obj,
        K="auto",
        rv_thresh=1,
        include_singletons=True,
        include_patches=False,
    ) -> mne.SourceEstimate:
        """Apply the inverse operator.

        Parameters
        ----------
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        K : int | "auto"
            Sparsity level (number of atoms to keep after pruning).
        rv_thresh : float
            Residual variance stopping threshold.
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
        source_mat = self.calc_cosamp_solution(
            data,
            K=K,
            rv_thresh=rv_thresh,
            include_singletons=include_singletons,
            include_patches=include_patches,
        )
        stc = self.source_to_object(source_mat)
        return stc

    def _select_atoms(self, R, K, include_singletons, include_patches):
        """Select candidate atoms from residual using aggregated correlations."""
        if include_patches and include_singletons:
            b_smooth = np.linalg.norm(self.leadfield_smooth_normed.T @ R, axis=1)
            b_sparse = np.linalg.norm(self.leadfield_normed.T @ R, axis=1)
            if b_sparse.max() > b_smooth.max():
                b_thresh = thresholding(b_sparse, K)
                return np.where(b_thresh != 0)[0]
            else:
                b_thresh = thresholding(b_smooth, K)
                best_idx = np.where(b_thresh != 0)[0]
                patches = []
                for idx in best_idx:
                    patch = np.where(self.adjacency[idx] != 0)[0]
                    patch = np.append(patch, idx)
                    patches.append(patch)
                return (
                    np.unique(np.concatenate(patches))
                    if patches
                    else np.array([], dtype=int)
                )
        elif include_patches:
            b_smooth = np.linalg.norm(self.leadfield_smooth_normed.T @ R, axis=1)
            b_thresh = thresholding(b_smooth, K)
            best_idx = np.where(b_thresh != 0)[0]
            patches = []
            for idx in best_idx:
                patch = np.where(self.adjacency[idx] != 0)[0]
                patch = np.append(patch, idx)
                patches.append(patch)
            return (
                np.unique(np.concatenate(patches))
                if patches
                else np.array([], dtype=int)
            )
        else:  # include_singletons only
            b_sparse = np.linalg.norm(self.leadfield_normed.T @ R, axis=1)
            b_thresh = thresholding(b_sparse, K)
            return np.where(b_thresh != 0)[0]

    def calc_cosamp_solution(
        self, y, K="auto", rv_thresh=1, include_singletons=True, include_patches=False
    ):
        """Calculates the CoSaMP inverse solution (MMV, patch-aware).

        Parameters
        ----------
        y : numpy.ndarray
            The data matrix, shape ``(n_chans,)`` or ``(n_chans, n_time)``.
        K : int | "auto"
            Sparsity level.
        rv_thresh : float
            Residual variance stopping threshold.
        include_singletons : bool
            Include individual-dipole atoms.
        include_patches : bool
            Include spatial-patch atoms.

        Return
        ------
        x_hat : numpy.ndarray
            The inverse solution.
        """
        if not include_singletons and not include_patches:
            raise ValueError(
                "At least one of include_patches/include_singletons must be True"
            )

        squeeze = False
        if y.ndim == 1:
            y = y[:, np.newaxis]
            squeeze = True

        n_chans, n_time = y.shape
        _, n_dipoles = self.leadfield_original.shape

        if K == "auto":
            K = min(8, max(2, n_chans // 10))
        K = int(K)
        if K <= 0:
            raise ValueError("K must be positive")

        x_hat = np.zeros((n_dipoles, n_time))
        x_hats = [deepcopy(x_hat)]
        r = y.copy()
        residuals = np.array([np.linalg.norm(y - self.leadfield_original @ x_hat)])
        unexplained_variance = np.array(
            [
                calc_residual_variance(self.leadfield_original @ x_hat, y),
            ]
        )

        for i in range(1, n_chans + 1):
            # Identify 2K candidate atoms
            omega = self._select_atoms(r, 2 * K, include_singletons, include_patches)

            # Merge with previous active set
            old_activations = np.where(np.linalg.norm(x_hats[i - 1], axis=1) != 0)[0]
            T = np.unique(np.concatenate([omega, old_activations]))

            # Solve least-squares on merged support
            b = np.zeros((n_dipoles, n_time))
            if len(T) > 0:
                L_T = self.leadfield_original[:, T]
                b[T] = self.robust_inverse_solution(L_T, y)

            # Prune to K atoms by aggregated coefficient norms
            b_norms = np.linalg.norm(b, axis=1)
            x_hat = np.zeros((n_dipoles, n_time))
            b_thresh = thresholding(b_norms, K)
            keep = np.where(b_thresh != 0)[0]
            if len(keep) > 0:
                # Re-solve on pruned support for clean coefficients
                L_keep = self.leadfield_original[:, keep]
                x_hat[keep] = self.robust_inverse_solution(L_keep, y)

            y_hat = self.leadfield_original @ x_hat
            r = y - y_hat

            residuals = np.append(residuals, np.linalg.norm(r))
            unexplained_variance = np.append(
                unexplained_variance,
                calc_residual_variance(y_hat, y),
            )
            x_hats.append(deepcopy(x_hat))

            if len(residuals) > 2 and (
                residuals[-1] >= residuals[-2] or unexplained_variance[-1] < rv_thresh
            ):
                break

        x_hat = best_index_residual(residuals, x_hats)

        if squeeze:
            x_hat = x_hat[:, 0]
        return x_hat
