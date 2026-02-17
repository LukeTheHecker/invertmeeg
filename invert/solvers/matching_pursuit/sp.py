import logging

import mne
import numpy as np

from ...util import (
    build_source_adjacency,
    thresholding,
)
from ..base import BaseSolver, SolverMeta

logger = logging.getLogger(__name__)


class SolverSP(BaseSolver):
    """Subspace Pursuit (SP) with MMV and patch-dictionary support.

    Greedy sparse recovery that alternates between support expansion and
    pruning, solving a least-squares fit on the candidate support each
    iteration.  All time points are used jointly (MMV) and spatial patches
    can be selected via an adjacency-based smooth dictionary.

    References
    ----------
    [1] Dai, W., & Milenkovic, O. (2009). Subspace pursuit for compressive
    sensing signal reconstruction. IEEE Transactions on Information Theory,
    55(5), 2230-2249.

    [2] Duarte, M. F., & Eldar, Y. C. (2011). Structured compressed sensing:
    From theory to applications. IEEE Transactions on Signal Processing, 59(9),
    4053-4085.
    """

    meta = SolverMeta(
        acronym="SP",
        full_name="Subspace Pursuit",
        category="Matching Pursuit",
        description=(
            "Greedy sparse recovery that alternates between support expansion and "
            "pruning (singletons or patches) using all time points jointly (MMV), "
            "solving a least-squares fit on the candidate support each iteration."
        ),
        references=[
            "Dai, W., & Milenkovic, O. (2009). Subspace pursuit for compressive sensing signal reconstruction. IEEE Transactions on Information Theory, 55(5), 2230–2249.",
            "Duarte, M. F., & Eldar, Y. C. (2011). Structured compressed sensing: From theory to applications. IEEE Transactions on Signal Processing, 59(9), 4053–4085.",
        ],
    )

    def __init__(self, name="Subspace Pursuit", **kwargs):
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
        include_singletons=True,
        include_patches=False,
    ) -> mne.SourceEstimate:
        """Apply the SP inverse solution.

        Parameters
        ----------
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        K : int | "auto"
            The sparsity level (number of atoms to keep after pruning).
        include_singletons : bool
            Include individual dipoles as candidate atoms.
        include_patches : bool
            Include spatial-patch atoms built from source adjacency.

        Return
        ------
        stc : mne.SourceEstimate
            The source estimate containing the inverse solution.
        """
        data = self.unpack_data_obj(mne_obj)
        self.validate_operator_data_compatibility(data)
        data = self._sensor_transform @ data
        source_mat = self.calc_sp_solution(
            data,
            K=K,
            include_singletons=include_singletons,
            include_patches=include_patches,
        )
        stc = self.source_to_object(source_mat)
        return stc

    def _select_atoms(self, R, K, include_singletons, include_patches):
        """Select K candidate atoms from residual using aggregated correlations."""
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

    def calc_sp_solution(
        self, y, K="auto", include_singletons=True, include_patches=False
    ):
        """Calculates the Subspace Pursuit (SP) inverse solution (MMV, patch-aware).

        Parameters
        ----------
        y : numpy.ndarray
            The data matrix, shape ``(n_chans,)`` or ``(n_chans, n_time)``.
        K : int | "auto"
            Sparsity level.
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

        # Initial atom selection
        T0 = self._select_atoms(y, K, include_singletons, include_patches)

        # Initial residual
        if len(T0) > 0:
            L_T0 = self.leadfield_original[:, T0]
            x_T0 = self.robust_inverse_solution(L_T0, y)
            R = y - L_T0 @ x_T0
        else:
            R = y.copy()

        T_list = [T0]
        R_list = [R]

        for i in range(1, n_chans + 1):
            # Select K new candidates from residual
            new_T = self._select_atoms(
                R_list[-1], K, include_singletons, include_patches
            )
            T_tilde = np.unique(np.concatenate([T_list[i - 1], new_T]))

            # Solve on expanded support and prune by coefficient norms
            if len(T_tilde) > 0:
                L_tilde = self.leadfield_original[:, T_tilde]
                xp = self.robust_inverse_solution(L_tilde, y)
                xp_norm = np.linalg.norm(xp, axis=1)
                T_l = T_tilde[np.where(thresholding(xp_norm, K) != 0)[0]]
            else:
                T_l = T_tilde

            T_list.append(T_l)

            # Calculate residual
            if len(T_l) > 0:
                L_final = self.leadfield_original[:, T_l]
                xp_final = self.robust_inverse_solution(L_final, y)
                R = y - L_final @ xp_final
            else:
                R = y.copy()

            R_list.append(R)

            # Early stopping
            if len(R_list) > 1 and (
                np.linalg.norm(R_list[-1]) >= np.linalg.norm(R_list[-2]) or i == n_chans
            ):
                T_l = T_list[-2]
                break
            else:
                T_l = T_list[-1]

        x_hat = np.zeros((n_dipoles, n_time))
        if len(T_l) > 0:
            L_final = self.leadfield_original[:, T_l]
            x_hat[T_l] = self.robust_inverse_solution(L_final, y)

        if squeeze:
            x_hat = x_hat[:, 0]
        return x_hat
