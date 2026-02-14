import logging
from copy import deepcopy

import mne
import numpy as np

from ...util import (
    build_source_adjacency,
    thresholding,
)
from ..base import BaseSolver, SolverMeta

logger = logging.getLogger(__name__)


class SolverREMBO(BaseSolver):
    """Reduce Multi-Measurement-Vector and Boost (ReMBo) with patch support.

    Randomly reduces a multi-measurement problem to repeated single-measurement
    OMP-style recovery (with patch-dictionary awareness), then boosts by
    re-fitting on the recovered joint support.

    References
    ----------
    [1] Mishali, M., & Eldar, Y. C. (2008). Reduce and boost: Recovering
    arbitrary sets of jointly sparse vectors. IEEE Transactions on Signal
    Processing, 56(10), 4692-4702.

    [2] Duarte, M. F., & Eldar, Y. C. (2011). Structured compressed sensing:
    From theory to applications. IEEE Transactions on Signal Processing, 59(9),
    4053-4085.
    """

    meta = SolverMeta(
        acronym="ReMBo",
        full_name="Reduce Multi-Measurement-Vector and Boost",
        category="Matching Pursuit",
        description=(
            "Randomly reduces a multi-measurement problem to repeated single-measurement "
            "OMP-style recovery with patch-dictionary awareness, then boosts by re-fitting "
            "on the recovered joint support."
        ),
        references=[
            "Mishali, M., & Eldar, Y. C. (2008). Reduce and boost: Recovering arbitrary sets of jointly sparse vectors. IEEE Transactions on Signal Processing, 56(10), 4692–4702.",
            "Duarte, M. F., & Eldar, Y. C. (2011). Structured compressed sensing: From theory to applications. IEEE Transactions on Signal Processing, 59(9), 4053–4085.",
        ],
    )

    def __init__(self, name="Reduce Multi-Measurement-Vector and Boost", **kwargs):
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
        self, mne_obj, K="auto", max_boost_iter=None, fit_tol="auto",
        include_singletons=True, include_patches=False,
    ) -> mne.SourceEstimate:
        """Apply the REMBO inverse solution.

        Parameters
        ----------
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        K : int | "auto"
            Maximum support size (sparsity budget) for the SMV solve.
        max_boost_iter : int | None
            Maximum number of boost iterations.
        fit_tol : float | "auto"
            Relative SMV fit tolerance for boost acceptance.
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
        source_mat = self.calc_rembo_solution(
            data, K=K, max_boost_iter=max_boost_iter, fit_tol=fit_tol,
            include_singletons=include_singletons,
            include_patches=include_patches,
        )
        stc = self.source_to_object(source_mat)
        return stc

    def calc_rembo_solution(self, y, K="auto", max_boost_iter=None, fit_tol="auto",
                            include_singletons=True, include_patches=False):
        """Calculate the REMBO inverse solution.

        Parameters
        ----------
        y : numpy.ndarray
            The EEG matrix (channels, time).
        K : int | "auto"
            Maximum support size for the SMV step.
        max_boost_iter : int | None
            Maximum number of random MMV->SMV reductions.
        fit_tol : float | "auto"
            Relative SMV fit tolerance.
        include_singletons : bool
            Include individual-dipole atoms.
        include_patches : bool
            Include spatial-patch atoms.

        Return
        ------
        x_hat : numpy.ndarray
            The source matrix (dipoles, time).
        """
        if not include_singletons and not include_patches:
            raise ValueError("At least one of include_patches/include_singletons must be True")

        n_chans, n_time = y.shape
        if K == "auto":
            K = min(8, max(2, n_chans // 10))
        K = int(K)
        if K <= 0:
            raise ValueError("K must be positive")

        if max_boost_iter is None:
            max_boost_iter = 50
        max_boost_iter = int(max_boost_iter)
        if max_boost_iter <= 0:
            raise ValueError("max_boost_iter must be positive")

        if fit_tol == "auto":
            fit_tol = 1e-3
        fit_tol = float(fit_tol)
        if fit_tol < 0:
            raise ValueError("fit_tol must be non-negative")

        n_dipoles = self.leadfield_original.shape[1]
        y_norm_eps = 1e-15
        best_support = np.array([], dtype=int)
        best_rel_resid = np.inf

        for _ in range(max_boost_iter):
            a = np.random.standard_normal(n_time)
            y_vec = y @ a
            x_vec = self.calc_omp_solution(
                y_vec, K=K,
                include_singletons=include_singletons,
                include_patches=include_patches,
            )
            S_hat = np.where(x_vec != 0)[0].astype(int)

            if len(S_hat) == 0 or len(S_hat) > K:
                continue

            rel_resid = np.linalg.norm(y_vec - self.leadfield_original @ x_vec) / max(
                np.linalg.norm(y_vec), y_norm_eps
            )
            if rel_resid < best_rel_resid:
                best_rel_resid = rel_resid
                best_support = S_hat

            if rel_resid <= fit_tol:
                break

        x_hat = np.zeros((n_dipoles, n_time))
        if len(best_support) == 0:
            return x_hat

        As_pinv = np.linalg.pinv(self.leadfield_original[:, best_support])
        x_hat[best_support, :] = As_pinv @ y

        return x_hat

    def calc_omp_solution(self, y, K="auto",
                          include_singletons=True, include_patches=False):
        """Inner OMP for a single-measurement vector (used by ReMBo).

        Parameters
        ----------
        y : numpy.ndarray
            The data vector (channels,).
        K : int | "auto"
            Maximum number of nonzero atoms.
        include_singletons : bool
            Include individual-dipole atoms.
        include_patches : bool
            Include spatial-patch atoms.

        Return
        ------
        x_hat : numpy.ndarray
            The inverse solution (dipoles,).
        """
        n_chans = len(y)
        _, n_dipoles = self.leadfield_original.shape

        if K == "auto":
            K = 1
        K = int(K)
        if K <= 0:
            raise ValueError("K must be positive")

        x_hat = np.zeros(n_dipoles)
        x_hats = [deepcopy(x_hat)]

        omega = np.array([], dtype=int)
        r = y.copy()
        residuals = np.array([np.linalg.norm(r)])

        max_iter = min(n_chans, K)
        for _ in range(max_iter):
            if include_patches and include_singletons:
                b_smooth = np.abs(self.leadfield_smooth_normed.T @ r)
                b_sparse = np.abs(self.leadfield_normed.T @ r)
                if b_sparse.max() > b_smooth.max():
                    b_thresh = thresholding(b_sparse, 1)
                    new_atoms = np.where(b_thresh != 0)[0]
                else:
                    b_thresh = thresholding(b_smooth, 1)
                    best_idx = np.where(b_thresh != 0)[0]
                    new_atoms = []
                    for idx in best_idx:
                        patch = np.where(self.adjacency[idx] != 0)[0]
                        patch = np.append(patch, idx)
                        new_atoms.append(patch)
                    new_atoms = np.unique(np.concatenate(new_atoms)) if new_atoms else np.array([], dtype=int)
            elif include_patches:
                b_smooth = np.abs(self.leadfield_smooth_normed.T @ r)
                b_thresh = thresholding(b_smooth, 1)
                best_idx = np.where(b_thresh != 0)[0]
                new_atoms = []
                for idx in best_idx:
                    patch = np.where(self.adjacency[idx] != 0)[0]
                    patch = np.append(patch, idx)
                    new_atoms.append(patch)
                new_atoms = np.unique(np.concatenate(new_atoms)) if new_atoms else np.array([], dtype=int)
            else:  # include_singletons only
                b_sparse = self.leadfield_normed.T @ r
                b_thresh = thresholding(b_sparse, 1)
                new_atoms = np.where(b_thresh != 0)[0]

            omega = np.unique(np.append(omega, new_atoms)).astype(int)

            if len(omega) > 0:
                L_omega = self.leadfield_original[:, omega]
                x_hat = np.zeros(n_dipoles)
                x_hat[omega] = self.robust_inverse_solution(L_omega, y)

            r = y - self.leadfield_original @ x_hat

            residuals = np.append(residuals, np.linalg.norm(r))
            x_hats.append(deepcopy(x_hat))

            if len(residuals) > 1 and residuals[-1] > residuals[-2]:
                break

        x_hat = x_hats[int(np.argmin(residuals))]
        return x_hat
