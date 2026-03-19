import logging

import mne
import numpy as np
from scipy.sparse import csr_matrix, issparse

from invert.util import build_source_adjacency

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverSSLOFO(BaseSolver):
    """Standardized Shrinking LORETA-FOCUSS (SSLOFO) inverse solver.

    SSLOFO is a hybrid EEG/MEG inverse method that seeds a high-resolution FOCUSS
    solver with an sLORETA estimate, standardizes the updates to reduce depth/smoothing
    bias, and shrinks (prunes) the source space iteratively with light smoothing to
    avoid over-focal local minima.

    Parameters
    ----------
    max_iter : int
        Maximum number of iterations for FOCUSS updates (default: 20)
    focuss_power : float
        Power parameter for FOCUSS weight updates (default: 1.0)
    percentile_threshold : float
        Percentile threshold for keeping "prominent" sources (default: 0.01)
    sparsity_threshold : int or None
        Minimum sparsity threshold; stop shrinking if fewer sources (default: None)
    convergence_tol : float
        Convergence tolerance for solution changes (default: 1e-4)
    smoothing_weight : float
        Weight for neighbor smoothing (0 = no smoothing, 1 = full smoothing) (default: 0.3)
    final_smoothing : bool
        Whether to apply final smoothing to the solution (default: True)
    spatial_temporal : bool
        Whether to use spatio-temporal variant (default: False)
    time_window : int or None
        Time window size for spatio-temporal variant (default: None)
    sloreta_alpha : float
        Regularization parameter for initial sLORETA estimate (default: 0.01)
    verbose : int
        Verbosity level (default: 0)

    References
    ----------
    [1] Liu, H., Schimpf, P. H., Dong, G., Gao, X., Yang, F., & Gao, S. (2005). 
    Standardized Shrinking LORETA-FOCUSS (SSLOFO): A new algorithm for spatio-temporal 
    EEG source reconstruction. IEEE Transactions on Biomedical Engineering, 52(10), 1681–1691.

    Notes
    -----
    The algorithm follows these steps:
    1. Compute initial current density with sLORETA
    2. Initialize FOCUSS weights from that estimate
    3. Run standardized FOCUSS update (normalized by resolution matrix)
    4. Keep prominent nonzero nodes and neighbors; apply light smoothing
    5. Shrink the problem: restrict leadfield to retained nodes
    6. Recompute weights from the new estimate
    7. Repeat steps 3-6 until convergence
    8. Optionally smooth the final solution

    For spatio-temporal variant:
    a) Run single-time-point SSLOFO at each sample in a window
    b) Sum the solutions and define common support as nonzeros of this sum
    c) Re-solve at each sample restricted to that support with fixed weights
    """

    meta = SolverMeta(
        acronym="SSLOFO",
        full_name="Standardized Shrinking LORETA-FOCUSS",
        category="Minimum Norm",
        description=(
            "Hybrid method combining sLORETA and reweighted (FOCUSS-style) updates "
            "with iterative source-space shrinking to obtain focal solutions."
        ),
        references=[
            "Liu, H., Schimpf, P. H., Dong, G., Gao, X., Yang, F., & Gao, S. (2005). Standardized Shrinking LORETA-FOCUSS (SSLOFO): A new algorithm for spatio-temporal EEG source reconstruction. IEEE Transactions on Biomedical Engineering, 52(10), 1681–1691.",
            "Pascual-Marqui, R. D. (2002). Standardized low-resolution brain electromagnetic tomography (sLORETA): technical details. Methods and Findings in Experimental and Clinical Pharmacology, 24(Suppl D), 5–12.",
        ],
    )

    def __init__(
        self,
        name="Standardized Shrinking LORETA-FOCUSS",
        max_iter=30,
        focuss_power=0.5,
        percentile_threshold=0.05,
        sparsity_threshold=50,
        convergence_tol=1e-5,
        smoothing_weight=0.1,
        final_smoothing=False,
        spatial_temporal=True,
        time_window=None,
        sloreta_alpha=0.01,
        **kwargs,
    ):
        self.name = name
        self.max_iter = max_iter
        self.focuss_power = focuss_power
        self.percentile_threshold = percentile_threshold
        self.sparsity_threshold = sparsity_threshold
        self.convergence_tol = convergence_tol
        self.smoothing_weight = smoothing_weight
        self.final_smoothing = final_smoothing
        self.spatial_temporal = spatial_temporal
        self.time_window = time_window
        self.sloreta_alpha = sloreta_alpha
        self.adjacency: csr_matrix | None = None
        self._adjacency_degree: np.ndarray | None = None
        self.leadfield_full: np.ndarray | None = None
        self._sloreta_operator: np.ndarray | None = None
        super().__init__(**kwargs)
        # The default BaseSolver r-grid (up to 1e1) can be too narrow once SSP
        # projection/whitening changes the effective scaling. Widen it so GCV
        # doesn't systematically saturate at the upper edge.
        self.r_values = np.logspace(-10, 4, int(max(self.n_reg_params, 1)))

    def make_inverse_operator(
        self,
        forward,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
        **kwargs,
    ):
        """Calculate inverse operator using SSLOFO.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        alpha : float or 'auto'
            The regularization parameter for weighted minimum norm updates.

        Return
        ------
        self : object returns itself for convenience
        """
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        self.prepare_whitened_forward(noise_cov)

        # Get adjacency matrix for smoothing operations
        self.adjacency = build_source_adjacency(
            forward["src"], verbose=self.verbose
        ).tocsr()
        self._adjacency_degree = self.adjacency.getnnz(axis=1).astype(float)

        # Store original leadfield
        self.leadfield_full = np.array(self.leadfield, copy=True)
        self._sloreta_operator = self._prepare_sloreta_operator(
            self.leadfield_full, self.sloreta_alpha
        )

        # Regularization candidates must be scaled to the *effective* leadfield
        # used for selection (whitened/projected space). BaseSolver computed
        # alphas before whitening, so recompute here.
        self.get_alphas(reference=self.leadfield_full @ self.leadfield_full.T)

        # Build MNE kernels for regularization selection. We intentionally run
        # selection on the unstandardized MNE operators and only use the chosen
        # alpha for the SSLOFO iterations (analogous to the sLORETA fix).
        leadfield = self.leadfield_full
        sensor_transform = self._sensor_transform
        n_chans = leadfield.shape[0]
        LLT = leadfield @ leadfield.T
        I = np.identity(n_chans)

        mne_operators = []
        for alpha_eff in self.alphas:
            inner_inv = np.linalg.pinv(LLT + alpha_eff * I)
            op_eff = leadfield.T @ inner_inv
            # Make operators compatible with BaseSolver regularisation selection:
            # regularise_* expects raw sensor data and uses _sensor_transform for
            # residuals. Therefore inverse operators must map raw->source.
            mne_operators.append(op_eff @ sensor_transform)

        self.inverse_operators = [
            InverseOperator(op, self.name) for op in mne_operators
        ]
        self.made_inverse_operator = True
        return self

    def apply_inverse_operator(self, mne_obj):
        """Apply the SSLOFO inverse operator to reconstruct sources.

        Parameters
        ----------
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.

        Return
        ------
        stc : mne.SourceEstimate
            The source estimate.
        """
        data_raw = self.unpack_data_obj(mne_obj)
        self.validate_operator_data_compatibility(data_raw)
        if data_raw.ndim == 1:
            data_raw = data_raw[:, np.newaxis]

        if self.use_last_alpha and self.last_reg_idx is not None:
            idx = int(self.last_reg_idx)
        else:
            # Select alpha on raw data. BaseSolver regularise_* methods will
            # apply _sensor_transform internally for residuals.
            idx = self._select_regularization_idx(data_raw)
            self.last_reg_idx = idx

        alpha = self.alphas[int(np.clip(idx, 0, len(self.alphas) - 1))]

        # SSLOFO iterations operate in whitened/projected sensor space.
        data = self._sensor_transform @ data_raw
        initial_sloreta = self._compute_sloreta(
            self.leadfield_full, data, self.sloreta_alpha
        )

        if self.spatial_temporal and data.ndim == 2:
            # Use spatio-temporal variant
            source_mat = self._solve_spatiotemporal(
                data, alpha, initial_sloreta=initial_sloreta
            )
        else:
            # Use single-timepoint SSLOFO
            # Solve for each timepoint
            n_sources = self.leadfield_full.shape[1]
            n_times = data.shape[1]
            source_mat = np.zeros((n_sources, n_times))

            for t in range(n_times):
                if self.verbose > 1:
                    logger.debug(f"Solving timepoint {t + 1}/{n_times}")
                source_mat[:, t] = self._solve_single_timepoint(
                    data[:, t],
                    alpha,
                    initial_sloreta=initial_sloreta[:, t],
                )

        # Convert to SourceEstimate
        stc = self.source_to_object(source_mat)
        return stc

    def _solve_single_timepoint(self, data_t, alpha, initial_sloreta=None):
        """Solve SSLOFO for a single timepoint.

        Parameters
        ----------
        data_t : numpy.ndarray
            Data vector for a single timepoint (n_channels,)
        alpha : float
            Regularization parameter

        Return
        ------
        J_final : numpy.ndarray
            Source estimate for this timepoint (n_sources,)
        """
        leadfield = self.leadfield_full
        n_sources = leadfield.shape[1]

        # Step 1: Compute initial sLORETA estimate
        if initial_sloreta is None:
            J_sloreta = self._compute_sloreta(leadfield, data_t, self.sloreta_alpha)
        else:
            J_sloreta = np.array(initial_sloreta, copy=True)

        # Step 2: Initialize FOCUSS weights from sLORETA estimate (full source space)
        W_full = np.abs(J_sloreta) ** self.focuss_power
        W_full = np.maximum(W_full, 1e-12)  # Avoid zeros

        # Initialize active source indices (all sources initially)
        active_idx = np.arange(n_sources)
        J_current = J_sloreta.copy()

        # Track iterations
        for iteration in range(self.max_iter):
            if self.verbose > 1:
                logger.debug(
                    f"  Iteration {iteration + 1}/{self.max_iter}, active sources: {len(active_idx)}"
                )

            # Extract weights for active sources
            W_active = W_full[active_idx]

            # Step 3: Standardized FOCUSS update
            J_new = self._standardized_focuss_update(
                leadfield[:, active_idx], data_t, W_active, alpha
            )

            # Step 4: Keep prominent nodes and neighbors, apply smoothing
            prominent_mask, J_smoothed = self._shrink_and_smooth(
                J_new,
                active_idx,
                self.adjacency,
                self.percentile_threshold,
                self.smoothing_weight,
            )

            # Check convergence
            J_full = np.zeros(n_sources)
            J_full[active_idx] = J_smoothed

            change = np.linalg.norm(J_full - J_current) / (
                np.linalg.norm(J_current) + 1e-12
            )
            J_current = J_full.copy()

            if self.verbose > 2:
                logger.debug(
                    f"    Change: {change:.6e}, Sparsity: {np.sum(prominent_mask)}"
                )

            # Step 5: Shrink the problem
            new_active_idx = active_idx[prominent_mask]

            # Check stopping criteria
            if change < self.convergence_tol:
                if self.verbose > 1:
                    logger.info(f"  Converged at iteration {iteration + 1}")
                break

            if len(new_active_idx) == 0:
                if self.verbose > 1:
                    logger.warning(f"  All sources pruned at iteration {iteration + 1}")
                break

            if (
                self.sparsity_threshold is not None
                and len(new_active_idx) < self.sparsity_threshold
            ):
                if self.verbose > 1:
                    logger.info(
                        f"  Sparsity threshold reached at iteration {iteration + 1}"
                    )
                break

            if len(new_active_idx) >= len(active_idx):
                # Sparsity not improving, stop shrinking
                if self.verbose > 1:
                    logger.info(
                        f"  Sparsity not improving at iteration {iteration + 1}"
                    )
                break

            # Step 6: Recompute weights from new estimate (update full weight array)
            active_idx = new_active_idx
            W_full = np.abs(J_current) ** self.focuss_power
            W_full = np.maximum(W_full, 1e-12)

        # Step 8: Optional final smoothing
        if self.final_smoothing:
            J_final = self._apply_final_smoothing(J_current, self.adjacency)
        else:
            J_final = J_current

        return J_final

    def _solve_sensor_system(self, system, rhs):
        """Solve the sensor-space linear system, falling back to a pseudo-inverse."""
        try:
            return np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(system) @ rhs

    def _prepare_sloreta_operator(self, leadfield, alpha):
        """Precompute the full-space standardized sLORETA operator.

        This is reused for every time point. Caching it avoids rebuilding the
        same full sensor-space inverse at the start of each SSLOFO solve.
        """
        system = leadfield @ leadfield.T
        system.flat[:: system.shape[0] + 1] += alpha

        solved_leadfield = self._solve_sensor_system(system, leadfield)
        kernel = solved_leadfield.T
        res_matrix_diag = np.einsum(
            "ij,ij->j", leadfield, solved_leadfield, optimize=True
        )
        res_scale = 1.0 / np.sqrt(np.maximum(res_matrix_diag, 1e-12))
        return kernel * res_scale[:, np.newaxis]

    def _compute_sloreta(self, leadfield, data, alpha):
        """Compute sLORETA estimate.

        Parameters
        ----------
        leadfield : numpy.ndarray
            Leadfield matrix (n_channels, n_sources)
        data : numpy.ndarray
            Data vector (n_channels,)
        alpha : float
            Regularization parameter

        Return
        ------
        J : numpy.ndarray
            sLORETA source estimate (n_sources,)
        """
        data_arr = np.asarray(data, dtype=float)
        data_was_vector = data_arr.ndim == 1
        if data_was_vector:
            data_arr = data_arr[:, np.newaxis]

        if (
            self._sloreta_operator is not None
            and leadfield is self.leadfield_full
            and np.isclose(alpha, self.sloreta_alpha)
        ):
            j_sloreta = self._sloreta_operator @ data_arr
        else:
            system = leadfield @ leadfield.T
            system.flat[:: system.shape[0] + 1] += alpha

            solved_data = self._solve_sensor_system(system, data_arr)
            solved_leadfield = self._solve_sensor_system(system, leadfield)

            j = leadfield.T @ solved_data
            res_matrix_diag = np.einsum(
                "ij,ij->j", leadfield, solved_leadfield, optimize=True
            )
            res_matrix_diag = np.maximum(res_matrix_diag, 1e-12)
            j_sloreta = j / np.sqrt(res_matrix_diag)[:, np.newaxis]

        return j_sloreta[:, 0] if data_was_vector else j_sloreta

    def _standardized_focuss_update(self, leadfield, data, weights, alpha):
        """Perform standardized FOCUSS update.

        Parameters
        ----------
        leadfield : numpy.ndarray
            Leadfield matrix (n_channels, n_sources_active)
        data : numpy.ndarray
            Data vector (n_channels,)
        weights : numpy.ndarray
            Current FOCUSS weights (n_sources_active,)
        alpha : float
            Regularization parameter

        Return
        ------
        J : numpy.ndarray
            Updated source estimate (n_sources_active,)
        """
        # L @ diag(weights) @ L.T via broadcasting (avoids d×d diag matrix)
        LW = leadfield * weights[np.newaxis, :]  # (m, d) broadcasting
        LWLT = LW @ leadfield.T  # (m, m)
        LWLT.flat[:: LWLT.shape[0] + 1] += alpha

        # Solve A x = b directly instead of forming A^{-1}. The system is only
        # sensor-sized, so this is faster and numerically better than building
        # an explicit inverse at every iteration.
        solved_data = self._solve_sensor_system(LWLT, data)
        solved_leadfield = self._solve_sensor_system(LWLT, leadfield)

        j = weights * (leadfield.T @ solved_data)

        # diag(diag(w) L^T A^{-1} L) = w_i * l_i^T A^{-1} l_i
        res_matrix_diag = weights * np.einsum(
            "ij,ij->j", leadfield, solved_leadfield, optimize=True
        )
        res_matrix_diag = np.maximum(res_matrix_diag, 1e-12)
        j_standardized = j / np.sqrt(res_matrix_diag)

        return j_standardized

    def _shrink_and_smooth(
        self, J, active_idx, adjacency, percentile, smoothing_weight
    ):
        """Keep prominent sources and neighbors, apply light smoothing.

        Parameters
        ----------
        J : numpy.ndarray
            Current source estimate (n_sources_active,)
        active_idx : numpy.ndarray
            Indices of active sources in full source space
        adjacency : numpy.ndarray
            Adjacency matrix (n_sources_full, n_sources_full)
        percentile : float
            Percentile threshold for keeping sources
        smoothing_weight : float
            Weight for neighbor averaging (0 = no smoothing, 1 = full averaging)

        Return
        ------
        prominent_mask : numpy.ndarray
            Boolean mask of sources to keep (n_sources_active,)
        J_smoothed : numpy.ndarray
            Smoothed source estimate (n_sources_active,)
        """
        adjacency = adjacency.tocsr() if issparse(adjacency) else csr_matrix(adjacency)

        # Find prominent sources
        max_val = np.abs(J).max()
        if max_val == 0:
            # No activity, keep all
            return np.ones(len(J), dtype=bool), J

        threshold = max_val * percentile
        prominent_local_idx = np.where(np.abs(J) > threshold)[0]

        if len(prominent_local_idx) == 0:
            # Keep at least the maximum
            prominent_local_idx = np.array([np.argmax(np.abs(J))])

        # Convert to global indices
        prominent_global_idx = active_idx[prominent_local_idx]

        # Sparse row gathering gives the union of all graph neighbors without
        # scanning the full dense adjacency matrix.
        neighbor_rows = adjacency[prominent_global_idx]
        neighbor_global_idx = np.unique(neighbor_rows.indices)

        # Combine prominent sources and neighbors
        keep_global_idx = np.unique(
            np.concatenate([prominent_global_idx, neighbor_global_idx])
        )

        # Create mask in local (active) space
        prominent_mask = np.isin(active_idx, keep_global_idx)

        # Apply light smoothing
        J_smoothed = J.copy()
        if smoothing_weight > 0:
            # Restrict smoothing to active sources only. This matches the
            # original logic but avoids rebuilding a global->local map in Python.
            adjacency_active = adjacency[active_idx][:, active_idx].tocsr()
            neighbor_count = adjacency_active.getnnz(axis=1)
            smooth_local_idx = np.where(prominent_mask & (neighbor_count > 0))[0]

            if len(smooth_local_idx) > 0:
                neighbor_sum = np.asarray(adjacency_active @ J).ravel()
                neighbor_avg = neighbor_sum[smooth_local_idx] / neighbor_count[
                    smooth_local_idx
                ]
                J_smoothed[smooth_local_idx] = (
                    (1 - smoothing_weight) * J[smooth_local_idx]
                    + smoothing_weight * neighbor_avg
                )

        return prominent_mask, J_smoothed

    def _apply_final_smoothing(self, J, adjacency):
        """Apply final smoothing to the solution.

        Parameters
        ----------
        J : numpy.ndarray
            Source estimate (n_sources,)
        adjacency : numpy.ndarray
            Adjacency matrix (n_sources, n_sources)

        Return
        ------
        J_smoothed : numpy.ndarray
            Smoothed source estimate (n_sources,)
        """
        adjacency = adjacency.tocsr() if issparse(adjacency) else csr_matrix(adjacency)
        J_smoothed = J.copy()
        active_sources = np.where(np.abs(J) > 0)[0]
        if len(active_sources) == 0:
            return J_smoothed

        if self._adjacency_degree is not None and len(self._adjacency_degree) == len(J):
            neighbor_count = self._adjacency_degree
        else:
            neighbor_count = adjacency.getnnz(axis=1).astype(float)

        valid_active = active_sources[neighbor_count[active_sources] > 0]
        if len(valid_active) == 0:
            return J_smoothed

        neighbor_sum = np.asarray(adjacency @ J).ravel()
        # This pass averages over all graph neighbors, including zeros for
        # inactive nodes, exactly like the original dense implementation.
        neighbor_avg = neighbor_sum[valid_active] / neighbor_count[valid_active]
        J_smoothed[valid_active] = 0.7 * J[valid_active] + 0.3 * neighbor_avg

        return J_smoothed

    def _solve_spatiotemporal(self, data, alpha, initial_sloreta=None):
        """Solve using spatio-temporal variant.

        Parameters
        ----------
        data : numpy.ndarray
            Data matrix (n_channels, n_times)

        Return
        ------
        source_mat : numpy.ndarray
            Source estimate matrix (n_sources, n_times)
        """
        n_times = data.shape[1]
        n_sources = self.leadfield_full.shape[1]

        # Determine time window
        if self.time_window is None:
            window_size = n_times
        else:
            window_size = min(self.time_window, n_times)

        if self.verbose > 0:
            logger.info(
                f"Solving spatio-temporal SSLOFO with window size {window_size}"
            )

        # Step a: Run single-time-point SSLOFO at each sample in window
        solutions = np.zeros((window_size, n_sources))
        for t in range(window_size):
            if self.verbose > 1:
                logger.debug(f"  Solving timepoint {t + 1}/{window_size}")
            seed_t = None if initial_sloreta is None else initial_sloreta[:, t]
            solutions[t, :] = self._solve_single_timepoint(
                data[:, t], alpha, initial_sloreta=seed_t
            )

        # Step b: Sum solutions and define common support
        J_sum = np.sum(np.abs(solutions), axis=0)
        support_mask = J_sum > 0
        support_idx = np.where(support_mask)[0]

        if len(support_idx) == 0:
            if self.verbose > 0:
                logger.warning("Empty support, returning zero solution")
            return np.zeros((n_sources, n_times))

        if self.verbose > 0:
            logger.info(f"Common support has {len(support_idx)} sources")

        # Step c: Re-solve all timepoints at once on the common support
        source_mat = np.zeros((n_sources, n_times))
        leadfield_support = self.leadfield_full[:, support_idx]

        # Use fixed weights from the summed solution
        weights = J_sum[support_idx] ** self.focuss_power
        weights = np.maximum(weights, 1e-12)

        # Since weights and leadfield are fixed, compute inverse operator once
        # and apply to all timepoints as a single matrix multiply
        J_all = self._weighted_minimum_norm(leadfield_support, data, weights, alpha)
        if J_all.ndim == 1:
            J_all = J_all[:, np.newaxis]
        source_mat[support_idx, :] = J_all

        return source_mat

    def _weighted_minimum_norm(self, leadfield, data, weights, alpha):
        """Compute weighted minimum norm solution.

        Parameters
        ----------
        leadfield : numpy.ndarray
            Leadfield matrix (n_channels, n_sources)
        data : numpy.ndarray
            Data vector or matrix (n_channels,) or (n_channels, n_times)
        weights : numpy.ndarray
            Source weights (n_sources,)
        alpha : float
            Regularization parameter

        Return
        ------
        J : numpy.ndarray
            Source estimate (n_sources,) or (n_sources, n_times)
        """
        # L @ diag(weights) @ L.T via broadcasting
        LW = leadfield * weights[np.newaxis, :]  # (m, d)
        LWLT = LW @ leadfield.T  # (m, m)
        LWLT.flat[:: LWLT.shape[0] + 1] += alpha

        solved_data = self._solve_sensor_system(LWLT, data)
        J = weights[:, None] * (leadfield.T @ solved_data)
        return J.squeeze()

    def _select_regularization_idx(self, data):
        """Select regularization index using the specified method.

        Parameters
        ----------
        data : numpy.ndarray
            Data matrix (n_channels, n_times) or a single sample vector (n_channels,).

        Return
        ------
        idx : int
            Selected regularization index into ``self.alphas``.
        """
        method = self.regularisation_method.lower()
        if method == "l":
            _, idx = self.regularise_lcurve(data, plot=self.plot_reg)
        elif method in {"gcv", "mgcv", "rgcv", "r1gcv", "composite"}:
            if method == "gcv":
                gamma = self.gcv_gamma
            elif method == "mgcv":
                gamma = self.mgcv_gamma
            elif method in {"rgcv", "composite"}:
                gamma = self.rgcv_gamma
            else:
                gamma = self.r1gcv_gamma
            _, idx = self.regularise_gcv(
                data, plot=self.plot_reg, gamma=gamma, method=method
            )
        elif method == "product":
            _, idx = self.regularise_product(data, plot=self.plot_reg)
        else:
            idx = len(self.alphas) // 2

        return int(np.clip(idx, 0, len(self.alphas) - 1))

    def _select_regularization(self, data_sample):
        """Backward-compatible helper returning the selected alpha value."""
        idx = self._select_regularization_idx(data_sample)
        return self.alphas[idx]
