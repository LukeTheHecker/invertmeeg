import logging
from copy import deepcopy

import mne
import numpy as np

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
    [1] Standardized Shrinking LORETA-FOCUSS (SSLOFO): A new method for spatio-temporal
        EEG source reconstruction

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
            "Wu, H., Gao, S., Li, J., & Li, X. (2005). A new method for spatio-temporal EEG source reconstruction: standardized shrinking LORETA-FOCUSS (SSLOFO). IEEE Transactions on Biomedical Engineering, 52(11), 1781–1792. https://doi.org/10.1109/TBME.2005.855720",
            "Pascual-Marqui, R. D. (2002). Standardized low-resolution brain electromagnetic tomography (sLORETA): technical details. Methods and Findings in Experimental and Clinical Pharmacology, 24(Suppl D), 5–12.",
            "Gorodnitsky, I. F., & Rao, B. D. (1997). Sparse signal reconstruction from limited data using FOCUSS: A re-weighted minimum norm algorithm. IEEE Transactions on Signal Processing, 45(3), 600–616.",
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
        super().__init__(**kwargs)

    def make_inverse_operator(self, forward, *args, alpha="auto", **kwargs):
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

        # Get adjacency matrix for smoothing operations
        self.adjacency = mne.spatial_src_adjacency(
            forward["src"], verbose=self.verbose
        ).toarray()

        # Store original leadfield
        self.leadfield_full = deepcopy(self.leadfield)

        # Build MNE kernels for regularization selection. We intentionally run
        # selection on the unstandardized MNE operators and only use the chosen
        # alpha for the SSLOFO iterations (analogous to the sLORETA fix).
        leadfield = self.leadfield_full
        n_chans = leadfield.shape[0]
        LLT = leadfield @ leadfield.T
        I = np.identity(n_chans)

        mne_operators = []
        for alpha_eff in self.alphas:
            inner_inv = np.linalg.pinv(LLT + alpha_eff * I)
            mne_operators.append(leadfield.T @ inner_inv)

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
        data = self.unpack_data_obj(mne_obj)
        if data.ndim == 1:
            data = data[:, np.newaxis]

        if self.use_last_alpha and self.last_reg_idx is not None:
            idx = int(self.last_reg_idx)
        else:
            idx = self._select_regularization_idx(data)
            self.last_reg_idx = idx

        alpha = self.alphas[int(np.clip(idx, 0, len(self.alphas) - 1))]

        if self.spatial_temporal and data.ndim == 2:
            # Use spatio-temporal variant
            source_mat = self._solve_spatiotemporal(data, alpha)
        else:
            # Use single-timepoint SSLOFO
            # Solve for each timepoint
            n_sources = self.leadfield_full.shape[1]
            n_times = data.shape[1]
            source_mat = np.zeros((n_sources, n_times))

            for t in range(n_times):
                if self.verbose > 1:
                    logger.debug(f"Solving timepoint {t + 1}/{n_times}")
                source_mat[:, t] = self._solve_single_timepoint(data[:, t], alpha)

        # Convert to SourceEstimate
        stc = self.source_to_object(source_mat)
        return stc

    def _solve_single_timepoint(self, data_t, alpha):
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
        leadfield = deepcopy(self.leadfield_full)
        n_chans, n_sources = leadfield.shape

        # Step 1: Compute initial sLORETA estimate
        J_sloreta = self._compute_sloreta(leadfield, data_t, self.sloreta_alpha)

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
        n_chans = leadfield.shape[0]
        I = np.identity(n_chans)

        # Compute MNE inverse operator: K = L.T @ inv(L @ L.T + alpha * I)
        LLT = leadfield @ leadfield.T
        try:
            inner_inv = np.linalg.inv(LLT + alpha * I)
        except np.linalg.LinAlgError:
            inner_inv = np.linalg.pinv(LLT + alpha * I)

        K_MNE = leadfield.T @ inner_inv

        # Compute resolution matrix diagonal for standardization
        # Rdiag = diag(K @ L) = sum of element-wise product of K and L.T along axis=1
        res_matrix_diag = np.sum(K_MNE * leadfield.T, axis=1)
        res_matrix_diag = np.maximum(res_matrix_diag, 1e-12)  # Avoid division by zero

        # Compute current density and standardize
        j = K_MNE @ data
        j_sloreta = j / np.sqrt(res_matrix_diag)

        return j_sloreta

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
        n_chans = leadfield.shape[0]

        # Construct weighted inverse operator: K = W @ L.T @ inv(L @ W @ L.T + alpha * I)
        W_diag = np.diag(weights)
        LW = leadfield @ W_diag  # L @ W
        LWLT = LW @ leadfield.T  # L @ W @ L.T (NOT W @ W!)
        I = np.identity(n_chans)

        try:
            inner_inv = np.linalg.inv(LWLT + alpha * I)
        except np.linalg.LinAlgError:
            inner_inv = np.linalg.pinv(LWLT + alpha * I)

        # Weighted inverse operator
        K_eff = W_diag @ leadfield.T @ inner_inv

        # Compute current density
        j = K_eff @ data

        # Standardize by resolution matrix to reduce depth bias
        # Rdiag = diag(K @ L) = sum of element-wise product
        res_matrix_diag = np.sum(K_eff * leadfield.T, axis=1)
        res_matrix_diag = np.maximum(res_matrix_diag, 1e-12)

        # Standardize
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

        # Find neighbors of prominent sources efficiently using vectorized operations
        # Extract relevant rows from adjacency matrix and combine
        neighbor_mask = adjacency[prominent_global_idx, :].sum(axis=0) > 0
        neighbor_global_idx = np.where(neighbor_mask)[0]

        # Combine prominent sources and neighbors
        keep_global_idx = np.unique(
            np.concatenate([prominent_global_idx, neighbor_global_idx])
        )

        # Create mask in local (active) space
        prominent_mask = np.isin(active_idx, keep_global_idx)

        # Apply light smoothing
        J_smoothed = J.copy()
        if smoothing_weight > 0:
            # Pre-compute which active indices to smooth
            smooth_local_idx = np.where(prominent_mask)[0]
            smooth_global_idx = active_idx[smooth_local_idx]

            # Vectorized neighbor finding: get adjacency rows for all sources to smooth
            # Create a mapping from global to local indices for fast lookup
            global_to_local = np.full(adjacency.shape[0], -1, dtype=np.int32)
            global_to_local[active_idx] = np.arange(len(active_idx))

            for local_idx, global_idx in zip(smooth_local_idx, smooth_global_idx):
                # Find neighbors in global space
                neighbor_global = adjacency[global_idx, :].nonzero()[0]

                # Map to local indices (filter out inactive neighbors)
                neighbor_local = global_to_local[neighbor_global]
                neighbor_local = neighbor_local[neighbor_local >= 0]

                if len(neighbor_local) > 0:
                    # Weighted average with neighbors
                    neighbor_avg = np.mean(J[neighbor_local])
                    J_smoothed[local_idx] = (1 - smoothing_weight) * J[
                        local_idx
                    ] + smoothing_weight * neighbor_avg

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
        J_smoothed = J.copy()
        active_sources = np.where(np.abs(J) > 0)[0]

        for idx in active_sources:
            neighbors = np.where(adjacency[idx, :] == 1)[0]
            if len(neighbors) > 0:
                neighbor_vals = J[neighbors]
                # Light smoothing: 70% original, 30% neighbor average
                J_smoothed[idx] = 0.7 * J[idx] + 0.3 * np.mean(neighbor_vals)

        return J_smoothed

    def _solve_spatiotemporal(self, data, alpha):
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
        n_chans, n_times = data.shape
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
        solutions = []
        for t in range(window_size):
            if self.verbose > 1:
                logger.debug(f"  Solving timepoint {t + 1}/{window_size}")
            J_t = self._solve_single_timepoint(data[:, t], alpha)
            solutions.append(J_t)

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

        # Step c: Re-solve at each sample restricted to common support
        source_mat = np.zeros((n_sources, n_times))
        leadfield_support = self.leadfield_full[:, support_idx]

        # Use fixed weights from the summed solution
        weights = J_sum[support_idx] ** self.focuss_power
        weights = np.maximum(weights, 1e-12)

        for t in range(n_times):
            if self.verbose > 1:
                logger.debug(
                    f"  Re-solving timepoint {t + 1}/{n_times} with fixed support"
                )

            # Weighted minimum norm on the support
            J_support = self._weighted_minimum_norm(
                leadfield_support, data[:, t], weights, alpha
            )
            source_mat[support_idx, t] = J_support

        return source_mat

    def _weighted_minimum_norm(self, leadfield, data, weights, alpha):
        """Compute weighted minimum norm solution.

        Parameters
        ----------
        leadfield : numpy.ndarray
            Leadfield matrix (n_channels, n_sources)
        data : numpy.ndarray
            Data vector (n_channels,)
        weights : numpy.ndarray
            Source weights (n_sources,)
        alpha : float
            Regularization parameter

        Return
        ------
        J : numpy.ndarray
            Source estimate (n_sources,)
        """
        n_chans = leadfield.shape[0]
        W_diag = np.diag(weights)
        LW = leadfield @ W_diag  # L @ W
        LWLT = LW @ leadfield.T  # L @ W @ L.T (NOT W @ W!)
        I = np.identity(n_chans)

        try:
            inner_inv = np.linalg.inv(LWLT + alpha * I)
        except np.linalg.LinAlgError:
            inner_inv = np.linalg.pinv(LWLT + alpha * I)

        # Weighted inverse operator: W @ L.T @ inv(...)
        J = W_diag @ leadfield.T @ inner_inv @ data
        return J

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
        if self.regularisation_method.lower() == "l":
            _, idx = self.regularise_lcurve(data, plot=self.plot_reg)
        elif self.regularisation_method.lower() in {"gcv", "mgcv"}:
            gamma = (
                self.gcv_gamma
                if self.regularisation_method.lower() == "gcv"
                else self.mgcv_gamma
            )
            _, idx = self.regularise_gcv(data, plot=self.plot_reg, gamma=gamma)
        elif self.regularisation_method.lower() == "product":
            _, idx = self.regularise_product(data, plot=self.plot_reg)
        else:
            idx = len(self.alphas) // 2

        return int(np.clip(idx, 0, len(self.alphas) - 1))

    def _select_regularization(self, data_sample):
        """Backward-compatible helper returning the selected alpha value."""
        idx = self._select_regularization_idx(data_sample)
        return self.alphas[idx]
