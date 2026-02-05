import logging

import mne
import numpy as np
from scipy.linalg import pinv

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverSelfRegularizedELORETA(BaseSolver):
    """Class for the Self-Regularized exact Low Resolution Tomography (SR-eLORETA)
    inverse solution, optimized for low SNR scenarios.

    This solver implements a three-step procedure:
    1. Compute eLORETA normally
    2. Compute the absolute average across timepoints and select the strongest activations
       using spatial peak detection to ensure multi-cluster coverage
    3. Recompute the inverse solution on these candidates using a reduced leadfield

    By leveraging information across multiple timepoints and spatial structure, this
    method provides better source localization in low SNR condi tions while avoiding
    the pitfall of selecting only sources from a single dominant cluster.

    References
    ----------
    [1] Pascual-Marqui, R. D. (2007). Discrete, 3D distributed, linear imaging
    methods of electric neuronal activity. Part 1: exact, zero error
    localization. arXiv preprint arXiv:0710.3341.
    """

    meta = SolverMeta(
        acronym="SR-eLORETA",
        full_name="Self-Regularized eLORETA",
        category="Experimental",
        description=(
            "Experimental extension of eLORETA that selects candidate sources "
            "from an initial estimate and re-solves on a reduced source space."
        ),
        references=[
            "Pascual-Marqui, R. D. (2007). Discrete, 3D distributed, linear imaging methods of electric neuronal activity. Part 1: exact, zero error localization. arXiv:0710.3341.",
            "tbd",
        ],
        internal=True,
    )

    def __init__(
        self,
        name="Self-Regularized Exact Low Resolution Tomography",
        selection_threshold=90,
        min_sources=50,
        max_sources=500,
        use_spatial_peaks=True,
        peak_distance=20,
        local_weight=0.5,
        **kwargs,
    ):
        self.name = name
        self.selection_threshold = selection_threshold
        self.min_sources = min_sources
        self.max_sources = max_sources
        self.use_spatial_peaks = use_spatial_peaks
        self.peak_distance = peak_distance  # mm distance for local maxima suppression
        self.local_weight = (
            local_weight  # Weight for combining global and local selection
        )
        return super().__init__(**kwargs)

    def make_inverse_operator(
        self,
        forward,
        *args,
        alpha="auto",
        verbose=0,
        stop_crit=1e-3,
        max_iter=100,
        **kwargs,
    ):
        """Calculate inverse operator for the initial eLORETA step.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        alpha : float or 'auto'
            The regularization parameter.
        stop_crit : float
            The convergence criterion to optimize the weight matrix.
        max_iter : int
            The stopping criterion to optimize the weight matrix.

        Return
        ------
        self : object returns itself for convenience
        """
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        self.stop_crit = stop_crit
        self.max_iter = max_iter

        leadfield = self.leadfield
        n_chans = leadfield.shape[0]

        # Some pre-calculations
        I = np.identity(n_chans)

        # No regularization leads to weird results with eLORETA
        if 0 in self.alphas and len(self.alphas) > 1:
            idx = self.alphas.index(0)
            self.alphas.pop(idx)
            self.r_values = np.delete(self.r_values, idx)
        elif 0 in self.alphas and len(self.alphas) == 1:
            idx = self.alphas.index(0)
            self.alphas = [0.01]

        inverse_operators = []
        for alpha in self.alphas:
            W = self.calc_W(alpha, max_iter=max_iter, stop_crit=stop_crit)

            # More efficient computation avoiding explicit W_inv matrix construction
            # Since W is diagonal, W_inv is also diagonal with reciprocal elements
            W_inv_diag = 1.0 / W.diagonal()

            # Compute leadfield @ W_inv more efficiently using broadcasting
            LW_inv = leadfield * W_inv_diag[np.newaxis, :]

            # Compute the final inverse operator
            inner_term = LW_inv @ leadfield.T + alpha * I
            inverse_operator = (W_inv_diag[:, np.newaxis] * leadfield.T) @ pinv(
                inner_term
            )

            inverse_operators.append(inverse_operator)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        return self

    def calc_W(self, alpha, max_iter=100, stop_crit=1e-3):
        """Calculate the weight matrix W for eLORETA.

        Parameters
        ----------
        alpha : float
            Regularization parameter
        max_iter : int
            Maximum number of iterations
        stop_crit : float
            Convergence criterion

        Returns
        -------
        W : scipy.sparse matrix
            Diagonal weight matrix
        """
        K = self.leadfield
        n_chans, n_dipoles = K.shape

        # Input validation
        if alpha <= 0:
            raise ValueError("Alpha must be positive")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if stop_crit <= 0:
            raise ValueError("stop_crit must be positive")

        # Use dense matrices for better performance in this iterative context
        I = np.identity(n_chans)
        W_diag = np.ones(n_dipoles)  # Store only diagonal elements

        # Pre-allocate arrays to avoid repeated memory allocation
        KT = K.T  # Cache transpose

        # Refine W iteratively
        for iter in range(max_iter):
            W_diag_old = W_diag.copy()

            # Ensure numerical stability by avoiding division by very small numbers
            W_inv_diag = 1.0 / np.maximum(W_diag, 1e-12)

            # Compute K @ W_inv @ K.T more efficiently
            # Since W is diagonal, K @ W_inv = K * W_inv_diag (broadcasting)
            KW_inv = K * W_inv_diag[np.newaxis, :]
            inner_matrix = KW_inv @ KT + alpha * I

            # Use more stable pseudo-inverse
            M = pinv(inner_matrix)

            # Compute diagonal elements more efficiently
            # diag(K.T @ M @ K) = sum(K.T * (M @ K), axis=0)
            MK = M @ K
            # KT is (n_dipoles, n_chans), MK is (n_chans, n_dipoles)
            # We need to compute the diagonal of KT @ MK
            diag_elements = np.sum(KT * MK.T, axis=1)

            # Ensure non-negative values before taking square root for numerical stability
            W_diag = np.sqrt(np.maximum(diag_elements, 1e-12))

            # More efficient convergence check using relative change
            rel_change = np.mean(np.abs(W_diag - W_diag_old) / (W_diag_old + 1e-12))

            if self.verbose > 1:
                logger.debug(f"iter {iter}: relative change = {rel_change:.6f}")

            if rel_change < stop_crit:
                if self.verbose > 0:
                    logger.info(f"eLORETA converged after {iter + 1} iterations")
                break
        else:
            if self.verbose > 0:
                logger.warning(
                    f"eLORETA reached max iterations ({max_iter}) without convergence"
                )

        # Return as sparse diagonal matrix for compatibility
        from scipy.sparse import diags

        return diags(W_diag, format="csr")

    def apply_inverse_operator(self, mne_obj) -> mne.SourceEstimate:
        """Apply the self-regularized inverse operator using a three-step procedure.

        Parameters
        ----------
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.

        Return
        ------
        stc : mne.SourceEstimate
            The mne SourceEstimate object with refined source estimates.
        """
        # Step 1: Compute eLORETA normally
        if self.verbose > 0:
            logger.info("Step 1: Computing initial eLORETA solution...")

        data = self.unpack_data_obj(mne_obj)

        # Apply regularization method to get initial source estimate
        if self.use_last_alpha and self.last_reg_idx is not None:
            initial_source_mat = self.inverse_operators[self.last_reg_idx].apply(data)
        else:
            if self.regularisation_method.lower() == "l":
                initial_source_mat, idx = self.regularise_lcurve(
                    data, plot=self.plot_reg
                )
                self.last_reg_idx = idx
            elif self.regularisation_method.lower() in {"gcv", "mgcv"}:
                gamma = (
                    self.gcv_gamma
                    if self.regularisation_method.lower() == "gcv"
                    else self.mgcv_gamma
                )
                initial_source_mat, idx = self.regularise_gcv(
                    data, plot=self.plot_reg, gamma=gamma
                )
                self.last_reg_idx = idx
            elif self.regularisation_method.lower() == "product":
                initial_source_mat, idx = self.regularise_product(
                    data, plot=self.plot_reg
                )
                self.last_reg_idx = idx
            else:
                msg = f"{self.regularisation_method} is no valid regularisation method."
                raise AttributeError(msg)

        # Step 2: Compute absolute average across timepoints and select strong activations
        if self.verbose > 0:
            logger.info(
                "Step 2: Selecting candidate sources based on temporal average and spatial peaks..."
            )

        # Compute the absolute average power across time
        temporal_average = np.mean(np.abs(initial_source_mat), axis=1)

        if self.use_spatial_peaks:
            # Use spatial peak detection to ensure multi-cluster coverage
            candidate_indices = self._select_spatial_peaks(temporal_average)
            n_candidates = len(candidate_indices)
            if self.verbose > 0:
                logger.info(
                    f"   Selected {n_candidates} candidate sources using spatial peak detection"
                )
        else:
            # Original global thresholding approach
            threshold = np.percentile(temporal_average, self.selection_threshold)
            candidate_indices = np.where(temporal_average >= threshold)[0]

            # Apply min/max constraints
            if len(candidate_indices) < self.min_sources:
                candidate_indices = np.argsort(temporal_average)[-self.min_sources :]
            elif len(candidate_indices) > self.max_sources:
                sorted_candidates = candidate_indices[
                    np.argsort(temporal_average[candidate_indices])
                ]
                candidate_indices = sorted_candidates[-self.max_sources :]

            n_candidates = len(candidate_indices)
            if self.verbose > 0:
                logger.info(
                    f"   Selected {n_candidates} candidate sources "
                    f"(threshold: {threshold:.2e}, percentile: {self.selection_threshold})"
                )

        # Step 3: Recompute inverse solution on selected candidates
        if self.verbose > 0:
            logger.info("Step 3: Recomputing inverse solution on candidate sources...")

        # Extract reduced leadfield for selected sources
        leadfield_reduced = self.leadfield[:, candidate_indices]

        # Get the alpha value from the regularization step
        if self.last_reg_idx is not None:
            alpha_refined = self.alphas[self.last_reg_idx]
        else:
            alpha_refined = self.alphas[0]

        # Compute refined weight matrix for candidates only
        n_chans = leadfield_reduced.shape[0]
        I = np.identity(n_chans)

        # Calculate W for the reduced leadfield
        W_reduced = self._calc_W_reduced(
            leadfield_reduced,
            alpha_refined,
            max_iter=self.max_iter,
            stop_crit=self.stop_crit,
        )

        # Compute refined inverse operator for candidates
        W_inv_diag = 1.0 / W_reduced.diagonal()
        LW_inv = leadfield_reduced * W_inv_diag[np.newaxis, :]
        inner_term = LW_inv @ leadfield_reduced.T + alpha_refined * I
        inverse_operator_refined = (
            W_inv_diag[:, np.newaxis] * leadfield_reduced.T
        ) @ pinv(inner_term)

        # Apply refined inverse operator to data
        source_mat_candidates = inverse_operator_refined @ data

        # Create full source matrix with zeros for non-selected sources
        n_sources = self.leadfield.shape[1]
        source_mat_refined = np.zeros((n_sources, data.shape[1]))
        source_mat_refined[candidate_indices, :] = source_mat_candidates

        # Convert to MNE SourceEstimate object
        stc = self.source_to_object(source_mat_refined)

        if self.verbose > 0:
            logger.info("Self-regularized eLORETA complete.")

        return stc

    def _calc_W_reduced(self, K_reduced, alpha, max_iter=100, stop_crit=1e-3):
        """Calculate the weight matrix W for a reduced leadfield.

        Parameters
        ----------
        K_reduced : numpy.ndarray
            Reduced leadfield matrix (n_channels, n_candidates)
        alpha : float
            Regularization parameter
        max_iter : int
            Maximum number of iterations
        stop_crit : float
            Convergence criterion

        Returns
        -------
        W : scipy.sparse matrix
            Diagonal weight matrix for reduced sources
        """
        n_chans, n_dipoles = K_reduced.shape

        # Use dense matrices for better performance
        I = np.identity(n_chans)
        W_diag = np.ones(n_dipoles)
        KT = K_reduced.T

        for iter in range(max_iter):
            W_diag_old = W_diag.copy()
            W_inv_diag = 1.0 / np.maximum(W_diag, 1e-12)

            KW_inv = K_reduced * W_inv_diag[np.newaxis, :]
            inner_matrix = KW_inv @ KT + alpha * I
            M = pinv(inner_matrix)

            MK = M @ K_reduced
            diag_elements = np.sum(KT * MK.T, axis=1)
            W_diag = np.sqrt(np.maximum(diag_elements, 1e-12))

            rel_change = np.mean(np.abs(W_diag - W_diag_old) / (W_diag_old + 1e-12))

            if self.verbose > 1:
                logger.debug(
                    f"   Refinement iter {iter}: relative change = {rel_change:.6f}"
                )

            if rel_change < stop_crit:
                if self.verbose > 1:
                    logger.debug(
                        f"   Refined eLORETA converged after {iter + 1} iterations"
                    )
                break

        from scipy.sparse import diags

        return diags(W_diag, format="csr")

    def _select_spatial_peaks(self, temporal_average):
        """Select candidate sources using spatial peak detection.

        This method combines global thresholding with local peak detection to ensure
        that multiple spatial clusters are represented in the candidate set, avoiding
        the problem of selecting only sources from a single broad cluster.

        Parameters
        ----------
        temporal_average : numpy.ndarray
            Average activation across time for each source (n_sources,)

        Returns
        -------
        candidate_indices : numpy.ndarray
            Indices of selected candidate sources
        """
        # Get source positions from the forward model
        src = self.forward["src"]

        # Extract vertex positions for both hemispheres
        positions = []
        vertex_to_index = []  # Maps from sequential index to position

        for _hemi_idx, hemi_src in enumerate(src):
            if hemi_src["type"] == "surf":
                # Surface source space
                rr = hemi_src["rr"][
                    hemi_src["vertno"]
                ]  # Get positions of active vertices
                positions.append(rr * 1000)  # Convert to mm
                vertex_to_index.extend(range(len(hemi_src["vertno"])))
            elif hemi_src["type"] == "vol":
                # Volume source space
                rr = hemi_src["rr"][hemi_src["vertno"]]
                positions.append(rr * 1000)  # Convert to mm
                vertex_to_index.extend(range(len(hemi_src["vertno"])))

        positions = np.vstack(positions)
        len(positions)

        # Step 1: Apply global threshold to get initial candidates
        global_threshold = np.percentile(temporal_average, self.selection_threshold)
        global_candidates = np.where(temporal_average >= global_threshold)[0]

        # Step 2: Find local maxima using spatial neighborhoods
        local_peaks = self._find_local_maxima(
            temporal_average, positions, self.peak_distance
        )

        if self.verbose > 1:
            logger.debug(
                f"   Found {len(global_candidates)} global candidates and "
                f"{len(local_peaks)} local peaks"
            )

        # Step 3: Combine global and local selections
        # Use weighted combination: take top sources from both methods
        n_from_global = int(self.max_sources * (1 - self.local_weight))
        n_from_local = int(self.max_sources * self.local_weight)

        # Select top sources from global candidates
        if len(global_candidates) > n_from_global:
            sorted_global = global_candidates[
                np.argsort(temporal_average[global_candidates])
            ]
            top_global = sorted_global[-n_from_global:]
        else:
            top_global = global_candidates

        # Select top sources from local peaks
        if len(local_peaks) > n_from_local:
            sorted_local = local_peaks[np.argsort(temporal_average[local_peaks])]
            top_local = sorted_local[-n_from_local:]
        else:
            top_local = local_peaks

        # Combine and remove duplicates
        candidate_indices = np.unique(np.concatenate([top_global, top_local]))

        # Apply min/max constraints
        if len(candidate_indices) < self.min_sources:
            # If too few sources, take the top min_sources globally
            candidate_indices = np.argsort(temporal_average)[-self.min_sources :]
            if self.verbose > 1:
                logger.debug(f"   Expanded to minimum {self.min_sources} sources")
        elif len(candidate_indices) > self.max_sources:
            # If too many sources, prioritize the strongest ones
            sorted_candidates = candidate_indices[
                np.argsort(temporal_average[candidate_indices])
            ]
            candidate_indices = sorted_candidates[-self.max_sources :]
            if self.verbose > 1:
                logger.debug(f"   Reduced to maximum {self.max_sources} sources")

        if self.verbose > 1:
            # Calculate coverage statistics
            n_global_unique = len(np.setdiff1d(top_global, top_local))
            n_local_unique = len(np.setdiff1d(top_local, top_global))
            n_overlap = len(np.intersect1d(top_global, top_local))
            logger.debug(
                f"   Selection breakdown: {n_global_unique} global-only, "
                f"{n_local_unique} local-only, {n_overlap} overlap"
            )

        return candidate_indices

    def _find_local_maxima(self, values, positions, distance_mm):
        """Find local maxima in source space using spatial distance criterion.

        Parameters
        ----------
        values : numpy.ndarray
            Activation values for each source
        positions : numpy.ndarray
            3D positions of sources (n_sources, 3) in mm
        distance_mm : float
            Minimum distance between peaks in mm

        Returns
        -------
        local_maxima : numpy.ndarray
            Indices of local maxima
        """
        n_sources = len(values)
        np.zeros(n_sources, dtype=bool)

        # Sort sources by value (highest first) to process strongest activations first
        sorted_indices = np.argsort(values)[::-1]

        local_maxima = []

        for idx in sorted_indices:
            if len(local_maxima) == 0:
                # First peak
                local_maxima.append(idx)
                continue

            # Check distance to all existing peaks
            distances = np.linalg.norm(positions[local_maxima] - positions[idx], axis=1)

            if np.all(distances >= distance_mm):
                # This is a new local maximum (far enough from existing peaks)
                local_maxima.append(idx)

            # Stop if we have enough peaks
            if len(local_maxima) >= self.max_sources:
                break

        return np.array(local_maxima)
