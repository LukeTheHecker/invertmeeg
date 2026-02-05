import logging

import mne
import numpy as np
from scipy.spatial.distance import cdist

from ...util import pos_from_forward
from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverAPSE(BaseSolver):
    """
    Adaptive Patch Source Estimation (APSE) - A hybrid inverse solution technique
    that combines the strengths of beamformers, subspace methods, and sparse
    Bayesian approaches for patch-sized source reconstruction.

    This solver:
    1. Uses subspace decomposition to estimate the number of active sources
    2. Applies LCMV-like beamforming with patch constraints
    3. Employs sparse Bayesian learning for source amplitude estimation
    4. Automatically determines patch sizes based on source clustering

    References
    ----------
    [1] Van Veen, B. D., & Buckley, K. M. (1988). Beamforming: A versatile
        approach to spatial filtering. IEEE assp magazine, 5(2), 4-24.
    [2] Wipf, D., & Nagarajan, S. (2009). A unified Bayesian framework for MEG/EEG
        source imaging. NeuroImage, 44(3), 947-966.
    [3] Mosher, J. C., & Leahy, R. M. (1999). Source localization using recursively
        applied and projected (RAP) MUSIC. IEEE TSP, 47(2), 332-340.
    """

    meta = SolverMeta(
        acronym="APSE",
        full_name="Adaptive Patch Source Estimation",
        category="Hybrid",
        description=(
            "Hybrid patch-based inverse solver combining subspace source-number estimation, "
            "beamforming-style spatial filtering, and sparse Bayesian updates to estimate "
            "patch-sized cortical activity."
        ),
        references=[
            "Lukas Hecker 2025, unpublished",
        ],
    )

    def __init__(
        self,
        name="Adaptive Patch Source Estimation",
        n_patches="auto",
        patch_size_percentile=30,
        sparsity_param=0.3,
        max_patch_size=50,
        min_patch_size=10,
        reduce_rank=True,
        rank="auto",
        **kwargs,
    ):
        self.name = name
        self.n_patches = n_patches
        self.patch_size_percentile = patch_size_percentile
        self.sparsity_param = sparsity_param
        self.max_patch_size = max_patch_size
        self.min_patch_size = min_patch_size
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(
        self,
        forward,
        mne_obj,
        *args,
        alpha="auto",
        max_iter=100,
        convergence_tol=1e-4,
        verbose=0,
        **kwargs,
    ):
        """
        Calculate the APSE inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object
        alpha : float or 'auto'
            The regularization parameter
        max_iter : int
            Maximum number of iterations for the Bayesian updates
        convergence_tol : float
            Convergence tolerance for the iterative algorithm
        verbose : int
            Verbosity level

        Return
        ------
        self : object
            Returns itself for convenience
        """
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)

        # Get data and leadfield
        data = self.unpack_data_obj(mne_obj)
        leadfield = self.leadfield
        n_chans, n_dipoles = leadfield.shape

        # Regularization scale should match the sensor-space covariance matrices
        # that are diagonal-loaded inside this solver (Sigma_y + alpha * I).
        data_cov = self.data_covariance(data, center=True, ddof=1)
        self.get_alphas(reference=data_cov)

        # Normalize leadfield columns
        leadfield_norm = leadfield / np.linalg.norm(leadfield, axis=0)

        # Step 1: Estimate number of sources using subspace decomposition
        n_sources = self._estimate_n_sources(data, leadfield_norm)
        if verbose > 0:
            logger.info(f"Estimated number of active sources: {n_sources}")

        # Step 2: Get source positions
        pos = pos_from_forward(self.forward)

        # Step 3: Create patches based on spatial proximity
        patches = self._create_adaptive_patches(
            data, leadfield_norm, pos, n_sources, verbose
        )
        self.patches = patches

        # Step 4: Create inverse operators for different regularization values
        inverse_operators = []

        for alpha in self.alphas:
            # Compute patch-constrained inverse operator
            inverse_op = self._compute_patch_inverse(
                data, leadfield, patches, alpha, max_iter, convergence_tol, verbose
            )
            inverse_operators.append(inverse_op)

        self.inverse_operators = [
            InverseOperator(op, self.name) for op in inverse_operators
        ]
        return self

    def _estimate_n_sources(self, data, leadfield):
        """
        Estimate the number of active sources using eigenvalue decomposition
        of the data covariance matrix.

        This implementation is specialized for patch-based source estimation
        and uses more conservative thresholds than the standard method.
        """
        if self.n_patches != "auto":
            return self.n_patches

        # Compute data covariance
        C = data @ data.T

        # SVD decomposition
        _, S, _ = np.linalg.svd(C, full_matrices=False)

        # Normalize eigenvalues
        S_norm = S / S.max()

        # Find elbow using multiple criteria
        # Criterion 1: Eigenvalue drop-off - more sensitive for patch sources
        diff_S = np.abs(np.diff(S_norm))
        drop_idx = np.where(diff_S < 0.02)[0]  # Slightly higher threshold for patches
        n_drop = drop_idx[0] + 1 if len(drop_idx) > 0 else len(S)

        # Criterion 2: L-curve method using base class method
        np.arange(min(len(S), 15))  # Limit to first 15 components for patches
        n_lcurve = self.get_comps_L(S[:15])

        # Criterion 3: Cumulative energy - lower threshold for patches
        cumsum_S = np.cumsum(S) / np.sum(S)
        n_energy = np.where(cumsum_S > 0.90)[0][0] + 1  # 90% instead of 95%

        # Combine criteria with conservative bias toward fewer sources for patches
        estimates = [n_drop, n_lcurve, n_energy]
        n_sources = int(
            np.percentile(estimates, 33)
        )  # Use 33rd percentile instead of median

        # Ensure reasonable bounds
        n_sources = np.clip(n_sources, 1, min(20, len(S) // 2))

        return n_sources

    def _create_adaptive_patches(self, data, leadfield, pos, n_sources, verbose):
        """
        Create adaptive patches based on beamformer peaks and spatial clustering.
        """
        n_chans, n_dipoles = leadfield.shape

        # Step 1: Compute initial source activity using LCMV-like beamformer
        C = data @ data.T
        reg = np.trace(C) / n_chans * 0.01  # Small regularization
        C_reg = C + reg * np.eye(n_chans)
        C_inv = np.linalg.inv(C_reg)

        # Beamformer spatial filter for each dipole
        source_power = np.zeros(n_dipoles)
        for i in range(n_dipoles):
            l = leadfield[:, i : i + 1]
            w = C_inv @ l / (l.T @ C_inv @ l)
            source_power[i] = np.real(w.T @ C @ w).item()

        # Step 2: Find peaks using adaptive thresholding
        # Normalize source power
        source_power_norm = source_power / source_power.max()

        # Dynamic threshold based on noise floor estimation - more aggressive for patches
        sorted_power = np.sort(source_power_norm)
        noise_floor = np.median(
            sorted_power[: len(sorted_power) // 3]
        )  # Use lower third instead of half
        threshold = noise_floor + 0.2 * (
            1 - noise_floor
        )  # Lower threshold for patch detection

        # Find peaks
        peak_indices = self._find_adaptive_peaks(
            source_power_norm, threshold, pos, n_sources
        )

        if verbose > 0:
            logger.info(f"Found {len(peak_indices)} peak locations")

        # Step 3: Create patches around peaks
        patches = self._grow_patches(peak_indices, pos, source_power_norm)

        return patches

    def _find_adaptive_peaks(self, source_power, threshold, pos, n_sources):
        """
        Find peak locations with adaptive spatial separation.
        """
        # Start with high threshold and decrease if needed
        peaks = []
        current_threshold = threshold

        while len(peaks) < n_sources and current_threshold > 0.1:
            candidates = np.where(source_power > current_threshold)[0]

            if len(candidates) == 0:
                current_threshold *= 0.8
                continue

            # Sort by power
            candidates = candidates[np.argsort(source_power[candidates])[::-1]]

            # Select peaks with minimum spatial separation
            min_distance = self._estimate_min_distance(pos)
            peaks = []

            for cand in candidates:
                # Check distance to existing peaks
                if len(peaks) == 0:
                    peaks.append(cand)
                else:
                    distances = cdist(pos[cand : cand + 1], pos[peaks])
                    if np.min(distances) > min_distance:
                        peaks.append(cand)

                if len(peaks) >= n_sources:
                    break

            current_threshold *= 0.8

        return np.array(peaks[:n_sources])

    def _estimate_min_distance(self, pos):
        """
        Estimate minimum distance between sources based on spatial distribution.
        """
        # Sample random pairs to estimate typical distances
        n_samples = min(1000, len(pos))
        indices = np.random.choice(len(pos), n_samples, replace=False)
        sample_pos = pos[indices]

        # Compute pairwise distances
        distances = cdist(sample_pos, sample_pos)
        distances[distances == 0] = np.inf  # Ignore self-distances

        # Use percentile of nearest neighbor distances
        min_distances = np.min(distances, axis=1)
        min_distance = np.percentile(min_distances, self.patch_size_percentile)

        return (
            min_distance * 1.5
        )  # Reduced scaling for patch sources (allow closer patches)

    def _grow_patches(self, peak_indices, pos, source_power):
        """
        Grow patches around peak locations based on spatial proximity and power.
        """
        patches = []
        used_vertices = set()

        # Estimate adaptive patch size with spatial correlation awareness
        patch_dists = []
        for peak in peak_indices:
            distances = cdist(pos[peak : peak + 1], pos).flatten()
            # Use larger percentile and add correlation-based scaling
            base_dist = np.percentile(
                distances[distances > 0], self.patch_size_percentile
            )
            # Scale based on local source power gradient for better patch extent
            local_indices = np.where(distances <= base_dist * 2)[0]
            if len(local_indices) > 1:
                local_power_std = np.std(source_power[local_indices])
                # Higher variability suggests larger patch needed
                scale_factor = 1.0 + 0.5 * local_power_std
                patch_dists.append(base_dist * scale_factor)
            else:
                patch_dists.append(base_dist)

        # Create patches
        for i, peak in enumerate(peak_indices):
            if peak in used_vertices:
                continue

            # Find vertices within adaptive distance
            distances = cdist(pos[peak : peak + 1], pos).flatten()
            patch_vertices = np.where(distances <= patch_dists[i])[0]

            # Apply power-based refinement with spatial weighting for patch sources
            patch_power = source_power[patch_vertices]
            patch_distances = distances[patch_vertices]

            # Use distance-weighted threshold that's more lenient for nearby vertices
            distance_weights = np.exp(-patch_distances / (patch_dists[i] * 0.5))
            adaptive_threshold = 0.05 * source_power[peak] * (1 + distance_weights)
            patch_vertices = patch_vertices[patch_power > adaptive_threshold]

            # Ensure patch size constraints
            if len(patch_vertices) > self.max_patch_size:
                # Keep closest vertices
                patch_distances = distances[patch_vertices]
                sorted_idx = np.argsort(patch_distances)
                patch_vertices = patch_vertices[sorted_idx[: self.max_patch_size]]
            elif len(patch_vertices) < self.min_patch_size:
                # Expand to include nearest neighbors
                sorted_idx = np.argsort(distances)
                patch_vertices = sorted_idx[: self.min_patch_size]

            # Mark vertices as used
            used_vertices.update(patch_vertices)
            patches.append(patch_vertices)

        return patches

    def _compute_patch_inverse(
        self, data, leadfield, patches, alpha, max_iter, convergence_tol, verbose
    ):
        """
        Compute the patch-constrained inverse operator using iterative
        sparse Bayesian learning.
        """
        n_chans, n_dipoles = leadfield.shape
        n_time = data.shape[1]

        # Initialize patch leadfields
        patch_leadfields = []
        patch_indices = []

        for patch in patches:
            if len(patch) > 0:
                patch_leadfield = leadfield[:, patch]
                patch_leadfields.append(patch_leadfield)
                patch_indices.extend(patch.tolist())

        # Construct combined patch leadfield
        L_patch = (
            np.hstack(patch_leadfields) if patch_leadfields else np.zeros((n_chans, 1))
        )
        n_patch_sources = L_patch.shape[1]

        # Initialize hyperparameters with better noise estimation
        gamma = np.ones(n_patch_sources)
        # Improved noise variance estimation based on residual after simple beamforming
        C = data @ data.T / n_time
        signal_subspace = min(10, n_chans // 2)  # Estimate signal subspace size
        eigenvals = np.linalg.eigvals(C)
        eigenvals = np.sort(eigenvals)[::-1]  # Sort descending
        # Noise variance from smallest eigenvalues
        noise_var = (
            np.mean(eigenvals[signal_subspace:])
            if len(eigenvals) > signal_subspace
            else eigenvals[-1] * 0.1
        )

        # Iterative sparse Bayesian learning
        for iteration in range(max_iter):
            gamma_old = gamma.copy()

            # E-step: Compute posterior mean and covariance
            Gamma = np.diag(gamma)
            Sigma_y = L_patch @ Gamma @ L_patch.T + noise_var * np.eye(n_chans)

            try:
                Sigma_y_inv = np.linalg.inv(Sigma_y + alpha * np.eye(n_chans))
            except np.linalg.LinAlgError:
                # Add more regularization if singular
                Sigma_y_inv = np.linalg.inv(Sigma_y + (alpha + 0.1) * np.eye(n_chans))

            # Posterior covariance
            Sigma_post = Gamma - Gamma @ L_patch.T @ Sigma_y_inv @ L_patch @ Gamma

            # Posterior mean
            mu_post = Gamma @ L_patch.T @ Sigma_y_inv @ data

            # M-step: Update hyperparameters with patch-aware updates
            for j in range(n_patch_sources):
                data_term = np.real(
                    np.linalg.norm(mu_post[j, :]) ** 2 / n_time + Sigma_post[j, j]
                )
                gamma[j] = (
                    1 - self.sparsity_param
                ) * data_term + self.sparsity_param * gamma[j]

            # Update noise variance with regularization to prevent over-fitting
            residual = data - L_patch @ mu_post
            residual_var = np.trace(residual @ residual.T) / (n_chans * n_time)
            # Blend with previous estimate for stability
            noise_var = 0.8 * noise_var + 0.2 * residual_var
            # Ensure minimum noise level
            noise_var = max(noise_var, 1e-6)

            # Check convergence
            if (
                np.linalg.norm(gamma - gamma_old) / np.linalg.norm(gamma_old)
                < convergence_tol
            ):
                if verbose > 0:
                    logger.info(f"Converged after {iteration + 1} iterations")
                break

        # Compute final inverse operator
        Gamma = np.diag(gamma)
        Sigma_y = L_patch @ Gamma @ L_patch.T + noise_var * np.eye(n_chans)
        Sigma_y_inv = np.linalg.inv(Sigma_y + alpha * np.eye(n_chans))

        # Create sparse inverse operator
        W_patch = np.real(Gamma @ L_patch.T @ Sigma_y_inv)

        # Map back to full source space with continuous spatial interpolation
        W = np.zeros((n_dipoles, n_chans))
        if len(patch_indices) > 0:
            W[patch_indices, :] = W_patch

        # Apply continuous spatial smoothing instead of discrete patch smoothing
        W_smooth = self._apply_continuous_spatial_smoothing(
            W, patches, pos_from_forward(self.forward)
        )

        return W_smooth

    def _smooth_within_patches(self, W, patches):
        """
        Apply smoothing within each patch to ensure coherent patch activations.
        """
        W_smooth = W.copy()

        for patch in patches:
            if len(patch) > 1:
                # Enhanced spatial smoothing for patch coherence
                patch_weights = W[patch, :]

                # Use power-weighted averaging for better patch coherence
                patch_norms = np.linalg.norm(patch_weights, axis=1)
                if np.sum(patch_norms) > 0:
                    weights = patch_norms / np.sum(patch_norms)
                    weighted_mean = np.average(patch_weights, axis=0, weights=weights)
                else:
                    weighted_mean = np.mean(patch_weights, axis=0)

                # More aggressive blending for patch sources
                blend_factor = 0.5  # 50% smoothing for better patch coherence
                for _i, vertex in enumerate(patch):
                    W_smooth[vertex, :] = (1 - blend_factor) * W[
                        vertex, :
                    ] + blend_factor * weighted_mean

        return W_smooth

    def _apply_continuous_spatial_smoothing(self, W, patches, pos):
        """
        Apply continuous spatial smoothing that extends beyond discrete patch boundaries.
        This creates smooth spatial gradients similar to Champagne's natural smoothness.
        """
        from scipy.spatial.distance import cdist

        W_smooth = W.copy()
        n_dipoles = W.shape[0]

        # For each patch, create smooth spatial gradients extending outward
        for patch in patches:
            if len(patch) == 0:
                continue

            # Get patch center and activity
            patch_pos = pos[patch]
            patch_weights = W[patch, :]

            # Find patch centroid
            patch_activity = np.linalg.norm(patch_weights, axis=1)
            if np.sum(patch_activity) > 0:
                centroid_weights = patch_activity / np.sum(patch_activity)
                centroid_pos = np.average(patch_pos, axis=0, weights=centroid_weights)
            else:
                centroid_pos = np.mean(patch_pos, axis=0)

            # Compute distances from centroid to all source locations
            distances = cdist(centroid_pos.reshape(1, -1), pos).flatten()

            # Create smooth spatial kernel extending beyond patch
            max_patch_dist = np.max(distances[patch]) if len(patch) > 0 else 1.0
            smooth_radius = max_patch_dist * 2.0  # Extend beyond patch boundaries

            # Gaussian-like smoothing kernel
            spatial_weights = np.exp(-((distances / smooth_radius) ** 2))

            # Get representative patch activity (weighted average)
            if np.sum(patch_activity) > 0:
                representative_activity = np.average(
                    patch_weights, axis=0, weights=patch_activity
                )
            else:
                representative_activity = np.mean(patch_weights, axis=0)

            # Apply smooth spatial distribution
            for i in range(n_dipoles):
                # Blend existing activity with smooth patch influence
                if i not in patch:  # Only modify locations outside the patch
                    blend_factor = spatial_weights[i] * 0.3  # 30% influence max
                    W_smooth[i, :] = (1 - blend_factor) * W_smooth[
                        i, :
                    ] + blend_factor * representative_activity

        return W_smooth

    def apply_inverse_operator(self, mne_obj) -> mne.SourceEstimate:
        """
        Apply the inverse operator with patch-based constraints.

        Parameters
        ----------
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object

        Return
        ------
        stc : mne.SourceEstimate
            The source estimate with patch structure
        """
        # Use parent class method with potential patch post-processing
        stc = super().apply_inverse_operator(mne_obj)

        # Optional: Apply continuous spatial smoothing to the solution
        if hasattr(self, "patches") and self.patches:
            stc = self._apply_continuous_solution_smoothing(stc)

        return stc

    def _apply_patch_smoothing(self, stc):
        """
        Apply post-hoc smoothing within detected patches.
        """
        data = stc.data.copy()

        for patch in self.patches:
            if len(patch) > 1:
                # Apply mild smoothing within patch
                patch_data = data[patch, :]

                # Spatial smoothing using Gaussian weights based on activation strength
                patch_power = np.linalg.norm(patch_data, axis=1)
                if patch_power.sum() > 0:
                    weights = patch_power / patch_power.sum()
                    weighted_mean = np.average(patch_data, axis=0, weights=weights)

                    # More aggressive blending for patch sources
                    blend = 0.3  # 30% smoothing for better patch coherence
                    for _i, vertex in enumerate(patch):
                        data[vertex, :] = (1 - blend) * data[
                            vertex, :
                        ] + blend * weighted_mean

        # Create new source estimate with smoothed data
        stc_smooth = mne.SourceEstimate(
            data=data,
            vertices=stc.vertices,
            tmin=stc.tmin,
            tstep=stc.tstep,
            subject=stc.subject,
        )

        return stc_smooth

    def _apply_continuous_solution_smoothing(self, stc):
        """
        Apply continuous spatial smoothing to the final solution, creating smooth
        gradients similar to Champagne's natural spatial distributions.
        """
        from scipy.spatial.distance import cdist

        data = stc.data.copy()
        pos = pos_from_forward(self.forward)

        # Create smooth spatial distributions around each patch
        for patch in self.patches:
            if len(patch) <= 1:
                continue

            # Get patch characteristics
            patch_data = data[patch, :]
            patch_pos = pos[patch]

            # Find patch centroid weighted by activity
            patch_power = np.linalg.norm(patch_data, axis=1)
            if patch_power.sum() > 0:
                weights = patch_power / patch_power.sum()
                centroid_pos = np.average(patch_pos, axis=0, weights=weights)
                representative_signal = np.average(patch_data, axis=0, weights=weights)
            else:
                centroid_pos = np.mean(patch_pos, axis=0)
                representative_signal = np.mean(patch_data, axis=0)

            # Compute distances from centroid to all sources
            distances = cdist(centroid_pos.reshape(1, -1), pos).flatten()

            # Create extended smooth kernel (larger than discrete patch)
            max_patch_dist = np.max(distances[patch])
            smooth_radius = max_patch_dist * 1.5  # Extend 50% beyond patch

            # Exponential decay kernel for smooth spatial gradients
            spatial_kernel = np.exp(-((distances / smooth_radius) ** 2))

            # Apply smooth spatial distribution to nearby sources
            for i in range(len(pos)):
                if (
                    i not in patch and spatial_kernel[i] > 0.1
                ):  # Threshold for efficiency
                    blend_factor = spatial_kernel[i] * 0.4  # Up to 40% influence
                    data[i, :] = (1 - blend_factor) * data[
                        i, :
                    ] + blend_factor * representative_signal

        # Create new source estimate with continuous spatial structure
        stc_continuous = mne.SourceEstimate(
            data=data,
            vertices=stc.vertices,
            tmin=stc.tmin,
            tstep=stc.tstep,
            subject=stc.subject,
        )

        return stc_continuous
