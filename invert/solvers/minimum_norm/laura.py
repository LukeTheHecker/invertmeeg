import logging

import mne
import numpy as np
from scipy.spatial.distance import cdist

from ...util import build_source_adjacency, pos_from_forward
from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverLAURA(BaseSolver):
    """Local AUtoRegressive Average (LAURA) inverse solution.

    LAURA uses spatially weighted source priors based on electromagnetic field
    decay laws (1/r^2 for potentials, 1/r^3 for currents) to enforce
    biophysically plausible spatial smoothness.

    Optional extensions (disabled by default for pure LAURA):
    - depth_weight: Depth bias correction (Lin et al. 2006)
    - use_mesh_adjacency: Restrict neighbors to mesh-connected sources

    References
    ----------
    [1] Grave de Peralta Menendez, R., et al. (2004). Electrical neuroimaging
        based on biophysical constraints. NeuroImage, 21(2), 527-539.
    [2] Lin, F. H., et al. (2006). Assessing and improving the spatial accuracy
        in MEG source localization by depth-weighted minimum-norm estimates.
    """

    meta = SolverMeta(
        acronym="LAURA",
        full_name="Local Auto-Regressive Average",
        category="Minimum Norm",
        description=(
            "Spatially weighted minimum-norm inverse using local neighborhood "
            "constraints (LAURA) to encourage physiologically plausible smoothness."
        ),
        references=[
            "Grave de Peralta Menendez, R., Murray, M. M., Michel, C. M., Martuzzi, R., & Gonzalez Andino, S. L. (2004). Electrical neuroimaging based on biophysical constraints. NeuroImage, 21(2), 527–539.",
        ],
    )

    def __init__(
        self,
        name="LAURA",
        depth_weight=0.5,
        use_mesh_adjacency=True,
        use_noise_whitener: bool = True,
        use_trace_normalization: bool = True,
        rank_tol: float = 1e-12,
        eps: float = 1e-15,
        **kwargs,
    ):
        self.name = name
        self.depth_weight = depth_weight
        self.use_mesh_adjacency = use_mesh_adjacency
        self.use_noise_whitener = bool(use_noise_whitener)
        self.use_trace_normalization = bool(use_trace_normalization)
        self.rank_tol = float(rank_tol)
        self.eps = float(eps)
        # LAURA handles depth weighting internally via W_j, so disable base class
        # depth weighting to avoid double compensation
        kwargs.setdefault("prep_leadfield", False)
        super().__init__(**kwargs)

    def make_inverse_operator(
        self,
        forward,
        *args,
        noise_cov: mne.Covariance | None = None,
        alpha="auto",
        drop_off=2,
        verbose=0,
        **kwargs,
    ):
        """Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        alpha : float or 'auto'
            The regularization parameter.
        noise_cov : mne.Covariance | None, optional
            The noise covariance matrix. If None, identity is used.
        drop_off : float, optional
            Controls the steepness of the spatial weighting distribution.
            Default is 2.
        verbose : int, optional
            Verbosity level.

        Return
        ------
        self : object returns itself for convenience
        """
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        _n_chans, n_dipoles = self.leadfield.shape
        pos = pos_from_forward(forward, verbose=verbose)

        if noise_cov is None:
            noise_cov = self.make_identity_noise_cov(list(self.forward.ch_names))
        noise_cov_mat, noise_cov_ch_names = self.coerce_noise_cov(noise_cov)
        forward_ch_names = list(self.forward.ch_names)
        if noise_cov_ch_names != forward_ch_names:
            noise_cov_mat = self.reorder_covariance_to_channels(
                noise_cov_mat, noise_cov_ch_names, forward_ch_names
            )
        noise_cov_mat = 0.5 * (noise_cov_mat + noise_cov_mat.T)

        if self.use_noise_whitener:
            wf = self.prepare_whitened_forward(
                noise_cov,
                trace_normalize=self.use_trace_normalization,
                rank_tol=self.rank_tol,
                eps=self.eps,
            )
            noise_cov_eff = np.eye(wf.n_eff, dtype=float)
        else:
            wf = self.prepare_whitened_forward(
                None,
                trace_normalize=self.use_trace_normalization,
                rank_tol=self.rank_tol,
                eps=self.eps,
            )
            noise_cov_eff = wf.projector @ noise_cov_mat @ wf.projector.T
            noise_cov_eff = 0.5 * (noise_cov_eff + noise_cov_eff.T)
        if wf.whitener_mode not in ("projected", "none"):
            logger.warning("LAURA whitener fallback used: %s", wf.whitener_mode)

        leadfield = wf.A if wf.A is not None else wf.G_white
        leadfield_scale = wf.trace_scale
        sensor_transform = wf.sensor_transform
        n_eff = wf.n_eff

        # Compute spatial distance matrix
        d = cdist(pos, pos)

        # Determine neighborhood structure
        if self.use_mesh_adjacency:
            # Use mesh connectivity (extension)
            adjacency = build_source_adjacency(forward["src"], verbose=verbose).toarray()
            d_adj = d * adjacency
        else:
            # Pure LAURA: use all sources (full distance matrix, exclude diagonal)
            d_adj = d.copy()
            np.fill_diagonal(d_adj, 0)

        # Compute spatial weighting matrix A following LAURA paper:
        # Off-diagonal: A_ik = -d_ki^{-e} (negative inverse distance for neighbors)
        # Diagonal: A_ii = (N/N_i) * sum_{k in neighbors} d_ki^{-e}
        # This creates a Laplacian-like structure enforcing local autoregressive averaging

        A = np.zeros((n_dipoles, n_dipoles))

        # Compute inverse distance weights for adjacent sources
        mask = d_adj > 0  # Only non-zero distances (neighbors)
        inv_dist_weights = np.zeros_like(d_adj)
        inv_dist_weights[mask] = d_adj[mask] ** (-drop_off)

        # Off-diagonal elements: NEGATIVE inverse distance weights
        A[mask] = -inv_dist_weights[mask]

        # Diagonal elements: (N/N_i) * sum of neighbor weights
        n_neighbors = (d_adj > 0).sum(axis=1)
        n_neighbors = np.maximum(n_neighbors, 1)  # avoid division by zero
        neighbor_weight_sums = inv_dist_weights.sum(axis=1)
        A[np.diag_indices(n_dipoles)] = (n_dipoles / n_neighbors) * neighbor_weight_sums

        # Source space metric: W_j = (M^T M)^{-1} where M = A
        # (W matrix is identity in standard LAURA formulation)
        M_j = A

        # Compute spatial prior covariance (source space metric). Use explicit
        # regularization and pseudo-inverse fallback to handle rank-deficient
        # neighborhood operators.
        metric = M_j.T @ M_j
        metric = 0.5 * (metric + metric.T)
        metric_scale = max(float(np.trace(metric)) / max(n_dipoles, 1), self.eps)
        metric += 1e-8 * metric_scale * np.identity(n_dipoles)
        try:
            W_j = np.linalg.inv(metric)
        except np.linalg.LinAlgError:
            W_j = np.linalg.pinv(metric)

        # Apply depth weighting to correct for depth bias
        # Key fix: Use NEGATIVE exponent to down-weight superficial sources
        if self.depth_weight > 0:
            # Compute source strengths (leadfield norms)
            source_norms = np.linalg.norm(leadfield, axis=0)
            # Avoid division by zero
            source_norms = np.maximum(source_norms, 1e-12)

            # Normalize to mean of 1 for numerical stability
            source_norms_normalized = source_norms / np.mean(source_norms)

            # Depth weighting: NEGATIVE exponent to down-weight superficial (strong) sources
            # depth_weight=0: no correction (weights all = 1)
            # depth_weight=1: full inverse weighting (deep sources up-weighted by leadfield ratio)
            depth_weights = source_norms_normalized ** (-self.depth_weight)

            if verbose > 0:
                logger.info(
                    f"LAURA: Depth weights range [{depth_weights.min():.3f}, {depth_weights.max():.3f}]"
                )
                logger.info(f"       Mean depth weight: {depth_weights.mean():.3f}")

            Depth = np.diag(depth_weights)
            # Incorporate depth weighting into spatial prior
            W_j = Depth @ W_j @ Depth

        # Ensure numerical stability
        # Add small regularization to ensure positive definiteness
        W_j += 1e-8 * np.trace(W_j) / n_dipoles * np.identity(n_dipoles)

        # Noise covariance in sensor space
        # Ensure it's symmetric and positive definite
        if not self.use_noise_whitener:
            noise_cov_eff += (
                1e-12 * np.trace(noise_cov_eff) / max(n_eff, 1) * np.identity(n_eff)
            )

        # LAURA inverse operator formula:
        # T = W_j @ L.T @ (L @ W_j @ L.T + alpha * Sigma_noise)^(-1)
        LW = leadfield @ W_j
        LWLT = LW @ leadfield.T

        # Scale alphas relative to LWLT
        self.get_alphas(reference=LWLT)

        inverse_operators = []
        I = np.identity(n_eff)
        for alpha in self.alphas:
            C = LWLT + float(alpha) * (I if self.use_noise_whitener else noise_cov_eff)
            # Inverse of regularized data covariance
            try:
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                if verbose > 0:
                    logger.warning(
                        f"Singular matrix for alpha={alpha}, using pseudo-inverse"
                    )
                C_inv = np.linalg.pinv(C)

            # Final inverse operator
            inverse_operator = (
                float(leadfield_scale) * (W_j @ leadfield.T @ C_inv)
            ) @ sensor_transform
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        return self
