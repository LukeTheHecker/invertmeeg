import logging

import mne
import numpy as np
from scipy.spatial.distance import cdist

from ...util import pos_from_forward
from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverLAURA(BaseSolver):
    """Improved Local AUtoRegressive Average (LAURA) inverse solution.

    This version fixes numerical stability issues and adds optional depth bias
    correction and adaptive regularization.

    LAURA uses spatially weighted source priors based on the distance between
    sources and their connectivity in the cortical mesh.

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

    def __init__(self, name="LAURA", depth_weight=0.5, adaptive_alpha=True, **kwargs):
        self.name = name
        self.depth_weight = depth_weight
        self.adaptive_alpha = adaptive_alpha
        return super().__init__(**kwargs)

    def make_inverse_operator(
        self,
        forward,
        *args,
        noise_cov=None,
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
        noise_cov : numpy.ndarray, optional
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
        adjacency = mne.spatial_src_adjacency(forward["src"], verbose=verbose).toarray()
        n_chans, n_dipoles = self.leadfield.shape
        pos = pos_from_forward(forward, verbose=verbose)

        if noise_cov is None:
            noise_cov = np.identity(n_chans)

        # Compute spatial distance matrix
        d = cdist(pos, pos)

        # Apply adjacency constraint
        d_adj = d * adjacency

        # Compute spatial weighting matrix with inverse distance weighting
        A = np.zeros_like(d_adj)
        mask = (
            d_adj > 0
        )  # Only non-zero distances (excluding diagonal and non-adjacent)
        A[mask] = d_adj[mask] ** (-drop_off)

        # Normalize each row to sum to 1 (probabilistic interpretation)
        row_sums = A.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero for isolated sources
        A_normalized = A / row_sums

        # Add diagonal for self-interaction (identity component)
        # This ensures the matrix is well-conditioned
        A_full = A_normalized + np.identity(n_dipoles)

        # Source space metric: W_j encodes local spatial smoothness
        # Following LAURA theory: M_j.T @ M_j represents the spatial correlation
        M_j = A_full

        # Compute spatial prior covariance (source space metric)
        # This is positive definite by construction
        W_j = M_j.T @ M_j

        # Apply depth weighting to correct for depth bias
        # Key fix: Use NEGATIVE exponent to down-weight superficial sources
        if self.depth_weight > 0:
            # Compute source strengths (leadfield norms)
            source_norms = np.linalg.norm(self.leadfield, axis=0)
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
        noise_cov = (noise_cov + noise_cov.T) / 2
        noise_cov += 1e-12 * np.trace(noise_cov) / n_chans * np.identity(n_chans)

        # Compute adaptive alpha scaling if requested
        if self.adaptive_alpha:
            # Scale based on the ratio of leadfield and noise power
            leadfield_power = np.trace(self.leadfield @ self.leadfield.T) / n_chans
            noise_power = np.trace(noise_cov) / n_chans
            alpha_scale = noise_power / max(leadfield_power, 1e-12)
        else:
            alpha_scale = 1e-6  # Original hard-coded value

        if verbose > 0:
            logger.info(f"LAURA: Using alpha_scale = {alpha_scale:.2e}")

        # NOTE: In LAURA, `alpha` acts as a dimensionless knob `r` (it enters as
        # r^2 in the noise term). We therefore do *not* scale it by leadfield
        # eigenvalues (unlike classic MNE where α is used in L Lᵀ + αI).
        if alpha == "auto":
            r_grid = np.asarray(self.r_values, dtype=float)
        else:
            r_grid = np.asarray([float(alpha)], dtype=float)
        self.alphas = list(r_grid)

        inverse_operators = []
        for r in r_grid:
            # LAURA inverse operator formula:
            # T = W_j @ L.T @ (L @ W_j @ L.T + alpha^2 * Sigma_noise)^(-1)
            # where:
            #   W_j = source space spatial prior covariance (with depth weighting)
            #   L = leadfield
            #   Sigma_noise = noise covariance
            #   alpha = regularization parameter

            # Compute middle term efficiently
            LW = self.leadfield @ W_j  # (n_chans x n_dipoles)
            LWLT = LW @ self.leadfield.T  # (n_chans x n_chans)

            # Regularization term
            reg_term = (float(r) ** 2 * alpha_scale) * noise_cov

            # Inverse of regularized data covariance
            try:
                C_inv = np.linalg.inv(LWLT + reg_term)
            except np.linalg.LinAlgError:
                if verbose > 0:
                    logger.warning(
                        f"Singular matrix for alpha={r}, using pseudo-inverse"
                    )
                C_inv = np.linalg.pinv(LWLT + reg_term)

            # Final inverse operator
            inverse_operator = W_j @ self.leadfield.T @ C_inv
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        return self
