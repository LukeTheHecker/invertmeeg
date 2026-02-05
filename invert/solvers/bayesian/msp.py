import logging

import mne
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import splu

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverMSP(BaseSolver):
    """Class for the Multiple Sparse Priors (MSP) inverse solution with
    Restricted Maximum Likelihood (ReML) [1].

    This method uses ReML to estimate the hyperparameters of multiple sparse
    priors defined over different spatial patterns in source space.

    References
    ----------
    [1] Friston, K., Harrison, L., Daunizeau, J., Kiebel, S., Phillips, C.,
    Trujillo-Barreto, N., ... & Mattout, J. (2008). Multiple sparse priors for
    the M/EEG inverse problem. NeuroImage, 39(3), 1104-1120.

    """

    meta = SolverMeta(
        acronym="MSP",
        full_name="Multiple Sparse Priors",
        category="Bayesian",
        description=(
            "Bayesian source imaging method that combines multiple spatial priors "
            "and estimates their hyperparameters with Restricted Maximum Likelihood "
            "(ReML)."
        ),
        references=[
            "Friston, K., Harrison, L., Daunizeau, J., Kiebel, S., Phillips, C., Trujillo-Barreto, N., & Mattout, J. (2008). Multiple sparse priors for the M/EEG inverse problem. NeuroImage, 39(3), 1104–1120.",
        ],
    )

    def __init__(
        self,
        patterns=None,
        name="Multiple Sparse Priors",
        include_sensor_noise=True,
        eta=-32.0,
        Pi=1.0 / 256.0,
        max_iter=512,
        tol=1e-4,
        n_patterns=None,
        diffusion_parameter=0.1,
        patch_order=2,
        reduce_rank=True,
        rank="auto",
        **kwargs,
    ):
        """
        Parameters
        ----------
        patterns : list, array, or None
            List of spatial patterns in source space. Each pattern should be
            a 1D array of shape (n_sources,). If None, patterns will be
            automatically generated using graph Laplacian smoothing.
        include_sensor_noise : bool
            Whether to include a sensor noise component in the model.
        eta : float
            Hyperprior mean (per component) in log-space.
        Pi : float
            Hyperprior precision (scalar, becomes Pi * I).
        max_iter : int
            Maximum number of ReML iterations.
        tol : float
            Convergence tolerance for relative change in hyperparameters.
        n_patterns : int or None
            Number of patterns to generate if patterns is None. If None,
            a reasonable default based on the source space will be used.
        diffusion_parameter : float
            Diffusion parameter (alpha) for graph smoothing when generating
            patterns automatically. Default is 0.1.
        patch_order : int
            Order of smoothing for generated patches (1 = small patches,
            2 = medium patches, etc.). Default is 1.
        """
        self.name = name
        self.patterns = patterns
        self.include_sensor_noise = include_sensor_noise
        self.eta = eta
        self.Pi = Pi
        self.max_iter = max_iter
        self.tol = tol
        self.n_patterns = n_patterns
        self.diffusion_parameter = diffusion_parameter
        self.patch_order = patch_order
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, alpha="auto", **kwargs):
        """Calculate inverse operator using Multiple Sparse Priors with ReML.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        alpha : float or 'auto'
            Regularization parameter (note: MSP uses its own hyperparameter
            optimization via ReML, so this is mainly for compatibility).

        Return
        ------
        self : object returns itself for convenience
        """
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)
        leadfield = self.leadfield

        # Generate patterns automatically if not provided
        if self.patterns is None:
            if self.verbose:
                logger.info(
                    "Generating spatial patterns using graph Laplacian smoothing..."
                )
            self.patterns = self._generate_patterns(forward)

        # Run MSP-ReML algorithm
        m, v = data.shape
        d = leadfield.shape[1]
        C = (data @ data.T) / float(v)  # sample covariance over time/epochs

        # Build rank-1 vectors u_i = L @ q_i and trace normalization factors
        # Instead of storing full m×m Q_i matrices, store m-vectors u_i
        # so that Q_i = u_i @ u_i^T / trace_i
        U_cols = []
        trace_factors = []

        for q in self.patterns:
            q = q.flatten()
            u = leadfield @ q  # m-vector
            trace_val = np.dot(u, u)  # trace of u @ u^T
            if trace_val > 0:
                U_cols.append(u / np.sqrt(trace_val))  # normalize so u@u^T has trace 1
            else:
                U_cols.append(u)
            trace_factors.append(trace_val)

        # U is m × n_patterns, each column is a normalized u_i
        U = np.column_stack(U_cols) if U_cols else np.empty((m, 0))
        n_source_components = U.shape[1]

        # Total number of components (with optional noise)
        has_noise = self.include_sensor_noise
        p = n_source_components + (1 if has_noise else 0)

        eta_vec = np.full(p, self.eta, dtype=float)
        P_hyper = self.Pi * np.eye(p)

        # Initialize lambdas at the hyperprior mean
        lambdas = np.full(p, self.eta, dtype=float)

        # Add small random perturbations to break symmetry
        rng = np.random.RandomState(42)
        lambdas += rng.randn(p) * 0.1

        if self.verbose:
            logger.info(
                f"Initialized {p} hyperparameters with lambda = {self.eta:.2f} (±0.1)"
            )
            logger.info(
                f"Running ReML optimization (max_iter={self.max_iter}, tol={self.tol})..."
            )

        # Add bounds to prevent numerical overflow
        lambda_min = -32.0
        lambda_max = 32.0

        # Pruning threshold for inactive components
        prune_threshold = 1e-12

        # ReML iterations (Fisher scoring)
        prev = None
        initial_grad_norm = None

        for it in range(1, self.max_iter + 1):
            # Clip lambdas to prevent overflow
            lambdas = np.clip(lambdas, lambda_min, lambda_max)
            scales = np.exp(lambdas)

            # Identify active components (pruning)
            active_mask = scales > prune_threshold

            # Build R = noise_scale * I/m + U_active @ diag(scales_active) @ U_active^T
            if has_noise:
                noise_scale = scales[0]
                source_scales = scales[1:]
                source_active = active_mask[1:]
            else:
                noise_scale = 0.0
                source_scales = scales
                source_active = active_mask

            active_idx = np.where(source_active)[0]
            U_active = U[:, active_idx]
            s_active = source_scales[active_idx]

            # R = noise_scale * I/m + U_active @ diag(s_active) @ U_active^T
            R = np.eye(m) * (noise_scale / m)
            if len(active_idx) > 0:
                # Efficiently: R += (U_active * s_active) @ U_active^T
                R += (U_active * s_active[np.newaxis, :]) @ U_active.T

            # Check for numerical issues in R
            if not np.isfinite(R).all():
                if self.verbose:
                    logger.warning(
                        f"[MSP {it:02d}] Warning: Non-finite values in R, stopping"
                    )
                break

            invR = self._solve_R(R, np.eye(m))

            # Check for numerical issues in invR
            if not np.isfinite(invR).all():
                if self.verbose:
                    logger.warning(
                        f"[MSP {it:02d}] Warning: Non-finite values in invR, stopping"
                    )
                break

            C_minus_R = C - R

            # Precompute W = invR @ U (m × n_source_components)
            W = invR @ U  # m × n_source_components

            # Precompute CmR_W = (C - R) @ W for gradient
            CmR_W = C_minus_R @ W  # m × n_source_components

            # Precompute V = R @ W for Hessian
            V = R @ W  # m × n_source_components

            # Gradient computation
            grad = np.empty(p)

            if has_noise:
                invR_CmR = invR @ C_minus_R  # reuse
                trace_noise = (noise_scale / m) * np.sum(invR * invR_CmR.T)
                grad[0] = 0.5 * v * trace_noise - P_hyper[0, 0] * (
                    lambdas[0] - eta_vec[0]
                )

                w_dot_cmrw = np.sum(W * CmR_W, axis=0)  # length n_source_components
                grad[1:] = 0.5 * v * source_scales * w_dot_cmrw - np.diag(P_hyper)[
                    1:
                ] * (lambdas[1:] - eta_vec[1:])
            else:
                w_dot_cmrw = np.sum(W * CmR_W, axis=0)
                grad[:] = 0.5 * v * source_scales * w_dot_cmrw - np.diag(P_hyper) * (
                    lambdas - eta_vec
                )

            # Hessian computation
            Fkk = np.empty((p, p))

            if has_noise:
                WtV = W.T @ V  # n_source × n_source
                ss_outer = np.outer(source_scales, source_scales)
                Fkk[1:, 1:] = -0.5 * v * ss_outer * (WtV**2) - P_hyper[1:, 1:]

                invR_sq_trace = np.sum(invR * invR.T)
                Fkk[0, 0] = (
                    -0.5 * v * (noise_scale / m) ** 2 * invR_sq_trace - P_hyper[0, 0]
                )

                w_norms_sq = np.sum(W**2, axis=0)  # ||w_j||^2 for each j
                cross = (noise_scale / m) * source_scales * w_norms_sq
                Fkk[0, 1:] = -0.5 * v * cross - P_hyper[0, 1:]
                Fkk[1:, 0] = Fkk[0, 1:]
            else:
                WtV = W.T @ V
                ss_outer = np.outer(source_scales, source_scales)
                Fkk[:, :] = -0.5 * v * ss_outer * (WtV**2) - P_hyper

            # Check for numerical issues
            if not np.isfinite(grad).all() or not np.isfinite(Fkk).all():
                if self.verbose:
                    logger.warning(
                        f"[MSP {it:02d}] Warning: Non-finite values in grad/Fkk, stopping"
                    )
                break

            # Newton step with regularization
            try:
                eigvals = np.linalg.eigvalsh(Fkk)
                cond_num = np.abs(eigvals).max() / (np.abs(eigvals).min() + 1e-12)

                if cond_num > 1e12 or np.min(eigvals) > -1e-8:
                    reg_factor = max(1e-6, 1e-4 * np.abs(np.diag(Fkk)).mean())
                    Fkk_reg = Fkk - reg_factor * np.eye(p)
                    step = -np.linalg.solve(Fkk_reg, grad)
                else:
                    step = -np.linalg.solve(Fkk, grad)
            except np.linalg.LinAlgError:
                step = -grad / (np.abs(np.diag(Fkk)).mean() + 1e-6)

            if not np.isfinite(step).all():
                if self.verbose:
                    logger.warning(f"[MSP {it:02d}] Warning: Non-finite step, stopping")
                break

            lambdas_new = np.clip(lambdas + step, lambda_min, lambda_max)

            # Convergence check
            delta = lambdas_new - lambdas
            rel = np.linalg.norm(delta) / (np.linalg.norm(lambdas) + 1e-12)
            grad_norm = np.linalg.norm(grad)

            if initial_grad_norm is None:
                initial_grad_norm = max(grad_norm, 1e-10)

            grad_rel = grad_norm / initial_grad_norm

            lambdas = lambdas_new

            if self.verbose:
                logger.debug(
                    f"[MSP {it:02d}] grad_norm={grad_norm:.3e} (rel={grad_rel:.3e})  "
                    f"change={rel:.3e}  lambda_range=[{lambdas.min():.2f}, {lambdas.max():.2f}]"
                )

            if prev is not None:
                if rel < self.tol and grad_rel < self.tol:
                    if self.verbose:
                        logger.info(
                            f"MSP converged after {it} iterations (change={rel:.3e}, grad_rel={grad_rel:.3e})"
                        )
                    break
                elif rel < 1e-8:
                    if self.verbose:
                        logger.info(
                            f"MSP stopped after {it} iterations (minimal change, rel={rel:.3e})"
                        )
                    break
            prev = lambdas.copy()

        # Final covariance
        lambdas = np.clip(lambdas, lambda_min, lambda_max)
        scales = np.exp(lambdas)

        if has_noise:
            noise_scale = scales[0]
            source_scales = scales[1:]
        else:
            noise_scale = 0.0
            source_scales = scales

        R = np.eye(m) * (noise_scale / m)
        if U.shape[1] > 0:
            R += (U * source_scales[np.newaxis, :]) @ U.T

        invR = self._solve_R(R, np.eye(m))

        # Build source-space prior Re = sum_i (scale_i / trace_i) * q_i @ q_i^T
        # Using rank-1 structure for efficiency
        Re = np.zeros((d, d))
        for i, q in enumerate(self.patterns):
            q = q.flatten()
            tf = trace_factors[i]
            coeff = source_scales[i] / tf if tf > 0 else source_scales[i]
            q_col = q.reshape(-1, 1)
            Re += coeff * (q_col @ q_col.T)

        # Posterior mean inverse operator: M = Re L^T R^{-1}
        M = Re @ leadfield.T @ invR

        # Store diagnostics
        self.lambdas = lambdas
        self.R = R
        self.invR = invR
        self.Re = Re

        # Create inverse operator (single operator for MSP)
        self.inverse_operators = [InverseOperator(M, self.name)]
        return self

    def _generate_patterns(self, forward):
        """Generate spatial patterns using graph Laplacian smoothing.

        This method creates localized spatial patterns by:
        1. Computing the adjacency matrix of the source space
        2. Applying graph Laplacian diffusion smoothing
        3. Selecting diverse spatial locations to center the patterns

        Parameters
        ----------
        forward : mne.Forward
            The forward solution containing source space information.

        Return
        ------
        patterns : list
            List of spatial patterns (numpy arrays).
        """
        # Get source space adjacency
        adjacency = mne.spatial_src_adjacency(forward["src"], verbose=0)
        adjacency = csr_matrix(adjacency)
        n_dipoles = adjacency.shape[0]

        # Build implicit diffusion operator: (I + alpha * L)
        from scipy.sparse import eye as speye

        L_sparse = laplacian(adjacency)
        smoother = speye(n_dipoles, format="csr") + self.diffusion_parameter * L_sparse

        # Determine number of patterns
        if self.n_patterns is None:
            self.n_patterns = int(np.clip(np.sqrt(n_dipoles), 300, 1000))

        if self.verbose:
            logger.info(
                f"Generating {self.n_patterns} spatial patterns with order {self.patch_order}"
            )

        # Farthest-point sampling on 3D source coordinates
        coords = []
        for src_hemi in forward["src"]:
            coords.append(src_hemi["rr"][src_hemi["vertno"]])
        coords = np.vstack(coords)  # (n_dipoles, 3)

        n_select = min(self.n_patterns, n_dipoles)
        rng = np.random.RandomState(0)
        pattern_centers = np.empty(n_select, dtype=int)
        pattern_centers[0] = rng.randint(n_dipoles)
        min_dist = np.full(n_dipoles, np.inf)
        for k in range(1, n_select):
            diff = coords - coords[pattern_centers[k - 1]]
            dist = np.sum(diff**2, axis=1)
            min_dist = np.minimum(min_dist, dist)
            pattern_centers[k] = np.argmax(min_dist)

        # Factor the smoother once with splu, then solve for each RHS
        lu = splu(smoother.tocsc())

        patterns = []
        for center in pattern_centers:
            rhs = np.zeros(n_dipoles)
            rhs[center] = 1.0

            pattern = rhs
            for _ in range(self.patch_order):
                pattern = lu.solve(pattern)

            norm = np.linalg.norm(pattern)
            if norm > 0:
                pattern /= norm

            patterns.append(pattern)

        if self.verbose:
            logger.info(
                f"Generated {len(patterns)} patterns, each with {n_dipoles} sources"
            )

        return patterns

    @staticmethod
    def _solve_R(R, X, reg=1e-12):
        """Solve R Z = X stably using Cholesky decomposition

        Parameters
        ----------
        R : array
            Covariance matrix to invert
        X : array
            Right-hand side
        reg : float
            Regularization parameter for numerical stability
        """
        m = R.shape[0]

        # Add small regularization for numerical stability
        R_reg = R + reg * np.trace(R) / m * np.eye(m)

        try:
            U = np.linalg.cholesky(R_reg)
            Y_ = np.linalg.solve(U, X)
            Z = np.linalg.solve(U.T, Y_)
            return Z
        except np.linalg.LinAlgError:
            # If Cholesky fails, try with more regularization
            R_reg = R + 1e-6 * np.trace(R) / m * np.eye(m)
            try:
                return np.linalg.solve(R_reg, X)
            except np.linalg.LinAlgError:
                # Last resort: use pseudoinverse
                return np.linalg.pinv(R) @ X
