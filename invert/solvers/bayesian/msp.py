import logging
from invert.util import build_source_adjacency

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
        max_iter=128,
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
            Convergence tolerance for predicted free energy change.
        n_patterns : int or None
            Number of patterns to generate if patterns is None. If None,
            a reasonable default based on the source space will be used.
        diffusion_parameter : float
            Diffusion parameter (alpha) for graph smoothing when generating
            patterns automatically. Default is 0.1.
        patch_order : int
            Order of smoothing for generated patches (1 = small patches,
            2 = medium patches, etc.). Default is 2.
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

    def make_inverse_operator(
        self,
        forward,
        mne_obj=None,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
        **kwargs,
    ):
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
        super().make_inverse_operator(forward, mne_obj, *args, alpha=alpha, **kwargs)
        wf = self.prepare_whitened_forward(noise_cov)
        data = self.unpack_data_obj(mne_obj)
        data = wf.sensor_transform @ data
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

        # Build diagonal source priors as sensor-space covariance components
        # Q_i = diag(q_i) in source space -> LQL_i = (L * q_i) @ L^T in sensor space
        n_pats = len(self.patterns)
        Q_src = np.array(self.patterns)  # n_pats x d

        LQL = np.empty((n_pats, m, m))
        tr = np.empty(n_pats)
        valid = np.ones(n_pats, dtype=bool)

        for i in range(n_pats):
            lq = leadfield * Q_src[i]  # m x d, broadcasting
            LQL[i] = lq @ leadfield.T  # m x m
            tr[i] = np.trace(LQL[i])
            if tr[i] > 1e-20:
                LQL[i] /= tr[i]
            else:
                valid[i] = False

        # Remove invalid patterns
        if not valid.all():
            LQL = LQL[valid]
            Q_src = Q_src[valid]
            tr = tr[valid]
            n_pats = int(valid.sum())

        # Add noise component
        has_noise = self.include_sensor_noise
        if has_noise:
            Q_noise = np.eye(m) / m  # trace-normalized identity
            LQL_all = np.concatenate([LQL, Q_noise[np.newaxis]], axis=0)
            n_comp = n_pats + 1
        else:
            LQL_all = LQL
            n_comp = n_pats

        # Data-driven initialization: h = log(trace(C) / n_comp)
        trC = max(np.trace(C), 1e-20)
        h_init = np.log(trC / n_comp)
        h = np.full(n_comp, h_init)
        hE = np.full(n_comp, self.eta)  # hyperprior mean
        Pi_val = self.Pi  # hyperprior precision (scalar)

        # Bounds: h_max so that a single component can explain ~exp(10)x the
        # data variance (generous margin); h_min at hyperprior mean
        lambda_min = -32.0
        lambda_max = np.log(trC) + 10.0

        if self.verbose:
            logger.info(
                f"MSP: {n_comp} components ({n_pats} source + "
                f"{'1 noise' if has_noise else '0 noise'}), "
                f"h_init={h_init:.2f}, h_bounds=[{lambda_min}, {lambda_max:.1f}]"
            )

        # ReML iterations (SPM-style Fisher scoring with damping)
        for it in range(1, self.max_iter + 1):
            scales = np.exp(h)

            # Model covariance: R = sum_i scales[i] * LQL_all[i]
            R = np.tensordot(scales, LQL_all, axes=([0], [0]))  # m x m

            # Precision
            P = self._solve_R(R, np.eye(m))
            if not np.isfinite(P).all():
                if self.verbose:
                    logger.warning(f"[MSP {it:02d}] Non-finite precision, stopping")
                break

            # Residual: W = I - P @ C
            W = np.eye(m) - P @ C

            # PQ[i] = P @ LQL_all[i] * scales[i]
            PQ = np.einsum('jk,ikl->ijl', P, LQL_all) * scales[:, None, None]

            # Vectorized gradient: g_i = -v/2 * trace(PQ[i] @ W)
            PQ_flat = PQ.reshape(n_comp, m * m)
            W_T_flat = W.T.ravel()
            g = -0.5 * v * (PQ_flat @ W_T_flat)

            # Vectorized Hessian: H_ij = -v/2 * trace(PQ[i] @ PQ[j])
            PQ_T_flat = PQ.transpose(0, 2, 1).reshape(n_comp, m * m)
            H = -0.5 * v * (PQ_flat @ PQ_T_flat.T)

            # Add hyperprior terms
            g_total = g - Pi_val * (h - hE)
            H_total = H - Pi_val * np.eye(n_comp)

            # Newton step
            try:
                dh = -np.linalg.solve(H_total, g_total)
            except np.linalg.LinAlgError:
                dh = -g_total / (np.abs(np.diag(H_total)).mean() + 1e-6)

            if not np.isfinite(dh).all():
                if self.verbose:
                    logger.warning(f"[MSP {it:02d}] Non-finite step, stopping")
                break

            # Per-component step clipping (prevents explosive steps when
            # Hessian is rank-deficient, i.e. n_comp > m)
            np.clip(dh, -4, 4, out=dh)

            # SPM-style damping
            dh /= np.log(it + 2)

            # Predicted free energy change
            dF = float(np.dot(dh, g_total) + 0.5 * dh @ H_total @ dh)

            # Update and bound
            h += dh
            np.clip(h, lambda_min, lambda_max, out=h)

            # Pruning: reset very inactive components to hyperprior mean
            prune_mask = h < -16
            h[prune_mask] = hE[prune_mask]

            if self.verbose:
                n_active = int(np.sum(np.exp(h) > 1e-6))
                logger.debug(
                    f"[MSP {it:02d}] dF={dF:.3e}  "
                    f"h_range=[{h.min():.1f}, {h.max():.1f}]  "
                    f"active={n_active}/{n_comp}"
                )

            # Convergence
            if abs(dF) < self.tol:
                if self.verbose:
                    logger.info(
                        f"MSP converged after {it} iterations (dF={dF:.3e})"
                    )
                break

        # Final reconstruction
        scales = np.exp(h)
        source_scales = scales[:n_pats]

        # Source prior diagonal: re[j] = sum_i source_scales[i] * Q_src[i,j] / tr[i]
        re = np.zeros(d)
        for i in range(n_pats):
            if tr[i] > 1e-20:
                re += source_scales[i] * Q_src[i] / tr[i]

        # Final model covariance and precision
        R = np.tensordot(scales, LQL_all, axes=([0], [0]))
        P = self._solve_R(R, np.eye(m))

        # MAP inverse operator: M = diag(re) @ L^T @ P
        # Efficiently: (re[:, None] * L^T) @ P  -- no d x d matrix needed
        M = (re[:, None] * leadfield.T) @ P

        # Store diagnostics
        self.lambdas = h
        self.R = R
        self.invR = P

        # Create inverse operator
        self.inverse_operators = [InverseOperator(M @ wf.sensor_transform, self.name)]
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
        adjacency = build_source_adjacency(forward["src"], verbose=0)
        adjacency = csr_matrix(adjacency)
        n_dipoles = adjacency.shape[0]

        # Build implicit diffusion operator: (I + alpha * L)
        from scipy.sparse import eye as speye

        L_sparse = laplacian(adjacency)
        smoother = speye(n_dipoles, format="csr") + self.diffusion_parameter * L_sparse

        # Determine number of patterns — use all dipoles when feasible
        # (one pattern per source = ARD), cap at 512 for large spaces
        if self.n_patterns is None:
            self.n_patterns = min(n_dipoles, 512)

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
