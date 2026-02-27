import logging

import mne
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import splu

from invert.util import build_source_adjacency
from invert.util.source_adjacency import source_space_signature

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)
_PATTERN_CACHE: dict[tuple[str, int, float, int], np.ndarray] = {}


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
        sensor_rank="auto",
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
        self.sensor_rank = sensor_rank
        # MSP operates on scalar source amplitudes; for free-orientation
        # volume/discrete forwards, reduce to scalar orientations by default.
        kwargs.setdefault("orientation", "pca")
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

        # Patterns are treated as *patch vectors* u_i in source space (d,).
        # Each covariance component is rank-1: Q_i = u_i u_i^T.
        # In sensor space: L Q_i L^T = (L u_i)(L u_i)^T, trace-normalized.
        try:
            patterns = np.asarray(self.patterns, dtype=float)
            if patterns.ndim != 2:
                raise ValueError
        except Exception:
            patterns = np.stack([np.asarray(p, dtype=float) for p in self.patterns])
        if patterns.shape[1] != d:
            raise ValueError(f"patterns have shape {patterns.shape}, expected (*, {d})")

        # Optional: project onto a low-dimensional sensor subspace to reduce
        # ReML costs, then map back to the full whitened sensor space.
        sensor_rank = getattr(self, "sensor_rank", "auto")
        use_projection = False
        if isinstance(sensor_rank, str):
            sensor_rank_str = sensor_rank.strip().lower()
            if sensor_rank_str in {"none", "full"}:
                r = m
            elif sensor_rank_str == "auto":
                r = min(m, 64, v)
                use_projection = r < m
            else:
                try:
                    r = int(sensor_rank_str)
                except Exception as err:
                    raise ValueError(
                        "sensor_rank must be 'auto', 'full', or an int, "
                        f"got {sensor_rank!r}"
                    ) from err
                r = int(np.clip(r, 1, min(m, v)))
                use_projection = r < m
        else:
            try:
                r = int(sensor_rank)
            except Exception as err:
                raise ValueError(
                    "sensor_rank must be 'auto', 'full', or an int, "
                    f"got {sensor_rank!r}"
                ) from err
            r = int(np.clip(r, 1, min(m, v)))
            use_projection = r < m

        if use_projection:
            U_sens, _s, _ = np.linalg.svd(data, full_matrices=False)
            basis = U_sens[:, :r]  # (m, r), orthonormal
            data_r = basis.T @ data  # (r, v)
            leadfield_r = basis.T @ leadfield  # (r, d)
            basis_T = basis.T  # (r, m)
        else:
            data_r = data
            leadfield_r = leadfield
            basis_T = None

        m_eff = int(data_r.shape[0])
        C = (data_r @ data_r.T) / float(v)  # sample covariance over time/epochs

        # Sensor-space pattern vectors: k_i = L u_i
        K = leadfield_r @ patterns.T  # (m_eff, n_pats)
        k_norm2 = np.sum(K * K, axis=0)
        valid = k_norm2 > 1e-20
        if not np.any(valid):
            raise ValueError("All MSP patterns are null in sensor space.")

        patterns = patterns[valid]
        k_norm2 = k_norm2[valid]
        K = K[:, valid]

        # Trace-normalize components by ||k_i||^2 (equivalently: u_i /= ||k_i||)
        k_scale = 1.0 / np.sqrt(k_norm2)
        K = K * k_scale[np.newaxis, :]
        patterns = patterns * k_scale[:, np.newaxis]

        n_pats = int(patterns.shape[0])

        has_noise = bool(self.include_sensor_noise)
        n_comp = n_pats + (1 if has_noise else 0)

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

        # ReML iterations (SPM-style Fisher scoring with damping).
        #
        # Important: MSP can have hundreds of components; building the full
        # Hessian scales as O(n_comp^2 * m^2) and is prohibitively expensive.
        # Use a diagonal Hessian approximation (per-component Fisher scoring),
        # which is fast and works well in practice for ARD-style updates.
        for it in range(1, self.max_iter + 1):
            scales = np.exp(h)
            scales_src = scales[:n_pats]

            # Model covariance in sensor space:
            #   R = K diag(scales_src) K^T + scale_noise * I/m
            K_w = K * np.sqrt(scales_src)[np.newaxis, :]
            R = K_w @ K_w.T
            if has_noise:
                scale_noise = float(scales[-1])
                R.flat[:: m_eff + 1] += scale_noise / float(m_eff)

            # Precision
            P = self._solve_R(R, np.eye(m_eff))
            if not np.isfinite(P).all():
                if self.verbose:
                    logger.warning(f"[MSP {it:02d}] Non-finite precision, stopping")
                break

            PC = P @ C
            # Pk = P @ K, Cpk = C @ Pk
            Pk = P @ K
            Cpk = C @ Pk

            # g_i = -v/2 * scales[i] * trace(P S_i W), with S_i = k_i k_i^T
            # trace(P S_i W) = k_i^T W P k_i = k_i^T P k_i - (P k_i)^T C (P k_i)
            a = np.sum(K * Pk, axis=0)  # k_i^T P k_i
            b = np.sum(Pk * Cpk, axis=0)  # (P k_i)^T C (P k_i)
            g_src = -0.5 * float(v) * scales_src * (a - b)

            # Diagonal Hessian approximation:
            # H_ii = -v/2 * scales[i]^2 * trace(P S_i P S_i) = -v/2 * scales[i]^2 * (k_i^T P k_i)^2
            H_src_diag = -0.5 * float(v) * (scales_src**2) * (a**2)

            if has_noise:
                # Noise component: S_noise = I/m, trace-normalized.
                trP = float(np.trace(P))
                trPPC = float(np.sum(P * PC.T))  # trace(P @ (P @ C))
                g_noise = (
                    -0.5 * float(v) * float(scales[-1]) * (trP - trPPC) / float(m_eff)
                )
                H_noise_diag = (
                    -0.5
                    * float(v)
                    * (float(scales[-1]) ** 2)
                    * float(np.sum(P * P))
                    / float(m_eff**2)
                )

            # Add hyperprior terms
            if has_noise:
                g = np.concatenate([g_src, np.array([g_noise])])
                H_diag = np.concatenate([H_src_diag, np.array([H_noise_diag])])
            else:
                g = g_src
                H_diag = H_src_diag

            g_total = g - Pi_val * (h - hE)
            H_total_diag = H_diag - Pi_val

            # Newton step
            denom = np.where(np.abs(H_total_diag) > 1e-30, H_total_diag, -1.0)
            dh = -g_total / denom

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
            dF = float(np.dot(dh, g_total) + 0.5 * np.dot(dh * H_total_diag, dh))

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
                    logger.info(f"MSP converged after {it} iterations (dF={dF:.3e})")
                break

        # Final reconstruction
        scales = np.exp(h)
        scales_src = scales[:n_pats]

        # Final model covariance and precision (in the reduced sensor space)
        K_w = K * np.sqrt(scales_src)[np.newaxis, :]
        R = K_w @ K_w.T
        if has_noise:
            R.flat[:: m_eff + 1] += float(scales[-1]) / float(m_eff)
        P = self._solve_R(R, np.eye(m_eff))

        # MAP inverse operator in reduced sensor space:
        #   M_r = (sum_i scales[i] u_i k_i^T) P = U @ (diag(scales) (K^T P))
        U_cols = patterns.T  # (d, n_pats)
        KP = K.T @ P  # (n_pats, m_eff)
        M_r = U_cols @ (scales_src[:, np.newaxis] * KP)  # (d, m_eff)

        # Map back to full whitened sensor space if projected.
        if basis_T is not None:
            M_white = M_r @ basis_T  # (d, m)
        else:
            M_white = M_r

        # Store diagnostics
        self.lambdas = h
        self.R = R
        self.invR = P

        # Create inverse operator
        self.inverse_operators = [
            InverseOperator(M_white @ wf.sensor_transform, self.name)
        ]
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
        # (one pattern per source = ARD), cap at 256 for large spaces
        n_patterns = (
            int(self.n_patterns) if self.n_patterns is not None else min(n_dipoles, 256)
        )
        n_select = min(n_patterns, n_dipoles)
        cache_key = (
            source_space_signature(forward["src"]),
            int(n_select),
            float(self.diffusion_parameter),
            int(self.patch_order),
        )
        cached_patterns = _PATTERN_CACHE.get(cache_key)
        if cached_patterns is not None:
            return [row.copy() for row in cached_patterns]

        if self.verbose:
            logger.info(
                f"Generating {n_select} spatial patterns with order {self.patch_order}"
            )

        # Farthest-point sampling on 3D source coordinates
        coords = []
        for src_hemi in forward["src"]:
            coords.append(src_hemi["rr"][src_hemi["vertno"]])
        coords = np.vstack(coords)  # (n_dipoles, 3)

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

        rhs = np.zeros((n_dipoles, n_select), dtype=float)
        rhs[pattern_centers, np.arange(n_select)] = 1.0
        smoothed = rhs
        for _ in range(self.patch_order):
            smoothed = lu.solve(smoothed)

        norms = np.linalg.norm(smoothed, axis=0)
        norms = np.maximum(norms, 1e-30)
        smoothed /= norms[np.newaxis, :]
        patterns = [smoothed[:, i].copy() for i in range(n_select)]

        if self.verbose:
            logger.info(
                f"Generated {len(patterns)} patterns, each with {n_dipoles} sources"
            )

        _PATTERN_CACHE[cache_key] = np.asarray(patterns, dtype=float)
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
