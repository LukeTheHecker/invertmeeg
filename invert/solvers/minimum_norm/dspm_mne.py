from __future__ import annotations

import logging

import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverDSPMMNE(BaseSolver):
    """MNE-Python-style dSPM.

    Implements the core pieces of MNE-Python's minimum-norm pipeline that are
    missing from the legacy :mod:`invert.solvers.minimum_norm.dspm` solver:

    - SSP projector integration in sensor space
    - PCA-space whitening using the noise covariance (optional)
    - Depth weighting as a source prior with dynamic-range clipping
    - Trace normalization so ``lambda2`` is interpretable as ``1 / snr**2``
    - SVD-based Tikhonov regularization (stable per-component shrinkage)
    - dSPM noise normalization derived from the whitened operator

    Notes
    -----
    Current `invertmeeg` infrastructure uses fixed source orientation.
    This solver therefore implements the fixed-orientation variant of MNE's
    depth weighting and dSPM normalization.
    """

    meta = SolverMeta(
        acronym="dSPM-MNE",
        full_name="Dynamic Statistical Parametric Mapping (MNE-style)",
        category="Minimum Norm",
        description=(
            "MNE-style dSPM with SSP-aware whitening, trace-normalized source "
            "prior, and SVD-based Tikhonov regularization."
        ),
        references=[
            "Dale, A. M., Liu, A. K., Fischl, B. R., Buckner, R. L., Belliveau, J. W., Lewine, J. D., & Halgren, E. (2000). Dynamic statistical parametric mapping: combining fMRI and MEG for high-resolution imaging of cortical activity. Neuron, 26(1), 55–67.",
            "Hämäläinen, M. S., & Ilmoniemi, R. J. (1994). Interpreting magnetic fields of the brain: minimum norm estimates. Medical & Biological Engineering & Computing, 32(1), 35–42.",
        ],
    )

    def __init__(
        self,
        name: str = "dSPM (MNE-style)",
        *,
        # Depth weighting (MNE-style): exponent and clipping limit.
        depth: float = 0.8,
        depth_limit: float = 10.0,
        # Numerical tolerances.
        rank_tol: float = 1e-12,
        eps: float = 1e-15,
        **kwargs,
    ):
        self.name = name
        # Re-use the existing toggle name but implement depth weighting as a
        # source prior (not by scaling the leadfield columns in prepare_forward).
        kwargs.setdefault("use_depth_weighting", True)
        super().__init__(**kwargs)
        self.depth = float(depth)
        self.depth_limit = float(depth_limit)
        self.rank_tol = float(rank_tol)
        self.eps = float(eps)

        # This operator depends only on the forward model (and optional covariances).
        self.require_recompute = False
        self.require_data = False

        # Stored for downstream selection/diagnostics.
        self._projector: np.ndarray | None = None
        self._whitener: np.ndarray | None = None  # (rank, n_chans)
        self._whiten_rank: int | None = None

    def prepare_forward(self) -> None:
        """Prepare forward model but skip BaseSolver's depth-weight scaling."""
        # BaseSolver.prepare_forward applies its own depth weighting when
        # `self.use_depth_weighting` is True. For this solver, depth weighting
        # is implemented as a source prior, so disable it for the call.
        orig = bool(self.use_depth_weighting)
        self.use_depth_weighting = False
        try:
            super().prepare_forward()
        finally:
            self.use_depth_weighting = orig

    def get_alphas(self, reference=None):  # noqa: ARG002
        """Return lambda2 grid without eigenvalue scaling.

        In MNE, the regularization parameter is typically expressed as
        ``lambda2 = 1 / snr**2`` and is meaningful after trace normalization.
        """
        if self.alpha == "auto":
            alphas = list(np.asarray(self.r_values, dtype=float))
        else:
            alphas = [float(self.alpha)]
        self.alphas = alphas
        return alphas

    @staticmethod
    def _as_diag(source_cov: np.ndarray | None, n_sources: int) -> np.ndarray:
        """Coerce source covariance to a diagonal vector."""
        if source_cov is None:
            return np.ones(n_sources, dtype=float)
        source_cov = np.asarray(source_cov, dtype=float)
        if source_cov.ndim == 0:
            return np.full(n_sources, float(source_cov), dtype=float)
        if source_cov.ndim == 1:
            if source_cov.shape[0] != n_sources:
                raise ValueError(
                    f"source_cov has length {source_cov.shape[0]}, expected {n_sources}"
                )
            return source_cov
        if source_cov.ndim == 2:
            if source_cov.shape != (n_sources, n_sources):
                raise ValueError(
                    f"source_cov has shape {source_cov.shape}, expected {(n_sources, n_sources)}"
                )
            off = source_cov - np.diag(np.diag(source_cov))
            if not np.allclose(off, 0.0):
                raise ValueError(
                    "Full (non-diagonal) source_cov is not supported in this solver. "
                    "Pass a 1D diagonal prior instead."
                )
            return np.diag(source_cov)
        raise ValueError(f"Invalid source_cov with ndim={source_cov.ndim}")

    def _compute_projector(self) -> np.ndarray:
        """Compute SSP projector from forward info (or identity if none)."""
        projs = []
        bads = []
        ch_names = []
        try:
            info = self.forward["info"]  # type: ignore[index]
            projs = info.get("projs", []) or []
            bads = info.get("bads", []) or []
            ch_names = info.get("ch_names", []) or []
        except Exception:
            pass

        n_chans = int(self.leadfield.shape[0])
        if not projs:
            return np.eye(n_chans)

        if not ch_names:
            # Fall back to the forward object's channel order if info is incomplete.
            try:
                ch_names = list(self.forward.ch_names)  # type: ignore[union-attr]
            except Exception:
                ch_names = [str(i) for i in range(n_chans)]

        try:
            import mne

            P, _nproj, _ = mne.make_projector(  # type: ignore[misc]
                projs, ch_names, bads=bads, verbose=0
            )
        except Exception as e:
            logger.warning("Failed to build SSP projector via MNE (%s); using I.", e)
            P = np.eye(n_chans)
        return np.asarray(P, dtype=float)

    def _compute_whitener(self, noise_cov: np.ndarray, projector: np.ndarray) -> np.ndarray:
        """Compute PCA-space whitening matrix.

        Returns
        -------
        W : np.ndarray
            Whitening matrix of shape (rank, n_chans) that maps sensor data to
            whitened PCA space: ``y_white = W @ (P @ y)``.
        """
        n_chans = int(self.leadfield.shape[0])
        noise_cov = np.asarray(noise_cov, dtype=float)
        if noise_cov.shape != (n_chans, n_chans):
            raise ValueError(
                f"noise_cov has shape {noise_cov.shape}, expected {(n_chans, n_chans)}"
            )

        P = np.asarray(projector, dtype=float)
        if P.shape != (n_chans, n_chans):
            raise ValueError(
                f"projector has shape {P.shape}, expected {(n_chans, n_chans)}"
            )

        Cn = 0.5 * (noise_cov + noise_cov.T)
        Cn_proj = P @ Cn @ P.T
        Cn_proj = 0.5 * (Cn_proj + Cn_proj.T)

        eigvals, eigvecs = np.linalg.eigh(Cn_proj)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        max_ev = float(np.max(eigvals)) if eigvals.size else 0.0
        if max_ev <= 0:
            # Degenerate covariance: treat as fully rejected.
            return np.zeros((0, n_chans), dtype=float)

        mask = eigvals > max(self.rank_tol * max_ev, self.eps)
        if not np.any(mask):
            return np.zeros((0, n_chans), dtype=float)

        W = (eigvecs[:, mask] / np.sqrt(eigvals[mask])).T
        return np.asarray(W, dtype=float)

    def _depth_prior(self, G_white: np.ndarray) -> np.ndarray:
        """Compute MNE-style depth prior for fixed orientation."""
        # Sensitivity proxy: ||g_k||^2 in whitened space.
        sens = np.sum(G_white * G_white, axis=0)
        sens = np.maximum(sens, self.eps)
        w = 1.0 / sens

        if self.depth_limit > 0:
            # Clip dynamic range so deep weights are bounded relative to shallow ones.
            w_min = float(np.min(w))
            w_max = w_min * float(self.depth_limit) ** 2
            w = np.minimum(w, w_max)

        # Apply exponent (depth=0 -> no weighting, depth~0.8 in MNE defaults).
        if self.depth != 0:
            w = w ** float(self.depth)
        return w

    def make_inverse_operator(
        self,
        forward,
        *args,
        alpha="auto",
        noise_cov=None,
        source_cov=None,
        verbose: int = 0,  # noqa: ARG002
        **kwargs,
    ):
        """Calculate inverse operators for a lambda2 grid."""
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)

        G = np.asarray(self.leadfield, dtype=float)
        n_chans, n_sources = G.shape

        if noise_cov is None:
            noise_cov = np.eye(n_chans, dtype=float)
        noise_cov = np.asarray(noise_cov, dtype=float)

        P = self._compute_projector()
        W = self._compute_whitener(noise_cov, P)
        self._projector = P
        self._whitener = W
        self._whiten_rank = int(W.shape[0])

        if W.shape[0] == 0:
            raise ValueError(
                "Whitening rank is zero (noise_cov/proj rejected all dimensions)."
            )

        # Operate entirely in whitened + projected sensor space.
        W_P = W @ P  # (rank, n_chans)
        G_white = W_P @ G  # (rank, n_sources)

        # Source prior (diagonal).
        prior_diag = self._as_diag(source_cov, n_sources)
        if self.use_depth_weighting:
            prior_diag = prior_diag * self._depth_prior(G_white)
        prior_diag = np.maximum(prior_diag, self.eps)
        R_sqrt = np.sqrt(prior_diag)

        # Trace normalization: trace(A A^T) == n_channels_effective.
        A = G_white * R_sqrt[np.newaxis, :]
        trace_AAT = float(np.sum(A * A))
        n_eff = int(W.shape[0])
        scale = np.sqrt(n_eff / max(trace_AAT, self.eps))
        R_sqrt = R_sqrt * scale
        A = A * scale

        # SVD of whitened, weighted forward operator.
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        V = Vt.T  # (n_sources, rank)
        Ut = U.T  # (rank, rank)

        inverse_operators = []
        for lambda2 in self.alphas:
            lambda2 = float(lambda2)
            if lambda2 < 0:
                raise ValueError(f"lambda2 must be >= 0, got {lambda2}")

            gamma = s / (s * s + lambda2)  # (rank,)

            # K_white maps whitened sensor data -> sources.
            VG = V * gamma[np.newaxis, :]
            K_white = (VG @ Ut) * R_sqrt[:, np.newaxis]  # (n_sources, rank)

            # Full operator maps raw sensor data -> sources (and projects/whitens internally).
            K_full = K_white @ W_P  # (n_sources, n_chans)

            # dSPM noise normalization (whitened noise cov is I).
            noise_var = np.sum(K_white * K_white, axis=1)
            noise_std = np.sqrt(np.maximum(noise_var, self.eps))
            K_dspm = K_full / noise_std[:, np.newaxis]

            inverse_operators.append(K_dspm)

        self.inverse_operators = [
            InverseOperator(op, self.name) for op in inverse_operators
        ]
        return self

