from __future__ import annotations

import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta


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

        P = self.compute_sensor_projector(forward_or_info=self.forward, n_chans=n_chans)
        W = self.compute_sensor_whitener(
            noise_cov,
            projector=P,
            rank_tol=self.rank_tol,
            eps=self.eps,
        )
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
        prior_diag = self.coerce_diag_source_prior(source_cov, n_sources)
        if self.use_depth_weighting:
            prior_diag = prior_diag * self.compute_depth_prior_whitened(
                G_white,
                depth=self.depth,
                depth_limit=self.depth_limit,
                eps=self.eps,
            )
        prior_diag = np.maximum(prior_diag, self.eps)
        R_sqrt = np.sqrt(prior_diag)

        # Trace normalization: trace(A A^T) == n_channels_effective.
        A = G_white * R_sqrt[np.newaxis, :]
        n_eff = int(W.shape[0])
        A, scale = self.trace_normalize_operator(A, target_rank=n_eff, eps=self.eps)
        R_sqrt = R_sqrt * scale

        svd = np.linalg.svd(A, full_matrices=False)

        inverse_operators = []
        for lambda2 in self.alphas:
            lambda2 = float(lambda2)
            K_white = self.solve_tikhonov_svd(
                A,
                lambda2,
                left_scale=R_sqrt,
                svd=svd,
                eps=self.eps,
            )

            # Full operator maps raw sensor data -> sources (and projects/whitens internally).
            K_full = K_white @ W_P  # (n_sources, n_chans)

            # dSPM noise normalization (whitened noise cov is I).
            K_dspm, _noise_std = self.noise_normalize_rows(
                K_white,
                K_full=K_full,
                eps=self.eps,
            )

            inverse_operators.append(K_dspm)

        self.inverse_operators = [
            InverseOperator(op, self.name) for op in inverse_operators
        ]
        return self
