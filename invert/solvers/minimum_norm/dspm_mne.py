from __future__ import annotations

import logging

import mne
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

    def make_inverse_operator(
        self,
        forward,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
        source_cov=None,
        verbose: int = 0,  # noqa: ARG002
        **kwargs,
    ):
        """Calculate inverse operators for a lambda2 grid."""
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        if noise_cov is None:
            noise_cov = self.make_identity_noise_cov(list(self.forward.ch_names))

        wf = self.prepare_whitened_forward(
            noise_cov,
            source_cov=source_cov,
            depth=self.depth if self.use_depth_weighting else None,
            depth_limit=self.depth_limit,
            trace_normalize=True,
            precompute_svd=True,
            rank_tol=self.rank_tol,
            eps=self.eps,
        )
        self._projector = wf.projector
        self._whitener = wf.sensor_transform
        self._whiten_rank = wf.n_eff

        if wf.whitener_mode != "projected" and wf.whitener_mode != "none":
            logger.warning("dSPM-MNE whitener fallback used: %s", wf.whitener_mode)

        inverse_operators = []
        tikhonov_matrix = wf.A if wf.A is not None else wf.G_white
        for lambda2 in self.alphas:
            lambda2 = float(lambda2)
            K_white = self.solve_tikhonov_svd(
                tikhonov_matrix,
                lambda2,
                left_scale=wf.R_sqrt,
                svd=wf.svd,
                eps=self.eps,
            )

            K_full = K_white @ wf.sensor_transform
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
