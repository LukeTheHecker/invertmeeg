from __future__ import annotations

import mne
import numpy as np

from invert.util import normalize_saliency

from ..base import BaseSolver, SolverMeta
from ..music.alternating_projections import SolverAlternatingProjections
from ..music.signal_subspace_matching import SolverSignalSubspaceMatching
from .graph_wiener_smooth_sparse_adaptive_robust import (
    SolverGraphWienerSmoothSparseAdaptiveRobustEB,
)


class SolverSparseSeededSupport(BaseSolver):
    """Support-only hybrid: sparse seeds plus smooth local patch shaping.

    The detector (`FLEX-SSM`) is good at identifying a small set of salient
    patch centers/orders, while `GW-SSAR` provides smoother local patch shape.
    This solver blends those two signals and intentionally ignores the diffuse
    background that dominates full-field metrics on `mixed_cortical_activity`.
    """

    meta = SolverMeta(
        acronym="SSS",
        full_name="Sparse Seeded Support",
        category="Subspace Methods",
        description=(
            "Support-focused hybrid that uses FLEX-SSM patch seeds to gate and "
            "blend a GW-SSAR saliency map, prioritizing dominant patch recovery "
            "instead of full-field source reconstruction."
        ),
        internal=True,
    )

    def __init__(
        self,
        name: str = "Sparse Seeded Support",
        *,
        seed_weight: float = 0.5,
        gate_max: float = 10.0,
        seed_strategy: str = "ap",
        seed_solver_kwargs: dict | None = None,
        shape_solver_kwargs: dict | None = None,
        **kwargs,
    ):
        self.name = name
        self.seed_weight = float(seed_weight)
        self.gate_max = float(gate_max)
        self.seed_strategy = str(seed_strategy).lower()
        self.seed_solver_kwargs = dict(seed_solver_kwargs or {})
        self.shape_solver_kwargs = dict(shape_solver_kwargs or {})
        self.seed_solver: BaseSolver | None = None
        self.shape_solver: SolverGraphWienerSmoothSparseAdaptiveRobustEB | None = None
        self._support_prior: np.ndarray | None = None
        super().__init__(**kwargs)
        self.require_recompute = True
        self.require_data = True

    def _build_seed_solver(self) -> BaseSolver:
        if self.seed_strategy == "ssm":
            return SolverSignalSubspaceMatching(**self.seed_solver_kwargs)
        if self.seed_strategy == "ap":
            return SolverAlternatingProjections(**self.seed_solver_kwargs)
        raise ValueError(
            f"seed_strategy must be 'ap' or 'ssm', got {self.seed_strategy!r}"
        )

    def _build_shape_solver(self) -> SolverGraphWienerSmoothSparseAdaptiveRobustEB:
        return SolverGraphWienerSmoothSparseAdaptiveRobustEB(**self.shape_solver_kwargs)

    @staticmethod
    def _candidate_support(seed_solver: BaseSolver) -> np.ndarray:
        n_vertices = int(seed_solver.leadfield.shape[1])
        support = np.zeros(n_vertices, dtype=float)
        for order, dipole in getattr(seed_solver, "candidates", []):
            atom = np.asarray(
                seed_solver.gradients[int(order)][int(dipole)].toarray(),
                dtype=float,
            ).ravel()
            support += atom
        return normalize_saliency(support, mode="sum")

    def make_inverse_operator(
        self,
        forward,
        mne_obj=None,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
        verbose=0,
        **kwargs,
    ):
        super().make_inverse_operator(forward, mne_obj, *args, alpha=alpha, **kwargs)
        self.seed_solver = self._build_seed_solver()
        self.shape_solver = self._build_shape_solver()

        self.seed_solver.make_inverse_operator(
            forward,
            mne_obj,
            *args,
            alpha=alpha,
            noise_cov=noise_cov,
            verbose=verbose,
            **kwargs,
        )
        self.shape_solver.make_inverse_operator(
            forward,
            mne_obj,
            *args,
            alpha=alpha,
            noise_cov=noise_cov,
            verbose=verbose,
            **kwargs,
        )
        self._support_prior = self._candidate_support(self.seed_solver)
        self.forward = self.shape_solver.forward
        self.obj_info = self.shape_solver.obj_info
        self.tmin = self.shape_solver.tmin
        return self

    def apply_inverse_operator(self, mne_obj) -> mne.SourceEstimate:
        if self.shape_solver is None or self.seed_solver is None:
            raise ValueError("Call make_inverse_operator() before apply_inverse_operator().")

        stc_shape = self.shape_solver.apply_inverse_operator(mne_obj)
        source_mat = np.asarray(stc_shape.data, dtype=float)
        shape_saliency = normalize_saliency(np.mean(np.abs(source_mat), axis=1), mode="sum")
        support_prior = np.asarray(self._support_prior, dtype=float)
        blended = normalize_saliency(
            (1.0 - self.seed_weight) * shape_saliency + self.seed_weight * support_prior,
            mode="sum",
        )

        ratio = blended / np.maximum(shape_saliency, 1e-12)
        ratio = np.clip(ratio, 0.0, self.gate_max)
        gated = source_mat * ratio[:, np.newaxis]
        return self.shape_solver.source_to_object(gated)
