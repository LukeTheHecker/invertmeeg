from __future__ import annotations

import logging
from typing import Any

import mne
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, cg

from invert.util import build_source_adjacency

from ..base import BaseSolver, SolverMeta

logger = logging.getLogger(__name__)


class SolverTotalVariation(BaseSolver):
    """Edge-preserving structured regularization via (Huber) graph total variation.

    Uses an iteratively reweighted quadratic approximation of a graph TV penalty
    on the source-space adjacency. Each IRLS step solves a symmetric positive
    definite linear system via conjugate gradients in implicit form.
    """

    meta = SolverMeta(
        acronym="TV",
        full_name="Graph Total Variation (Huber)",
        category="Structured Sparsity",
        description=(
            "Iteratively reweighted graph-TV (edge-preserving) regularizer on the "
            "source-space mesh adjacency."
        ),
        references=[
            "Rudin, L. I., Osher, S., & Fatemi, E. (1992). Nonlinear total variation based noise removal algorithms. Physica D: Nonlinear Phenomena, 60(1–4), 259–268.",
            "Huber, P. J. (1964). Robust estimation of a location parameter. The Annals of Mathematical Statistics, 35(1), 73–101.",
        ],
    )

    def __init__(
        self,
        name: str = "Total Variation",
        default_tv_weight: float = 0.01,
        default_ridge: float | None = None,
        default_n_irls: int = 8,
        default_eps: float = 1e-3,
        default_cg_tol: float = 2e-4,
        default_cg_max_iter: int = 200,
        auto_scale_hyperparams: bool = True,
        **kwargs: Any,
    ) -> None:
        self.name = name
        self._edges_i: np.ndarray | None = None
        self._edges_j: np.ndarray | None = None
        self._leadfield_gram: np.ndarray | None = None
        self._leadfield_gram_diag: np.ndarray | None = None
        self._sensor_gram: np.ndarray | None = None
        self.default_tv_weight = float(default_tv_weight)
        self.default_ridge = (
            None if default_ridge is None else max(float(default_ridge), 1e-15)
        )
        self.default_n_irls = int(default_n_irls)
        self.default_eps = float(default_eps)
        self.default_cg_tol = float(default_cg_tol)
        self.default_cg_max_iter = int(default_cg_max_iter)
        self.auto_scale_hyperparams = bool(auto_scale_hyperparams)
        super().__init__(**kwargs)
        # TV's make_inverse_operator only depends on the forward/noise model, not
        # on the individual sample. Marking it reusable lets the benchmark runner
        # build once and call apply_inverse_operator for each sample instead of
        # recomputing all caches 50 times.
        self.require_recompute = False

    @staticmethod
    def _auto_scaled_hyperparams(
        L: np.ndarray,
        edges_i: np.ndarray,
        edges_j: np.ndarray,
        *,
        tv_weight: float,
        ridge: float | None,
    ) -> tuple[float, float]:
        """Map dimensionless TV/ridge knobs onto the current whitened operator scale."""
        LTL_diag = np.sum(L * L, axis=0)
        data_scale = max(float(np.median(LTL_diag)), 1e-12)

        degree = np.bincount(
            np.concatenate([edges_i, edges_j]), minlength=L.shape[1]
        ).astype(float)
        lap_scale = max(float(np.median(degree)), 1.0)

        lam = max(float(tv_weight), 0.0) * (data_scale / lap_scale)
        if ridge is None:
            # A small data-scale ridge stabilizes CG while preserving TV structure.
            ridge_eff = 0.01 * data_scale
        else:
            ridge_eff = max(float(ridge), 1e-15)
        return lam, ridge_eff

    def make_inverse_operator(
        self,
        forward,
        *args: Any,
        alpha: str | float = "auto",
        noise_cov: mne.Covariance | None = None,
        **kwargs: Any,
    ):
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        self.prepare_whitened_forward(noise_cov)

        adjacency = build_source_adjacency(self.forward["src"], verbose=0).tocoo()
        i = adjacency.row.astype(int)
        j = adjacency.col.astype(int)
        mask = i < j
        self._edges_i = i[mask]
        self._edges_j = j[mask]
        self._leadfield_gram = self.leadfield.T @ self.leadfield
        self._leadfield_gram_diag = np.einsum(
            "ij,ij->j", self.leadfield, self.leadfield, optimize=True
        )
        self._sensor_gram = self.leadfield @ self.leadfield.T
        return self

    def _build_weighted_laplacian(self, weights: np.ndarray) -> coo_matrix:
        """Build the weighted graph Laplacian for the current IRLS step."""
        if self._edges_i is None or self._edges_j is None:
            raise RuntimeError(
                "Call make_inverse_operator() before building the TV Laplacian."
            )

        edges_i = self._edges_i
        edges_j = self._edges_j
        rows = np.concatenate([edges_i, edges_j, edges_i, edges_j])
        cols = np.concatenate([edges_i, edges_j, edges_j, edges_i])
        vals = np.concatenate([weights, weights, -weights, -weights])
        return coo_matrix(
            (vals, (rows, cols)),
            shape=(self.leadfield.shape[1], self.leadfield.shape[1]),
        ).tocsr()

    @staticmethod
    def _solve_sensor_system(system: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        """Solve the small sensor-space SPD system used for the warm start."""
        try:
            chol, lower = cho_factor(system, lower=True, check_finite=False)
            return cho_solve((chol, lower), rhs, check_finite=False)
        except np.linalg.LinAlgError:
            return np.linalg.solve(system, rhs)

    def apply_inverse_operator(
        self,
        mne_obj,
        tv_weight: float | None = None,
        n_irls: int | None = None,
        eps: float | None = None,
        ridge: float | None = None,
        cg_tol: float | None = None,
        cg_max_iter: int | None = None,
        auto_scale_hyperparams: bool | None = None,
    ):  # type: ignore[override]
        if self._edges_i is None or self._edges_j is None:
            raise RuntimeError(
                "Call make_inverse_operator() before apply_inverse_operator()."
            )
        if (
            self._leadfield_gram is None
            or self._leadfield_gram_diag is None
            or self._sensor_gram is None
        ):
            raise RuntimeError(
                "TV solver cache is missing. Rebuild the inverse operator first."
            )

        Y = self.unpack_data_obj(mne_obj)
        self.validate_operator_data_compatibility(Y)
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]
        Y = self._sensor_transform @ Y
        L = self.leadfield
        n_chans = L.shape[0]

        if tv_weight is None:
            tv_weight = self.default_tv_weight
        if n_irls is None:
            n_irls = self.default_n_irls
        if eps is None:
            eps = self.default_eps
        if cg_tol is None:
            cg_tol = self.default_cg_tol
        if cg_max_iter is None:
            cg_max_iter = self.default_cg_max_iter
        if ridge is None:
            ridge = self.default_ridge
        if auto_scale_hyperparams is None:
            auto_scale_hyperparams = self.auto_scale_hyperparams

        if auto_scale_hyperparams:
            lam, ridge_eff = self._auto_scaled_hyperparams(
                L,
                self._edges_i,
                self._edges_j,
                tv_weight=float(tv_weight),
                ridge=ridge,
            )
        else:
            lam = max(float(tv_weight), 0.0)
            if ridge is None:
                ridge_eff = 1e-6
            else:
                ridge_eff = max(float(ridge), 1e-15)

        LTL = self._leadfield_gram
        LTL_diag = self._leadfield_gram_diag
        LLT = self._sensor_gram

        b = L.T @ Y  # (n_sources, n_time)
        sensor_system = LLT + ridge_eff * np.eye(n_chans)
        # The MNE warm start solves one sensor-sized system for all timepoints
        # instead of forming an explicit kernel matrix.
        X = L.T @ self._solve_sensor_system(sensor_system, Y)

        for _it in range(int(n_irls)):
            diff = X[self._edges_i] - X[self._edges_j]
            grad_sq = np.sum(diff * diff, axis=1)
            w = 1.0 / np.sqrt(grad_sq + float(eps) ** 2)

            lap = self._build_weighted_laplacian(w)
            # Jacobi preconditioning is cheap and can reduce CG work on
            # poorly conditioned IRLS steps without changing the objective.
            lap_diag = np.asarray(lap.diagonal()).ravel()
            inv_diag = 1.0 / np.maximum(LTL_diag + lam * lap_diag + ridge_eff, 1e-15)
            preconditioner = LinearOperator(
                (len(inv_diag), len(inv_diag)),
                matvec=lambda v, d=inv_diag: d * v,
                dtype=np.float64,
            )

            def matvec(v: np.ndarray, lap_val=lap) -> np.ndarray:
                return LTL @ v + lam * (lap_val @ v) + ridge_eff * v

            operator = LinearOperator(
                (L.shape[1], L.shape[1]), matvec=matvec, dtype=np.float64
            )

            X_new = np.empty_like(X)
            for t in range(X.shape[1]):
                x_t, info = cg(
                    operator,
                    b[:, t],
                    x0=X[:, t],
                    rtol=float(cg_tol),
                    maxiter=int(cg_max_iter),
                    M=preconditioner,
                )
                if info != 0:
                    logger.debug("CG did not fully converge at t=%s (info=%s)", t, info)
                X_new[:, t] = x_t

            rel = float(np.linalg.norm(X_new - X)) / max(
                float(np.linalg.norm(X)), 1e-15
            )
            X = X_new
            if rel < 1e-3:
                break

        return self.source_to_object(X)
