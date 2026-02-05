from __future__ import annotations

import logging
from typing import Any

import mne
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, cg

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

    def __init__(self, name: str = "Total Variation", **kwargs: Any) -> None:
        self.name = name
        self._edges_i: np.ndarray | None = None
        self._edges_j: np.ndarray | None = None
        super().__init__(**kwargs)

    def make_inverse_operator(
        self, forward, *args: Any, alpha: str | float = "auto", **kwargs: Any
    ):
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)

        adjacency = mne.spatial_src_adjacency(self.forward["src"], verbose=0).tocoo()
        i = adjacency.row.astype(int)
        j = adjacency.col.astype(int)
        mask = i < j
        self._edges_i = i[mask]
        self._edges_j = j[mask]
        return self

    def apply_inverse_operator(
        self,
        mne_obj,
        tv_weight: float = 0.1,
        n_irls: int = 8,
        eps: float = 1e-3,
        ridge: float | None = None,
        cg_tol: float = 1e-4,
        cg_max_iter: int = 200,
    ):  # type: ignore[override]
        if self._edges_i is None or self._edges_j is None:
            raise RuntimeError(
                "Call make_inverse_operator() before apply_inverse_operator()."
            )

        Y = self.unpack_data_obj(mne_obj)
        L = self.leadfield
        n_chans, n_sources = L.shape
        n_time = Y.shape[1]

        if ridge is None:
            ridge = float(self.alphas[0]) if hasattr(self, "alphas") and self.alphas else 1e-6
        ridge = max(float(ridge), 1e-15)
        lam = max(float(tv_weight), 0.0)

        LLT = L @ L.T
        K_mne = np.linalg.solve(LLT + ridge * np.eye(n_chans), L).T
        X = K_mne @ Y

        edges_i = self._edges_i
        edges_j = self._edges_j

        b = L.T @ Y  # (n_sources, n_time)

        def build_weighted_laplacian(weights: np.ndarray) -> coo_matrix:
            rows = np.concatenate([edges_i, edges_j, edges_i, edges_j])
            cols = np.concatenate([edges_i, edges_j, edges_j, edges_i])
            vals = np.concatenate([weights, weights, -weights, -weights])
            return coo_matrix((vals, (rows, cols)), shape=(n_sources, n_sources)).tocsr()

        for _it in range(int(n_irls)):
            diff = X[edges_i] - X[edges_j]
            grad_sq = np.sum(diff * diff, axis=1)
            w = 1.0 / np.sqrt(grad_sq + float(eps) ** 2)

            Lap = build_weighted_laplacian(w)

            def matvec(v: np.ndarray, Lap_val: np.ndarray = Lap) -> np.ndarray:
                return L.T @ (L @ v) + lam * (Lap_val @ v) + ridge * v

            A = LinearOperator((n_sources, n_sources), matvec=matvec, dtype=np.float64)

            X_new = np.empty_like(X)
            for t in range(n_time):
                x_t, info = cg(
                    A,
                    b[:, t],
                    x0=X[:, t],
                    rtol=cg_tol,
                    maxiter=int(cg_max_iter),
                )
                if info != 0:
                    logger.debug("CG did not fully converge at t=%s (info=%s)", t, info)
                X_new[:, t] = x_t

            rel = float(np.linalg.norm(X_new - X)) / max(float(np.linalg.norm(X)), 1e-15)
            X = X_new
            if rel < 1e-3:
                break

        return self.source_to_object(X)
