from __future__ import annotations

import logging
from typing import Any

import mne
import numpy as np
from scipy import linalg
from scipy.sparse import coo_matrix, csr_matrix

from invert.util import build_source_adjacency

from ..base import BaseSolver, SolverMeta

logger = logging.getLogger(__name__)


def _soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - float(threshold), 0.0)


class SolverSISSY(BaseSolver):
    """Source Imaging based on Structured Sparsity (SISSY).

    Solves, in whitened space:

        min_s  (lambda / 2) ||X - G s||_F^2 + ||T s||_1 + alpha ||s||_1

    with ADMM updates as described by Aud'hui et al. (2025), where ``T`` is a
    graph-difference operator over adjacent source locations.
    """

    meta = SolverMeta(
        acronym="SISSY",
        full_name="Source Imaging based on Structured Sparsity",
        category="Structured Sparsity",
        description=(
            "ADMM solver with sparsity on both source amplitudes and spatial "
            "source gradients (graph total-variation-like differences)."
        ),
        references=[
            "Aud'hui, M., Kachenoura, A., Albera, L., Merlet, I., Yochum, M., Van Bogaert, P., ... & Benquet, P. (2025). Fully Automated EEG Source Imaging Using Structured Sparsity for Single and Multiple Synchronous Epileptic Activities. bioRxiv, 2025-11.",
        ],
    )

    def __init__(
        self,
        name: str = "SISSY",
        default_lambda: float = 1.0,
        default_alpha_l1: float = 5e-2,
        default_rho1: float = 1.0,
        default_rho2: float = 1.0,
        default_rho3: float = 1.0,
        default_max_iter: int = 80,
        default_abs_tol: float = 1e-5,
        default_rel_tol: float = 1e-4,
        default_lambda_floor: float = 1e-8,
        default_lambda_ceil: float = 1e8,
        adjacency_type: str = "spatial",
        adjacency_distance: float = 3e-3,
        **kwargs: Any,
    ) -> None:
        self.name = name
        self.default_lambda = float(default_lambda)
        self.default_alpha_l1 = float(default_alpha_l1)
        self.default_rho1 = float(default_rho1)
        self.default_rho2 = float(default_rho2)
        self.default_rho3 = float(default_rho3)
        self.default_max_iter = int(default_max_iter)
        self.default_abs_tol = float(default_abs_tol)
        self.default_rel_tol = float(default_rel_tol)
        self.default_lambda_floor = float(default_lambda_floor)
        self.default_lambda_ceil = float(default_lambda_ceil)
        self.adjacency_type = str(adjacency_type)
        self.adjacency_distance = float(adjacency_distance)

        self._T: csr_matrix | None = None
        self._Tt: csr_matrix | None = None
        self._Gt: np.ndarray | None = None
        self._GtG: np.ndarray | None = None
        self._L: np.ndarray | None = None
        self._I: np.ndarray | None = None
        self._factor_key: tuple[float, float, float] | None = None
        self._factor: tuple[np.ndarray, bool] | None = None

        self.last_solver_info: dict[str, Any] = {}
        super().__init__(**kwargs)

    @staticmethod
    def _build_incidence(
        *,
        src: Any,
        n_sources: int,
        adjacency_type: str,
        adjacency_distance: float,
    ) -> csr_matrix:
        adjacency = build_source_adjacency(
            src,
            adjacency_type=adjacency_type,
            adjacency_distance=adjacency_distance,
            verbose=0,
        ).tocoo()

        i = adjacency.row.astype(int)
        j = adjacency.col.astype(int)
        mask = i < j
        i = i[mask]
        j = j[mask]
        n_edges = int(i.size)
        if n_edges == 0:
            raise ValueError("SISSY requires a non-empty source adjacency graph.")

        n_locations = int(adjacency.shape[0])
        if n_locations <= 0 or n_sources % n_locations != 0:
            raise ValueError(
                "SISSY source dimensions are incompatible with adjacency: "
                f"n_sources={n_sources}, n_locations={n_locations}."
            )
        n_orient = n_sources // n_locations

        if n_orient == 1:
            rows = np.repeat(np.arange(n_edges, dtype=int), 2)
            cols = np.empty(2 * n_edges, dtype=int)
            cols[0::2] = i
            cols[1::2] = j
            vals = np.empty(2 * n_edges, dtype=float)
            vals[0::2] = 1.0
            vals[1::2] = -1.0
            return coo_matrix((vals, (rows, cols)), shape=(n_edges, n_sources)).tocsr()

        edge_idx = np.repeat(np.arange(n_edges, dtype=int), n_orient)
        orient_idx = np.tile(np.arange(n_orient, dtype=int), n_edges)
        n_rows = n_edges * n_orient

        col_i = i[edge_idx] * n_orient + orient_idx
        col_j = j[edge_idx] * n_orient + orient_idx

        rows = np.repeat(np.arange(n_rows, dtype=int), 2)
        cols = np.empty(2 * n_rows, dtype=int)
        cols[0::2] = col_i
        cols[1::2] = col_j
        vals = np.empty(2 * n_rows, dtype=float)
        vals[0::2] = 1.0
        vals[1::2] = -1.0
        return coo_matrix((vals, (rows, cols)), shape=(n_rows, n_sources)).tocsr()

    def _get_factor(self, rho1: float, rho2: float, rho3: float) -> tuple[np.ndarray, bool]:
        key = (float(rho1), float(rho2), float(rho3))
        if self._factor_key == key and self._factor is not None:
            return self._factor
        if self._GtG is None or self._L is None or self._I is None:
            raise RuntimeError("SISSY was not initialized. Call make_inverse_operator first.")
        A = float(rho1) * self._GtG + float(rho2) * self._L + float(rho3) * self._I
        self._factor = linalg.cho_factor(A, lower=True, check_finite=False)
        self._factor_key = key
        return self._factor

    def make_inverse_operator(
        self,
        forward,
        *args: Any,
        alpha: str | float = "auto",
        noise_cov: mne.Covariance | None = None,
        adjacency_type: str | None = None,
        adjacency_distance: float | None = None,
        **kwargs: Any,
    ):
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        self.prepare_whitened_forward(noise_cov)

        n_sources = int(self.leadfield.shape[1])
        adj_type = self.adjacency_type if adjacency_type is None else str(adjacency_type)
        adj_dist = (
            self.adjacency_distance
            if adjacency_distance is None
            else float(adjacency_distance)
        )
        self._T = self._build_incidence(
            src=self.forward["src"],
            n_sources=n_sources,
            adjacency_type=adj_type,
            adjacency_distance=adj_dist,
        )
        self._Tt = self._T.T.tocsr()
        self._Gt = self.leadfield.T
        self._GtG = self._Gt @ self.leadfield
        self._L = (self._Tt @ self._T).toarray()
        self._I = np.eye(n_sources, dtype=float)

        self._factor = None
        self._factor_key = None
        self.inverse_operators = []
        return self

    def _run_sissy_admm(
        self,
        x: np.ndarray,
        *,
        lambda_init: float,
        alpha_l1: float,
        rho1: float,
        rho2: float,
        rho3: float,
        max_iter: int,
        abs_tol: float,
        rel_tol: float,
        adaptive_lambda: bool,
        discrepancy_c: float | None,
        lambda_floor: float,
        lambda_ceil: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self._T is None or self._Tt is None or self._Gt is None:
            raise RuntimeError("SISSY was not initialized. Call make_inverse_operator first.")

        G = self.leadfield
        Gt = self._Gt
        T = self._T
        Tt = self._Tt

        rho1 = float(rho1)
        rho2 = float(rho2)
        rho3 = float(rho3)
        if (
            not np.isfinite(rho1)
            or not np.isfinite(rho2)
            or not np.isfinite(rho3)
            or rho1 <= 0
            or rho2 <= 0
            or rho3 <= 0
        ):
            raise ValueError("rho1, rho2, rho3 must be finite positive scalars.")
        max_iter = max(int(max_iter), 1)
        inv_rho1 = 1.0 / rho1
        inv_rho2 = 1.0 / rho2
        inv_rho3 = 1.0 / rho3
        alpha_l1 = max(float(alpha_l1), 0.0)
        lambda_floor = max(float(lambda_floor), 0.0)
        lambda_ceil = max(float(lambda_ceil), lambda_floor)

        factor = self._get_factor(rho1, rho2, rho3)

        n_chans, n_sources = G.shape
        n_time = int(x.shape[1])
        n_edges = int(T.shape[0])

        s = np.zeros((n_sources, n_time), dtype=float)
        a = np.asarray(x, dtype=float).copy()
        y = np.zeros((n_edges, n_time), dtype=float)
        z = np.zeros((n_sources, n_time), dtype=float)
        u = np.zeros((n_chans, n_time), dtype=float)
        v = np.zeros((n_edges, n_time), dtype=float)
        w = np.zeros((n_sources, n_time), dtype=float)

        lambda_curr = float(lambda_init)
        lambda_curr = min(max(lambda_curr, lambda_floor), lambda_ceil)
        discrepancy_c = None if discrepancy_c is None else float(discrepancy_c)
        if adaptive_lambda and (discrepancy_c is None or discrepancy_c <= 0):
            raise ValueError("APESISSY requires discrepancy_c > 0.")

        lambda_path: list[float] = []
        primal_norm = np.inf
        dual_norm = np.inf
        converged = False

        for iteration in range(max_iter):
            a_prev = a.copy()
            y_prev = y.copy()
            z_prev = z.copy()

            rhs = rho1 * (Gt @ (a - inv_rho1 * u))
            rhs += rho2 * (Tt @ (y - inv_rho2 * v))
            rhs += rho3 * (z - inv_rho3 * w)
            s = linalg.cho_solve(factor, rhs, check_finite=False)

            Gs = G @ s
            if adaptive_lambda:
                discrepancy = Gs + inv_rho1 * u - x
                lambda_curr = (rho1 / discrepancy_c) * np.linalg.norm(discrepancy) - rho1
                lambda_curr = min(max(float(lambda_curr), lambda_floor), lambda_ceil)

            lambda_path.append(float(lambda_curr))
            denom = max(lambda_curr + rho1, 1e-15)
            a = (lambda_curr * x + rho1 * (Gs + inv_rho1 * u)) / denom

            Ts = T @ s
            y = _soft_threshold(Ts + inv_rho2 * v, inv_rho2)
            z = _soft_threshold(s + inv_rho3 * w, alpha_l1 * inv_rho3)

            r1 = Gs - a
            r2 = Ts - y
            r3 = s - z
            u += rho1 * r1
            v += rho2 * r2
            w += rho3 * r3

            primal_norm = np.sqrt(
                np.linalg.norm(r1) ** 2 + np.linalg.norm(r2) ** 2 + np.linalg.norm(r3) ** 2
            )
            dual_norm = np.sqrt(
                (rho1 * np.linalg.norm(Gt @ (a - a_prev))) ** 2
                + (rho2 * np.linalg.norm(Tt @ (y - y_prev))) ** 2
                + (rho3 * np.linalg.norm(z - z_prev)) ** 2
            )

            eps_primal = np.sqrt(a.size + y.size + z.size) * abs_tol + rel_tol * max(
                np.linalg.norm(Gs),
                np.linalg.norm(a),
                np.linalg.norm(Ts),
                np.linalg.norm(y),
                np.linalg.norm(s),
                np.linalg.norm(z),
            )
            eps_dual = np.sqrt(s.size) * abs_tol + rel_tol * (
                np.linalg.norm(Gt @ u) + np.linalg.norm(Tt @ v) + np.linalg.norm(w)
            )
            if primal_norm <= eps_primal and dual_norm <= eps_dual:
                converged = True
                break

        info = {
            "n_iter": int(iteration + 1),
            "converged": bool(converged),
            "primal_residual": float(primal_norm),
            "dual_residual": float(dual_norm),
            "lambda_last": float(lambda_curr),
            "lambda_path": np.asarray(lambda_path, dtype=float),
        }
        return s, info

    def apply_inverse_operator(
        self,
        mne_obj,
        *,
        lambda_reg: float | None = None,
        alpha_l1: float | None = None,
        rho1: float | None = None,
        rho2: float | None = None,
        rho3: float | None = None,
        max_iter: int | None = None,
        abs_tol: float | None = None,
        rel_tol: float | None = None,
    ) -> mne.SourceEstimate:
        data = self.unpack_data_obj(mne_obj)
        self.validate_operator_data_compatibility(data)
        x = self._sensor_transform @ data

        s, info = self._run_sissy_admm(
            np.asarray(x, dtype=float),
            lambda_init=self.default_lambda if lambda_reg is None else float(lambda_reg),
            alpha_l1=self.default_alpha_l1 if alpha_l1 is None else float(alpha_l1),
            rho1=self.default_rho1 if rho1 is None else float(rho1),
            rho2=self.default_rho2 if rho2 is None else float(rho2),
            rho3=self.default_rho3 if rho3 is None else float(rho3),
            max_iter=self.default_max_iter if max_iter is None else int(max_iter),
            abs_tol=self.default_abs_tol if abs_tol is None else float(abs_tol),
            rel_tol=self.default_rel_tol if rel_tol is None else float(rel_tol),
            adaptive_lambda=False,
            discrepancy_c=None,
            lambda_floor=self.default_lambda_floor,
            lambda_ceil=self.default_lambda_ceil,
        )
        self.last_solver_info = info
        return self.source_to_object(s)
