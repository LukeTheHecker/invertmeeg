from __future__ import annotations

import logging
from collections.abc import Iterable
from copy import deepcopy
from typing import Any

import numpy as np

_TORCH_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    import torch  # type: ignore[import-not-found]
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc

from ...simulate.spatial import build_adjacency, build_spatial_basis
from ..base import SolverMeta
from .covcnn_kl import SolverCovCNNKL
from .torch_utils import get_torch_device

logger = logging.getLogger(__name__)


def _max_sources_from_config(n_sources: int | tuple[int, int]) -> int:
    if isinstance(n_sources, (tuple, list)):
        return int(n_sources[1])
    return int(n_sources)


def _parse_orders(n_orders: int | tuple[int, int]) -> tuple[int, int]:
    """Match SimulationGenerator's (min_order, max_order) parsing."""
    if isinstance(n_orders, (tuple, list)):
        min_order, max_order = int(n_orders[0]), int(n_orders[1])
        if min_order == max_order:
            max_order += 1
        return min_order, max_order
    return 0, int(n_orders)


def _ridge_fit(A: np.ndarray, Y: np.ndarray, *, ridge: float) -> np.ndarray:
    """Solve min_S ||Y - A S||_F^2 + λ ||S||_F^2 with λ scaled by tr(A^T A)."""
    n_atoms = int(A.shape[1])
    if n_atoms == 0:
        return np.zeros((0, Y.shape[1]), dtype=Y.dtype)
    G = A.T @ A
    tr = float(np.trace(G))
    scale = tr / float(n_atoms) if np.isfinite(tr) and tr > 0 else 1.0
    reg = float(ridge) * float(scale)
    return np.linalg.solve(G + reg * np.eye(n_atoms), A.T @ Y)


class SolverCovCNNKLFlexOMP(SolverCovCNNKL):
    """NN-guided FLEX-OMP: iterative explain-away using CovCNN-KL as scorer.

    High-level idea
    ---------------
    1) Use the CovCNN-KL network to score likely source *centers* from the
       (residual) sensor covariance.
    2) Convert each chosen center into a basis candidate (dipole for n_orders==0,
       or patch-like basis atoms for n_orders>0).
    3) Do an OMP-style *joint* refit of timecourses on the selected atoms and
       update the residual.
    4) (Optional) refinement pass inspired by generalized_iterative.py: revisit
       each selected atom and allow replacement given the others.
    """

    meta = SolverMeta(
        acronym="CovCNN-KL-FLEXOMP",
        full_name="CovCNN-KL (NN-guided FLEX-OMP)",
        category="Neural Networks",
        description=(
            "Uses a CovCNN-KL network as an iterative scorer inside an OMP-style "
            "explain-away loop (with optional generalized-iterative refinement). "
            "For patch datasets, selects patch-shaped spatial basis atoms."
        ),
        references=["Lukas Hecker 2026, unpublished"],
    )

    def __init__(
        self,
        name: str = "CovCNN-KL (FLEX-OMP)",
        *,
        reduce_rank: bool = False,
        use_shrinkage: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            reduce_rank=reduce_rank,
            use_shrinkage=use_shrinkage,
            **kwargs,
        )
        self._adjacency: Any = None
        self._basis_dense: Any = None
        self._atoms: Any = None
        self._n_dipoles: int | None = None
        self._n_orders_included: int | None = None
        self._min_order: int | None = None

    def make_inverse_operator(  # type: ignore[override]
        self,
        forward,
        simulation_config,
        *args,
        max_iter: int | None = None,
        min_relative_improvement: float = 0.01,
        ridge: float = 1e-6,
        center_topk: int = 5,
        nms_hops: int = 0,
        refine_solution: bool = True,
        refine_max_iter: int = 2,
        blend_with_kl: bool = True,
        blend_weight_patch: float = 0.85,
        blend_weight_dipole: float = 0.5,
        prior_from_flex: bool = False,
        prior_power: float = 1.0,
        alpha: str | float = "auto",
        **kwargs,
    ):
        # Train the CovCNN-KL model as usual.
        super().make_inverse_operator(
            forward,
            simulation_config,
            *args,
            alpha=alpha,
            **kwargs,
        )

        self.max_iter = (
            int(max_iter)
            if max_iter is not None
            else _max_sources_from_config(getattr(simulation_config, "n_sources", 5))
        )
        self.min_relative_improvement = float(min_relative_improvement)
        self.ridge = float(ridge)
        self.center_topk = int(center_topk)
        self.nms_hops = int(nms_hops)
        self.refine_solution = bool(refine_solution)
        self.refine_max_iter = int(refine_max_iter)
        self.blend_with_kl = bool(blend_with_kl)
        self.blend_weight_patch = float(blend_weight_patch)
        self.blend_weight_dipole = float(blend_weight_dipole)
        self.prior_from_flex = bool(prior_from_flex)
        self.prior_power = float(prior_power)

        # Build candidate basis and sensor-space atoms consistent with the simulator config.
        n_dipoles = int(self.leadfield.shape[1])
        min_order, max_order = _parse_orders(getattr(simulation_config, "n_orders", 0))
        adjacency = build_adjacency(
            forward, verbose=getattr(simulation_config, "verbose", 0)
        )
        _sources_sparse, sources_dense, _gradient = build_spatial_basis(
            adjacency,
            n_dipoles,
            min_order,
            max_order,
            diffusion_smoothing=bool(
                getattr(simulation_config, "diffusion_smoothing", True)
            ),
            diffusion_parameter=float(
                getattr(simulation_config, "diffusion_parameter", 0.1)
            ),
        )

        self._adjacency = adjacency
        self._basis_dense = sources_dense.astype(
            np.float32, copy=False
        )  # (n_candidates, n_dipoles)
        self._n_dipoles = int(n_dipoles)
        self._n_orders_included = int(self._basis_dense.shape[0] // n_dipoles)
        self._min_order = int(min_order)

        # Sensor-space atoms A = L @ B^T (n_channels, n_candidates).
        self._atoms = (self.leadfield @ self._basis_dense.T).astype(
            np.float64, copy=False
        )
        return self

    def _predict_center_scores(
        self, Y: np.ndarray, *, prior: np.ndarray | None
    ) -> np.ndarray:
        """Return per-dipole center scores from CovCNN-KL on covariance(Y)."""
        if _TORCH_IMPORT_ERROR is not None:  # pragma: no cover
            raise ImportError(
                "PyTorch is required for neural-network solvers. "
                'Install it via `pip install "invertmeeg[ann]"` (or install `torch` directly).'
            ) from _TORCH_IMPORT_ERROR
        if self.model is None:
            raise RuntimeError("Model not initialized.")

        y = deepcopy(Y)
        y = y - y.mean(axis=1, keepdims=True)
        C = self.compute_covariance(y)
        max_abs = float(np.abs(C).max())
        if max_abs > 0:
            C = C / max_abs
        C = C[np.newaxis, np.newaxis, :, :].astype(np.float32, copy=False)

        device = self.device or get_torch_device()
        self.model.eval()
        with torch.no_grad():
            logits = self.model(
                torch.as_tensor(C, dtype=torch.float32, device=device)
            ) / float(self.temperature)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]

        scores = probs.astype(np.float64, copy=False)
        if prior is not None:
            prior = np.asarray(prior, dtype=np.float64)
            if prior.shape != scores.shape:
                raise ValueError(
                    f"prior shape {prior.shape} does not match scores shape {scores.shape}"
                )
            prior_max = float(np.max(prior))
            if np.isfinite(prior_max) and prior_max > 0:
                scores = scores * (prior / prior_max)

        return scores

    def _forbidden_from_centers(self, centers: Iterable[int]) -> set[int]:
        """Expand forbidden centers by nms_hops over the adjacency graph."""
        centers_set = {int(c) for c in centers}
        adjacency = getattr(self, "_adjacency", None)
        if self.nms_hops <= 0 or adjacency is None:
            return centers_set

        from scipy import sparse

        A = sparse.csr_matrix(adjacency)
        frontier = set(centers_set)
        out = set(centers_set)
        for _ in range(int(self.nms_hops)):
            if not frontier:
                break
            next_frontier: set[int] = set()
            for c in frontier:
                row = A.getrow(int(c))
                neigh = row.indices.tolist()
                for n in neigh:
                    if n not in out:
                        out.add(int(n))
                        next_frontier.add(int(n))
            frontier = next_frontier
        return out

    def _candidate_indices_for_center(self, center: int) -> list[int]:
        n_dipoles = getattr(self, "_n_dipoles", None)
        n_orders_included = getattr(self, "_n_orders_included", None)
        if n_dipoles is None or n_orders_included is None:
            raise RuntimeError(
                "Dictionary not initialized; call make_inverse_operator() first."
            )
        n = int(n_dipoles)
        return [int(o * n + center) for o in range(int(n_orders_included))]

    def _fit_support(
        self, Y: np.ndarray, support: list[int]
    ) -> tuple[np.ndarray, np.ndarray, float]:
        if len(support) == 0:
            R = Y
            return np.zeros((0, Y.shape[1]), dtype=np.float64), R, float(np.sum(R * R))
        atoms = getattr(self, "_atoms", None)
        if atoms is None:
            raise RuntimeError(
                "Dictionary not initialized; call make_inverse_operator() first."
            )
        A = atoms[:, support]
        S = _ridge_fit(A, Y, ridge=self.ridge)
        R = Y - (A @ S)
        return S, R, float(np.sum(R * R))

    def _choose_next_candidate(
        self,
        Y: np.ndarray,
        *,
        residual: np.ndarray,
        support: list[int],
        banned_centers: set[int],
        prior: np.ndarray | None,
    ) -> tuple[int | None, float]:
        scores = self._predict_center_scores(residual, prior=prior)
        if len(banned_centers) > 0:
            scores = scores.copy()
            for c in banned_centers:
                if 0 <= int(c) < scores.shape[0]:
                    scores[int(c)] = -np.inf

        # Candidate centers to evaluate.
        n_centers = int(scores.shape[0])
        k = max(1, min(int(self.center_topk), n_centers))
        centers_sorted = np.argsort(scores)[::-1]
        centers_eval: list[int] = []
        for c in centers_sorted:
            if not np.isfinite(scores[int(c)]):
                continue
            centers_eval.append(int(c))
            if len(centers_eval) >= k:
                break
        if len(centers_eval) == 0:
            return None, float("inf")

        # Evaluate best (center, order) by actual residual reduction.
        best_cand = None
        best_norm = float("inf")
        for center in centers_eval:
            for cand in self._candidate_indices_for_center(center):
                if cand in support:
                    continue
                _S, _R, norm2 = self._fit_support(Y, support + [cand])
                if norm2 < best_norm:
                    best_norm = norm2
                    best_cand = int(cand)
        return best_cand, best_norm

    def _refine_support(
        self, Y: np.ndarray, support: list[int], *, prior: np.ndarray | None
    ) -> list[int]:
        if len(support) <= 1:
            return support

        cur = list(support)
        for _iter in range(int(self.refine_max_iter)):
            changed = False
            for j in range(len(cur)):
                keep = [cur[i] for i in range(len(cur)) if i != j]
                _S, R, _norm2 = self._fit_support(Y, keep)

                n_dipoles = getattr(self, "_n_dipoles", None)
                if n_dipoles is None:
                    raise RuntimeError(
                        "Dictionary not initialized; call make_inverse_operator() first."
                    )
                other_centers = {int(c % int(n_dipoles)) for c in keep}
                banned = self._forbidden_from_centers(other_centers)
                cand, _cand_norm = self._choose_next_candidate(
                    Y, residual=R, support=keep, banned_centers=banned, prior=prior
                )
                if cand is None:
                    continue
                if cand != cur[j]:
                    cur[j] = int(cand)
                    changed = True

            if not changed:
                break
        return cur

    def _prior_from_support(self, support: list[int]) -> np.ndarray:
        basis_dense = getattr(self, "_basis_dense", None)
        n_dipoles = getattr(self, "_n_dipoles", None)
        if basis_dense is None or n_dipoles is None:
            raise RuntimeError(
                "Dictionary not initialized; call make_inverse_operator() first."
            )
        if len(support) == 0:
            return np.zeros((int(n_dipoles),), dtype=np.float64)
        prior_vec = (
            basis_dense[np.asarray(support, dtype=int)]
            .sum(axis=0)
            .astype(np.float64, copy=False)
        )
        if getattr(self, "prior_power", 1.0) != 1.0:
            prior_vec = prior_vec ** float(self.prior_power)
        max_val = float(np.max(prior_vec))
        if np.isfinite(max_val) and max_val > 0:
            prior_vec = prior_vec / max_val
        return prior_vec

    def _apply_flexomp_only(
        self,
        data: np.ndarray,
        *,
        prior_arr: np.ndarray | None,
    ) -> tuple[np.ndarray, list[int]]:
        atoms = getattr(self, "_atoms", None)
        basis_dense = getattr(self, "_basis_dense", None)
        n_dipoles = getattr(self, "_n_dipoles", None)
        if atoms is None or basis_dense is None or n_dipoles is None:
            raise RuntimeError(
                "Dictionary not initialized; call make_inverse_operator() first."
            )

        Y = deepcopy(data).astype(np.float64, copy=False)
        Y = Y - Y.mean(axis=1, keepdims=True)

        support: list[int] = []
        _S, R, prev_norm2 = self._fit_support(Y, support)
        base_norm2 = (
            float(prev_norm2) if np.isfinite(prev_norm2) and prev_norm2 > 0 else 1.0
        )

        max_iter = max(1, int(self.max_iter))
        for _it in range(max_iter):
            selected_centers = {int(c % int(n_dipoles)) for c in support}
            banned = self._forbidden_from_centers(selected_centers)

            cand, cand_norm2 = self._choose_next_candidate(
                Y, residual=R, support=support, banned_centers=banned, prior=prior_arr
            )
            if cand is None:
                break

            rel_impr = (prev_norm2 - cand_norm2) / float(
                prev_norm2 if prev_norm2 > 0 else base_norm2
            )
            if not np.isfinite(rel_impr) or rel_impr < float(
                self.min_relative_improvement
            ):
                break

            support.append(int(cand))
            _S, R, prev_norm2 = self._fit_support(Y, support)

        if self.refine_solution and len(support) > 1:
            support = self._refine_support(Y, support, prior=prior_arr)

        S_hat, _R, _norm2 = self._fit_support(Y, support)
        if len(support) == 0:
            return np.zeros((int(n_dipoles), Y.shape[1]), dtype=np.float64), support

        B_sel = basis_dense[np.asarray(support, dtype=int)]  # (m, n_dipoles)
        X_hat = (B_sel.T @ S_hat).astype(np.float64, copy=False)
        return X_hat, support

    def apply_model(self, data: np.ndarray, prior=None) -> np.ndarray:  # type: ignore[override]
        n_dipoles = getattr(self, "_n_dipoles", None)
        if n_dipoles is None:
            raise RuntimeError(
                "Dictionary not initialized; call make_inverse_operator() first."
            )

        prior_arr = None
        if prior is not None:
            prior_arr = np.asarray(prior, dtype=np.float64).ravel()
            if prior_arr.shape[0] != int(n_dipoles):
                raise ValueError(
                    f"prior must have shape ({int(n_dipoles)},), got {prior_arr.shape}"
                )

        X_flex, support = self._apply_flexomp_only(data, prior_arr=prior_arr)
        if not getattr(self, "blend_with_kl", True):
            return X_flex

        # Optionally turn the FLEX-OMP support into a spatial prior for the KL solver.
        combined_prior = prior_arr
        if getattr(self, "prior_from_flex", True) and len(support) > 0:
            flex_prior = self._prior_from_support(support)
            if combined_prior is None:
                combined_prior = flex_prior
            else:
                combined_prior = flex_prior * combined_prior

        X_kl = super().apply_model(data, prior=combined_prior)

        is_patch = bool(getattr(self, "_n_orders_included", 1) > 1)
        w = float(self.blend_weight_patch if is_patch else self.blend_weight_dipole)
        w = float(np.clip(w, 0.0, 1.0))
        return (w * X_kl + (1.0 - w) * X_flex).astype(np.float64, copy=False)
