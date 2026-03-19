"""Forge: Flex-Optimized Refined Greedy Estimation.

Combines three complementary algorithms:

1. **Greedy-ML detection** — multi-start greedy forward selection with ML
   criterion on a flexible-extent dictionary.  Proven best source detector
   (MLE rank 2 across 6 benchmark datasets).

2. **NL-Champagne refinement** — sparse Bayesian learning on the detected
   support expanded with graph neighbours.  Produces calibrated per-source
   variances (gamma) and per-channel noise (lambda), yielding the best
   spatial-extent estimates (EMD rank 1 across 6 datasets for NL-Champagne).

3. **BIC-adaptive smoothing** — post-hoc graph diffusion whose order is
   selected per-sample via BIC.  Helps for extended/NMM sources while
   leaving focal sources untouched (smoothing order 0 = no smoothing).

The design is *not* a gated hybrid — it is a single pipeline where every
sample passes through all three stages.  The adaptivity comes from the
BIC selection in stages 1 (how many sources) and 3 (how much smoothing).
"""

from __future__ import annotations

import logging

import mne
import numpy as np
from scipy.sparse import csr_matrix, identity as sparse_identity
from scipy.sparse.csgraph import laplacian

from invert.util import build_source_adjacency

from ..base import BaseSolver, SolverMeta
from ..music.greedy_ml import greedy_ml_search

logger = logging.getLogger(__name__)


class SolverForge(BaseSolver):
    """Forge: GreedyML detection → NL-Champagne refinement → adaptive smoothing."""

    meta = SolverMeta(
        slug="forge",
        full_name="Forge (GreedyML + NL-Champagne + Adaptive Smoothing)",
        category="Hybrid",
        description=(
            "Three-stage solver: greedy ML source detection on a flexible-extent "
            "dictionary, NL-Champagne Bayesian refinement on expanded support, "
            "and BIC-adaptive graph diffusion smoothing."
        ),
        references=[
            "Wax, M., & Kailath, T. (1985). Detection of signals by information "
            "theoretic criteria. IEEE Trans. ASSP, 33(2), 387-392.",
            "Cai, C., et al. (2021). Robust estimation of noise for electromagnetic "
            "brain imaging with the champagne algorithm. NeuroImage, 225, 117411.",
        ],
    )

    def __init__(
        self,
        name: str = "Forge",
        *,
        # Greedy-ML detection params
        k_max: int = 5,
        n_starts: int = 50,
        n_refine_iters: int = 3,
        penalty_mode: str = "bic_per_timepoint",
        n_orders: int = 3,
        diffusion_parameter: float = 0.1,
        adjacency_distance: float = 3e-3,
        # Support expansion params
        neighbor_hops: int = 2,
        include_order0: bool = True,
        # NL-Champagne refinement params
        max_iter_nlc: int = 500,
        pruning_thresh: float = 1e-4,
        convergence_criterion: float = 1e-8,
        # Adaptive smoothing params
        smoothing_orders: tuple[int, ...] = (0, 1, 2, 3, 4, 6),
        smoothing_diffusion: float = 0.15,
        **kwargs,
    ) -> None:
        self.name = name
        # Detection
        self._k_max = int(k_max)
        self._n_starts = int(n_starts)
        self._n_refine_iters = int(n_refine_iters)
        self._penalty_mode = str(penalty_mode)
        self._n_orders = int(n_orders)
        self._diffusion_parameter = float(diffusion_parameter)
        self._adjacency_distance = float(adjacency_distance)
        # Expansion
        self._neighbor_hops = int(neighbor_hops)
        self._include_order0 = bool(include_order0)
        # Refinement
        self._max_iter_nlc = int(max_iter_nlc)
        self._pruning_thresh = float(pruning_thresh)
        self._convergence_criterion = float(convergence_criterion)
        # Smoothing
        self._smoothing_orders = tuple(int(o) for o in smoothing_orders)
        self._smoothing_diffusion = float(smoothing_diffusion)
        super().__init__(**kwargs)
        self.require_recompute = True
        self.require_data = True
        self._noise_cov: mne.Covariance | None = None

    def make_inverse_operator(
        self,
        forward,
        mne_obj=None,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
        **kwargs,
    ):
        super().make_inverse_operator(forward, mne_obj, *args, alpha=alpha, **kwargs)
        self._noise_cov = noise_cov
        _ = self.unpack_data_obj(mne_obj)
        self.inverse_operators = []
        return self

    def apply_inverse_operator(self, mne_obj):
        wf = self.prepare_whitened_forward(self._noise_cov)
        data = self.unpack_data_obj(mne_obj)
        self.validate_operator_data_compatibility(data)

        data_w = wf.sensor_transform @ data
        G_w = wf.G_white
        n_eff, n_dipoles = G_w.shape
        T = data_w.shape[1]

        # ============================================================
        # Stage 1: Greedy-ML detection on flexible-extent dictionary
        # ============================================================
        leadfields, gradients = self._build_patch_dictionary(
            G_w, self.forward, self._n_orders,
            self._diffusion_parameter, self._adjacency_distance,
        )
        n_total_orders = len(leadfields)

        G_ext = np.hstack(leadfields)
        n_ext = G_ext.shape[1]

        Gram = G_ext.T @ G_ext
        eps_gram = 1e-10 * np.trace(Gram) / max(n_ext, 1)
        Gram[np.diag_indices_from(Gram)] += eps_gram
        R = G_ext.T @ data_w
        Q = R @ R.T
        total_var = np.sum(data_w ** 2)

        def _excluded_fn(selected):
            blocked = set()
            for e in selected:
                dipole = e % n_dipoles
                for o in range(n_total_orders):
                    blocked.add(o * n_dipoles + dipole)
            return blocked

        selected_ext = greedy_ml_search(
            Gram, Q, total_var, n_eff, T, n_ext,
            self._k_max, self._n_starts, self._n_refine_iters,
            self._penalty_mode, excluded_fn=_excluded_fn,
        )

        # Decode (order, dipole) pairs
        candidates = [
            (e // n_dipoles, e % n_dipoles) for e in selected_ext
        ]

        # ============================================================
        # Stage 2: Expand support + NL-Champagne refinement
        # ============================================================
        expanded = self._expand_candidates(
            candidates, n_dipoles, n_total_orders
        )

        # Build expanded leadfield in whitened space
        L_sel = np.stack(
            [leadfields[int(o)][:, int(d)] for o, d in expanded], axis=1,
        )

        # Run NL-Champagne on reduced problem
        from ..bayesian.nl_champagne import SolverNLChampagne
        refiner = SolverNLChampagne(verbose=getattr(self, "verbose", 0))
        refiner.leadfield = np.asarray(L_sel, dtype=float)
        K_low = refiner.nl_champagne(
            np.asarray(data_w, dtype=float),
            alpha=0.01,
            max_iter=self._max_iter_nlc,
            prune=True,
            pruning_thresh=self._pruning_thresh,
            convergence_criterion=self._convergence_criterion,
        )  # (k_expanded, n_eff)

        # Map back to full source space via diffusion gradients
        grad_matrix = np.stack(
            [gradients[int(o)][int(d)].toarray().ravel()
             for o, d in expanded],
            axis=1,
        )  # (n_dipoles, k_expanded)

        inverse_operator_w = grad_matrix @ K_low  # (n_dipoles, n_eff)
        source_est = inverse_operator_w @ data_w  # (n_dipoles, T)

        # ============================================================
        # Stage 3: BIC-adaptive post-hoc smoothing
        # ============================================================
        source_est = self._adaptive_smooth(
            source_est, data_w, G_w, n_eff, T, total_var,
        )

        return self.source_to_object(source_est)

    # ------------------------------------------------------------------
    # Patch dictionary (same as flex-greedy-ml)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_patch_dictionary(
        G_w, forward, n_orders, diffusion_parameter, adjacency_distance
    ):
        n_dipoles = G_w.shape[1]
        I = np.identity(n_dipoles)

        leadfields = [G_w.copy()]
        gradients = [csr_matrix(I)]

        if n_orders == 0:
            return leadfields, gradients

        adjacency = build_source_adjacency(
            forward["src"],
            adjacency_type="spatial",
            adjacency_distance=adjacency_distance,
            verbose=0,
        )

        LL = laplacian(adjacency)
        smoothing_op = csr_matrix(I - diffusion_parameter * LL)

        for i in range(n_orders):
            S_i = smoothing_op ** (i + 1)
            leadfields.append(G_w @ S_i.toarray())
            gradients.append(gradients[0] @ S_i)

        for i in range(len(gradients)):
            row_sums = gradients[i].sum(axis=1).ravel()
            scaling = 1.0 / np.maximum(
                np.abs(np.asarray(row_sums).ravel()), 1e-12
            )
            gradients[i] = csr_matrix(
                gradients[i].multiply(scaling.reshape(-1, 1))
            )

        return leadfields, gradients

    # ------------------------------------------------------------------
    # Support expansion (from subspace-sbl-plus)
    # ------------------------------------------------------------------
    def _expand_candidates(
        self,
        base_candidates: list[tuple[int, int]],
        n_dipoles: int,
        n_orders: int,
    ) -> list[tuple[int, int]]:
        """Expand detected (order, dipole) candidates with graph neighbors."""
        base = [(int(o), int(d)) for o, d in base_candidates]
        if self._neighbor_hops <= 0 or len(base) <= 1:
            if not self._include_order0:
                return base
            expanded = {(0, d) for _o, d in base}
            expanded.update(base)
            return sorted(expanded, key=lambda x: (x[0], x[1]))

        adjacency = csr_matrix(
            build_source_adjacency(
                self.forward["src"],
                adjacency_type="spatial",
                adjacency_distance=self._adjacency_distance,
                verbose=0,
            )
        )

        def hop_neighbors(seed: int, n_hops: int) -> set[int]:
            visited: set[int] = {seed}
            frontier: set[int] = {seed}
            for _ in range(n_hops):
                new_frontier: set[int] = set()
                for node in frontier:
                    for nb in adjacency[node].indices:
                        nb_i = int(nb)
                        if nb_i not in visited:
                            visited.add(nb_i)
                            new_frontier.add(nb_i)
                frontier = new_frontier
                if not frontier:
                    break
            return visited

        expanded: set[tuple[int, int]] = set()
        for order, dipole in base:
            order_i = max(0, min(int(order), n_orders - 1))
            for d2 in hop_neighbors(int(dipole), self._neighbor_hops):
                if self._include_order0:
                    expanded.add((0, d2))
                expanded.add((order_i, d2))

        return sorted(expanded, key=lambda x: (x[0], x[1]))

    # ------------------------------------------------------------------
    # BIC-adaptive graph diffusion smoothing
    # ------------------------------------------------------------------
    def _adaptive_smooth(
        self,
        source_est: np.ndarray,
        data_w: np.ndarray,
        G_w: np.ndarray,
        n_eff: int,
        T: int,
        total_var: float,
    ) -> np.ndarray:
        if len(self._smoothing_orders) <= 1:
            return source_est

        n_dipoles = source_est.shape[0]

        adjacency = build_source_adjacency(
            self.forward["src"],
            adjacency_type="spatial",
            adjacency_distance=self._adjacency_distance,
            verbose=0,
        )
        LL = laplacian(adjacency)
        I_sp = sparse_identity(n_dipoles, format="csr", dtype=float)

        def _bic_score(X_src):
            residual = data_w - G_w @ X_src
            rss = np.sum(residual ** 2)
            power = np.sum(X_src ** 2, axis=1)
            k = max(1, int(np.sum(power > 1e-10 * power.max())))
            return n_eff * T * np.log(max(rss, 1e-30) / (n_eff * T)) + k * np.log(n_eff * T)

        best_bic = _bic_score(source_est)
        best_est = source_est

        for order in self._smoothing_orders:
            if order == 0:
                continue
            S = (I_sp - self._smoothing_diffusion * LL) ** order
            smoothed = np.asarray(S @ source_est, dtype=float)
            bic = _bic_score(smoothed)
            if bic < best_bic:
                best_bic = bic
                best_est = smoothed

        return best_est
