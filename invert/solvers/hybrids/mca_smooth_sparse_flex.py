from __future__ import annotations

import mne
import numpy as np
from scipy.sparse import csc_matrix, identity
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import splu

from invert.util import build_source_adjacency

from ..base import BaseSolver, InverseOperator, SolverMeta
from ..music.flex_greedy_ml import SolverFlexGreedyML
from ..music.greedy_ml import greedy_ml_search


class SolverMCASmoothSparseFLEX(BaseSolver):
    """Morphological component analysis (MCA): smooth background + sparse patches.

    Goal: treat smooth ongoing activity as nuisance and recover salient patch
    support. Uses an alternating scheme:
      1) estimate smooth background y_bg with a graph-Laplacian Tikhonov prior
      2) estimate sparse patch component y_patch via FLEX-GreedyML on residual

    This is intentionally "math-first": it matches the simulator's mixture story
    (smooth background + localized patches) instead of forcing everything into a
    sparse atom model.
    """

    meta = SolverMeta(
        acronym="MCA-Smooth+FLEX",
        full_name="MCA Smooth Background + FLEX-GreedyML Patches",
        category="Hybrids",
        description=(
            "Alternating decomposition into a graph-smooth background component and "
            "a sparse patch component estimated with FLEX-GreedyML on the residual."
        ),
        internal=True,
    )

    def __init__(self, name="MCA Smooth+FLEX", **kwargs):
        self.name = str(name)
        super().__init__(**kwargs)
        self.require_recompute = True
        self.require_data = True
        self._source_mat: np.ndarray | None = None

    @staticmethod
    def _build_laplacian(forward, *, adjacency_distance: float) -> csc_matrix:
        adjacency = build_source_adjacency(
            forward["src"],
            adjacency_type="spatial",
            adjacency_distance=float(adjacency_distance),
            verbose=0,
        )
        L = laplacian(adjacency).astype(float)
        # Normalize so mean diagonal is ~1 for stable lambda defaults.
        diag = np.asarray(L.diagonal(), dtype=float)
        scale = float(np.mean(diag)) if diag.size else 1.0
        if np.isfinite(scale) and scale > 0:
            L = L / scale
        return csc_matrix(L)

    @staticmethod
    def _smooth_bg_estimate(
        G_w: np.ndarray,
        Y_w: np.ndarray,
        *,
        L: csc_matrix,
        lam: float,
        ridge: float,
        jitter: float = 1e-9,
    ) -> np.ndarray:
        """Compute y_bg using Woodbury with sparse Laplacian prior."""
        n_eff, n_src = G_w.shape
        lam = float(max(lam, 0.0))
        ridge = float(max(ridge, 1e-12))

        M = lam * L + float(ridge) * identity(n_src, format="csc", dtype=float)
        # Sparse direct solve is cheap at ico3 sizes; fallback can be added later.
        lu = splu(M)

        # X = M^{-1} G^T  (n_src x n_eff)
        X = lu.solve(G_w.T)
        K = G_w @ X  # (n_eff x n_eff)
        A = np.eye(n_eff, dtype=float) + K
        A.flat[:: n_eff + 1] += float(jitter)

        Z = np.linalg.solve(A, Y_w)  # (n_eff x T)
        y_bg = X @ Z  # (n_src x T)
        return np.asarray(y_bg, dtype=float)

    @staticmethod
    def _flex_greedy_patch_estimate(
        G_w: np.ndarray,
        Y_res: np.ndarray,
        forward,
        *,
        k_max: int,
        n_starts: int,
        n_refine_iters: int,
        penalty_mode: str,
        n_orders: int,
        diffusion_parameter: float,
        adjacency_distance: float,
    ) -> np.ndarray:
        """Run FLEX-GreedyML selection on residual and return y_patch (n_src x T)."""
        n_eff, n_dipoles = G_w.shape
        T = int(Y_res.shape[1])

        leadfields, gradients = SolverFlexGreedyML._build_patch_dictionary(
            G_w,
            forward,
            int(n_orders),
            float(diffusion_parameter),
            float(adjacency_distance),
        )
        n_total_orders = len(leadfields)
        G_ext = np.hstack(leadfields)
        n_ext = int(G_ext.shape[1])

        Gram = G_ext.T @ G_ext
        eps_gram = 1e-10 * np.trace(Gram) / max(n_ext, 1)
        Gram[np.diag_indices_from(Gram)] += eps_gram

        R = G_ext.T @ Y_res
        Q = R @ R.T
        total_var = float(np.sum(Y_res**2))

        def _excluded_fn(selected):
            blocked = set()
            for e in selected:
                dipole = int(e % n_dipoles)
                for o in range(n_total_orders):
                    blocked.add(o * n_dipoles + dipole)
            return blocked

        selected_ext = greedy_ml_search(
            Gram,
            Q,
            total_var,
            int(n_eff),
            int(T),
            n_ext,
            int(k_max),
            int(n_starts),
            int(n_refine_iters),
            str(penalty_mode),
            excluded_fn=_excluded_fn,
        )

        selected_ext_arr = np.asarray(selected_ext, dtype=np.intp)
        G_sel = G_ext[:, selected_ext_arr]
        inv_ext = np.linalg.lstsq(G_sel, np.eye(n_eff), rcond=None)[0]  # (K, n_eff)

        # Map from selected atoms back to full source space.
        sel_orders = [int(e // n_dipoles) for e in selected_ext]
        sel_dipoles = [int(e % n_dipoles) for e in selected_ext]
        M_map = np.zeros((n_dipoles, len(selected_ext)), dtype=float)
        for i, (o, d) in enumerate(zip(sel_orders, sel_dipoles, strict=False)):
            M_map[:, i] = gradients[o][d].toarray().ravel()

        W_patch = M_map @ inv_ext  # (n_src, n_eff)
        return W_patch @ Y_res

    def make_inverse_operator(
        self,
        forward,
        mne_obj=None,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
        # Decomposition hyperparameters
        n_mca_iters: int = 2,
        bg_lambda: float = 10.0,
        bg_ridge: float = 1.0,
        bg_adjacency_distance: float = 3e-3,
        # Patch estimator hyperparameters (FLEX-GreedyML on residual)
        k_max: int = 5,
        n_starts: int = 30,
        n_refine_iters: int = 3,
        penalty_mode: str = "bic_support_only",
        n_orders: int = 3,
        diffusion_parameter: float = 0.12,
        patch_adjacency_distance: float = 3e-3,
        **kwargs,
    ):
        super().make_inverse_operator(forward, mne_obj, *args, alpha=alpha, **kwargs)
        wf = self.prepare_whitened_forward(noise_cov)

        Y = self.unpack_data_obj(mne_obj)
        Y_w = wf.sensor_transform @ Y  # (n_eff, T)
        G_w = wf.G_white  # (n_eff, n_src)
        n_eff, n_src = G_w.shape

        L = self._build_laplacian(forward, adjacency_distance=float(bg_adjacency_distance))

        y_patch = np.zeros((n_src, int(Y_w.shape[1])), dtype=float)
        y_bg = np.zeros_like(y_patch)

        for _ in range(int(max(1, n_mca_iters))):
            # Background given current patch.
            Y_bg_fit = Y_w - (G_w @ y_patch)
            y_bg = self._smooth_bg_estimate(
                G_w,
                Y_bg_fit,
                L=L,
                lam=float(bg_lambda),
                ridge=float(bg_ridge),
            )
            # Patch given background.
            Y_res = Y_w - (G_w @ y_bg)
            y_patch = self._flex_greedy_patch_estimate(
                G_w,
                Y_res,
                forward,
                k_max=int(k_max),
                n_starts=int(n_starts),
                n_refine_iters=int(n_refine_iters),
                penalty_mode=str(penalty_mode),
                n_orders=int(n_orders),
                diffusion_parameter=float(diffusion_parameter),
                adjacency_distance=float(patch_adjacency_distance),
            )

        # We only expose the patch component (support-focused).
        self._source_mat = np.asarray(y_patch, dtype=float)

        # Provide a dummy operator for API consistency (not used by apply override).
        inv = np.zeros((n_src, n_eff), dtype=float) @ wf.sensor_transform
        self.inverse_operators = [InverseOperator(inv, self.name)]
        return self

    def apply_inverse_operator(self, mne_obj) -> mne.SourceEstimate:  # type: ignore[override]
        if self._source_mat is None:
            return super().apply_inverse_operator(mne_obj)
        return self.source_to_object(self._source_mat)

