from __future__ import annotations

import mne
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, laplacian

from invert.util import build_source_adjacency

from ..base import BaseSolver, InverseOperator, SolverMeta


def _invsqrt_psd(M: np.ndarray, *, eps: float = 1e-10) -> np.ndarray:
    """Compute (M + ridge I)^(-1/2) for a symmetric PSD matrix (small n)."""
    M = np.asarray(M, dtype=float)
    M = 0.5 * (M + M.T)
    n = int(M.shape[0])
    ridge = float(eps) * float(np.trace(M)) / max(1, n)
    evals, evecs = np.linalg.eigh(M + ridge * np.eye(n))
    evals = np.maximum(evals, ridge)
    inv_sqrt = (evecs * (1.0 / np.sqrt(evals))[None, :]) @ evecs.T
    return 0.5 * (inv_sqrt + inv_sqrt.T)


def _largest_connected_component(idx: np.ndarray, adjacency) -> np.ndarray:
    """Keep only the largest connected component within idx (if any)."""
    idx = np.asarray(idx, dtype=int).reshape(-1)
    if idx.size <= 1:
        return idx
    sub = adjacency[idx][:, idx]
    n_comp, labels = connected_components(sub, directed=False, return_labels=True)
    if n_comp <= 1:
        return idx
    # Pick largest component; ties broken by keeping lower label.
    sizes = np.bincount(labels, minlength=n_comp)
    keep_label = int(np.argmax(sizes))
    return idx[labels == keep_label]


def _hard_threshold_connected(v: np.ndarray, *, k: int, adjacency) -> np.ndarray:
    """Hard-threshold vector v to k largest |.| entries, then enforce connectedness."""
    v = np.asarray(v, dtype=float).reshape(-1)
    n = int(v.size)
    k = int(np.clip(k, 1, n))
    order = np.argsort(np.abs(v))[::-1][:k]
    keep = _largest_connected_component(order, adjacency)
    out = np.zeros_like(v)
    out[keep] = v[keep]
    return out


class SolverGraphSparsePCA(BaseSolver):
    """Graph-regularized sparse low-rank factorization for salient patches.

    This is a data-adaptive alternative to fixed patch dictionaries: it learns a
    small number of spatial maps u_k (contiguous/smooth) whose sensor topographies
    a_k = G u_k explain the dominant variance in the measurements. It treats
    background as nuisance and focuses on recovering a compact support set.
    """

    meta = SolverMeta(
        acronym="G-SPCA",
        full_name="Graph Sparse PCA (Support)",
        category="Hybrids",
        description=(
            "Learns contiguous/smooth sparse spatial maps directly from data via a "
            "graph-regularized sparse PCA/power method on sensor covariance, then "
            "builds a low-rank inverse operator from these maps."
        ),
        internal=True,
    )

    def __init__(self, name="Graph Sparse PCA (Support)", **kwargs):
        self.name = name
        super().__init__(**kwargs)
        self.require_recompute = True
        self.require_data = True

    def make_inverse_operator(
        self,
        forward,
        mne_obj=None,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
        n_components=4,
        n_iters=25,
        support_size=60,
        smooth_weight=0.15,
        # Optional background whitening (GLS) using a graph-smooth prior.
        bg_whiten=True,
        bg_diffusion_parameter=0.1,
        bg_eps=1e-10,
        adjacency_distance=3e-3,
        **kwargs,
    ):
        super().make_inverse_operator(forward, mne_obj, *args, alpha=alpha, **kwargs)
        wf = self.prepare_whitened_forward(noise_cov)

        Y = wf.sensor_transform @ self.unpack_data_obj(mne_obj)  # (n_eff, T)
        G = wf.G_white  # (n_eff, n_src)
        n_eff, n_src = G.shape
        T = int(Y.shape[1])

        adjacency = build_source_adjacency(
            forward["src"],
            adjacency_type="spatial",
            adjacency_distance=float(adjacency_distance),
            verbose=0,
        ).tocsr()
        L = laplacian(adjacency).astype(float).tocsr()

        sensor_transform = wf.sensor_transform
        if bool(bg_whiten):
            # Background model: y_bg = (I - a*L) w, w iid.
            S_bg = csr_matrix(np.eye(n_src, dtype=float)) - float(bg_diffusion_parameter) * L
            GS = (S_bg.T @ G.T).T  # (n_eff, n_src)
            C_bg = GS @ GS.T
            Cx = (Y @ Y.T) / max(1, T)
            tr_bg = float(np.trace(C_bg))
            tr_x = float(np.trace(Cx))
            scale = (tr_x / tr_bg) if (tr_bg > 0 and tr_x > 0) else 1.0
            W_bg = _invsqrt_psd(scale * C_bg, eps=float(bg_eps))
            Y = W_bg @ Y
            G = W_bg @ G
            sensor_transform = W_bg @ wf.sensor_transform

        n_components = int(np.clip(n_components, 1, min(n_eff, 8)))
        n_iters = int(np.clip(n_iters, 1, 200))
        support_size = int(np.clip(support_size, 1, n_src))
        smooth_weight = float(max(0.0, smooth_weight))

        U_list: list[np.ndarray] = []
        A_list: list[np.ndarray] = []

        Y_res = np.asarray(Y, dtype=float).copy()

        rng = np.random.default_rng(0)
        for _k in range(n_components):
            # Use the current residual covariance.
            C = (Y_res @ Y_res.T) / max(1, T)
            # Init from a random backprojection (helps avoid repeated components).
            u = rng.standard_normal(n_src)
            u /= max(np.linalg.norm(u), 1e-12)

            for _ in range(n_iters):
                a = G @ u  # (n_eff,)
                b = C @ a  # (n_eff,)
                v = G.T @ b  # (n_src,)
                if smooth_weight > 0:
                    v = v - smooth_weight * (L @ v)
                u = _hard_threshold_connected(v, k=support_size, adjacency=adjacency)
                nu = float(np.linalg.norm(u))
                if not np.isfinite(nu) or nu <= 0:
                    u = rng.standard_normal(n_src)
                    nu = float(np.linalg.norm(u))
                u = u / max(nu, 1e-12)

            a = G @ u
            denom = float(a @ a)
            if not np.isfinite(denom) or denom <= 1e-20:
                continue

            # Deflate rank-1 sensor component.
            c = (a.T @ Y_res) / denom  # (T,)
            Y_res = Y_res - np.outer(a, c)

            U_list.append(u.copy())
            A_list.append(a.copy())

        if not U_list:
            # Fallback to identity-like behavior: pick max-norm leadfield column.
            j = int(np.argmax(np.sum(G * G, axis=0)))
            inverse_operator = np.zeros((n_src, n_eff), dtype=float)
            denom = float(G[:, j].T @ G[:, j])
            if denom > 0:
                inverse_operator[j, :] = G[:, j] / denom
            self.inverse_operators = [
                InverseOperator(inverse_operator @ sensor_transform, self.name)
            ]
            return self

        U = np.stack(U_list, axis=1)  # (n_src, K)
        A = np.stack(A_list, axis=1)  # (n_eff, K)

        # Low-rank inverse: x_hat = U * pinv(A) * Y
        pinvA = np.linalg.pinv(A)  # (K, n_eff)
        inverse_operator = (U @ pinvA) @ sensor_transform  # (n_src, n_chans)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name)]
        return self
