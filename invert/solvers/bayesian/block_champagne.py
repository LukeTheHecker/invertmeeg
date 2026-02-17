"""Block-Champagne (local overlapping blocks) sparse Bayesian solver.

This implements the "compute-ready" equations provided by the user (Eq. 1–22),
in a practical form compatible with Invert's solver API.

Scope (current implementation)
------------------------------
- Local overlapping blocks with constant within-block correlation ``r`` (Eq. 2–4).
- Convexity/MM-style block gamma updates (Eq. 16).
- Diagonal residual noise learning (Eq. 18), akin to NL-Champagne.
- Optional augmented noise-activity variables ``E`` with fixed prior variance
  (Eq. 1, 6, 9). (No learning rule for ``v`` was provided in the equations.)

Performance note (exact algebraic optimization)
----------------------------------------------
The paper expresses the model using an expanded latent variable ``hatZ`` with
leadfield ``L_hat = L @ (G M)`` and a diagonal prior on ``hatZ`` (Eq. 7–10).
Building ``L_hat`` explicitly can be extremely expensive because its number of
columns scales with the sum of block sizes.

For the special case implemented here (constant-correlation blocks ``C_i(r)``
and scalar ``gamma_i`` repeated within each block), the same sensor covariance
``Sigma_b`` can be computed *exactly* without constructing ``L_hat``:

Let ``H`` be the neighborhood incidence matrix where ``H[i, j] = 1`` iff voxel
``j`` is in the block of voxel ``i`` (including ``i`` itself). Let ``L`` be the
(projected) leadfield with columns ``l_j``. Define:

  u_i = sum_{j in block(i)} l_j   (stacked as columns in U)
  w_j = sum_{i: j in block(i)} gamma_i   (w = H.T @ gamma)

Then (using C_i = (1-r)I + r 11^T) we have:

  sum_i gamma_i * L_block(i) C_i L_block(i)^T
    = (1-r) * L diag(w) L^T  +  r * U diag(gamma) U^T

This reduces per-iteration complexity by ~factor (average block size) while
preserving the math exactly.

Not implemented (yet)
---------------------
- Connectome (remote) covariance blocks ``Psi`` (Eq. 5, 15, 17).
- EM updates for ``C_i`` and correlation ``r`` (Eq. 21–22).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mne
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

from invert.util import build_source_adjacency

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _BlockGeometry:
    """Precomputed block geometry and mapping for Block-Champagne."""

    H: csr_matrix  # (n_sources, n_sources) neighborhood incidence
    H_T: csc_matrix  # H.T for fast H.T @ x and H.T @ M
    block_sizes: np.ndarray  # (n_sources,) sizes of each block


def _chol_constant_corr(p: int, r: float, *, jitter: float) -> np.ndarray:
    """Cholesky factor of a constant-correlation matrix of size p x p."""
    if p <= 0:
        raise ValueError(f"p must be positive, got {p}")
    r = float(r)
    if p == 1:
        return np.ones((1, 1), dtype=float)
    # PSD condition for constant-correlation: -1/(p-1) < r < 1.
    r_min = -1.0 / float(p - 1) + 1e-12
    r = float(np.clip(r, r_min, 1.0 - 1e-12))
    C = (1.0 - r) * np.eye(p, dtype=float) + r * np.ones((p, p), dtype=float)
    C.flat[:: p + 1] += float(jitter)
    try:
        return np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        # If r is very close to 1, numerical issues can occur; add more jitter.
        C.flat[:: p + 1] += 10.0 * float(jitter)
        return np.linalg.cholesky(C)


def _safe_slogdet_sym(A: np.ndarray) -> float:
    """Return log|A| for a symmetric matrix with a safe fallback."""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sign, log_det = np.linalg.slogdet(A)
    if sign <= 0 or not np.isfinite(log_det):
        return float(np.finfo(float).max / 2)
    return float(log_det)


class SolverBlockChampagne(BaseSolver):
    """Block-Champagne: Champagne with local overlapping blocks and noise learning."""

    meta = SolverMeta(
        slug="block-champagne",
        full_name="Block-Champagne",
        category="Bayesian",
        description=(
            "Block-Champagne sparse Bayesian learning with overlapping local blocks "
            "(local homogeneity prior) and diagonal residual noise learning."
        ),
        references=[
            "Feng, Z., Guan, C., & Sun, Y. (2025). Block-Champagne: A Novel Bayesian Framework for Imaging Extended E/MEG Source. IEEE Transactions on Medical Imaging.",
        ],
    )

    def __init__(
        self,
        name: str = "BlockChampagne",
        *,
        adjacency_type: str = "spatial",
        adjacency_distance: float = 3e-3,
        r: float = 0.95,
        augment_noise_activity: bool = False,
        v_noise_activity: float = 1.0,
        max_iter: int = 500,
        pruning_thresh: float = 1e-3,
        convergence_criterion: float = 1e-8,
        damping: float = 0.0,
        eps: float = 1e-15,
        rank_tol: float = 1e-12,
        **kwargs,
    ) -> None:
        self.name = str(name)
        self.adjacency_type = str(adjacency_type)
        self.adjacency_distance = float(adjacency_distance)
        self.r = float(r)
        self.augment_noise_activity = bool(augment_noise_activity)
        self.v_noise_activity = float(v_noise_activity)
        self.max_iter = int(max_iter)
        self.pruning_thresh = float(pruning_thresh)
        self.convergence_criterion = float(convergence_criterion)
        self.damping = float(damping)
        self.eps = float(eps)
        self.rank_tol = float(rank_tol)

        self._block_geometry: _BlockGeometry | None = None
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def make_inverse_operator(
        self,
        forward,
        mne_obj=None,
        *args,
        alpha: str | float = "auto",
        noise_cov: mne.Covariance | None = None,
        **kwargs,
    ):
        super().make_inverse_operator(forward, mne_obj, *args, alpha=alpha, **kwargs)

        # SSP projection only (no whitening). Block-Champagne learns a diagonal
        # residual noise covariance internally (Eq. 18).
        wf = self.prepare_whitened_forward(None)
        self._sensor_projector = wf.projector

        data = self.unpack_data_obj(mne_obj)
        data_proj = wf.sensor_transform @ np.asarray(data, dtype=float)
        leadfield_proj = np.asarray(wf.G_white, dtype=float)

        # Data centering helps identifiability for noise learning variants.
        data_proj = data_proj - data_proj.mean(axis=1, keepdims=True)

        data_cov = self.data_covariance(data_proj, center=True, ddof=1)
        # Block-Champagne learns diagonal noise (beta) internally via
        # Convexity/sqrt updates, converging to the same solution regardless
        # of the initial noise level.  A single alpha avoids a wasted grid
        # search and spurious edge-of-grid GCV warnings.
        if self.alpha == "auto":
            n_chans = data_proj.shape[0]
            self.alphas = [float(np.trace(data_cov) / n_chans)]
        else:
            self.get_alphas(reference=data_cov)

        geom = self._prepare_blocks(n_sources=leadfield_proj.shape[1])
        # U = L @ H.T, where each u_i is the sum of leadfield columns in block(i).
        # Compute once per make_inverse_operator and reuse across alphas.
        U = (geom.H @ leadfield_proj.T).T  # (n_chans, n_sources)

        inverse_operators: list[np.ndarray] = []
        for a in self.alphas:
            W_proj = self._fit_block_champagne(
                B=data_proj,
                L=leadfield_proj,
                U=U,
                geom=geom,
                alpha=float(a),
            )
            inverse_operators.append(W_proj @ wf.sensor_transform)

        self.inverse_operators = [
            InverseOperator(op, self.name) for op in inverse_operators
        ]
        return self

    # ------------------------------------------------------------------
    # Block geometry
    # ------------------------------------------------------------------
    def _prepare_blocks(self, *, n_sources: int) -> _BlockGeometry:
        if self._block_geometry is not None:
            if self._block_geometry.H.shape[0] == int(n_sources):
                return self._block_geometry

        adjacency = build_source_adjacency(
            self.forward["src"],
            adjacency_type=self.adjacency_type,
            adjacency_distance=self.adjacency_distance,
            verbose=0,
        ).tocsr()
        if adjacency.shape[0] != int(n_sources):
            raise ValueError(
                f"Adjacency shape {adjacency.shape} does not match n_sources={n_sources}."
            )

        block_sizes = np.empty(n_sources, dtype=int)
        blocks: list[np.ndarray] = []
        for i in range(n_sources):
            start, end = adjacency.indptr[i], adjacency.indptr[i + 1]
            neigh = adjacency.indices[start:end].astype(int, copy=False)
            block = np.unique(np.concatenate([np.array([i], dtype=int), neigh]))
            blocks.append(block)
            block_sizes[i] = int(block.size)

        # Build neighborhood incidence matrix H (rows: blocks i, cols: voxels j).
        # H[i, j] = 1 if j is in block(i).
        rows: list[int] = []
        cols: list[int] = []
        for i, block in enumerate(blocks):
            rows.extend([i] * int(block.size))
            cols.extend(block.tolist())
        data = np.ones(len(rows), dtype=float)
        H = csr_matrix(
            (data, (np.asarray(rows), np.asarray(cols))), shape=(n_sources, n_sources)
        )
        H.sum_duplicates()
        H.eliminate_zeros()
        H_T = H.T.tocsc()

        geom = _BlockGeometry(H=H, H_T=H_T, block_sizes=block_sizes)
        self._block_geometry = geom
        return geom

    # ------------------------------------------------------------------
    # Core loop (Eq. 10, 16, 18)
    # ------------------------------------------------------------------
    def _fit_block_champagne(
        self,
        *,
        B: np.ndarray,  # (n_chans, n_times)
        L: np.ndarray,  # (n_chans, n_sources)
        U: np.ndarray,  # (n_chans, n_sources) where U[:, i] = sum_{j in block(i)} L[:, j]
        geom: _BlockGeometry,
        alpha: float,
    ) -> np.ndarray:
        n_chans, _n_times = B.shape
        n_sources = L.shape[1]
        if int(geom.H.shape[0]) != int(n_sources):
            raise ValueError("Block geometry does not match leadfield n_sources.")
        if U.shape != (n_chans, n_sources):
            raise ValueError(f"U has shape {U.shape}, expected {(n_chans, n_sources)}")

        # Initialize hyperparameters
        gamma_blocks = np.ones(n_sources, dtype=float)
        beta = np.full(
            n_chans, max(alpha, self.eps), dtype=float
        )  # residual diag noise
        v = float(max(self.v_noise_activity, self.eps))

        n_times = int(B.shape[1])
        eye_ch = np.eye(n_chans, dtype=float)
        loss_prev: float | None = None
        best_state: tuple[float, np.ndarray, np.ndarray] | None = None

        # Preallocate scaled matrices to avoid per-iteration allocations.
        Lw = np.empty_like(L)
        Ug = np.empty_like(U)

        for i_iter in range(self.max_iter):
            # ------------------------------------------------------------------
            # Model covariance Sigma_b (Eq. 10), computed via the exact algebraic
            # simplification described in the module docstring:
            #   Sigma_sources = (1-r) L diag(w) L^T  +  r U diag(gamma) U^T
            # where w = H.T @ gamma and U[:, i] = sum_{j in block(i)} L[:, j].
            # ------------------------------------------------------------------
            w = np.asarray(geom.H_T @ gamma_blocks, dtype=float).ravel()  # (n_sources,)
            w = np.maximum(w, 0.0)

            np.multiply(L, w[np.newaxis], out=Lw)
            Sigma_b = (1.0 - self.r) * (Lw @ L.T)

            np.multiply(U, gamma_blocks[np.newaxis], out=Ug)
            Sigma_b = Sigma_b + self.r * (Ug @ U.T)

            Sigma_b = Sigma_b + np.diag(beta)
            if self.augment_noise_activity:
                Sigma_b = Sigma_b + v * eye_ch
            Sigma_b = 0.5 * (Sigma_b + Sigma_b.T)

            # Invert (with rank-tolerant fallback)
            try:
                Sigma_b_inv = np.linalg.inv(Sigma_b)
            except np.linalg.LinAlgError:
                Sigma_b_inv = np.linalg.pinv(Sigma_b, rcond=self.rank_tol)

            # Posterior mean (Eq. 10) for hatZ (and optionally E)
            SiB = Sigma_b_inv @ B  # (n_chans, n_times)
            # We do not explicitly form hatZ. For the diagonal-noise update we only
            # need the posterior mean residual:
            #   residual = B - L * E[S] = Sigma_n * Sigma_b^{-1} B = diag(beta) * SiB.
            # This identity follows from Sigma_b = L Cov(S) L^T + Sigma_n (+ vI).
            residual = beta[:, np.newaxis] * SiB

            # Loss (Eq. 11)
            data_fit = float(np.trace(B.T @ SiB) / n_times)
            log_det = _safe_slogdet_sym(Sigma_b)
            loss = data_fit + log_det

            if best_state is None or loss < best_state[0]:
                best_state = (loss, gamma_blocks.copy(), beta.copy())

            # Convergence check
            if loss_prev is not None:
                rel_change = (loss_prev - loss) / (abs(loss_prev) + 1e-20)
                if rel_change > 0 and rel_change < self.convergence_criterion:
                    loss_prev = loss
                    break
            loss_prev = loss

            # ------------------------------------------------------------------
            # Gamma updates (Eq. 16) without forming hatZ explicitly.
            #
            # For block i, let A_i be the leadfield restricted to block(i) columns.
            # The convexity/MM update needs:
            #   upper_i = (1/T) ||mu_hatZ_i||_F^2
            #   lower_i = trace(L_i^T Sigma_b^{-1} L_i)
            # with L_i = A_i M_i and M_i M_i^T = C_i(r) = (1-r)I + r 11^T.
            #
            # Using C_i directly yields (exactly):
            #   lower_i = (1-r) * sum_{j in block(i)} z_j  +  r * u_i^T Sigma_b^{-1} u_i
            #   upper_i = gamma_i^2 * [ (1-r) * sum_{j in block(i)} t_j
            #                          +  r * u_i^T (SiB SiB^T) u_i ] / T
            # where:
            #   z_j = l_j^T Sigma_b^{-1} l_j              (diag of Fisher info)
            #   t_j = ||l_j^T Sigma_b^{-1} B||_2^2        (time-energy in SiB space)
            #   u_i = sum_{j in block(i)} l_j             (U column i)
            # ------------------------------------------------------------------
            SiL = Sigma_b_inv @ L  # (n_chans, n_sources)
            z = np.sum(L * SiL, axis=0)  # (n_sources,)

            # SiB SiB^T is small (n_chans x n_chans) and reused several times.
            G = SiB @ SiB.T  # (n_chans, n_chans)
            GL = G @ L
            t = np.sum(L * GL, axis=0)  # (n_sources,) == diag(L.T G L)

            # Neighborhood sums: (H @ z)[i] = sum_{j in block(i)} z_j.
            z_sum = np.asarray(geom.H @ z, dtype=float).ravel()
            t_sum = np.asarray(geom.H @ t, dtype=float).ravel()

            # u_i^T Sigma_b^{-1} u_i and u_i^T G u_i for all i via diagonal of U.T M U.
            SiU = Sigma_b_inv @ U
            uSu = np.sum(U * SiU, axis=0)
            GU = G @ U
            uGu = np.sum(U * GU, axis=0)

            upper = (gamma_blocks**2) * (
                (1.0 - self.r) * (t_sum / max(n_times, 1))
                + self.r * (uGu / max(n_times, 1))
            )
            lower = (1.0 - self.r) * z_sum + self.r * uSu
            lower = np.maximum(lower, self.eps)
            new_gamma_blocks = np.sqrt(np.maximum(upper, 0.0) / lower)

            # Damping (optional): gamma <- (1-d)*new + d*old
            if self.damping > 0:
                d = float(np.clip(self.damping, 0.0, 0.99))
                new_gamma_blocks = (1.0 - d) * new_gamma_blocks + d * gamma_blocks

            new_gamma_blocks[~np.isfinite(new_gamma_blocks)] = 0.0
            new_gamma_blocks = np.maximum(new_gamma_blocks, 0.0)

            # Pruning (block-level)
            if float(new_gamma_blocks.max(initial=0.0)) > 0.0:
                thresh = self.pruning_thresh * float(new_gamma_blocks.max())
                new_gamma_blocks[new_gamma_blocks < thresh] = 0.0

            gamma_blocks = new_gamma_blocks

            # Residual for noise update (Eq. 18)
            q_hat = np.diag(Sigma_b_inv)  # (n_chans,)
            numer = np.mean(residual**2, axis=1)
            denom = np.maximum(q_hat, self.eps)
            beta_new = np.sqrt(np.maximum(numer, 0.0) / denom)
            beta_new[~np.isfinite(beta_new)] = float(np.max(beta))
            beta = np.maximum(beta_new, self.eps)

            if self.verbose > 1:
                logger.debug(
                    "iter %d: loss=%.4e, nnz_gamma=%d, beta=[%.3e..%.3e]",
                    i_iter,
                    loss,
                    int(np.count_nonzero(gamma_blocks)),
                    float(beta.min()),
                    float(beta.max()),
                )

        # If the loop ended in a worse state (e.g., numerical issues), restore best.
        if best_state is not None:
            _best_loss, gamma_blocks, beta = best_state

        # Final operator W_proj mapping sensors -> cortex sources:
        #   W = CovS L^T Sigma_b^{-1}
        # with CovS = (1-r) diag(w) + r H.T diag(gamma) H  (exact for constant-corr blocks).
        w = np.asarray(geom.H_T @ gamma_blocks, dtype=float).ravel()
        w = np.maximum(w, 0.0)

        np.multiply(L, w[np.newaxis], out=Lw)
        Sigma_b = (1.0 - self.r) * (Lw @ L.T)
        np.multiply(U, gamma_blocks[np.newaxis], out=Ug)
        Sigma_b = Sigma_b + self.r * (Ug @ U.T)
        Sigma_b = Sigma_b + np.diag(beta)
        if self.augment_noise_activity:
            Sigma_b = Sigma_b + v * eye_ch
        Sigma_b = 0.5 * (Sigma_b + Sigma_b.T)
        try:
            Sigma_b_inv = np.linalg.inv(Sigma_b)
        except np.linalg.LinAlgError:
            Sigma_b_inv = np.linalg.pinv(Sigma_b, rcond=self.rank_tol)

        A = (Sigma_b_inv @ L).T  # (n_sources, n_chans) == L^T Sigma_b^{-1}

        # W1 = (1-r) diag(w) A
        W1 = (1.0 - self.r) * (w[:, np.newaxis] * A)

        # W2 = r H.T diag(gamma) H A
        HA = geom.H @ A  # (n_sources, n_chans)
        gamma_HA = gamma_blocks[:, np.newaxis] * HA
        W2 = self.r * (geom.H_T @ gamma_HA)

        W_cortex = (W1 + W2).astype(float, copy=False)
        return W_cortex
