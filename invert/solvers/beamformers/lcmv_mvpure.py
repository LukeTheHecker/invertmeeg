import logging
from typing import Any

import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


def _top_eigenspace_projector_spd(S: np.ndarray, rank: int) -> np.ndarray:
    """Return orthogonal projector onto top-`rank` eigenspace of an SPD matrix."""
    if rank <= 0:
        raise ValueError("rank must be >= 1")
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("S must be a square matrix")
    l = S.shape[0]
    if rank > l:
        raise ValueError(f"rank ({rank}) cannot exceed matrix size ({l})")

    # eigh is stable for symmetric matrices and returns ascending eigenvalues.
    _, evecs = np.linalg.eigh(S)
    u = evecs[:, -rank:]
    return u @ u.T


def _lcmv_multisource_weights_from_inv_cov(
    C_inv: np.ndarray,
    H: np.ndarray,
    *,
    cond_threshold: float = 1e12,
    regularization: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute multi-source LCMV weights for constraint matrix H.

    Implements W = (H^T C^{-1} H)^{-1} H^T C^{-1}, returning (W, S) where
    S = H^T C^{-1} H.
    """
    if H.ndim != 2 or C_inv.ndim != 2:
        raise ValueError("C_inv and H must be 2D arrays")
    if C_inv.shape[0] != C_inv.shape[1]:
        raise ValueError("C_inv must be square")
    if C_inv.shape[0] != H.shape[0]:
        raise ValueError("C_inv and H must have matching channel dimension")

    S = H.T @ C_inv @ H
    cond_num = np.linalg.cond(S) if S.size else np.inf
    if not np.isfinite(cond_num) or cond_num > cond_threshold:
        S_inv = np.linalg.inv(S + regularization * np.eye(S.shape[0]))
    else:
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.inv(S + regularization * np.eye(S.shape[0]))

    W = S_inv @ (H.T @ C_inv)
    return W, S


def _mvpure_projected_lcmv_weights_from_inv_cov(
    C_inv: np.ndarray,
    H: np.ndarray,
    *,
    rank: int,
    cond_threshold: float = 1e12,
    regularization: float = 1e-12,
) -> np.ndarray:
    """MV-PURE projected multi-source LCMV: W_r = P_r(S) W."""
    W, S = _lcmv_multisource_weights_from_inv_cov(
        C_inv, H, cond_threshold=cond_threshold, regularization=regularization
    )
    if rank >= S.shape[0]:
        return W
    P = _top_eigenspace_projector_spd(S, rank)
    return P @ W


class SolverLCMVMVPURE(BaseSolver):
    """MV-PURE projected multi-source LCMV (alternative to vanilla single-source LCMV).

    Implements the Stage-2 MV-PURE spatial filter:

        W_LCMV = (H^T R^{-1} H)^{-1} H^T R^{-1}
        W_MVP(r) = P_r(H^T R^{-1} H) W_LCMV

    Since this package's :class:`SolverLCMV` is single-source (per dipole), this
    solver computes a joint filter on a selected small set of dipoles and embeds
    those weights into the full source space (non-selected dipoles get zero).
    """

    meta = SolverMeta(
        slug="lcmv-mvpure",
        full_name="MV-PURE (Projected) LCMV",
        category="Beamformers",
        description=(
            "Reduced-rank MV-PURE variant of multi-source LCMV: computes a joint "
            "LCMV filter on a selected source set and projects out ill-conditioned "
            "output directions."
        ),
        references=[
            "Jurkowska, J., Dreszer, J., Lewandowska, M., & Piotrowski, T. (2025). "
            "Multi-Source Neural Activity Indices and Spatial Filters for EEG/MEG "
            "Inverse Problem: An Extension to MNE-Python. arXiv preprint arXiv:2509.14118."
        ],
    )

    def __init__(
        self,
        name: str = "LCMV MV-PURE Beamformer",
        *,
        mvp_n_sources: int | str = "auto",
        mvp_rank: int | str = "auto",
        mvp_max_sources: int = 10,
        spectrum_threshold: float = 0.05,
        source_indices: list[int] | np.ndarray | None = None,
        reduce_rank: bool = True,
        rank: str = "auto",
        **kwargs,
    ):
        self.name = name
        self.mvp_n_sources = mvp_n_sources
        self.mvp_rank = mvp_rank
        self.mvp_max_sources = int(mvp_max_sources)
        self.spectrum_threshold = float(spectrum_threshold)
        self.source_indices = (
            None if source_indices is None else np.asarray(source_indices, dtype=int)
        )
        super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    @staticmethod
    def _estimate_noise_floor_from_cov(R: np.ndarray) -> float:
        # Robust scalar noise-floor estimate from the bottom third of eigenvalues.
        evals = np.linalg.eigvalsh(R)
        k = max(1, len(evals) // 3)
        floor = float(np.median(evals[:k]))
        if floor > 0:
            return floor
        positive = evals[evals > 0]
        if positive.size:
            return float(np.min(positive))
        return 1.0

    def _infer_l0_and_rank(self, R: np.ndarray) -> tuple[int, int]:
        # Heuristic consistent with RN^{-1} (> 1) when N ~ sigma^2 I.
        sigma2 = self._estimate_noise_floor_from_cov(R)
        evals = np.linalg.eigvalsh(R)
        ratios = evals / sigma2
        n_gt1 = int(np.sum(ratios > (1.0 + self.spectrum_threshold)))

        l0 = max(1, min(n_gt1, self.mvp_max_sources, int(R.shape[0])))
        if isinstance(self.mvp_n_sources, (int, np.integer)):
            l0 = min(int(self.mvp_n_sources), int(R.shape[0]))

        if isinstance(self.mvp_rank, (int, np.integer)):
            r = int(self.mvp_rank)
        else:
            r = max(1, min(n_gt1, l0))

        r = max(1, min(r, l0))
        return l0, r

    def make_inverse_operator(  # type: ignore[override]
        self,
        forward: Any,
        mne_obj: Any,
        *args: Any,
        alpha: str | float = "auto",
        weight_norm: bool = True,
        verbose: int = 0,
        **kwargs: Any,
    ) -> Any:
        self.weight_norm = bool(weight_norm)
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)

        leadfield = self.leadfield
        leadfield /= np.linalg.norm(leadfield, axis=0)
        n_chans, n_dipoles = leadfield.shape

        y = data - data.mean(axis=1, keepdims=True)
        I = np.identity(n_chans)
        R = self.data_covariance(y, center=False, ddof=1)
        self.get_alphas(reference=R)

        inverse_operators = []
        selected_sources_per_alpha: list[np.ndarray] = []
        selected_rank_per_alpha: list[int] = []
        for alpha_val in self.alphas:
            R_inv = self.robust_inverse(R + alpha_val * I)

            if self.source_indices is not None:
                sel = np.unique(self.source_indices)
                if sel.size == 0:
                    raise ValueError("source_indices must be non-empty when provided")
                l0 = int(sel.size)
                r = (
                    int(self.mvp_rank)
                    if isinstance(self.mvp_rank, (int, np.integer))
                    else l0
                )
                r = max(1, min(r, l0))
            else:
                l0, r = self._infer_l0_and_rank(R)

                # Fast selection: top single-source LCMV output power.
                upper = R_inv @ leadfield
                lower = np.einsum("ij,jk,ki->i", leadfield.T, R_inv, leadfield)
                W_vanilla = upper / lower
                q_hat = W_vanilla.T @ y
                scores = np.mean(q_hat**2, axis=1)
                sel = np.argsort(scores)[::-1][:l0]
                sel = np.unique(sel)
                l0 = int(sel.size)
                r = max(1, min(r, l0))

            selected_sources_per_alpha.append(sel.copy())
            selected_rank_per_alpha.append(int(r))

            H0 = leadfield[:, sel]
            W_sel = _mvpure_projected_lcmv_weights_from_inv_cov(
                R_inv, H0, rank=r, cond_threshold=1e12, regularization=1e-12
            )  # (l0, n_chans)

            K_full = np.zeros((n_dipoles, n_chans), dtype=W_sel.dtype)
            K_full[sel, :] = W_sel

            if self.weight_norm:
                row_norm = np.linalg.norm(K_full, axis=1)
                nz = row_norm > 0
                K_full[nz, :] = (K_full[nz, :].T / row_norm[nz]).T

            inverse_operators.append(K_full)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        # Expose selection details for debugging/benchmarks.
        self.selected_sources_ = selected_sources_per_alpha
        self.selected_rank_ = selected_rank_per_alpha
        return self
