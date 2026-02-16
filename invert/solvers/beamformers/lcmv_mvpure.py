from __future__ import annotations

import logging
from typing import Any

import mne
import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta
from .utils import (
    _infer_mvpure_n_sources_and_rank,
    _mvpure_projected_lcmv_weights_from_inv_cov,
    _select_top_diverse_indices,
    build_covariance_candidates,
)

logger = logging.getLogger(__name__)


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
    R_VALUE_EXPONENTS = (-10.0, 4.0)

    def __init__(
        self,
        name: str = "LCMV MV-PURE Beamformer",
        *,
        mvp_n_sources: int | str = "auto",
        mvp_rank: int | str = "auto",
        mvp_max_sources: int = 4,
        spectrum_threshold: float = 0.05,
        selection_corr_threshold: float = 0.97,
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
        self.selection_corr_threshold = float(selection_corr_threshold)
        self.source_indices = (
            None if source_indices is None else np.asarray(source_indices, dtype=int)
        )
        super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def _infer_l0_and_rank(self, R: np.ndarray) -> tuple[int, int]:
        return _infer_mvpure_n_sources_and_rank(
            R,
            mvp_n_sources=self.mvp_n_sources,
            mvp_rank=self.mvp_rank,
            mvp_max_sources=self.mvp_max_sources,
            spectrum_threshold=self.spectrum_threshold,
        )

    def make_inverse_operator(  # type: ignore[override]
        self,
        forward: Any,
        mne_obj: Any,
        *args: Any,
        alpha: str | float = "auto",
        weight_norm: bool = True,
        noise_cov: mne.Covariance | None = None,
        cov_reg: str = "oas",
        cov_reg_beta: float = 0.05,
        cov_reg_cond_target: float = 1e4,
        verbose: int = 0,
        **kwargs: Any,
    ) -> Any:
        self.weight_norm = bool(weight_norm)
        super().make_inverse_operator(forward, mne_obj, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)

        if noise_cov is None:
            # Match vanilla LCMV exactly in the no-noise-covariance path.
            sensor_transform = np.eye(int(self.leadfield.shape[0]), dtype=float)
            leadfield = self.leadfield
        else:
            wf = self.prepare_whitened_forward(noise_cov)
            sensor_transform = wf.sensor_transform
            leadfield = wf.G_white

        leadfield_norm = np.linalg.norm(leadfield, axis=0, keepdims=True)
        leadfield_norm = np.where(leadfield_norm > 0, leadfield_norm, 1.0)
        leadfield /= leadfield_norm
        n_chans, n_dipoles = leadfield.shape
        leadfield_similarity = np.abs(leadfield.T @ leadfield)

        y = sensor_transform @ data
        y -= y.mean(axis=1, keepdims=True)
        I = np.identity(n_chans)
        R = self.data_covariance(y, center=False, ddof=1)
        cov_mats, self.alphas, cov_meta = build_covariance_candidates(
            C=R,
            I=I,
            alpha=self.alpha,
            get_alphas_fn=self.get_alphas,
            n_samples=int(y.shape[1]),
            cov_reg=cov_reg,
            cov_reg_beta=float(cov_reg_beta),
            cov_reg_cond_target=float(cov_reg_cond_target),
        )
        if "oas_shrinkage" in cov_meta:
            self._cov_reg_oas_shrinkage = float(cov_meta["oas_shrinkage"])

        inverse_operators = []
        selected_sources_per_alpha: list[np.ndarray] = []
        selected_rank_per_alpha: list[int] = []
        for cov_mat in cov_mats:
            R_inv = self.robust_inverse(cov_mat)

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
                scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
                sel = _select_top_diverse_indices(
                    scores,
                    leadfield_similarity,
                    l0,
                    corr_threshold=self.selection_corr_threshold,
                )
                sel = np.unique(sel)
                l0 = int(sel.size)
                r = max(1, min(r, l0))

            selected_sources_per_alpha.append(sel.copy())
            selected_rank_per_alpha.append(int(r))

            H0 = leadfield[:, sel]
            if H0.shape[1] == 1 and r == 1:
                # For the single-source case, MV-PURE reduces exactly to
                # vanilla LCMV; keep the closed form to avoid projection
                # round-off differences.
                h = H0[:, [0]]
                denom = float(np.asarray(h.T @ R_inv @ h).item())
                W_sel = (R_inv @ h / denom).T
            else:
                W_sel = _mvpure_projected_lcmv_weights_from_inv_cov(
                    R_inv, H0, rank=r, cond_threshold=1e12, regularization=1e-12
                )  # (l0, n_chans)

            K_full = np.zeros((n_dipoles, n_chans), dtype=W_sel.dtype)
            K_full[sel, :] = W_sel

            if self.weight_norm:
                row_norm = np.linalg.norm(K_full, axis=1)
                nz = row_norm > 0
                K_full[nz, :] = (K_full[nz, :].T / row_norm[nz]).T

            inverse_operators.append(K_full @ sensor_transform)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        # Expose selection details for debugging/benchmarks.
        self.selected_sources_ = selected_sources_per_alpha
        self.selected_rank_ = selected_rank_per_alpha
        return self
