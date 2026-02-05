from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..base import BaseSolver, SolverMeta
from .flex_esmv2 import SolverFlexESMV2
from .utils import _estimate_rank_mdl


@dataclass(frozen=True)
class _SafeContrastParams:
    eps: float = 1e-12
    gamma_base_max: float = (
        1.2  # target gamma in [1, 1+gamma_base_max] before safety clip
    )
    gamma_fscale: float = 0.05
    patchiness_threshold: float = 0.25
    min_keep: float = 0.26  # keep K-th peak above this (>= EMD threshold)
    max_sources: int = 4


class SolverSafeFlexESMV(BaseSolver):
    """FlexESMV6: FlexESMV2 + *safe* adaptive contrast.

    Unlike FlexESMV5, this caps the contrast exponent per sample so that the
    K-th strongest peak (K estimated via MDL on sensor covariance) stays above
    the EMD support threshold. This aims to reduce dispersion without
    collapsing multi-source structure (which spikes EMD).
    """

    meta = SolverMeta(
        slug="safe_flex_esmv",
        full_name="Safe Flexible-Extent ESMV",
        category="Beamformers",
        description=(
            "Flex-extent ESMV variant with a capped ('safe') adaptive contrast "
            "mechanism designed to avoid collapsing multi-source structure."
        ),
        references=[
            "Lukas Hecker (2025). Unpublished.",
            "Jonmohamadi, Y., Poudel, G., Innes, C., Weiss, D., Krueger, R., & Jones, R. "
            "(2014). Comparison of beamformers for EEG source signal reconstruction. "
            "Biomedical Signal Processing and Control, 14, 175-188.",
        ],
    )

    def __init__(
        self,
        name: str = "SafeFlexESMV (FlexESMV2+SafeContrast) Beamformer",
        params: _SafeContrastParams | None = None,
        reduce_rank: bool = True,
        rank: str | int = "auto",
        **kwargs,
    ):
        if params is None:
            params = _SafeContrastParams()
        self.name = name
        self._scp = params
        self._inner_kwargs = dict(kwargs)
        self._inner_kwargs.pop("verbose", None)
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, alpha="auto", **kwargs):
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        self._x_cached = self.unpack_data_obj(mne_obj)
        self.inverse_operators = []
        return self

    def apply_inverse_operator(self, mne_obj):  # type: ignore[override]
        base = SolverFlexESMV2(
            reduce_rank=self.reduce_rank,
            rank=self.rank,
            verbose=self.verbose,
            **self._inner_kwargs,
        )
        base.make_inverse_operator(self.forward, mne_obj, alpha="auto")
        stc = base.apply_inverse_operator(mne_obj)

        y = stc.data
        if y.ndim != 2 or y.size == 0:
            return stc

        scp = self._scp
        p = np.mean(np.abs(y), axis=1)
        pmax = float(np.max(p)) + scp.eps
        rel = np.clip(p / pmax, scp.eps, 1.0)

        patchiness = float(np.mean(rel > scp.patchiness_threshold))
        gamma_target = 1.0 + scp.gamma_base_max * float(
            np.exp(-patchiness / scp.gamma_fscale)
        )

        # Estimate K via MDL on sensor covariance and cap gamma so top-K stay above min_keep
        x = getattr(self, "_x_cached", None)
        k_hat = 1
        if x is not None:
            n_times = x.shape[1]
            Cx = (x @ x.T) / max(1, n_times)
            eigvals = np.linalg.eigvalsh(Cx)[::-1]
            k_hat = _estimate_rank_mdl(
                eigvals, n_samples=n_times, max_rank=scp.max_sources
            )

        k_hat = int(np.clip(k_hat, 1, scp.max_sources))
        r_k = (
            float(np.sort(rel)[::-1][k_hat - 1])
            if rel.size >= k_hat
            else float(np.max(rel))
        )
        if r_k < 1.0 - 1e-9:
            gamma_max_allowed = float(np.log(scp.min_keep) / np.log(r_k))
            gamma = float(
                np.clip(min(gamma_target, gamma_max_allowed), 1.0, gamma_target)
            )
        else:
            gamma = float(gamma_target)

        scale = rel ** (gamma - 1.0)
        stc.data[:] = y * scale[:, None]
        return stc
