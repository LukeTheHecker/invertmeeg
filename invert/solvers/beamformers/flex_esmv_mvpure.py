from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np

from ..base import BaseSolver, SolverMeta
from .esmv_mvpure import SolverESMVMVPURE


@dataclass(frozen=True)
class _ContrastParams:
    eps: float = 1e-12
    gamma_max: float = 1.6  # gamma in [1, 1+gamma_max]
    gamma_fscale: float = 0.05
    patchiness_threshold: float = 0.25


class SolverFlexESMVMVPURE(BaseSolver):
    """FlexESMV-style adaptive contrast wrapper around ESMV MV-PURE."""

    meta = SolverMeta(
        slug="flex-esmv-mvpure",
        full_name="Flexible-Extent ESMV (MV-PURE)",
        category="Beamformers",
        description=(
            "FlexESMV-style adaptive contrast reweighting applied on top of an "
            "ESMV MV-PURE beamformer core."
        ),
        references=[
            "Lukas Hecker (2025). Unpublished.",
            "Jonmohamadi, Y., Poudel, G., Innes, C., Weiss, D., Krueger, R., & Jones, R. "
            "(2014). Comparison of beamformers for EEG source signal reconstruction. "
            "Biomedical Signal Processing and Control, 14, 175-188.",
            "Jurkowska, J., Dreszer, J., Lewandowska, M., & Piotrowski, T. (2025). "
            "Multi-Source Neural Activity Indices and Spatial Filters for EEG/MEG "
            "Inverse Problem: An Extension to MNE-Python. arXiv preprint arXiv:2509.14118.",
        ],
    )

    def __init__(
        self,
        name: str = "FlexESMV-MVPURE (ESMV-MVPURE+Contrast) Beamformer",
        params: _ContrastParams | None = None,
        *,
        mvp_n_sources: int | str = "auto",
        mvp_rank: int | str = "auto",
        mvp_max_sources: int = 4,
        spectrum_threshold: float = 0.05,
        selection_corr_threshold: float = 0.97,
        source_indices: list[int] | np.ndarray | None = None,
        reduce_rank: bool = True,
        rank: str | int = "auto",
        **kwargs,
    ):
        if params is None:
            params = _ContrastParams()
        self.name = name
        self._cp = params
        self.mvp_n_sources = mvp_n_sources
        self.mvp_rank = mvp_rank
        self.mvp_max_sources = int(mvp_max_sources)
        self.spectrum_threshold = float(spectrum_threshold)
        self.selection_corr_threshold = float(selection_corr_threshold)
        self.source_indices = (
            None if source_indices is None else np.asarray(source_indices, dtype=int)
        )
        self._inner_kwargs = dict(kwargs)
        self._inner_kwargs.pop("verbose", None)
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(
        self,
        forward,
        mne_obj=None,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
        **kwargs,
    ):
        # Keep forward/info handling consistent and defer heavy work to apply.
        super().make_inverse_operator(forward, mne_obj, *args, alpha=alpha, **kwargs)
        if noise_cov is not None:
            self.coerce_noise_cov(noise_cov)
        self._noise_cov = noise_cov
        _ = self.unpack_data_obj(mne_obj)
        self.inverse_operators = []
        return self

    def apply_inverse_operator(self, mne_obj):  # type: ignore[override]
        data = self.unpack_data_obj(mne_obj)
        self.validate_operator_data_compatibility(data)

        base = SolverESMVMVPURE(
            mvp_n_sources=self.mvp_n_sources,
            mvp_rank=self.mvp_rank,
            mvp_max_sources=self.mvp_max_sources,
            spectrum_threshold=self.spectrum_threshold,
            selection_corr_threshold=self.selection_corr_threshold,
            source_indices=self.source_indices,
            reduce_rank=self.reduce_rank,
            rank=self.rank,
            verbose=self.verbose,
            **self._inner_kwargs,
        )
        base.make_inverse_operator(
            self.forward, mne_obj, alpha="auto", noise_cov=self._noise_cov
        )
        stc = base.apply_inverse_operator(mne_obj)
        self.selected_sources_ = getattr(base, "selected_sources_", None)
        self.selected_rank_ = getattr(base, "selected_rank_", None)

        y = stc.data
        if y.ndim != 2 or y.size == 0:
            return stc

        cp = self._cp
        p = np.mean(np.abs(y), axis=1)
        pmax = float(np.max(p)) + cp.eps
        patchiness = float(np.mean((p / pmax) > cp.patchiness_threshold))
        gamma = 1.0 + cp.gamma_max * float(np.exp(-patchiness / cp.gamma_fscale))

        rel = np.clip(p / pmax, cp.eps, 1.0)
        scale = rel ** (gamma - 1.0)
        stc.data[:] = y * scale[:, None]
        return stc
