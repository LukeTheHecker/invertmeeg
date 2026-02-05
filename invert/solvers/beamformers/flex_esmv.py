from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..base import BaseSolver, SolverMeta
from .flex_esmv2 import SolverFlexESMV2


@dataclass(frozen=True)
class _ContrastParams:
    eps: float = 1e-12
    gamma_max: float = 1.6  # gamma in [1, 1+gamma_max]
    gamma_fscale: float = 0.05
    patchiness_threshold: float = 0.25


class SolverFlexESMV(BaseSolver):
    """FlexESMV5: FlexESMV2 + adaptive contrast (power-law) to reduce dispersion.

    This is a lightweight, monotone reweighting on the per-dipole mean |y|
    map intended to suppress mid-level leakage (which drives the 50%-FWHM
    dispersion metric) while keeping high-confidence sources intact.
    """

    meta = SolverMeta(
        slug="flex_esmv",
        full_name="Flexible-Extent ESMV",
        category="Beamformers",
        description=(
            "Flex-extent ESMV variant with an adaptive contrast reweighting "
            "postprocess intended to reduce spatial dispersion."
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
        name: str = "FlexESMV (FlexESMV2+Contrast) Beamformer",
        params: _ContrastParams | None = None,
        reduce_rank: bool = True,
        rank: str | int = "auto",
        **kwargs,
    ):
        if params is None:
            params = _ContrastParams()
        self.name = name
        self._cp = params
        self._inner_kwargs = dict(kwargs)
        self._inner_kwargs.pop("verbose", None)
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, alpha="auto", **kwargs):
        # Just ensure forward/info handling is consistent
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        _ = self.unpack_data_obj(mne_obj)
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

        cp = self._cp
        p = np.mean(np.abs(y), axis=1)
        pmax = float(np.max(p)) + cp.eps
        patchiness = float(np.mean((p / pmax) > cp.patchiness_threshold))
        gamma = 1.0 + cp.gamma_max * float(np.exp(-patchiness / cp.gamma_fscale))

        rel = np.clip(p / pmax, cp.eps, 1.0)
        scale = rel ** (gamma - 1.0)
        stc.data[:] = y * scale[:, None]
        return stc
