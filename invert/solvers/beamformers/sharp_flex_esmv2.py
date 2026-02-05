from __future__ import annotations

import numpy as np

from ..base import BaseSolver, SolverMeta
from .deblur_flex_esmv import SolverDeblurFlexESMV
from .sharp_flex_esmv import _anchored_power, _AnchoredContrastParams


class SolverSharpFlexESMV2(BaseSolver):
    """FlexESMV8: FlexESMV3 + anchored contrast shaping.

    Start from the graph-deblurred FlexESMV3 (often best EMD), then apply the
    anchored contrast shaping from FlexESMV7 to reduce FWHM blur without
    collapsing EMD support.
    """

    meta = SolverMeta(
        slug="sharp_flex_esmv2",
        full_name="Sharp Flexible-Extent ESMV (variant 2)",
        category="Beamformers",
        description=(
            "Sharp flex-extent ESMV variant built on graph-deblurred FlexESMV, "
            "followed by anchored contrast shaping."
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
        name: str = "SharpFlexESMV2 (DeblurFlexESMV+AnchoredContrast) Beamformer",
        params: _AnchoredContrastParams | None = None,
        reduce_rank: bool = True,
        rank: str | int = "auto",
        **kwargs,
    ):
        if params is None:
            params = _AnchoredContrastParams()
        self.name = name
        self._acp = params
        self._inner_kwargs = dict(kwargs)
        self._inner_kwargs.pop("verbose", None)
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, alpha="auto", **kwargs):
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        _ = self.unpack_data_obj(mne_obj)
        self.inverse_operators = []
        return self

    def apply_inverse_operator(self, mne_obj):  # type: ignore[override]
        base = SolverDeblurFlexESMV(
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

        acp = self._acp
        p = np.mean(np.abs(y), axis=1)
        pmax = float(np.max(p)) + acp.eps
        rel = np.clip(p / pmax, 0.0, 1.0)

        core = float(np.mean(rel > acp.core_threshold))
        gamma = acp.gamma_min + (acp.gamma_max - acp.gamma_min) * (
            1.0 - float(np.exp(-core / acp.gamma_fscale))
        )

        rel_t = _anchored_power(rel, anchor=acp.anchor, gamma=gamma)
        scale = np.divide(rel_t, rel, out=np.zeros_like(rel_t), where=rel > acp.eps)
        stc.data[:] = y * scale[:, None]
        return stc
