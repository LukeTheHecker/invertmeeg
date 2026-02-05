from __future__ import annotations

import logging
from dataclasses import dataclass

import mne
import numpy as np
from scipy.sparse import identity as sparse_identity
from scipy.sparse.csgraph import laplacian as sparse_laplacian
from scipy.sparse.linalg import splu

from ..base import BaseSolver, SolverMeta
from .flex_esmv2 import SolverFlexESMV2

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _FlexSharpParams:
    eps: float = 1e-12
    smooth_alpha: float = 0.12
    tau_max: float = 1.2
    tau_fscale: float = 0.05
    patchiness_threshold: float = 0.25
    max_scale: float = 3.0


class SolverDeblurFlexESMV(BaseSolver):
    """FlexESMV3: FlexESMV2 + graph unsharp-mask deblurring.

    FlexESMV2 is very strong on MLE/EMD/AP but tends to over-blur in the
    FWHM-based spatial dispersion metric. This variant applies a light,
    graph-based unsharp mask to reduce leakage/blur while keeping the time
    courses from FlexESMV2.
    """

    meta = SolverMeta(
        slug="deblur_flex_esmv",
        full_name="Deblurred Flexible-Extent ESMV",
        category="Beamformers",
        description=(
            "Flex-extent ESMV variant with a graph-based unsharp-mask deblurring "
            "postprocess to reduce spatial dispersion/leakage."
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
        name: str = "DeblurFlexESMV (FlexESMV2+GraphSharp) Beamformer",
        params: _FlexSharpParams | None = None,
        reduce_rank: bool = True,
        rank: str | int = "auto",
        **kwargs,
    ):
        if params is None:
            params = _FlexSharpParams()
        self.name = name
        self._fs = params
        self._inner_kwargs = dict(kwargs)
        self._inner_kwargs.pop("verbose", None)
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, alpha="auto", **kwargs):
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        _ = self.unpack_data_obj(mne_obj)
        adjacency = mne.spatial_src_adjacency(self.forward["src"], verbose=0)
        self._src_laplacian = sparse_laplacian(adjacency, normed=False).tocsc()
        self._src_n_dipoles = int(self._src_laplacian.shape[0])
        self.inverse_operators = []
        return self

    def apply_inverse_operator(self, mne_obj):  # type: ignore[override]
        # Base solution
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

        fs = self._fs
        p = np.mean(np.abs(y), axis=1)
        pmax = float(np.max(p)) + fs.eps
        patchiness = float(np.mean((p / pmax) > fs.patchiness_threshold))
        tau = fs.tau_max * float(np.exp(-patchiness / fs.tau_fscale))

        n_dipoles = y.shape[0]
        if (
            hasattr(self, "_src_laplacian")
            and getattr(self, "_src_n_dipoles", n_dipoles) == n_dipoles
        ):
            A = (
                sparse_identity(n_dipoles, format="csc")
                + fs.smooth_alpha * self._src_laplacian
            )
            try:
                lu = splu(A)
                p_smooth = lu.solve(p)
                p_sharp = (1.0 + tau) * p - tau * p_smooth
                p_sharp = np.clip(p_sharp, 0.0, None)
                scale = np.clip(p_sharp / (p + fs.eps), 0.0, fs.max_scale)
                stc.data[:] = y * scale[:, None]
            except Exception as e:
                logger.debug("DeblurFlexESMV graph-sharpen failed, skipping: %s", e)

        return stc
