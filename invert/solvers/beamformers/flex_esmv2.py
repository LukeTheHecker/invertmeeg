from __future__ import annotations

from ..base import BaseSolver, SolverMeta


class SolverFlexESMV2(BaseSolver):
    """Flex-extent beamformer tuned for the benchmark generator.

    This is a thin wrapper around `SolverAdaptFlexESMV` with `n_orders=2`
    (i.e., consider smoothing orders 0/1/2), matching the default synthetic
    `multi_patch` dataset which uses orders in [1, 2].
    """

    meta = SolverMeta(
        slug="flex_esmv2",
        full_name="Flexible-Extent ESMV (variant 2)",
        category="Beamformers",
        description=(
            "Flex-extent ESMV wrapper around AdaptFlexESMV using a fixed set of "
            "diffusion smoothing orders (0/1/2)."
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
        name: str = "FlexESMV2 (orders 0..2) Beamformer",
        reduce_rank: bool = True,
        rank: str | int = "auto",
        **kwargs,
    ):
        self.name = name
        # Keep kwargs for the inner solver, but avoid duplicating BaseSolver args
        self._inner_kwargs = dict(kwargs)
        self._inner_kwargs.pop("verbose", None)
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, alpha="auto", **kwargs):
        from .adapt_flex_esmv import SolverAdaptFlexESMV

        # Delegate to the tuned inner solver and expose its inverse operators
        self._inner = SolverAdaptFlexESMV(
            n_orders=2,
            diffusion_parameter=0.1,
            reduce_rank=self.reduce_rank,
            rank=self.rank,
            verbose=self.verbose,
            **self._inner_kwargs,
        )
        self._inner.make_inverse_operator(
            forward, mne_obj, *args, alpha=alpha, **kwargs
        )
        self.inverse_operators = self._inner.inverse_operators
        self.forward = self._inner.forward
        self.leadfield = self._inner.leadfield
        return self

    def apply_inverse_operator(self, mne_obj):  # type: ignore[override]
        return self._inner.apply_inverse_operator(mne_obj)
