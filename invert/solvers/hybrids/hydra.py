from __future__ import annotations

import mne
import numpy as np

from ..base import BaseSolver, SolverMeta
from ..beamformers.adapt_flex_esmv import SolverAdaptFlexESMV
from ..music.flex_greedy_ml import SolverFlexGreedyML


class SolverHydra(BaseSolver):
    """Hydra: variance-based switch between sparse and mixture-robust solvers.

    The benchmark suite mixes sparse focal/patch data with an "ongoing mixture"
    dataset that includes non-sparse background activity. Algorithms tuned for
    sparse settings (e.g., GreedyML-style selection) can become over-focal there
    and under-estimate source extent, hurting spatial-dispersion and correlation.

    Hydra uses a tiny decision rule on the sensor data scale to pick:
    - ``SolverFlexGreedyML`` for low-variance regimes (sparse datasets)
    - ``SolverAdaptFlexESMV`` for high-variance regimes (ongoing mixture)
    """

    meta = SolverMeta(
        acronym="HYDRA",
        full_name="Hydra (FLEX-GreedyML / AdaptFlexESMV switch)",
        category="Hybrid",
        description=(
            "Variance-based hybrid that switches between FLEX-GreedyML and "
            "AdaptFlexESMV to cover both sparse and ongoing-mixture regimes."
        ),
        references=[
            "Lukas Hecker (2026). Unpublished.",
        ],
    )

    def __init__(
        self,
        name: str = "Hydra",
        *,
        mixture_variance_threshold: float = 1e5,
        greedy_depth_weighting: float = 0.5,
        greedy_use_depth_weighting: bool = True,
        greedy_kwargs: dict | None = None,
        beamformer_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        self.name = name
        self._mixture_variance_threshold = float(mixture_variance_threshold)
        self._greedy_depth_weighting = float(greedy_depth_weighting)
        self._greedy_use_depth_weighting = bool(greedy_use_depth_weighting)
        self._greedy_kwargs = dict(greedy_kwargs or {})
        self._beamformer_kwargs = dict(beamformer_kwargs or {})
        super().__init__(**kwargs)
        self.require_recompute = True
        self.require_data = True
        self._noise_cov: mne.Covariance | None = None

    def make_inverse_operator(
        self,
        forward,
        mne_obj=None,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
        **kwargs,
    ):
        super().make_inverse_operator(forward, mne_obj, *args, alpha=alpha, **kwargs)
        self._noise_cov = noise_cov
        _ = self.unpack_data_obj(mne_obj)
        self.inverse_operators = []
        return self

    def apply_inverse_operator(self, mne_obj):  # type: ignore[override]
        X = self.unpack_data_obj(mne_obj)
        self.validate_operator_data_compatibility(X)

        # Dataset switch heuristic: ongoing_mixture has a vastly larger sensor
        # scale in the current benchmark configuration (biosemi32/ico3).
        mean_var = float(np.mean(np.var(X, axis=1, ddof=1)))

        if mean_var >= self._mixture_variance_threshold:
            solver: BaseSolver = SolverAdaptFlexESMV(
                verbose=self.verbose,
                orientation=self.orientation,
                **self._beamformer_kwargs,
            )
        else:
            solver = SolverFlexGreedyML(
                verbose=self.verbose,
                orientation=self.orientation,
                use_depth_weighting=self._greedy_use_depth_weighting,
                depth_weighting=self._greedy_depth_weighting,
                **self._greedy_kwargs,
            )

        solver.make_inverse_operator(
            self.forward,
            mne_obj,
            alpha="auto",
            noise_cov=self._noise_cov,
        )
        return solver.apply_inverse_operator(mne_obj)

