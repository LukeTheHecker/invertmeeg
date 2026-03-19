import mne
import numpy as np

from ..base import SolverMeta
from .flex_greedy_ml_temporal_diff import SolverFlexGreedyMLTemporalDiff


class SolverFlexGreedyMLTemporalDiff2(SolverFlexGreedyMLTemporalDiff):
    """Second temporal difference variant (stronger high-pass)."""

    meta = SolverMeta(
        acronym="FLEX-GreedyML-Δ²t",
        full_name="FLEX Greedy Maximum-Likelihood (Second Temporal Difference)",
        category="Subspace Methods",
        description=(
            "Runs FLEX-GreedyML on second-differenced sensor data to more strongly "
            "suppress low-frequency background."
        ),
        references=SolverFlexGreedyMLTemporalDiff.meta.references,
        internal=True,
    )

    def __init__(self, name="FLEX-GreedyML-TemporalDiff2", **kwargs):
        super().__init__(name=name, **kwargs)

    @staticmethod
    def _diff_data(data: np.ndarray) -> np.ndarray:  # type: ignore[override]
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[1] < 3:
            return data
        return np.diff(data, n=2, axis=1)

