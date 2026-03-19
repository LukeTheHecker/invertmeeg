import mne
import numpy as np

from ..base import SolverMeta
from .flex_greedy_ml import SolverFlexGreedyML


class SolverFlexGreedyMLTemporalDiff(SolverFlexGreedyML):
    """FLEX-GreedyML on temporally differenced data (simple high-pass).

    mixed_cortical_activity background is temporally 1/f-like; a first-difference
    (discrete derivative) suppresses low-frequency components and can improve
    support recovery when ongoing background is treated as nuisance.
    """

    meta = SolverMeta(
        acronym="FLEX-GreedyML-Δt",
        full_name="FLEX Greedy Maximum-Likelihood (Temporal Difference)",
        category="Subspace Methods",
        description=(
            "Runs FLEX-GreedyML on first-differenced sensor data to reduce "
            "low-frequency smooth background dominance."
        ),
        references=SolverFlexGreedyML.meta.references,
        internal=True,
    )

    def __init__(self, name="FLEX-GreedyML-TemporalDiff", **kwargs):
        self.name = name
        super().__init__(**kwargs)
        self._diff_applied = True

    @staticmethod
    def _diff_data(data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[1] < 2:
            return data
        return np.diff(data, axis=1)

    def make_inverse_operator(  # type: ignore[override]
        self,
        forward,
        mne_obj=None,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
        **kwargs,
    ):
        # Avoid BaseSolver.unpack_data_obj here because it expects self.forward
        # to already be populated. FLEX-GreedyML will call BaseSolver.make_inverse_operator
        # when we delegate below.
        if hasattr(mne_obj, "get_data"):
            data = mne_obj.get_data()
        else:
            data = getattr(mne_obj, "data", None)
        data = np.asarray(data, dtype=float)
        data_d = self._diff_data(data)
        evoked_d = mne.EvokedArray(data_d, mne_obj.info, tmin=getattr(mne_obj, "tmin", 0.0), verbose=0)
        return super().make_inverse_operator(
            forward,
            evoked_d,
            *args,
            alpha=alpha,
            noise_cov=noise_cov,
            **kwargs,
        )

    def apply_inverse_operator(self, mne_obj):  # type: ignore[override]
        data = self.unpack_data_obj(mne_obj)
        self.validate_operator_data_compatibility(data)
        data_d = self._diff_data(data)
        source_mat, _idx = self._compute_source_mat(data_d)
        return self.source_to_object(source_mat)
