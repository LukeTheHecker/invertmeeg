import numpy as np

from .base import BaseSolver, SolverMeta


class SolverRandomNoise(BaseSolver):
    """Baseline solver that returns random Gaussian noise as source estimate."""

    meta = SolverMeta(
        acronym="RAND",
        full_name="Random Noise Baseline",
        category="Baseline",
        description=(
            "Baseline solver that returns random Gaussian noise with the correct "
            "source dimensionality. Useful for sanity-checking pipelines."
        ),
        references=["Lukas Hecker 2025, unpublished"],
    )

    def __init__(self, name="Random Noise", **kwargs):
        self.name = name
        super().__init__(**kwargs)
        self.require_recompute = False
        self.require_data = False

    def make_inverse_operator(self, forward, *args, alpha="auto", **kwargs):
        super().make_inverse_operator(
            forward, *args, reference=None, alpha=alpha, **kwargs
        )
        self.n_sources = self.leadfield.shape[1]
        return self

    def apply_inverse_operator(self, mne_obj):
        data = self.unpack_data_obj(mne_obj)
        n_time = data.shape[1] if data.ndim > 1 else 1
        source_mat = np.random.randn(self.n_sources, n_time)
        return self.source_to_object(source_mat)
