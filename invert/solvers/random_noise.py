import numpy as np

from .base import BaseSolver, InverseOperator, SolverMeta


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
        n_sources = self.leadfield.shape[1]
        self.n_sources = n_sources
        # Store a single dummy inverse operator so the benchmark runner can call .apply()
        self.inverse_operators = [
            InverseOperator(self._RandomOperator(n_sources), self.name)
        ]
        return self

    class _RandomOperator:
        """Thin wrapper that returns random noise with the correct number of source rows."""

        def __init__(self, n_sources):
            self.n_sources = n_sources

        def __matmul__(self, M):
            n_time = M.shape[1] if M.ndim > 1 else 1
            return np.random.randn(self.n_sources, n_time)
