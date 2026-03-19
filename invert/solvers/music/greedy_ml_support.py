import mne

from ..base import SolverMeta
from .greedy_ml import SolverGreedyML


class SolverGreedyMLSupport(SolverGreedyML):
    """GreedyML tuned for support recovery.

    Uses a BIC variant that penalizes only support size (k) rather than k*T.
    This is more appropriate when the goal is to recover salient support across
    time rather than to fit per-timepoint amplitudes.
    """

    meta = SolverMeta(
        acronym="GreedyML-Support",
        full_name="Greedy Maximum-Likelihood (Support BIC)",
        category="Subspace Methods",
        description=(
            "GreedyML with BIC penalty scaling for support recovery (penalize k, "
            "not k*T). Intended for benchmarks where background activity is "
            "treated as nuisance and the goal is to recover dominant regions."
        ),
        references=SolverGreedyML.meta.references,
        internal=True,
    )

    def make_inverse_operator(  # type: ignore[override]
        self,
        forward,
        mne_obj=None,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
        k_max=5,
        n_starts=50,
        n_refine_iters=3,
        penalty_mode="bic_support_only",
        **kwargs,
    ):
        return super().make_inverse_operator(
            forward,
            mne_obj,
            *args,
            alpha=alpha,
            noise_cov=noise_cov,
            k_max=k_max,
            n_starts=n_starts,
            n_refine_iters=n_refine_iters,
            penalty_mode=penalty_mode,
            **kwargs,
        )

