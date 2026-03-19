import mne

from ..base import SolverMeta
from .flex_greedy_ml import SolverFlexGreedyML


class SolverFlexGreedyMLSupport(SolverFlexGreedyML):
    """FLEX-GreedyML tuned for support recovery.

    Default BIC penalty is set to support-only (penalize k, not k*T). This
    typically increases recall for multi-source samples at longer T.
    """

    meta = SolverMeta(
        acronym="FLEX-GreedyML-Support",
        full_name="FLEX Greedy Maximum-Likelihood (Support BIC)",
        category="Subspace Methods",
        description=(
            "FLEX-GreedyML with a support-only BIC penalty (penalize k, not k*T). "
            "Intended for salient support recovery tasks."
        ),
        references=SolverFlexGreedyML.meta.references,
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
        n_orders=3,
        diffusion_parameter=0.1,
        adjacency_distance=3e-3,
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
            n_orders=n_orders,
            diffusion_parameter=diffusion_parameter,
            adjacency_distance=adjacency_distance,
            **kwargs,
        )

