import mne
import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta


class SolverEPIFOCUS(BaseSolver):
    """Class for the EPIFOCUS inverse solution [1].

    EPIFOCUS normalizes leadfield columns to unit norm before computing the
    inverse, then divides the result by the original column norms.  This
    promotes focal solutions because sources with small leadfield norms (deep
    sources) receive a larger weight in the final estimate.

    References
    ----------
    [1] Menendez, R. G. D. P., Andino, S. G., Lantz, G., Michel, C. M., &
    Landis, T. (2001). Noninvasive localization of electromagnetic epileptic
    activity. I. Method descriptions and simulations. Brain topography, 14(2),
    131-137.

    """

    meta = SolverMeta(
        acronym="EPIFOCUS",
        full_name="EPIFOCUS",
        category="Minimum Norm",
        description=(
            "Epileptic focus localization method that uses source-wise "
            "normalization/weighting to promote focal solutions."
        ),
        references=[
            "Grave de Peralta Menendez, R., Gonzalez Andino, S., Lantz, G., Michel, C. M., & Landis, T. (2001). Noninvasive localization of electromagnetic epileptic activity. I. Method descriptions and simulations. Brain Topography, 14(2), 131–137.",
        ],
    )
    SUPPORTS_VECTOR_ORIENTATION: bool = True

    def __init__(self, name="EPIFOCUS", eps=1e-15, **kwargs):
        self.name = name
        self.eps = float(eps)
        return super().__init__(**kwargs)

    def make_inverse_operator(
        self,
        forward,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
        **kwargs,
    ):
        """Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        alpha : float
            The regularization parameter.

        Return
        ------
        self : object returns itself for convenience
        """
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        wf = self.prepare_whitened_forward(noise_cov)
        leadfield = wf.G_white

        n_chans, n_dipoles = leadfield.shape

        # Column normalization: normalize each source's leadfield to unit norm
        col_norms = np.linalg.norm(leadfield, axis=0)
        col_norms = np.maximum(col_norms, self.eps)
        L_norm = leadfield / col_norms

        # Compute regularized inverse of the normalized leadfield.
        # The 1/col_norms weighting promotes focal (deep) sources.
        LLT = L_norm @ L_norm.T
        self.get_alphas(reference=LLT)

        inverse_operators = []
        for alpha in self.alphas:
            K = np.linalg.solve(LLT + float(alpha) * np.identity(n_chans), L_norm).T
            # Denormalize: divide by col_norms to weight toward focal sources
            K = (1.0 / col_norms[:, np.newaxis]) * K
            inverse_operators.append(K @ wf.sensor_transform)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        return self
