import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta


class SolverEPIFOCUS(BaseSolver):
    """Class for the EPIFOCUS inverse solution [1].

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

    def __init__(self, name="EPIFOCUS", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, *args, alpha="auto", **kwargs):
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
        leadfield = self.leadfield
        leadfield -= leadfield.mean(axis=0)

        n_chans, _ = leadfield.shape

        W = np.diag(1 / np.linalg.norm(leadfield, axis=0))
        T = leadfield @ W
        inverse_operator = np.array([np.linalg.pinv(Ti[:, np.newaxis]) for Ti in T.T])[
            :, 0
        ]
        inverse_operators = [
            inverse_operator,
        ]

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        return self
