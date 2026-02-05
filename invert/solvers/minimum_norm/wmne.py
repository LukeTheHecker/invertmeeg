import logging

import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverWMNE(BaseSolver):
    """Class for the Weighted Minimum Norm Estimate (wMNE) inverse solution
        [1].

    References
    ----------
    [1] Grech, R., Cassar, T., Muscat, J., Camilleri, K. P., Fabri, S. G.,
    Zervakis, M., ... & Vanrumste, B. (2008). Review on solving the inverse
    problem in EEG source analysis. Journal of neuroengineering and
    rehabilitation, 5(1), 1-33.
    """

    meta = SolverMeta(
        acronym="wMNE",
        full_name="Weighted Minimum Norm Estimate",
        category="Minimum Norm",
        description=(
            "Minimum-norm inverse with depth/weighting to reduce superficial bias "
            "by scaling the source prior or leadfield columns."
        ),
        references=[
            "Hämäläinen, M. S., & Ilmoniemi, R. J. (1994). Interpreting magnetic fields of the brain: minimum norm estimates. Medical & Biological Engineering & Computing, 32(1), 35–42.",
            "Lin, F.-H., Witzel, T., Ahlfors, S. P., Stufflebeam, S. M., Belliveau, J. W., & Hämäläinen, M. S. (2006). Assessing and improving the spatial accuracy in MEG source localization by depth-weighted minimum-norm estimates. NeuroImage, 31(1), 160–171.",
        ],
    )

    def __init__(self, name="Weighted Minimum Norm Estimate", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, *args, alpha="auto", verbose=0, **kwargs):
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
        W = np.diag(np.linalg.norm(self.leadfield, axis=0))
        WTW = np.linalg.inv(W.T @ W)
        LWTWL = self.leadfield @ WTW @ self.leadfield.T
        n_chans, _ = self.leadfield.shape

        inverse_operators = []
        for alpha in self.alphas:
            inverse_operator = (
                WTW
                @ np.linalg.solve(
                    LWTWL + alpha * np.identity(n_chans), self.leadfield
                ).T
            )
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        return self
