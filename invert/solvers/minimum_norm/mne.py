import logging

import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverMNE(BaseSolver):
    """Class for the Minimum Norm Estimate (MNE) inverse solution [1].

    The formulas provided by [2] were used for implementation.

    References
    ----------
    [1] Pascual-Marqui, R. D. (1999). Review of methods for solving the EEG
        inverse problem. International journal of bioelectromagnetism, 1(1), 75-86.

    [2] Grech, R., Cassar, T., Muscat, J., Camilleri, K. P., Fabri, S. G.,
        Zervakis, M., ... & Vanrumste, B. (2008). Review on solving the inverse
        problem in EEG source analysis. Journal of neuroengineering and
        rehabilitation, 5(1), 1-33.
    """

    meta = SolverMeta(
        acronym="MNE",
        full_name="Minimum Norm Estimate",
        category="Minimum Norm",
        description=(
            "Classic L2 minimum-norm inverse solution. Estimates source currents "
            "by minimising the L2 norm of the source distribution subject to the "
            "data fit constraint."
        ),
        references=[
            "Hämäläinen, M. S., & Ilmoniemi, R. J. (1994). Interpreting magnetic fields of the brain: minimum norm estimates. Medical & Biological Engineering & Computing, 32(1), 35–42.",
        ],
    )

    def __init__(self, name="Minimum Norm Estimate", **kwargs):
        self.name = name
        super().__init__(**kwargs)
        self.require_recompute = False
        self.require_data = False

    def make_inverse_operator(self, forward, *args, alpha="auto", verbose=0, **kwargs):
        """Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        alpha : float
            The regularization parameter. When set to a float, it is scaled
            by the largest eigenvalue of L @ L.T.

        Return
        ------
        self : object returns itself for convenience
        """
        super().make_inverse_operator(
            forward, *args, reference=None, alpha=alpha, **kwargs
        )

        leadfield = self.leadfield
        n_chans, _ = leadfield.shape

        LLT = leadfield @ leadfield.T
        inverse_operators = []
        for alpha in self.alphas:
            inverse_operator = np.linalg.solve(
                LLT + alpha * np.identity(n_chans), leadfield
            ).T
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        return self
