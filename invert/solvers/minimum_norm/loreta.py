import logging

import mne
import numpy as np
from scipy.sparse.csgraph import laplacian

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverLORETA(BaseSolver):
    """Class for the Low Resolution Tomography (LORETA) inverse solution.

    References
    ----------
    [1] Pascual-Marqui, R. D. (1999). Review of methods for solving the EEG
    inverse problem. International journal of bioelectromagnetism, 1(1), 75-86.

    """

    meta = SolverMeta(
        acronym="LORETA",
        full_name="Low Resolution Electromagnetic Tomography",
        category="Minimum Norm",
        description=(
            "Smoothness-constrained minimum-norm inverse that penalizes spatial "
            "roughness via a Laplacian operator on the source space."
        ),
        references=[
            "Pascual-Marqui, R. D., Michel, C. M., & Lehmann, D. (1994). Low resolution electromagnetic tomography: a new method for localizing electrical activity in the brain. International Journal of Psychophysiology, 18(1), 49–65.",
        ],
    )

    def __init__(self, name="Low Resolution Tomography", **kwargs):
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
        LTL = leadfield.T @ leadfield
        B = np.eye(leadfield.shape[1])
        adjacency = mne.spatial_src_adjacency(
            forward["src"], verbose=self.verbose
        ).toarray()
        laplace_operator = laplacian(adjacency)
        BLapTLapB = B @ laplace_operator.T @ laplace_operator @ B

        if alpha == "auto":
            r_grid = np.asarray(self.r_values, dtype=float)
        else:
            r_grid = np.asarray([float(alpha)], dtype=float)
        max_eig_LTL = float(np.linalg.svd(LTL, compute_uv=False).max())
        max_eig_penalty = float(np.linalg.svd(BLapTLapB, compute_uv=False).max())
        scale = max_eig_LTL / max(max_eig_penalty, 1e-15)
        self.alphas = list(r_grid * scale)

        inverse_operators = []
        for alpha in self.alphas:
            inverse_operator = np.linalg.inv(LTL + (alpha) * BLapTLapB) @ leadfield.T
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        return self
