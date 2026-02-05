import logging

import numpy as np
from scipy.spatial.distance import cdist

from ...util import pos_from_forward
from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverBackusGilbert(BaseSolver):
    """Class for the Backus Gilbert inverse solution.

"""

    meta = SolverMeta(
        acronym="BG",
        full_name="Backus–Gilbert",
        category="Minimum Norm",
        description=(
            "Resolution-optimizing linear inverse method that trades off spatial "
            "resolution versus noise amplification using Backus–Gilbert theory."
        ),
        references=[
            "Backus, G., & Gilbert, F. (1968). The resolving power of gross earth data. Geophysical Journal of the Royal Astronomical Society, 16(2), 169–205.",
        ],
    )

    def __init__(self, name="Backus-Gilbert", **kwargs):
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
        _, n_dipoles = self.leadfield.shape
        pos = pos_from_forward(forward, verbose=self.verbose)
        dist = cdist(pos, pos)

        W_BG = []
        for i in range(n_dipoles):
            W_gamma_BG = np.diag(dist[i, :])
            W_BG.append(W_gamma_BG)

        C = []
        for i in range(n_dipoles):
            C_gamma = self.leadfield @ W_BG[i] @ self.leadfield.T
            C.append(C_gamma)

        F = self.leadfield @ self.leadfield.T

        E = []
        for i in range(n_dipoles):
            E_gamma = C[i] + F
            E.append(E_gamma)

        L = self.leadfield @ np.ones((n_dipoles, 1))

        T = []
        for i in range(n_dipoles):
            E_gamma_pinv = np.linalg.pinv(E[i])
            T_gamma = (E_gamma_pinv @ L) / (L.T @ E_gamma_pinv @ L)
            T.append(T_gamma)

        inverse_operators = [
            np.stack(T, axis=0)[:, :, 0],
        ]

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        return self
