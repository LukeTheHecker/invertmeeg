import mne
import numpy as np
from scipy.sparse.csgraph import laplacian

from ..base import BaseSolver, InverseOperator, SolverMeta


class SolverSMAP(BaseSolver):
    """Class for the Quadratic regularization and spatial regularization
        (S-MAP) inverse solution [1].

    References
    ----------
    [1] Baillet, S., & Garnero, L. (1997). A Bayesian approach to introducing
    anatomo-functional priors in the EEG/MEG inverse problem. IEEE transactions
    on Biomedical Engineering, 44(5), 374-385.

    """

    meta = SolverMeta(
        acronym="S-MAP",
        full_name="S-MAP (Quadratic + Spatial MAP)",
        category="Minimum Norm",
        description=(
            "MAP-style quadratic inverse with spatial regularization via a "
            "graph Laplacian over the source space."
        ),
        references=[
            "Baillet, S., & Garnero, L. (1997). A Bayesian approach to introducing anatomo-functional priors in the EEG/MEG inverse problem. IEEE Transactions on Biomedical Engineering, 44(5), 374–385.",
        ],
    )

    def __init__(self, name="S-MAP", **kwargs):
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
        LTL = self.leadfield.T @ self.leadfield
        # n_chans, n_dipoles = self.leadfield.shape

        adjacency = mne.spatial_src_adjacency(self.forward["src"], verbose=0)
        gradient = laplacian(adjacency)
        GTG = gradient.T @ gradient

        if alpha == "auto":
            r_grid = np.asarray(self.r_values, dtype=float)
        else:
            r_grid = np.asarray([float(alpha)], dtype=float)
        max_eig_LTL = float(np.linalg.svd(LTL, compute_uv=False).max())
        max_eig_penalty = float(np.linalg.svd(GTG, compute_uv=False).max())
        scale = max_eig_LTL / max(max_eig_penalty, 1e-15)
        self.alphas = list(r_grid * scale)

        inverse_operators = []
        # GG_inv = np.linalg.inv(GTG)
        for alpha in self.alphas:
            inverse_operator = np.linalg.inv(LTL + alpha * GTG) @ self.leadfield.T
            # inverse_operator = GG_inv @ self.leadfield.T @ np.linalg.inv(self.leadfield @ GG_inv @ self.leadfield.T + alpha * np.identity(n_chans))
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]

        return self
