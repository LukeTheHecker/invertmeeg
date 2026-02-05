import logging

import mne
import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverGFTMNE(BaseSolver):
    """Class for the Minimum Norm Estimate (MNE) inverse solution [1] with graph fourier transform (GFT).
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
        acronym="GFT-MNE",
        full_name="Graph Fourier MNE",
        category="Minimum Norm",
        description=(
            "Minimum-norm inverse performed in a graph Fourier basis (Laplacian "
            "eigenvectors), typically retaining low graph frequencies to impose "
            "spatial smoothness."
        ),
        references=[
            "Hämäläinen, M. S., & Ilmoniemi, R. J. (1994). Interpreting magnetic fields of the brain: minimum norm estimates. Medical & Biological Engineering & Computing, 32(1), 35–42.",
            "Shuman, D. I., Narang, S. K., Frossard, P., Ortega, A., & Vandergheynst, P. (2013). The emerging field of signal processing on graphs: Extending high-dimensional data analysis to networks and other irregular domains. IEEE Signal Processing Magazine, 30(3), 83–98.",
        ],
        internal=True,
    )

    def __init__(self, name="GFT Minimum Norm Estimate", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(
        self, forward, *args, alpha="auto", cutoff=0.3, verbose=0, **kwargs
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
        super().make_inverse_operator(
            forward, *args, reference=None, alpha=alpha, **kwargs
        )

        leadfield = self.leadfield
        n_chans, _ = leadfield.shape

        # Get Adjacency matrix

        adjacency = mne.spatial_src_adjacency(forward["src"], verbose=0)
        lap = laplacian(adjacency).astype(float)

        num_eigenvalues = lap.shape[0]
        cutoff_index = int(num_eigenvalues * cutoff)
        logger.info(f"Keeping {cutoff_index}/{num_eigenvalues} eigenvalues")
        eigenvalues, U = eigsh(lap, k=cutoff_index, which="SM")
        U = np.real(U)

        # Transform leadfield
        leadfield_gft = leadfield @ U
        leadfield_gft /= np.linalg.norm(leadfield_gft, axis=0)

        LLT = leadfield_gft @ leadfield_gft.T
        # Regularization should match the transformed system (leadfield_gft).
        self.get_alphas(reference=LLT)
        inverse_operators = []
        for alpha in self.alphas:
            inverse_operator = np.linalg.solve(
                LLT + alpha * np.identity(n_chans), leadfield_gft
            ).T
            inverse_operators.append(U @ inverse_operator)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        return self
