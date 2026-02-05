import logging

import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverLCMV(BaseSolver):
    """Class for the Linearly Constrained Minimum Variance Beamformer (LCMV) inverse solution.

    References
    ----------
    [1] Van Veen, B. D., & Buckley, K. M. (1988). Beamforming: A versatile
        approach to spatial filtering. IEEE ASSP Magazine, 5(2), 4-24.
    """

    meta = SolverMeta(
        slug="lcmv",
        full_name="Linearly Constrained Minimum Variance",
        category="Beamformers",
        description=(
            "Classic time-domain linearly constrained minimum-variance (LCMV) "
            "beamformer / spatial filter."
        ),
        references=[
            "Van Veen, B. D., van Drongelen, W., Yuchtman, M., & Suzuki, A. (1997). "
            "Localization of brain electrical activity via linearly constrained minimum "
            "variance spatial filtering. IEEE Transactions on Biomedical Engineering, "
            "44(9), 867-880.",
            "Van Veen, B. D., & Buckley, K. M. (1988). Beamforming: A versatile "
            "approach to spatial filtering. IEEE ASSP Magazine, 5(2), 4-24.",
        ],
    )

    def __init__(self, name="LCMV Beamformer", reduce_rank=True, rank="auto", **kwargs):
        self.name = name
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(
        self,
        forward,
        mne_obj,
        *args,
        alpha="auto",
        weight_norm=True,
        verbose=0,
        **kwargs,
    ):
        """Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        weight_norm : bool
            Normalize the filter weight matrix W to unit length of the columns.
        alpha : float
            The regularization parameter.

        Return
        ------
        self : object returns itself for convenience

        """
        self.weight_norm = weight_norm
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)
        leadfield = self.leadfield
        leadfield /= np.linalg.norm(leadfield, axis=0)
        # leadfield -= leadfield.mean(axis=0)
        n_chans, n_dipoles = self.leadfield.shape

        y = data

        I = np.identity(n_chans)
        y -= y.mean(axis=1, keepdims=True)
        C = self.data_covariance(y, center=False, ddof=1)
        # C = OAS(assume_centered=False).fit(C.T).covariance_.T
        # C = LedoitWolf(assume_centered=False).fit(C.T).covariance_.T

        # Recompute regularization based on the max eigenvalue of the Covariance
        # Matrix (opposed to that of the leadfield)
        # self.alphas = np.logspace(-4, 1, self.n_reg_params) * np.diagonal(y@y.T).mean()
        self.get_alphas(reference=C)

        inverse_operators = []
        for alpha in self.alphas:
            C_inv = self.robust_inverse(C + alpha * I)

            # W = (C_inv @ leadfield) / np.diagonal(leadfield.T @ C_inv @ leadfield)
            upper = C_inv @ leadfield
            lower = np.einsum("ij,jk,ki->i", leadfield.T, C_inv, leadfield)
            W = upper / lower

            # C_inv_L = C_inv @ leadfield
            # diagonal_elements = np.einsum('ij,ji->i', leadfield.T, C_inv_L)
            # W = C_inv_L / diagonal_elements

            if self.weight_norm:
                W /= np.linalg.norm(W, axis=0)

            inverse_operator = W.T
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        return self
