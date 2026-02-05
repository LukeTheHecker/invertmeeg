import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta


class SolverSAM(BaseSolver):
    """Class for the Synthetic Aperture Magnetometry Beamformer (SAM) inverse
    solution [1].

    References
    ----------
    [1] Robinson, S. E. V. J. (1999). Functional neuroimaging by synthetic
    aperture magnetometry (SAM). Recent advances in biomagnetism.

    """

    meta = SolverMeta(
        slug="sam",
        full_name="Synthetic Aperture Magnetometry",
        category="Beamformers",
        description=(
            "Synthetic Aperture Magnetometry (SAM) beamformer implementation for "
            "time-domain source power estimation."
        ),
        references=[
            "Robinson, S. E., & Vrba, J. (1999). Functional neuroimaging by synthetic "
            "aperture magnetometry (SAM). In Recent Advances in Biomagnetism.",
        ],
    )

    def __init__(self, name="SAM Beamformer", reduce_rank=True, rank="auto", **kwargs):
        self.name = name
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(
        self,
        forward,
        mne_obj,
        *args,
        weight_norm=True,
        alpha="auto",
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
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)

        self.weight_norm = weight_norm
        leadfield = self.leadfield
        n_chans, n_dipoles = leadfield.shape

        y = data
        I = np.identity(n_chans)
        C = self.data_covariance(y, center=True, ddof=1)
        self.get_alphas(reference=C)

        inverse_operators = []
        for alpha in self.alphas:
            C_inv = self.robust_inverse(C + alpha * I)
            W = []
            for i in range(n_dipoles):
                l = leadfield[:, i][:, np.newaxis]
                w = (C_inv @ l) / (l.T @ C_inv @ l)
                W.append(w)
            W = np.stack(W, axis=1)[:, :, 0]
            if self.weight_norm:
                W = W / np.linalg.norm(W, axis=0)
            inverse_operator = W.T
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        return self
