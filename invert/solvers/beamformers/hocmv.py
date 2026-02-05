import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta


class SolverHOCMV(BaseSolver):
    """Class for the Higher-Order Covariance Minimum Variance (HOCMV)
        Beamformer inverse solution [1].

    References
    ----------
    [1] Jonmohamadi, Y., Poudel, G., Innes, C., Weiss, D., Krueger, R., & Jones,
    R. (2014). Comparison of beamformers for EEG source signal reconstruction.
    Biomedical Signal Processing and Control, 14, 175-188.

    """

    meta = SolverMeta(
        slug="hocmv",
        full_name="Higher-Order Covariance Minimum Variance",
        category="Beamformers",
        description=(
            "Minimum-variance beamformer using higher-order covariance statistics "
            "(as implemented here)."
        ),
        references=[
            "Jonmohamadi, Y., Poudel, G., Innes, C., Weiss, D., Krueger, R., & Jones, R. "
            "(2014). Comparison of beamformers for EEG source signal reconstruction. "
            "Biomedical Signal Processing and Control, 14, 175-188.",
        ],
    )

    def __init__(
        self, name="HOCMV Beamformer", reduce_rank=True, rank="auto", **kwargs
    ):
        self.name = name
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(
        self,
        forward,
        mne_obj,
        *args,
        weight_norm=True,
        alpha="auto",
        order=3,
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
        order : int
            The order of the covariance matrix. Should be a positive integer not
            evenly divisible by two {3, 5, 7, ...}

        Return
        ------
        self : object returns itself for convenience
        """
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)

        leadfield = self.leadfield
        leadfield /= np.linalg.norm(leadfield, axis=0)
        n_chans, n_dipoles = self.leadfield.shape

        self.weight_norm = weight_norm

        y = data
        I = np.identity(n_chans)

        # Recompute regularization based on the max eigenvalue of the Covariance
        # Matrix (opposed to that of the leadfield)
        y -= y.mean(axis=1, keepdims=True)
        C = self.data_covariance(y, center=False, ddof=1)
        self.alphas = self.get_alphas(reference=C)

        inverse_operators = []
        for alpha in self.alphas:
            C_inv = self.robust_inverse(C + alpha * I)
            C_inv_n = np.linalg.matrix_power(C_inv, order)

            upper = C_inv @ leadfield
            lower = np.sqrt(
                abs(np.einsum("ij,jk,ki->i", leadfield.T, C_inv_n, leadfield))
            )
            W = upper / lower

            if self.weight_norm:
                W /= np.linalg.norm(W, axis=0)

            inverse_operator = W.T
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        return self
