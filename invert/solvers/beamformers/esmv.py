import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta


class SolverESMV(BaseSolver):
    """Class for the Eigenspace-based Minimum Variance (ESMV) Beamformer
        inverse solution [1].

    References
    ----------
    [1] Jonmohamadi, Y., Poudel, G., Innes, C., Weiss, D., Krueger, R., & Jones,
    R. (2014). Comparison of beamformers for EEG source signal reconstruction.
    Biomedical Signal Processing and Control, 14, 175-188.

    """

    meta = SolverMeta(
        slug="esmv",
        full_name="Eigenspace-based Minimum Variance",
        category="Beamformers",
        description=(
            "Eigenspace-projected minimum-variance beamformer (ESMV), using a "
            "signal-subspace projection of the classic MV filter."
        ),
        references=[
            "Jonmohamadi, Y., Poudel, G., Innes, C., Weiss, D., Krueger, R., & Jones, R. "
            "(2014). Comparison of beamformers for EEG source signal reconstruction. "
            "Biomedical Signal Processing and Control, 14, 175-188.",
        ],
    )

    def __init__(self, name="ESMV Beamformer", reduce_rank=True, rank="auto", **kwargs):
        self.name = name
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, alpha="auto", **kwargs):
        """Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        alpha : float
            The regularization parameter.

        Return
        ------
        self : object returns itself for convenience

        """
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)

        leadfield = self.leadfield
        leadfield /= np.linalg.norm(leadfield, axis=0)
        n_chans, n_dipoles = leadfield.shape

        y = data
        I = np.identity(n_chans)

        # Recompute regularization based on the max eigenvalue of the Covariance
        # Matrix (opposed to that of the leadfield)
        y -= y.mean(axis=1, keepdims=True)
        C = self.data_covariance(y, center=False, ddof=1)

        # Reduce rank of covariance matrix
        C_sub = self.select_signal_subspace(C)

        self.alphas = self.get_alphas(reference=C)

        inverse_operators = []
        for alpha in self.alphas:
            C_inv = self.robust_inverse(C + alpha * I)
            C_inv_leadfield = C_inv @ leadfield
            diag_elements = np.einsum("ij,ji->i", leadfield.T, C_inv_leadfield)
            W = C_inv_leadfield / diag_elements
            W_ESMV = C_sub @ W

            inverse_operator = W_ESMV.T
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        return self
