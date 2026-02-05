import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta


class SolverESMV2(BaseSolver):
    """Class for the Eigenspace-based Minimum Variance (ESMV) Beamformer
        inverse solution [1].

    References
    ----------
    [1] Jonmohamadi, Y., Poudel, G., Innes, C., Weiss, D., Krueger, R., & Jones,
    R. (2014). Comparison of beamformers for EEG source signal reconstruction.
    Biomedical Signal Processing and Control, 14, 175-188.

    """

    meta = SolverMeta(
        slug="esmv2",
        full_name="ESMV (variant 2)",
        category="Beamformers",
        description=(
            "A project-specific ESMV variant (see implementation for details) based "
            "on eigenspace-projected minimum-variance beamforming."
        ),
        references=[
            "Lukas Hecker (2025). Unpublished.",
            "Jonmohamadi, Y., Poudel, G., Innes, C., Weiss, D., Krueger, R., & Jones, R. "
            "(2014). Comparison of beamformers for EEG source signal reconstruction. "
            "Biomedical Signal Processing and Control, 14, 175-188.",
        ],
    )

    def __init__(
        self, name="ESMV2 Beamformer", reduce_rank=True, rank="auto", **kwargs
    ):
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

        self.alphas = self.get_alphas(reference=C)
        epsilon = 1e-15

        inverse_operators = []
        for alpha in self.alphas:
            C_reg = C + alpha * I

            # 2. Eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C_reg)
            idx = np.argsort(eigvals)[::-1]
            eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

            n_comp = self.estimate_n_sources(C_reg, method="auto")
            E_S = eigvecs[:, :n_comp]

            C_inv = self.robust_inverse(C_reg)

            C_inv_leadfield = C_inv @ leadfield
            diag_elements = np.einsum("ij,ji->i", leadfield.T, C_inv_leadfield)
            W_mv = C_inv_leadfield / (diag_elements + epsilon)

            W_ESMV = E_S @ (E_S.T @ W_mv)

            inverse_operator = W_ESMV.T
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        return self
