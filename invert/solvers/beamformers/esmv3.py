import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta
from .utils import bayesian_pca_covariance


class SolverESMV3(BaseSolver):
    """Class for the Eigenspace-based Minimum Variance (ESMV) Beamformer
        inverse solution [1].

    References
    ----------
    [1] Jonmohamadi, Y., Poudel, G., Innes, C., Weiss, D., Krueger, R., & Jones,
    R. (2014). Comparison of beamformers for EEG source signal reconstruction.
    Biomedical Signal Processing and Control, 14, 175-188.

    """

    meta = SolverMeta(
        slug="esmv3",
        full_name="ESMV (variant 3)",
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
        self, name="ESMV3 Beamformer", reduce_rank=True, rank="auto", **kwargs
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
        data.shape[1]
        epsilon = 1e-8

        y = data
        I = np.identity(n_chans)

        # CAR
        # y -= y.mean(axis=0, keepdims=True)
        # leadfield -= leadfield.mean(axis=0, keepdims=True)
        # y = apply_rest(y, leadfield)

        # Recompute regularization based on the max eigenvalue of the Covariance
        # Matrix (opposed to that of the leadfield)
        # C = (y@y.T) / n_times
        C = bayesian_pca_covariance(y, rank=None, var_threshold=0.95)[0]

        # Shrinking (not better than no shrinking in my experience)
        # C = LedoitWolf(assume_centered=True).fit(C).covariance_
        # C = OAS(assume_centered=True).fit(C).covariance_
        # C = ShrunkCovariance().fit(C).covariance_

        self.alphas = self.get_alphas(reference=C)
        # self.alphas = [0]

        inverse_operators = []
        for alpha in self.alphas:
            C_reg = C + alpha * I

            # Robust inverse
            C_inv = self.robust_inverse(C_reg)

            # Whitened inverse (C**(-0.5))
            # eigvals_reg, eigvecs_reg = np.linalg.eigh(C_reg)
            # C_inv = eigvecs_reg @ np.diag(1.0 / np.sqrt(np.maximum(eigvals_reg, 1e-12))) @ eigvecs_reg.T

            # Eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C_reg)
            idx = np.argsort(eigvals)[::-1]
            eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

            # Select signal subspace
            n_comp = self.estimate_n_sources(C_reg, method="auto")
            E_S = eigvecs[:, :n_comp]
            Λ_S = np.diag(abs(eigvals[:n_comp]))

            # Compute LCMV weights
            C_inv_leadfield = C_inv @ leadfield
            diag_elements = np.einsum("ij,ji->i", leadfield.T, C_inv_leadfield)
            W_mv = C_inv_leadfield / (diag_elements + epsilon)

            # Weighted eigenspace projection (mathematically sound)
            # Option 1: Square root of eigenvalues (like Mahalanobis whitening)
            W_ESMV = E_S @ np.sqrt(Λ_S) @ (E_S.T @ W_mv)

            # # Option 2: Full eigenvalues (like the "wrong" version but explicit)
            # W_ESMV = E_S @ Λ_S @ (E_S.T @ W_mv)

            # # Option 3: Tunable weighting with exponent β
            # β = 0.25  # β=0 → standard ESMV, β=1 → covariance weighted, β=1.5 → more aggressive weighting
            # W_ESMV = E_S @ (Λ_S ** β) @ (E_S.T @ W_mv)

            inverse_operator = W_ESMV.T
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        return self
