import mne
import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta


class SolverUnitNoiseGain(BaseSolver):
    """Class for the Unit Noise Gain (UNIG) Beamformer
    inverse solution [1].

    References
    ----------
    [1]
    """

    meta = SolverMeta(
        slug="unit_noise_gain",
        full_name="Unit-Noise-Gain Beamformer",
        category="Beamformers",
        description=(
            "Minimum-variance beamformer variant using unit-noise-gain weight "
            "normalization (as implemented here)."
        ),
        references=["tbd"],
    )

    def __init__(self, name="UNIG Beamformer", reduce_rank=True, rank="auto", **kwargs):
        self.name = name
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(
        self,
        forward,
        mne_obj,
        *args,
        weight_norm=True,
        noise_cov: mne.Covariance | None = None,
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

        wf = self.prepare_whitened_forward(noise_cov)
        leadfield = wf.G_white
        sensor_transform = wf.sensor_transform
        n_chans, n_dipoles = leadfield.shape

        self.weight_norm = weight_norm

        y = sensor_transform @ data
        I = np.identity(n_chans)

        # Recompute regularization based on the max eigenvalue of the Covariance
        # Matrix (opposed to that of the leadfield)
        y -= y.mean(axis=1, keepdims=True)
        C = self.data_covariance(y, center=False, ddof=1)
        self.alphas = self.get_alphas(reference=C)
        inverse_operators = []
        for alpha in self.alphas:
            C_inv = np.linalg.inv(C + alpha * I)
            C_inv_sq = C_inv @ C_inv
            leadfield_C_inv_sq = leadfield.T @ C_inv_sq

            # Use np.einsum to compute the diagonal elements
            diag_elements = np.einsum("ij,ji->i", leadfield_C_inv_sq, leadfield)

            W = C_inv @ leadfield * (1 / np.sqrt(diag_elements))

            if self.weight_norm:
                W /= np.linalg.norm(W, axis=0)

            # Map back to raw sensor space
            inverse_operator = (sensor_transform.T @ W).T
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]

        return self
