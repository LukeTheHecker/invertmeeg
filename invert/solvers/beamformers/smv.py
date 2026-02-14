import logging

import mne
import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverSMV(BaseSolver):
    """Class for the Standardized Minimum Variance (SMV) Beamformer inverse
        solution [1].

    References
    ----------
    [1] Jonmohamadi, Y., Poudel, G., Innes, C., Weiss, D., Krueger, R., & Jones, R.
    (2014). Comparison of beamformers for EEG source signal reconstruction.
    Biomedical Signal Processing and Control, 14, 175-188.

    """

    meta = SolverMeta(
        slug="smv",
        full_name="Standardized Minimum Variance",
        category="Beamformers",
        description=(
            "Standardized minimum-variance beamformer variant (as implemented here), "
            "including eigenspace projection options."
        ),
        references=[
            "Jonmohamadi, Y., Poudel, G., Innes, C., Weiss, D., Krueger, R., & Jones, R. "
            "(2014). Comparison of beamformers for EEG source signal reconstruction. "
            "Biomedical Signal Processing and Control, 14, 175-188.",
        ],
    )

    def __init__(self, name="SMV Beamformer", reduce_rank=True, rank="auto", **kwargs):
        kwargs.setdefault("regularisation_method", "L")
        self.name = name
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(
        self,
        forward,
        mne_obj=None,
        *args,
        weight_norm=True,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
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
        super().make_inverse_operator(forward, mne_obj, *args, alpha=alpha, **kwargs)
        wf = self.prepare_whitened_forward(noise_cov)
        data = self.unpack_data_obj(mne_obj)

        leadfield = wf.G_white
        leadfield /= np.linalg.norm(leadfield, axis=0)
        n_chans, n_dipoles = leadfield.shape

        self.weight_norm = weight_norm

        y = wf.sensor_transform @ data
        I = np.identity(n_chans)

        # Recompute regularization based on the max eigenvalue of the Covariance
        # Matrix (opposed to that of the leadfield)
        y -= y.mean(axis=1, keepdims=True)
        C = self.data_covariance(y, center=False, ddof=1)
        self.alphas = self.get_alphas(reference=C)
        logger.debug("bing")
        inverse_operators = []
        for alpha in self.alphas:
            C_inv = self.robust_inverse(C + alpha * I)
            W = (C_inv @ leadfield) / np.sqrt(
                np.diagonal(leadfield.T @ C_inv @ leadfield)
            )

            if self.weight_norm:
                W /= np.linalg.norm(W, axis=0)
            inverse_operator = W.T @ wf.sensor_transform
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        return self
