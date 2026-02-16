import mne
import numpy as np

from ..base import InverseOperator, SolverMeta
from .base_beamformer import BaseBeamformer
from .utils import (
    build_covariance_candidates,
)


class SolverSAM(BaseBeamformer):
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
        mne_obj=None,
        *args,
        weight_norm=True,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
        cov_reg: str = "oas",
        cov_reg_beta: float = 0.05,
        cov_reg_cond_target: float = 1e4,
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
        super().make_inverse_operator(forward, mne_obj, *args, alpha=alpha, **kwargs)
        wf = self.prepare_whitened_forward(noise_cov)
        data = self.unpack_data_obj(mne_obj)

        self.weight_norm = weight_norm
        leadfield = wf.G_white
        n_chans, n_dipoles = leadfield.shape

        y = wf.sensor_transform @ data
        I = np.identity(n_chans)
        C = self.data_covariance(y, center=True, ddof=1)
        cov_mats, self.alphas, cov_meta = build_covariance_candidates(
            C=C,
            I=I,
            alpha=self.alpha,
            get_alphas_fn=self.get_alphas,
            n_samples=int(y.shape[1]),
            cov_reg=cov_reg,
            cov_reg_beta=float(cov_reg_beta),
            cov_reg_cond_target=float(cov_reg_cond_target),
        )
        if "oas_shrinkage" in cov_meta:
            self._cov_reg_oas_shrinkage = float(cov_meta["oas_shrinkage"])

        inverse_operators = []
        for cov_mat in cov_mats:
            C_inv = self.robust_inverse(cov_mat)
            weights: list[np.ndarray] = []
            for i in range(n_dipoles):
                l = leadfield[:, i][:, np.newaxis]
                w = (C_inv @ l) / (l.T @ C_inv @ l)
                weights.append(w)
            W = np.stack(weights, axis=1)[:, :, 0]
            if self.weight_norm:
                W = W / np.linalg.norm(W, axis=0)
            inverse_operator = W.T @ wf.sensor_transform
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        return self
