import mne
import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta
from .utils import build_covariance_candidates


class SolverMVAB(BaseSolver):
    """Class for the Minimum Variance Adaptive Beamformer (MVAB) inverse solution.

    Per-source beamformer with NAI weight normalization in whitened sensor space.

    References
    ----------
    [1] Sekihara, K., Nagarajan, S. S., Poeppel, D., & Marantz, A. (2004).
        Asymptotic SNR of scalar and vector minimum-variance beamformers for
        neuromagnetic source reconstruction. IEEE Transactions on Biomedical
        Engineering, 51(10), 1726-1734.
    """

    meta = SolverMeta(
        slug="mvab",
        full_name="Minimum Variance Adaptive Beamformer",
        category="Beamformers",
        description=(
            "Per-source minimum-variance adaptive beamformer (Sekihara) with "
            "NAI weight normalization in whitened sensor space."
        ),
        references=[
            "Sekihara, K., Nagarajan, S. S., Poeppel, D., & Marantz, A. (2004). "
            "Asymptotic SNR of scalar and vector minimum-variance beamformers for "
            "neuromagnetic source reconstruction. IEEE Transactions on Biomedical "
            "Engineering, 51(10), 1726-1734.",
        ],
    )

    def __init__(
        self,
        name="Minimum Variance Adaptive Beamformer",
        reduce_rank=True,
        rank="auto",
        **kwargs,
    ):
        kwargs.setdefault("regularisation_method", "L")
        self.name = name
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(
        self,
        forward,
        mne_obj=None,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
        cov_reg: str = "oas",
        cov_reg_beta: float = 0.05,
        cov_reg_cond_target: float = 1e4,
        weight_norm=True,
        **kwargs,
    ):
        """Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        alpha : float
            The regularization parameter.
        weight_norm : bool
            Apply NAI weight normalization (in whitened space, equivalent to
            dividing by ||w||).

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

        y = wf.sensor_transform @ data
        I = np.identity(n_chans)
        eps = 1e-15

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

            # Per-source scalar MVAB: w_i = R⁻¹ l_i / (l_i^T R⁻¹ l_i)
            CiL = C_inv @ leadfield
            denom = np.einsum("ij,ji->i", leadfield.T, CiL)  # l_i^T R⁻¹ l_i
            W = np.zeros_like(CiL)
            valid = np.abs(denom) > eps
            if np.any(valid):
                W[:, valid] = CiL[:, valid] / denom[valid]

            # NAI weight normalization: in whitened space noise cov = I,
            # so NAI reduces to dividing by ||w||.
            if weight_norm:
                norms = np.sqrt(np.sum(W * W, axis=0))
                norms = np.maximum(norms, eps)
                W /= norms

            # Map back to raw sensor space
            inverse_operator = (wf.sensor_transform.T @ W).T
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        return self
