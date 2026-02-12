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

    def __init__(
        self,
        name="LCMV Beamformer",
        reduce_rank=True,
        rank="auto",
        use_robust_covariance: bool = False,
        rank_tol: float = 1e-12,
        eps: float = 1e-15,
        **kwargs,
    ):
        self.name = name
        self.use_robust_covariance = bool(use_robust_covariance)
        self.rank_tol = float(rank_tol)
        self.eps = float(eps)
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(
        self,
        forward,
        mne_obj,
        *args,
        alpha="auto",
        noise_cov=None,
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
        # Match legacy behavior: column-normalize leadfield *in place* so
        # downstream solvers/tests that relied on this side-effect keep working.
        leadfield = self.leadfield
        lead_norms = np.linalg.norm(leadfield, axis=0)
        leadfield /= np.maximum(lead_norms, self.eps)

        n_chans_raw = int(leadfield.shape[0])
        y_raw = data
        noise_cov_raw = None
        if noise_cov is not None:
            noise_cov_raw = np.asarray(noise_cov, dtype=float)
            if noise_cov_raw.shape != (n_chans_raw, n_chans_raw):
                msg = (
                    f"noise_cov has shape {noise_cov_raw.shape}, "
                    f"expected {(n_chans_raw, n_chans_raw)}"
                )
                raise ValueError(msg)
            noise_cov_raw = 0.5 * (noise_cov_raw + noise_cov_raw.T)

        # Optional robust path: whiten + project in sensor space using noise_cov.
        use_robust_covariance = bool(self.use_robust_covariance)
        if use_robust_covariance:
            if noise_cov_raw is None:
                noise_cov_raw = np.eye(n_chans_raw, dtype=float)
            wf = self.prepare_whitened_forward(
                noise_cov_raw,
                rank_tol=self.rank_tol,
                eps=self.eps,
            )
            if wf.whitener_mode not in ("projected", "none"):
                logger.warning("LCMV whitener fallback used: %s", wf.whitener_mode)
            sensor_transform = wf.sensor_transform
            leadfield_eff = wf.G_white
            y_eff = sensor_transform @ y_raw
        else:
            sensor_transform = None
            leadfield_eff = leadfield
            y_eff = y_raw

        n_chans_eff = int(leadfield_eff.shape[0])
        I = np.identity(n_chans_eff)

        y_eff -= y_eff.mean(axis=1, keepdims=True)
        C = self.data_covariance(y_eff, center=False, ddof=1)
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
            upper = C_inv @ leadfield_eff
            lower = np.einsum("ij,jk,ki->i", leadfield_eff.T, C_inv, leadfield_eff)
            W_eff = np.zeros_like(upper)
            valid = np.abs(lower) > self.eps
            if np.any(valid):
                W_eff[:, valid] = upper[:, valid] / lower[valid]

            # C_inv_L = C_inv @ leadfield
            # diagonal_elements = np.einsum('ij,ji->i', leadfield.T, C_inv_L)
            # W = C_inv_L / diagonal_elements

            # Map weights back to raw sensor space so the operator can be applied
            # to raw sensor data (matches dSPM-MNE pattern: K_full = K_white @ W_P).
            if self.weight_norm:
                if sensor_transform is not None:
                    # In whitened coordinates the sensor-noise covariance is I.
                    noise_power = np.sum(W_eff * W_eff, axis=0)
                    W_eff = W_eff / np.sqrt(np.maximum(noise_power, self.eps))
                    W_raw = sensor_transform.T @ W_eff
                else:
                    W_raw = W_eff
                    if noise_cov_raw is not None:
                        noise_power = np.sum(
                            W_raw * (noise_cov_raw @ W_raw),
                            axis=0,
                        )
                    else:
                        noise_power = np.sum(W_raw * W_raw, axis=0)
                    W_raw = W_raw / np.sqrt(np.maximum(noise_power, self.eps))
            elif sensor_transform is None:
                W_raw = W_eff
            else:
                W_raw = sensor_transform.T @ W_eff

            inverse_operator = W_raw.T
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        return self
