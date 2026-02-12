import logging

import mne
import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        name="ESMV Beamformer",
        reduce_rank=True,
        rank="auto",
        use_robust_covariance: bool = False,
        use_noise_whitening: bool = True,
        subspace_weighting: str = "eig",
        enforce_unit_gain_after_subspace: bool = False,
        rank_tol: float = 1e-12,
        eps: float = 1e-15,
        **kwargs,
    ):
        self.name = name
        self.use_robust_covariance = bool(use_robust_covariance)
        self.use_noise_whitening = bool(use_noise_whitening)
        self.subspace_weighting = str(subspace_weighting).lower()
        self.enforce_unit_gain_after_subspace = bool(enforce_unit_gain_after_subspace)
        self.rank_tol = float(rank_tol)
        self.eps = float(eps)
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(
        self,
        forward,
        mne_obj,
        *args,
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
        alpha : float
            The regularization parameter.

        Return
        ------
        self : object returns itself for convenience

        """
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        if noise_cov is not None:
            self.coerce_noise_cov(noise_cov)
        data = self.unpack_data_obj(mne_obj)

        leadfield = self.leadfield
        _n_chans_raw, _n_dipoles = leadfield.shape
        y_raw = np.asarray(data, dtype=float)

        lead_norms = np.linalg.norm(leadfield, axis=0)
        leadfield /= np.maximum(lead_norms, self.eps)

        if self.use_robust_covariance:
            if self.use_noise_whitening:
                if noise_cov is None:
                    noise_cov = self.make_identity_noise_cov(
                        list(self.forward.ch_names)
                    )
                wf = self.prepare_whitened_forward(
                    noise_cov,
                    rank_tol=self.rank_tol,
                    eps=self.eps,
                )
                if wf.whitener_mode not in ("projected", "none"):
                    logger.warning("ESMV whitener fallback used: %s", wf.whitener_mode)
            else:
                wf = self.prepare_whitened_forward(None)
            sensor_transform = wf.sensor_transform
            leadfield_eff = wf.G_white
            y_eff = sensor_transform @ y_raw
        else:
            sensor_transform = None
            leadfield_eff = leadfield
            y_eff = y_raw
        n_chans_eff = int(leadfield_eff.shape[0])
        I = np.identity(n_chans_eff)

        # Recompute regularization based on the max eigenvalue of the Covariance
        # Matrix (opposed to that of the leadfield)
        y_eff -= y_eff.mean(axis=1, keepdims=True)
        C = self.data_covariance(y_eff, center=False, ddof=1)
        eigvals, eigvecs = np.linalg.eigh(C)
        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order]
        if isinstance(self.rank, str):
            n_comp = self.estimate_n_sources(C, method=self.rank)
        elif isinstance(self.rank, int):
            n_comp = self.rank
        else:
            raise ValueError(
                f"Unknown rank: {self.rank}. Must be 'auto', 'L', 'drop', 'mean', or a positive integer."
            )
        n_comp = int(np.clip(n_comp, 1, n_chans_eff))
        signal_subspace = eigvecs[:, :n_comp]
        signal_eigs = np.maximum(eigvals[order][:n_comp], 0.0)
        if self.subspace_weighting == "projector":
            subspace_weights = np.ones(n_comp, dtype=float)
        elif self.subspace_weighting == "sqrt_eig":
            max_ev = max(float(np.max(signal_eigs)), self.eps)
            subspace_weights = np.sqrt(signal_eigs / max_ev)
        elif self.subspace_weighting == "eig":
            max_ev = max(float(np.max(signal_eigs)), self.eps)
            subspace_weights = signal_eigs / max_ev
        else:
            raise ValueError(
                "subspace_weighting must be one of {'projector', 'sqrt_eig', 'eig'}"
            )
        subspace_operator = signal_subspace @ (
            subspace_weights[:, np.newaxis] * signal_subspace.T
        )

        self.alphas = self.get_alphas(reference=C)

        inverse_operators = []
        for alpha in self.alphas:
            C_inv = self.robust_inverse(C + alpha * I)
            C_inv_leadfield = C_inv @ leadfield_eff
            diag_elements = np.einsum("ij,ji->i", leadfield_eff.T, C_inv_leadfield)
            W_mv = np.zeros_like(C_inv_leadfield)
            valid = np.abs(diag_elements) > self.eps
            if np.any(valid):
                W_mv[:, valid] = C_inv_leadfield[:, valid] / diag_elements[valid]
            W_esmv_eff = subspace_operator @ W_mv

            if self.enforce_unit_gain_after_subspace:
                gains = np.einsum("ij,ji->i", leadfield_eff.T, W_esmv_eff)
                valid_gain = np.abs(gains) > self.eps
                if np.any(valid_gain):
                    W_esmv_eff[:, valid_gain] = (
                        W_esmv_eff[:, valid_gain] / gains[valid_gain]
                    )
                if np.any(~valid_gain):
                    W_esmv_eff[:, ~valid_gain] = W_mv[:, ~valid_gain]

            inverse_operator = W_esmv_eff.T
            # If we computed the filter in whitened/projected sensor space,
            # map it back so the operator can be applied to raw sensor data.
            if sensor_transform is not None:
                inverse_operator = inverse_operator @ sensor_transform
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        return self
