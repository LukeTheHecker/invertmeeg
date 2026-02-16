import logging

import mne
import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta
from .utils import (
    build_covariance_candidates,
)

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
        free_orientation_collapse: str = "optimal",
        rank_tol: float = 1e-12,
        eps: float = 1e-15,
        **kwargs,
    ):
        kwargs.setdefault("regularisation_method", "L")
        self.name = name
        self.use_robust_covariance = bool(use_robust_covariance)
        self.free_orientation_collapse = str(free_orientation_collapse).lower()
        if self.free_orientation_collapse not in {"amplitude", "optimal"}:
            raise ValueError(
                "free_orientation_collapse must be one of {'amplitude','optimal'}, "
                f"got {free_orientation_collapse!r}"
            )
        self.rank_tol = float(rank_tol)
        self.eps = float(eps)
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    # LCMV supports true free orientation (vector weights) and can optionally
    # collapse per-location outputs using an optimal-orientation criterion.
    SUPPORTS_VECTOR_ORIENTATION: bool = True

    def _compute_source_mat(self, data: np.ndarray):  # type: ignore[override]
        """Beamformer-specific alpha selection.

        The BaseSolver's generic GCV/L-curve criteria are designed for
        regularized linear inverse operators where ``G @ x ≈ y``. For
        beamformers, those residual-based criteria are not meaningful.

        For now we use the smallest diagonal-loading value in the search grid
        (or the single user-provided alpha).
        """
        if self.use_last_alpha and self.last_reg_idx is not None:
            idx = int(self.last_reg_idx)
            return self.inverse_operators[idx].apply(data), idx

        idx = 0
        self.last_reg_idx = idx
        return self.inverse_operators[idx].apply(data), idx

    def _unpack_cov_samples(
        self,
        mne_obj,
        *,
        pick_types: list[str] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Return a (n_channels, n_samples) matrix for covariance estimation.

        For beamformers, estimating covariance from an Evoked average can be
        brittle (especially with near noise-free phantom/sim data). If Epochs
        are provided, concatenate all epochs in time to increase sample count.
        """
        return self.unpack_covariance_samples(
            mne_obj,
            pick_types=pick_types,
            require_forward_channel_match=True,
            context="LCMV",
        )

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
        super().make_inverse_operator(forward, mne_obj, *args, alpha=alpha, **kwargs)
        noise_cov_raw = None
        if noise_cov is not None:
            noise_cov_raw, noise_cov_ch_names = self.coerce_noise_cov(noise_cov)
            try:
                noise_cov_raw = self.reorder_covariance_to_channels(
                    noise_cov_raw,
                    noise_cov_ch_names,
                    list(self.forward.ch_names),
                )
            except KeyError:
                logger.warning(
                    "LCMV: noise_cov channels do not align with forward channels; "
                    "falling back to Euclidean weight normalization."
                )
                noise_cov_raw = None
            else:
                noise_cov_raw = 0.5 * (noise_cov_raw + noise_cov_raw.T)
        # Use more appropriate covariance samples when possible.
        y_raw, kept_ch_names = self._unpack_cov_samples(mne_obj)
        self._operator_ch_names = tuple(kept_ch_names)
        # Historical / test-contract behavior: normalize the (fixed-orientation)
        # leadfield columns prior to beamformer construction. For free
        # orientation, avoid block normalization because it can distort
        # max-power orientation selection on volume/discrete grids.
        leadfield = self.leadfield
        if not (
            getattr(self, "_free_orientation", False)
            and int(getattr(self, "_n_orient", 1)) == 3
        ):
            lead_norms = np.linalg.norm(leadfield, axis=0)
            leadfield /= np.maximum(lead_norms, self.eps)

        _n_chans_raw = int(leadfield.shape[0])
        # Optional robust path: whiten + project in sensor space using noise_cov.
        use_robust_covariance = bool(self.use_robust_covariance)
        if use_robust_covariance:
            if noise_cov is None:
                noise_cov = self.make_identity_noise_cov(list(self.forward.ch_names))
            wf = self.prepare_whitened_forward(
                noise_cov,
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
        cov_mats, self.alphas, cov_meta = build_covariance_candidates(
            C=C,
            I=I,
            alpha=self.alpha,
            get_alphas_fn=self.get_alphas,
            n_samples=int(y_eff.shape[1]),
            cov_reg=cov_reg,
            cov_reg_beta=float(cov_reg_beta),
            cov_reg_cond_target=float(cov_reg_cond_target),
        )
        if "oas_shrinkage" in cov_meta:
            self._cov_reg_oas_shrinkage = float(cov_meta["oas_shrinkage"])

        inverse_operators = []
        for cov_mat in cov_mats:
            C_inv = self.robust_inverse(cov_mat)

            if getattr(self, "_free_orientation", False) and int(getattr(self, "_n_orient", 1)) == 3:
                # Vector LCMV: per-location 3x3 solve
                # W_i^T = C_inv L_i (L_i^T C_inv L_i)^-1
                L = np.asarray(leadfield_eff, dtype=float)
                m, n_cols = L.shape
                n_src = int(n_cols // 3)
                Lb = L.reshape(m, n_src, 3)
                CiL = (C_inv @ L).reshape(m, n_src, 3)
                G = np.einsum("mni,mnj->nij", Lb, CiL, optimize=True)  # (n,3,3)

                collapse_opt = self.free_orientation_collapse == "optimal"
                if collapse_opt:
                    # "max-power" optimal orientation (MNE-style):
                    #   q = argmax_q  (q^T B q) / (q^T A q),  with
                    #   A = L^T C_inv L  and  B = L^T C_inv C C_inv L
                    # Then scalar filter weights are:
                    #   w = (C_inv L q) / (q^T A q)
                    #
                    # This yields a single (scalar) filter per location.
                    CiL_mat = CiL.reshape(m, n_cols)  # (m, 3*n)
                    CCiL = (C @ CiL_mat).reshape(m, n_src, 3)
                    B = np.einsum("mni,mnj->nij", CiL, CCiL, optimize=True)  # (n,3,3)

                    q = np.zeros((n_src, 3), dtype=float)
                    for i in range(n_src):
                        Ai = 0.5 * (G[i] + G[i].T)
                        Bi = 0.5 * (B[i] + B[i].T)
                        try:
                            M = np.linalg.solve(Ai, Bi)
                        except np.linalg.LinAlgError:
                            M = np.linalg.pinv(Ai, rcond=1e-12) @ Bi
                        M = 0.5 * (M + M.T)
                        _evals, evecs = np.linalg.eigh(M)
                        qi = evecs[:, -1]
                        j = int(np.argmax(np.abs(qi)))
                        if qi[j] < 0:
                            qi = -qi
                        q[i] = qi

                    num = np.einsum("mni,ni->mn", CiL, q, optimize=True)  # (m,n)
                    den = np.einsum("ni,nij,nj->n", q, G, q, optimize=True)
                    den = np.maximum(den, self.eps)
                    W_eff = num / den[np.newaxis, :]
                else:
                    # Invert 3x3 blocks robustly.
                    G_inv = np.zeros_like(G)
                    for i in range(n_src):
                        Gi = G[i]
                        try:
                            G_inv[i] = np.linalg.inv(Gi)
                        except np.linalg.LinAlgError:
                            G_inv[i] = np.linalg.pinv(Gi, rcond=1e-12)

                    W_eff = np.einsum("mni,nij->mnj", CiL, G_inv, optimize=True).reshape(
                        m, n_cols
                    )
            else:
                # Scalar LCMV: classic unit-gain per source
                # W = (C_inv @ leadfield) / diag(leadfield^T C_inv leadfield)
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
        # If we collapsed free-orientation filters to scalar weights, mark this
        # solver instance as scalar so SourceEstimate construction doesn't try
        # to collapse again.
        if (
            getattr(self, "_free_orientation", False)
            and int(getattr(self, "_n_orient", 1)) == 3
            and self.free_orientation_collapse == "optimal"
        ):
            self._free_orientation = False
            self._n_orient = 1
        return self

    def apply_inverse_operator(self, mne_obj) -> mne.SourceEstimate:
        """Apply LCMV and optionally collapse free-orientation outputs optimally."""
        if (
            getattr(self, "_free_orientation", False)
            and int(getattr(self, "_n_orient", 1)) == 3
            and self.free_orientation_collapse == "optimal"
        ):
            data = self.unpack_data_obj(mne_obj)
            self.validate_operator_data_compatibility(data)
            source_mat, _idx = self._compute_source_mat(data)

            # source_mat is (3*n_src, T). Collapse per location via PCA on J_i.
            source_mat = np.asarray(source_mat, dtype=float)
            if source_mat.ndim == 1:
                source_mat = source_mat[:, np.newaxis]
            n_src = int(source_mat.shape[0] // 3)
            J = source_mat.reshape(n_src, 3, -1)
            C = np.einsum("nit,njt->nij", J, J, optimize=True)  # (n,3,3)
            _evals, evecs = np.linalg.eigh(C)
            q = evecs[:, :, -1]  # (n,3)
            # Deterministic sign: largest-magnitude component positive.
            idx = np.argmax(np.abs(q), axis=1)
            sel = q[np.arange(n_src), idx]
            flip = np.where(sel < 0, -1.0, 1.0)
            q = q * flip[:, np.newaxis]
            self._beamformer_q = q
            x = np.einsum("ni,nit->nt", q, J, optimize=True)  # signed scalar (n,T)

            # Build scalar STC without triggering norm-collapse again.
            was_free = bool(getattr(self, "_free_orientation", False))
            was_n_orient = int(getattr(self, "_n_orient", 3))
            try:
                self._free_orientation = False
                self._n_orient = 1
                return super().source_to_object(x)
            finally:
                self._free_orientation = was_free
                self._n_orient = was_n_orient

        return super().apply_inverse_operator(mne_obj)
