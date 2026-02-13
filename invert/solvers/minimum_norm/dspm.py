import logging

import mne
import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverDSPM(BaseSolver):
    """Class for the Dynamic Statistical Parametric Mapping (dSPM) inverse
        solution [1,2].  The formulas provided by [3] were used for
        implementation.

    References
    ----------
    [1] Dale, A. M., Liu, A. K., Fischl, B. R., Buckner, R. L., Belliveau, J.
    W., Lewine, J. D., & Halgren, E. (2000). Dynamic statistical parametric
    mapping: combining fMRI and MEG for high-resolution imaging of cortical
    activity. neuron, 26(1), 55-67.

    [2] Dale, A. M., Fischl, B., & Sereno, M. I. (1999). Cortical surface-based
    analysis: I. Segmentation and surface reconstruction. Neuroimage, 9(2),
    179-194.

    [3] Grech, R., Cassar, T., Muscat, J., Camilleri, K. P., Fabri, S. G.,
    Zervakis, M., ... & Vanrumste, B. (2008). Review on solving the inverse
    problem in EEG source analysis. Journal of neuroengineering and
    rehabilitation, 5(1), 1-33.
    """

    meta = SolverMeta(
        acronym="dSPM",
        full_name="Dynamic Statistical Parametric Mapping",
        category="Minimum Norm",
        description=(
            "Noise-normalized minimum-norm inverse (MNE) that yields statistical "
            "maps by scaling the MNE estimate with an estimate of its variance."
        ),
        references=[
            "Dale, A. M., Liu, A. K., Fischl, B. R., Buckner, R. L., Belliveau, J. W., Lewine, J. D., & Halgren, E. (2000). Dynamic statistical parametric mapping: combining fMRI and MEG for high-resolution imaging of cortical activity. Neuron, 26(1), 55–67.",
        ],
    )

    def __init__(
        self,
        name="Dynamic Statistical Parametric Mapping",
        *,
        depth: float = 0.8,
        depth_limit: float = 10.0,
        use_noise_whitener: bool = True,
        use_trace_normalization: bool = False,
        rank_tol: float = 1e-12,
        eps: float = 1e-15,
        **kwargs,
    ):
        self.name = name
        self.depth = float(depth)
        self.depth_limit = float(depth_limit)
        self.use_noise_whitener = bool(use_noise_whitener)
        self.use_trace_normalization = bool(use_trace_normalization)
        self.rank_tol = float(rank_tol)
        self.eps = float(eps)

        kwargs.setdefault("use_depth_weighting", False)
        super().__init__(**kwargs)

        # dSPM depends only on the forward model (and optional covariances),
        # so it can be computed once per dataset.
        self.require_recompute = False
        self.require_data = False
        return None

    def prepare_forward(self) -> None:
        """Prepare forward model but skip BaseSolver's depth-weight scaling."""
        # BaseSolver.prepare_forward applies its own depth weighting when
        # `self.use_depth_weighting` is True. For this solver, depth weighting
        # is implemented as a source prior, so disable it for the call.
        orig = bool(self.use_depth_weighting)
        self.use_depth_weighting = False
        try:
            super().prepare_forward()
        finally:
            self.use_depth_weighting = orig

    def make_inverse_operator(
        self,
        forward,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
        source_cov=None,
        verbose=0,
        **kwargs,
    ):
        """Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        alpha : float
            The regularization parameter.
        noise_cov : numpy.ndarray
            The noise covariance matrix (channels x channels).
        source_cov : numpy.ndarray
            The source covariance matrix (dipoles x dipoles). This can be used if
            prior information, e.g., from fMRI images, is available.

        Return
        ------
        self : object returns itself for convenience
        """
        if "depth_weighting" in kwargs and "depth" not in kwargs:
            kwargs["depth"] = kwargs.pop("depth_weighting")
        if "depth" in kwargs:
            self.depth = float(kwargs.pop("depth"))

        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)

        if self.use_noise_whitener:
            # If no covariance is provided, do not silently project; behave like
            # MNE/wMNE and use an identity transform in that case.
            if noise_cov is None:
                wf = self.prepare_whitened_forward(
                    None,
                    apply_projector_when_no_cov=False,
                    source_cov=source_cov,
                    depth=self.depth if self.use_depth_weighting else None,
                    depth_limit=self.depth_limit,
                    trace_normalize=self.use_trace_normalization,
                    precompute_svd=True,
                    rank_tol=self.rank_tol,
                    eps=self.eps,
                )
            else:
                wf = self.prepare_whitened_forward(
                    noise_cov,
                    source_cov=source_cov,
                    depth=self.depth if self.use_depth_weighting else None,
                    depth_limit=self.depth_limit,
                    trace_normalize=self.use_trace_normalization,
                    precompute_svd=True,
                    rank_tol=self.rank_tol,
                    eps=self.eps,
                )
        else:
            wf = self.prepare_whitened_forward(
                None,
                source_cov=source_cov,
                depth=self.depth if self.use_depth_weighting else None,
                depth_limit=self.depth_limit,
                trace_normalize=self.use_trace_normalization,
                precompute_svd=True,
                rank_tol=self.rank_tol,
                eps=self.eps,
            )
        if wf.whitener_mode != "projected" and wf.whitener_mode != "none":
            logger.warning("dSPM whitener fallback used: %s", wf.whitener_mode)

        if wf.A is not None:
            tikhonov_matrix = wf.A
            left_scale = wf.R_sqrt
            svd = wf.svd
        elif wf.R_sqrt is not None:
            tikhonov_matrix = wf.G_white * wf.R_sqrt[np.newaxis, :]
            left_scale = wf.R_sqrt
            # Precomputed SVD is for G_white in this branch, not G_white*R_sqrt.
            svd = None
        else:
            tikhonov_matrix = wf.G_white
            left_scale = None
            svd = wf.svd

        # Scale regularization candidates to match the effective matrix used for
        # the Tikhonov solve (whitened/projected and optionally prior-weighted).
        self.get_alphas(reference=tikhonov_matrix @ tikhonov_matrix.T)

        inverse_operators = []
        mne_operators = []
        for alpha_eff in self.alphas:
            alpha_eff = float(alpha_eff)
            K_white = self.solve_tikhonov_svd(
                tikhonov_matrix,
                alpha_eff,
                left_scale=left_scale,
                svd=svd,
                eps=self.eps,
            )

            # Full operator maps raw sensor data -> sources (and projects/whitens internally).
            K_full = K_white @ wf.sensor_transform  # (n_sources, n_chans)

            # dSPM noise normalization (whitened noise cov is I).
            K_dspm, _noise_std = self.noise_normalize_rows(
                K_white,
                K_full=K_full,
                eps=self.eps,
            )

            inverse_operators.append(K_dspm)
            mne_operators.append(K_full)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        # Keep the underlying MNE kernels around for regularization selection.
        # dSPM kernels are noise-normalized statistical maps; their "hat matrix"
        # can have trace > n_channels, which breaks GCV/L-curve assumptions.
        self._mne_inverse_operators = [
            InverseOperator(mne_operator, f"{self.name} (MNE kernel)")
            for mne_operator in mne_operators
        ]
        return self

    def apply_inverse_operator(self, mne_obj) -> "mne.SourceEstimate":  # type: ignore[name-defined]
        """Apply dSPM operator with regularization chosen on the MNE kernel.

        dSPM is a *noise-normalized* version of the MNE estimate. Regularization
        selection methods like GCV, L-curve, or the product criterion are
        defined in terms of data fit and should therefore operate on the
        underlying (non-normalized) MNE kernel. Selecting α on the dSPM kernel
        can yield invalid effective degrees of freedom (e.g. trace(H) > n),
        producing only ``inf`` GCV values.
        """
        data = self.unpack_data_obj(mne_obj)
        self.validate_operator_data_compatibility(data)

        dspm_ops = self.inverse_operators
        reg_ops = getattr(self, "_mne_inverse_operators", None) or dspm_ops

        if self.use_last_alpha and self.last_reg_idx is not None:
            idx = int(self.last_reg_idx)
        else:
            original_ops = self.inverse_operators
            try:
                # Run the selection on the unnormalized MNE kernels.
                self.inverse_operators = reg_ops
                method = self.regularisation_method.lower()
                if method == "l":
                    _, idx = self.regularise_lcurve(data, plot=self.plot_reg)
                elif method in {"gcv", "mgcv", "rgcv", "r1gcv", "composite"}:
                    if method == "gcv":
                        gamma = self.gcv_gamma
                    elif method == "mgcv":
                        gamma = self.mgcv_gamma
                    elif method in {"rgcv", "composite"}:
                        gamma = self.rgcv_gamma
                    else:
                        gamma = self.r1gcv_gamma
                    _, idx = super().regularise_gcv(
                        data, plot=self.plot_reg, gamma=gamma, method=method
                    )
                elif method == "product":
                    _, idx = self.regularise_product(data, plot=self.plot_reg)
                else:
                    msg = f"{self.regularisation_method} is no valid regularisation method."
                    raise AttributeError(msg)
            finally:
                self.inverse_operators = original_ops

            self.last_reg_idx = int(idx)
            idx = int(idx)

        source_mat = dspm_ops[idx].apply(data)
        return self.source_to_object(source_mat)
