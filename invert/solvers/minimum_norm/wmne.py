import logging

import mne
import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverWMNE(BaseSolver):
    """Class for the Weighted Minimum Norm Estimate (wMNE) inverse solution
        [1].

    References
    ----------
    [1] Grech, R., Cassar, T., Muscat, J., Camilleri, K. P., Fabri, S. G.,
    Zervakis, M., ... & Vanrumste, B. (2008). Review on solving the inverse
    problem in EEG source analysis. Journal of neuroengineering and
    rehabilitation, 5(1), 1-33.
    """

    meta = SolverMeta(
        acronym="wMNE",
        full_name="Weighted Minimum Norm Estimate",
        category="Minimum Norm",
        description=(
            "Minimum-norm inverse with depth/weighting to reduce superficial bias "
            "by scaling the source prior or leadfield columns."
        ),
        references=[
            "Hämäläinen, M. S., & Ilmoniemi, R. J. (1994). Interpreting magnetic fields of the brain: minimum norm estimates. Medical & Biological Engineering & Computing, 32(1), 35–42.",
            "Lin, F.-H., Witzel, T., Ahlfors, S. P., Stufflebeam, S. M., Belliveau, J. W., & Hämäläinen, M. S. (2006). Assessing and improving the spatial accuracy in MEG source localization by depth-weighted minimum-norm estimates. NeuroImage, 31(1), 160–171.",
        ],
    )
    SUPPORTS_VECTOR_ORIENTATION: bool = True

    def __init__(
        self,
        name="Weighted Minimum Norm Estimate",
        use_noise_whitener: bool = True,
        rank_tol: float = 1e-12,
        eps: float = 1e-15,
        **kwargs,
    ):
        self.name = name
        self.use_noise_whitener = bool(use_noise_whitener)
        self.rank_tol = float(rank_tol)
        self.eps = float(eps)
        kwargs.setdefault("depth_weighting", 0.5)
        super().__init__(**kwargs)

    def make_inverse_operator(
        self,
        forward,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
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
        noise_cov : numpy.ndarray | None
            Optional sensor noise covariance used for whitening when
            ``use_noise_whitener=True``.

        Return
        ------
        self : object returns itself for convenience
        """
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        if noise_cov is not None:
            self.coerce_noise_cov(noise_cov)

        if self.use_noise_whitener:
            if noise_cov is None:
                noise_cov = self.make_identity_noise_cov(list(self.forward.ch_names))
            wf = self.prepare_whitened_forward(
                noise_cov,
                rank_tol=self.rank_tol,
                eps=self.eps,
            )
        else:
            wf = self.prepare_whitened_forward(None)
        if wf.whitener_mode not in ("projected", "none"):
            logger.warning("wMNE whitener fallback used: %s", wf.whitener_mode)

        leadfield = wf.G_white
        sensor_transform = wf.sensor_transform

        # Keep alpha scaling consistent with the transformed leadfield.
        self.get_alphas(reference=leadfield @ leadfield.T)

        col_norms = np.linalg.norm(leadfield, axis=0) ** float(self.depth_weighting)
        col_norms = np.maximum(col_norms, self.eps)
        WTW = np.diag(1.0 / (col_norms**2))
        LWTWL = leadfield @ WTW @ leadfield.T
        n_eff = int(leadfield.shape[0])

        inverse_operators = []
        for alpha in self.alphas:
            inverse_operator = (
                WTW
                @ np.linalg.solve(
                    LWTWL + float(alpha) * np.identity(n_eff), leadfield
                ).T
            )
            inverse_operators.append(inverse_operator @ sensor_transform)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        return self
