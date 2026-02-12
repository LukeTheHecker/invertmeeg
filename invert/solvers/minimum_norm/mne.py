import logging

import mne
import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverMNE(BaseSolver):
    """Class for the Minimum Norm Estimate (MNE) inverse solution [1].

    The formulas provided by [2] were used for implementation.

    References
    ----------
    [1] Pascual-Marqui, R. D. (1999). Review of methods for solving the EEG
        inverse problem. International journal of bioelectromagnetism, 1(1), 75-86.

    [2] Grech, R., Cassar, T., Muscat, J., Camilleri, K. P., Fabri, S. G.,
        Zervakis, M., ... & Vanrumste, B. (2008). Review on solving the inverse
        problem in EEG source analysis. Journal of neuroengineering and
        rehabilitation, 5(1), 1-33.
    """

    meta = SolverMeta(
        acronym="MNE",
        full_name="Minimum Norm Estimate",
        category="Minimum Norm",
        description=(
            "Classic L2 minimum-norm inverse solution. Estimates source currents "
            "by minimising the L2 norm of the source distribution subject to the "
            "data fit constraint."
        ),
        references=[
            "Hämäläinen, M. S., & Ilmoniemi, R. J. (1994). Interpreting magnetic fields of the brain: minimum norm estimates. Medical & Biological Engineering & Computing, 32(1), 35–42.",
        ],
    )

    def __init__(
        self,
        name="Minimum Norm Estimate",
        use_noise_whitener: bool = True,
        rank_tol: float = 1e-12,
        eps: float = 1e-15,
        **kwargs,
    ):
        self.name = name
        self.use_noise_whitener = bool(use_noise_whitener)
        self.rank_tol = float(rank_tol)
        self.eps = float(eps)
        super().__init__(**kwargs)
        self.require_recompute = False
        self.require_data = False

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
            The regularization parameter. When set to a float, it is scaled
            by the largest eigenvalue of L @ L.T.
        noise_cov : numpy.ndarray | None
            Optional sensor noise covariance used for whitening when
            ``use_noise_whitener=True``.

        Return
        ------
        self : object returns itself for convenience
        """
        super().make_inverse_operator(
            forward, *args, reference=None, alpha=alpha, **kwargs
        )
        if noise_cov is not None:
            self.coerce_noise_cov(noise_cov)

        if self.use_noise_whitener:
            if noise_cov is None:
                wf = self.prepare_whitened_forward(
                    None,
                    apply_projector_when_no_cov=False,
                    rank_tol=self.rank_tol,
                    eps=self.eps,
                )
            else:
                wf = self.prepare_whitened_forward(
                    noise_cov,
                    rank_tol=self.rank_tol,
                    eps=self.eps,
                )
        else:
            wf = self.prepare_whitened_forward(None)
        if wf.whitener_mode not in ("projected", "none"):
            logger.warning("MNE whitener fallback used: %s", wf.whitener_mode)

        leadfield_sensor = wf.G_white
        sensor_transform = wf.sensor_transform

        # Keep alpha scaling consistent with the actual leadfield used to build K.
        self.get_alphas(reference=leadfield_sensor @ leadfield_sensor.T)

        inverse_operators = []
        if self.use_noise_whitener:
            svd = np.linalg.svd(leadfield_sensor, full_matrices=False)
            for alpha in self.alphas:
                kernel_eff = self.solve_tikhonov_svd(
                    leadfield_sensor,
                    float(alpha),
                    svd=svd,
                    eps=self.eps,
                )
                inverse_operators.append(kernel_eff @ sensor_transform)
        else:
            LLT = leadfield_sensor @ leadfield_sensor.T
            I = np.identity(int(leadfield_sensor.shape[0]))
            for alpha in self.alphas:
                kernel_eff = np.linalg.solve(LLT + float(alpha) * I, leadfield_sensor).T
                inverse_operators.append(kernel_eff @ sensor_transform)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        return self
