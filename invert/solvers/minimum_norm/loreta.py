import logging

import mne
import numpy as np
from scipy.sparse.csgraph import laplacian
from invert.util import build_source_adjacency

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverLORETA(BaseSolver):
    """Class for the Low Resolution Tomography (LORETA) inverse solution.

    References
    ----------
    [1] Pascual-Marqui, R. D. (1999). Review of methods for solving the EEG
    inverse problem. International journal of bioelectromagnetism, 1(1), 75-86.

    """

    meta = SolverMeta(
        acronym="LORETA",
        full_name="Low Resolution Electromagnetic Tomography",
        category="Minimum Norm",
        description=(
            "Smoothness-constrained minimum-norm inverse that penalizes spatial "
            "roughness via a Laplacian operator on the source space."
        ),
        references=[
            "Pascual-Marqui, R. D., Michel, C. M., & Lehmann, D. (1994). Low resolution electromagnetic tomography: a new method for localizing electrical activity in the brain. International Journal of Psychophysiology, 18(1), 49–65.",
        ],
    )

    def __init__(
        self,
        name="Low Resolution Tomography",
        use_noise_whitener: bool = True,
        use_trace_normalization: bool = True,
        rank_tol: float = 1e-12,
        eps: float = 1e-15,
        **kwargs,
    ):
        self.name = name
        self.use_noise_whitener = bool(use_noise_whitener)
        self.use_trace_normalization = bool(use_trace_normalization)
        self.rank_tol = float(rank_tol)
        self.eps = float(eps)
        super().__init__(**kwargs)
        # LORETA's smoothness penalty can push the optimal alpha above the
        # BaseSolver grid under low SNR; widen to avoid edge saturation.
        self.r_values = np.logspace(-10, 4, int(max(self.n_reg_params, 1)))

    def make_inverse_operator(
        self,
        forward,
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
        alpha : float
            The regularization parameter.

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
                trace_normalize=self.use_trace_normalization,
                rank_tol=self.rank_tol,
                eps=self.eps,
            )
        else:
            wf = self.prepare_whitened_forward(
                None,
                trace_normalize=self.use_trace_normalization,
                rank_tol=self.rank_tol,
                eps=self.eps,
            )
        if wf.whitener_mode not in ("projected", "none"):
            logger.warning("LORETA whitener fallback used: %s", wf.whitener_mode)

        leadfield = wf.A if wf.A is not None else wf.G_white
        leadfield_scale = wf.trace_scale
        sensor_transform = wf.sensor_transform

        LTL = leadfield.T @ leadfield
        B = np.eye(leadfield.shape[1])
        adjacency = build_source_adjacency(forward["src"], verbose=self.verbose).toarray()
        laplace_operator = laplacian(adjacency)
        BLapTLapB = B @ laplace_operator.T @ laplace_operator @ B

        if alpha == "auto":
            r_grid = np.asarray(self.r_values, dtype=float)
        else:
            r_grid = np.asarray([float(alpha)], dtype=float)
        max_eig_LTL = float(np.linalg.svd(LTL, compute_uv=False).max())
        max_eig_penalty = float(np.linalg.svd(BLapTLapB, compute_uv=False).max())
        scale = max_eig_LTL / max(max_eig_penalty, 1e-15)
        self.alphas = list(r_grid * scale)

        inverse_operators = []
        for alpha in self.alphas:
            kernel_eff = np.linalg.solve(LTL + float(alpha) * BLapTLapB, leadfield.T)
            inverse_operators.append(
                (float(leadfield_scale) * kernel_eff) @ sensor_transform
            )

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        return self
