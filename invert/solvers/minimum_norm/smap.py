import logging

import mne
import numpy as np
from scipy.sparse.csgraph import laplacian

from invert.util import build_source_adjacency

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverSMAP(BaseSolver):
    """Class for the Quadratic regularization and spatial regularization
        (S-MAP) inverse solution [1].

    References
    ----------
    [1] Baillet, S., & Garnero, L. (1997). A Bayesian approach to introducing
    anatomo-functional priors in the EEG/MEG inverse problem. IEEE transactions
    on Biomedical Engineering, 44(5), 374-385.

    """

    meta = SolverMeta(
        acronym="S-MAP",
        full_name="S-MAP (Quadratic + Spatial MAP)",
        category="Minimum Norm",
        description=(
            "MAP-style quadratic inverse with spatial regularization via a "
            "graph Laplacian over the source space."
        ),
        references=[
            "Baillet, S., & Garnero, L. (1997). A Bayesian approach to introducing anatomo-functional priors in the EEG/MEG inverse problem. IEEE Transactions on Biomedical Engineering, 44(5), 374–385.",
        ],
    )

    def __init__(
        self,
        name="S-MAP",
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
        # The BaseSolver default r-grid tops out at 1e1, which is often too
        # narrow for SMAP's scaled Laplacian penalty. Widen the auto grid.
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
            logger.warning("S-MAP whitener fallback used: %s", wf.whitener_mode)

        leadfield = wf.A if wf.A is not None else wf.G_white
        leadfield_scale = wf.trace_scale
        sensor_transform = wf.sensor_transform

        LTL = leadfield.T @ leadfield

        adjacency = build_source_adjacency(self.forward["src"], verbose=0)
        # First-order Laplacian smoothness: x^T L x (graph Laplacian), rather
        # than squaring the Laplacian as in LORETA-style penalties.
        GTG = laplacian(adjacency).astype(float).toarray()

        if alpha == "auto":
            r_grid = np.asarray(self.r_values, dtype=float)
        else:
            r_grid = np.asarray([float(alpha)], dtype=float)
        max_eig_LTL = float(np.linalg.svd(LTL, compute_uv=False).max())
        max_eig_penalty = float(np.linalg.svd(GTG, compute_uv=False).max())
        scale = max_eig_LTL / max(max_eig_penalty, 1e-15)
        self.alphas = list(r_grid * scale)

        inverse_operators = []
        # GG_inv = np.linalg.inv(GTG)
        for alpha in self.alphas:
            kernel_eff = np.linalg.solve(LTL + float(alpha) * GTG, leadfield.T)
            # inverse_operator = GG_inv @ self.leadfield.T @ np.linalg.inv(self.leadfield @ GG_inv @ self.leadfield.T + alpha * np.identity(n_chans))
            inverse_operators.append(
                (float(leadfield_scale) * kernel_eff) @ sensor_transform
            )

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]

        return self
