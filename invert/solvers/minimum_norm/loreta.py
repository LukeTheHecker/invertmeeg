import logging

import mne
import numpy as np
from scipy.sparse.csgraph import laplacian

from invert.util import build_source_adjacency

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverLORETA(BaseSolver):
    """Class for the Low Resolution Tomography (LORETA) inverse solution.

    Solves ``(L^T L + α * Lap^p) J = L^T Y`` where ``Lap`` is the graph
    Laplacian on the source mesh. The ``laplacian_power`` parameter ``p``
    controls the smoothness order via the fractional matrix power
    ``Lap^p = U diag(λ_i^p) U^T``:

    - ``p < 1``: less smoothness than standard LORETA (more focal solutions)
    - ``p = 1``: standard LORETA (2nd-order smoothness, default)
    - ``p > 1``: stronger smoothness (e.g. ``p = 2`` gives the biharmonic)

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
        laplacian_power: float = 1.0,
        **kwargs,
    ):
        self.name = name
        self.use_noise_whitener = bool(use_noise_whitener)
        self.use_trace_normalization = bool(use_trace_normalization)
        self.rank_tol = float(rank_tol)
        self.eps = float(eps)
        self.laplacian_power = float(laplacian_power)
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
        adjacency = build_source_adjacency(
            forward["src"], verbose=self.verbose
        ).toarray()
        Lap = laplacian(adjacency)

        # Fractional Laplacian power: L^p = U diag(λ_i^p) U^T
        p = self.laplacian_power
        eigvals, eigvecs = np.linalg.eigh(Lap)
        eigvals = np.maximum(eigvals, 0.0)  # clamp numerical noise
        if p == 1.0:
            penalty = Lap
        else:
            penalty = eigvecs * (eigvals**p) @ eigvecs.T

        # Free orientation: expand penalty from (n_pos, n_pos) to (n_dipoles, n_dipoles)
        n_orient = getattr(self, "_n_orient", 1)
        if n_orient > 1 and penalty.shape[0] != LTL.shape[0]:
            penalty = np.kron(penalty, np.eye(n_orient))

        if alpha == "auto":
            r_grid = np.asarray(self.r_values, dtype=float)
        else:
            r_grid = np.asarray([float(alpha)], dtype=float)
        max_eig_LTL = float(np.linalg.svd(LTL, compute_uv=False).max())
        max_eig_penalty = float(eigvals.max() ** p)
        scale = max_eig_LTL / max(max_eig_penalty, 1e-15)
        self.alphas = list(r_grid * scale)

        inverse_operators = []
        for alpha in self.alphas:
            M = LTL + float(alpha) * penalty
            # Add small diagonal for numerical stability (Laplacian has zero eigenvalues)
            M += self.eps * np.trace(M) / M.shape[0] * np.eye(M.shape[0])
            kernel_eff = np.linalg.solve(M, leadfield.T)
            inverse_operators.append(
                (float(leadfield_scale) * kernel_eff) @ sensor_transform
            )

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        return self
