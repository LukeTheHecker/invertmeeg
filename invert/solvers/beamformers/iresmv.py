import logging

import mne
import numpy as np

from ..base import BaseSolver, SolverMeta

logger = logging.getLogger(__name__)


class SolverIRESMV(BaseSolver):
    """Iteratively Reweighted Eigenspace Minimum Variance (IR-ESMV) Beamformer.

    Applies ESMV beamforming iteratively, reweighting the leadfield at each
    step to promote sparsity. After each ESMV pass, source amplitudes are
    used to downweight unlikely source locations, effectively focusing the
    beamformer on the true support.

    This is inspired by FOCUSS / iteratively reweighted least squares (IRLS)
    applied to the beamformer output rather than to a minimum-norm solution.

    Algorithm
    ---------
    1. Compute initial ESMV solution.
    2. Use source power to build diagonal reweighting matrix W.
    3. Form reweighted leadfield L_w = L @ W and repeat ESMV.
    4. After convergence, rescale amplitudes by W.

    References
    ----------
    ESMV: Jonmohamadi et al. (2014). Comparison of beamformers for EEG.
    FOCUSS: Gorodnitsky & Rao (1997). Sparse signal reconstruction.
    """

    meta = SolverMeta(
        slug="iresmv",
        full_name="Iteratively Reweighted ESMV",
        category="Beamformers",
        description=(
            "Iterative reweighting wrapper around ESMV that promotes sparsity by "
            "downweighting low-power source locations (FOCUSS/IRLS-inspired)."
        ),
        references=[
            "Lukas Hecker (2025). Unpublished.",
            "Jonmohamadi, Y., Poudel, G., Innes, C., Weiss, D., Krueger, R., & Jones, R. "
            "(2014). Comparison of beamformers for EEG source signal reconstruction. "
            "Biomedical Signal Processing and Control, 14, 175-188.",
            "Gorodnitsky, I. F., & Rao, B. D. (1997). Sparse signal reconstruction from "
            "limited data using FOCUSS: A re-weighted minimum norm algorithm. "
            "IEEE Transactions on Signal Processing, 45(3), 600-616.",
        ],
    )

    def __init__(
        self,
        name="IR-ESMV",
        reduce_rank=True,
        rank="auto",
        n_iterations=5,
        sparsity_exponent=0.5,
        **kwargs,
    ):
        self.name = name
        self.n_iterations = n_iterations
        self.sparsity_exponent = sparsity_exponent
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, alpha="auto", **kwargs):
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        self.inverse_operators = []
        return self

    def apply_inverse_operator(self, mne_obj) -> mne.SourceEstimate:
        data = self.unpack_data_obj(mne_obj)
        source_mat = self._solve(data)
        stc = self.source_to_object(source_mat)
        return stc

    def _solve(self, y):
        leadfield = self.leadfield.copy()
        leadfield /= np.linalg.norm(leadfield, axis=0)
        n_chans, n_dipoles = leadfield.shape
        I = np.eye(n_chans)

        C = y @ y.T
        C_sub = self.select_signal_subspace(C)
        eigs = np.linalg.svd(C, compute_uv=False)

        # Weights start uniform
        w = np.ones(n_dipoles)

        for _iteration in range(self.n_iterations):
            L_w = leadfield * w[np.newaxis, :]

            # ESMV on reweighted leadfield
            alpha_val = eigs.max() * 1e-3
            C_inv = self.robust_inverse(C + alpha_val * I)
            C_inv_L = C_inv @ L_w
            diag = np.einsum("ij,ji->i", L_w.T, C_inv_L)
            diag = np.maximum(diag, 1e-12)
            W_bf = C_inv_L / diag
            W_esmv = C_sub @ W_bf

            x = W_esmv.T @ y  # (n_dipoles, n_time)

            # Source power for reweighting
            power = np.sqrt(np.mean(x**2, axis=1))
            power_max = power.max()
            if power_max < 1e-15:
                break

            # Update weights with sparsity-promoting exponent
            w = (power / power_max) ** self.sparsity_exponent
            w = np.maximum(w, 1e-6)  # floor to avoid zero columns

        # Final solution with current weights
        x_hat = w[:, np.newaxis] * x
        return x_hat
