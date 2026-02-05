import logging

import mne
import numpy as np

from ..base import BaseSolver, SolverMeta

logger = logging.getLogger(__name__)


class SolverSSPIRESMV(BaseSolver):
    """SSP-initialized Iteratively Reweighted ESMV (SSP-IR-ESMV).

    Two-phase solver:
    1. SSP greedy pursuit identifies an initial sparse support set.
    2. Iteratively reweighted ESMV refines amplitudes and support,
       initialized with the SSP support as a warm start.

    This combines SSP's localization accuracy with IR-ESMV's ability
    to refine source amplitudes through iterative reweighting.
    """

    meta = SolverMeta(
        slug="ssp_iresmv",
        full_name="Subspace Pursuit + Iteratively Reweighted ESMV",
        category="Beamformers",
        description=(
            "Two-phase solver that uses Subspace Pursuit for initialization and "
            "iteratively reweighted ESMV (FOCUSS/IRLS-inspired) for refinement."
        ),
        references=[
            "Lukas Hecker (2025). Unpublished.",
            "Dai, W., & Milenkovic, O. (2009). Subspace pursuit for compressive sensing "
            "signal reconstruction. IEEE Transactions on Information Theory, 55(5), "
            "2230-2249.",
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
        name="SSP-IR-ESMV",
        reduce_rank=True,
        rank="auto",
        n_iterations=4,
        sparsity_exponent=0.5,
        **kwargs,
    ):
        self.name = name
        self.n_iterations = n_iterations
        self.sparsity_exponent = sparsity_exponent
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, alpha="auto", **kwargs):
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        self.leadfield_original = self.leadfield.copy()
        self.leadfield_normed = self.robust_normalize_leadfield(self.leadfield)
        self.inverse_operators = []
        return self

    def apply_inverse_operator(self, mne_obj) -> mne.SourceEstimate:
        data = self.unpack_data_obj(mne_obj)
        source_mat = self._solve(data)
        stc = self.source_to_object(source_mat)
        return stc

    def _solve(self, y):
        leadfield = self.leadfield_original
        leadfield_normed = self.leadfield_normed
        n_chans, n_dipoles = leadfield.shape
        n_time = y.shape[1]
        I = np.eye(n_chans)

        K = max(1, n_chans // 2)

        # --- Phase 1: SSP for initial support ---
        b = np.linalg.norm(leadfield_normed.T @ y, axis=1)
        T = self._top_k_indices(b, K)
        R = self._residual(y, leadfield[:, T])

        T_prev = T
        for _i in range(n_chans):
            b = np.linalg.norm(leadfield_normed.T @ R, axis=1)
            new_atoms = self._top_k_indices(b, K)
            T_tilde = np.unique(np.concatenate([T_prev, new_atoms]))

            if len(T_tilde) > 0:
                xp = self.robust_inverse_solution(leadfield[:, T_tilde], y)
                xp_norm = np.linalg.norm(xp, axis=1)
                T_candidate = T_tilde[self._top_k_indices(xp_norm, K)]
            else:
                T_candidate = T_tilde

            if len(T_candidate) > 0:
                xp_final = self.robust_inverse_solution(leadfield[:, T_candidate], y)
                R_new = y - leadfield[:, T_candidate] @ xp_final
            else:
                R_new = y

            if np.linalg.norm(R_new) >= np.linalg.norm(R):
                break
            R = R_new
            T_prev = T_candidate
            T = T_candidate

        # --- Phase 2: IR-ESMV with SSP warm start ---
        # Initialize weights from SSP support
        L = leadfield / np.linalg.norm(leadfield, axis=0)
        C = y @ y.T
        C_sub = self.select_signal_subspace(C)
        eigs = np.linalg.svd(C, compute_uv=False)

        # Warm start: boost SSP-identified sources
        w = np.ones(n_dipoles) * 0.1
        w[T] = 1.0

        x = None
        for _iteration in range(self.n_iterations):
            L_w = L * w[np.newaxis, :]

            alpha_val = eigs.max() * 1e-3
            C_inv = self.robust_inverse(C + alpha_val * I)
            C_inv_L = C_inv @ L_w
            diag = np.einsum("ij,ji->i", L_w.T, C_inv_L)
            diag = np.maximum(diag, 1e-12)
            W_bf = C_inv_L / diag
            W_esmv = C_sub @ W_bf

            x = W_esmv.T @ y

            power = np.sqrt(np.mean(x**2, axis=1))
            power_max = power.max()
            if power_max < 1e-15:
                break

            w = (power / power_max) ** self.sparsity_exponent
            w = np.maximum(w, 1e-6)

        if x is None:
            return np.zeros((n_dipoles, n_time))

        return w[:, np.newaxis] * x

    @staticmethod
    def _top_k_indices(values, k):
        k = min(k, len(values))
        return np.argpartition(values, -k)[-k:]

    def _residual(self, y, L_sub):
        if L_sub.shape[1] == 0:
            return y.copy()
        return self.robust_residual(y, L_sub)
