import logging

import mne
import numpy as np

from ..base import BaseSolver, SolverMeta

logger = logging.getLogger(__name__)


class SolverSSPESMV(BaseSolver):
    """Hybrid Simultaneous Subspace Pursuit + Eigenspace Minimum Variance solver.

    Combines SSP's greedy sparse support identification with ESMV's
    eigenspace-projected beamforming for amplitude recovery. This targets
    the highly underdetermined scenario (e.g., 4-channel Muse headband)
    where localization accuracy (SSP's strength) and amplitude fidelity
    (ESMV's strength) are both critical.

    Algorithm
    ---------
    1. Normalize leadfield columns for unbiased atom selection.
    2. Run SSP iterations to identify the sparse support set T.
    3. Build a reduced leadfield from the support set.
    4. Apply ESMV-style eigenspace beamforming on the reduced problem
       to get accurate amplitude estimates.

    References
    ----------
    SSP: Dai & Milenkovic (2009). Subspace pursuit for compressive sensing.
    ESMV: Jonmohamadi et al. (2014). Comparison of beamformers for EEG.
    """

    meta = SolverMeta(
        slug="ssp_esmv",
        full_name="Subspace Pursuit + ESMV",
        category="Beamformers",
        description=(
            "Hybrid solver that uses Subspace Pursuit to identify a sparse support "
            "set, then applies ESMV-style beamforming on the reduced problem."
        ),
        references=[
            "Lukas Hecker (2025). Unpublished.",
            "Dai, W., & Milenkovic, O. (2009). Subspace pursuit for compressive sensing "
            "signal reconstruction. IEEE Transactions on Information Theory, 55(5), "
            "2230-2249.",
            "Jonmohamadi, Y., Poudel, G., Innes, C., Weiss, D., Krueger, R., & Jones, R. "
            "(2014). Comparison of beamformers for EEG source signal reconstruction. "
            "Biomedical Signal Processing and Control, 14, 175-188.",
        ],
    )

    def __init__(self, name="SSP-ESMV", reduce_rank=True, rank="auto", **kwargs):
        self.name = name
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
        n_chans, n_time = y.shape
        leadfield = self.leadfield_original
        leadfield_normed = self.leadfield_normed
        _, n_dipoles = leadfield.shape

        K = max(1, n_chans // 2)

        # --- Phase 1: SSP support identification ---
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

        # --- Phase 2: ESMV amplitude recovery on the identified support ---
        x_hat = np.zeros((n_dipoles, n_time))
        if len(T) == 0:
            return x_hat

        L_T = leadfield[:, T]
        L_T_norm = L_T / np.linalg.norm(L_T, axis=0)

        # Eigenspace beamformer on the reduced problem
        C = y @ y.T
        C_sub = self.select_signal_subspace(C)

        eigs = np.linalg.svd(C, compute_uv=False)
        alpha_val = eigs.max() * 1e-3  # moderate regularization

        I = np.eye(n_chans)
        C_inv = self.robust_inverse(C + alpha_val * I)
        C_inv_L = C_inv @ L_T_norm
        diag = np.einsum("ij,ji->i", L_T_norm.T, C_inv_L)
        diag = np.maximum(diag, 1e-12)
        W = C_inv_L / diag
        W_esmv = C_sub @ W

        x_hat[T] = W_esmv.T @ y

        return x_hat

    @staticmethod
    def _top_k_indices(values, k):
        k = min(k, len(values))
        return np.argpartition(values, -k)[-k:]

    def _residual(self, y, L_sub):
        if L_sub.shape[1] == 0:
            return y.copy()
        return self.robust_residual(y, L_sub)
