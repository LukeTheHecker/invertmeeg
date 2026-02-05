"""SSM-NLC: Signal Subspace Matching with Noise-Learning Champagne Refinement.

A two-stage solver that combines the best source detector (SSM) with the
best amplitude estimator (NLChampagne):

Stage 1 - SSM Detection:
    Uses the full Signal Subspace Matching algorithm with multi-order diffusion
    basis and iterative refinement to detect source locations and extents.
    SSM's greedy orthogonal-projection approach is the most reliable method
    for finding multiple sources in the presence of noise.

Stage 2 - NLChampagne Amplitude Refinement:
    Given the k detected sources, runs NLChampagne's Sparse Bayesian Learning
    on the low-rank problem (k sources only) to:
    - Learn per-source variances γ (optimal amplitude weighting)
    - Learn per-channel noise covariance Λ (robust to realistic noise)
    This replaces SSM's simple minimum-norm amplitude estimation with a
    statistically optimal Bayesian estimate.

The combination is principled: SSM solves the combinatorial source detection
problem (which atom from the dictionary?), while NLChampagne solves the
continuous amplitude estimation problem (how much from each source?).
"""

import logging
from copy import deepcopy

import mne
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverSubspaceSBL(BaseSolver):
    """SSM source detection + NLChampagne amplitude refinement."""

    meta = SolverMeta(
        slug="subspace-sbl",
        full_name="SubspaceSBL (SSM + NL-Champagne)",
        category="Bayesian",
        description=(
            "Two-stage solver that detects sources with signal subspace matching "
            "and refines amplitudes/noise parameters using NL-Champagne on the "
            "reduced problem."
        ),
        references=[
            "Lukas Hecker 2025, unpublished",
        ],
    )

    def __init__(
        self,
        name="SubspaceSBL",
        n_orders=3,
        scale_leadfield=False,
        diffusion_parameter=0.1,
        adjacency_type="spatial",
        adjacency_distance=3e-3,
        **kwargs,
    ):
        self.name = name
        self.n_orders = n_orders
        self.scale_leadfield = scale_leadfield
        self.diffusion_parameter = diffusion_parameter
        self.adjacency_type = adjacency_type
        self.adjacency_distance = adjacency_distance
        self.is_prepared = False
        super().__init__(**kwargs)

    def make_inverse_operator(
        self,
        forward,
        mne_obj,
        *args,
        alpha="auto",
        n="enhanced",
        max_iter_ssm=5,
        max_iter_nlc=500,
        lambda_reg1=0.001,
        lambda_reg2=0.0001,
        lambda_reg3=0.0,
        pruning_thresh=1e-3,
        convergence_criterion=1e-8,
        **kwargs,
    ):
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)

        if not self.is_prepared:
            self._prepare_flex()

        inverse_operator = self._ssm_nlc(
            data,
            n=n,
            max_iter_ssm=max_iter_ssm,
            max_iter_nlc=max_iter_nlc,
            lambda_reg1=lambda_reg1,
            lambda_reg2=lambda_reg2,
            lambda_reg3=lambda_reg3,
            pruning_thresh=pruning_thresh,
            conv_crit=convergence_criterion,
        )
        self.inverse_operators = [InverseOperator(inverse_operator, self.name)]
        return self

    # ================================================================
    # Stage 1: SSM source detection (exact copy of SSM algorithm)
    # ================================================================

    def _prepare_flex(self):
        n_dipoles = self.leadfield.shape[1]
        I = np.identity(n_dipoles)

        self.leadfields = [deepcopy(self.leadfield)]
        self.gradients = [csr_matrix(I)]

        if self.n_orders == 0:
            self.is_prepared = True
            return

        if self.adjacency_type == "spatial":
            adjacency = mne.spatial_src_adjacency(self.forward["src"], verbose=0)
        else:
            adjacency = mne.spatial_dist_adjacency(
                self.forward["src"], self.adjacency_distance, verbose=None
            )

        LL = laplacian(adjacency)
        if self.diffusion_parameter == "auto":
            alphas = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175]
            smoothing_operators = [csr_matrix(I - a * LL) for a in alphas]
        else:
            smoothing_operators = [
                csr_matrix(I - self.diffusion_parameter * LL),
            ]

        for smoothing_operator in smoothing_operators:
            for i in range(self.n_orders):
                S_i = smoothing_operator ** (i + 1)
                new_lf = self.leadfields[0] @ S_i
                new_grad = self.gradients[0] @ S_i
                if self.scale_leadfield:
                    new_lf /= np.linalg.norm(new_lf, axis=0)
                self.leadfields.append(new_lf)
                self.gradients.append(new_grad)

        for i in range(len(self.gradients)):
            row_sums = self.gradients[i].sum(axis=1).ravel()
            scaling = 1.0 / np.maximum(np.abs(np.asarray(row_sums).ravel()), 1e-12)
            self.gradients[i] = csr_matrix(
                self.gradients[i].multiply(scaling.reshape(-1, 1))
            )

        self.is_prepared = True

    def _ssm_detect(
        self,
        Y,
        n="enhanced",
        max_iter=5,
        lambda_reg1=0.001,
        lambda_reg2=0.0001,
        lambda_reg3=0.0,
    ):
        """Run SSM to detect source locations and extents.

        Returns list of (order, dipole) tuples.
        """
        n_chans, n_dipoles = self.leadfield.shape
        n_time = Y.shape[1]
        leadfields = self.leadfields

        # Determine number of sources
        if isinstance(n, str):
            n_comp = self.estimate_n_sources(Y, method=n)
        else:
            n_comp = deepcopy(n)

        # Scale per channel type
        Y_work = deepcopy(Y)
        channel_types = self.forward["info"].get_channel_types()
        for ch_type in set(channel_types):
            sel = np.where(np.array(channel_types) == ch_type)[0]
            C_ch = Y_work[sel] @ Y_work[sel].T
            scaler = np.sqrt(np.trace(C_ch)) / C_ch.shape[0]
            Y_work[sel] /= scaler

        # SSM data projection matrix
        M_Y = Y_work.T @ Y_work
        YY = M_Y + lambda_reg1 * np.trace(M_Y) * np.eye(n_time)
        P_Y = (Y_work @ np.linalg.inv(YY)) @ Y_work.T
        C = P_Y.T @ P_Y

        P_A = np.zeros((n_chans, n_chans))

        S_SSM = []
        A_q = []

        # Initial source
        S_SSM.append(self._get_source_ssm(C, P_A, leadfields, lambda_reg=lambda_reg3))
        for _ in range(1, n_comp):
            order, location = S_SSM[-1]
            A_q.append(leadfields[order][:, location])
            P_A = self._compute_projection_matrix(A_q, lambda_reg=lambda_reg2)
            S_SSM.append(
                self._get_source_ssm(C, P_A, leadfields, S_SSM, lambda_reg=lambda_reg3)
            )
        A_q.append(leadfields[S_SSM[-1][0]][:, S_SSM[-1][1]])

        # Refinement phase
        S_SSM_2 = deepcopy(S_SSM)
        if len(S_SSM_2) > 1:
            S_prev = deepcopy(S_SSM_2)
            for _j in range(max_iter):
                A_q_j = A_q.copy()
                for qq in range(n_comp):
                    A_temp = np.delete(A_q_j, qq, axis=0)
                    qq_temp = np.delete(S_SSM_2, qq, axis=0)
                    P_A = self._compute_projection_matrix(A_temp, lambda_reg=lambda_reg2)
                    S_SSM_2[qq] = self._get_source_ssm(
                        C, P_A, leadfields, qq_temp, lambda_reg=lambda_reg3
                    )
                    A_q_j[qq] = leadfields[S_SSM_2[qq][0]][:, S_SSM_2[qq][1]]
                if S_SSM_2 == S_prev:
                    break
                S_prev = deepcopy(S_SSM_2)

        return S_SSM_2

    def _get_source_ssm(
        self,
        C,
        P_A,
        leadfields,
        q_ignore=None,
        lambda_reg=0.0,
    ):
        if q_ignore is None:
            q_ignore = []
        n_dipoles = leadfields[0].shape[1]
        n_orders = len(leadfields)

        R = np.eye(P_A.shape[0]) - P_A
        expression = np.zeros((n_orders, n_dipoles))

        for jj in range(n_orders):
            a_s = R @ leadfields[jj]
            upper = np.einsum("ij,ij->j", a_s, C @ a_s)
            lower = np.einsum("ij,ij->j", a_s, a_s) + lambda_reg
            expression[jj] = upper / lower

        if len(q_ignore) > 0:
            for order, dipole in q_ignore:
                expression[order, dipole] = np.nan

        order, dipole = np.unravel_index(np.nanargmax(expression), expression.shape)
        return order, dipole

    @staticmethod
    def _compute_projection_matrix(A_q, lambda_reg=0.0001):
        A_q = np.stack(A_q, axis=1)
        M_A = A_q.T @ A_q
        AA = M_A + lambda_reg * np.trace(M_A) * np.eye(M_A.shape[0])
        P_A = (A_q @ np.linalg.inv(AA)) @ A_q.T
        return P_A

    # ================================================================
    # Stage 2: NLChampagne amplitude refinement
    # ================================================================

    def _nlc_refine(self, Y, candidates, max_iter=500, pruning_thresh=1e-3, conv_crit=1e-8):
        """Run NLChampagne on detected sources to refine amplitudes.

        Parameters
        ----------
        Y : array (n_chans, n_times)
        candidates : list of (order, dipole) tuples from SSM

        Returns
        -------
        gamma_refined : array of per-source variances
        llambda : array of per-channel noise variances
        """
        n_chans = Y.shape[0]
        n_times = Y.shape[1]
        k = len(candidates)

        # Build low-rank leadfield from detected sources
        L_sel = np.stack(
            [self.leadfields[order][:, dipole] for order, dipole in candidates], axis=1
        )  # (n_chans, k)

        # Scale data
        Y_scaled = deepcopy(Y)
        Y_scaled /= abs(Y_scaled).mean() + 1e-12

        # Initialize
        alpha = np.ones(k)
        C_y = self.data_covariance(Y_scaled, center=True, ddof=1)
        llambda = np.ones(n_chans) * float(np.trace(C_y) / (n_chans * 100))

        loss_list = []
        for _ in range(max_iter):
            prev_alpha = deepcopy(alpha)

            Sigma_y = (L_sel * alpha) @ L_sel.T + np.diag(llambda)
            Sigma_y = 0.5 * (Sigma_y + Sigma_y.T)
            try:
                Sigma_y_inv = np.linalg.inv(Sigma_y)
            except np.linalg.LinAlgError:
                Sigma_y_inv = np.linalg.pinv(Sigma_y)

            # Alpha update (Convexity/MM)
            s_bar = (L_sel.T @ Sigma_y_inv @ Y_scaled) * alpha[:, None]
            z_hat = np.sum(L_sel * (Sigma_y_inv @ L_sel), axis=0)
            C_s_bar = np.sum(s_bar**2, axis=1) / n_times
            alpha = np.sqrt(C_s_bar / (z_hat + 1e-20))
            alpha[~np.isfinite(alpha)] = 0.0
            alpha = np.maximum(alpha, 0.0)

            # Lambda update (Convex Bound)
            Y_hat = L_sel @ s_bar
            residual_sq = np.sum((Y_scaled - Y_hat) ** 2, axis=1) / n_times
            diag_inv = np.diag(Sigma_y_inv)
            llambda = np.sqrt(residual_sq / (diag_inv + 1e-20))
            llambda = np.maximum(llambda, 1e-10)

            # Convergence
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                sign, log_det = np.linalg.slogdet(Sigma_y)
            if sign <= 0:
                log_det = -np.inf
            summation = np.sum(
                np.einsum("ti,ij,tj->t", Y_scaled.T, Sigma_y_inv, Y_scaled.T)
            ) / n_times
            loss = float(log_det + summation)
            loss_list.append(loss)

            if loss == float("-inf") or loss == float("inf") or np.linalg.norm(alpha) == 0:
                alpha = prev_alpha
                break

            if len(loss_list) > 1:
                change = abs(1 - loss_list[-1] / (loss_list[-2] + 1e-20))
                if change < conv_crit:
                    break

        return alpha, llambda

    # ================================================================
    # Combined pipeline
    # ================================================================

    def _ssm_nlc(
        self,
        Y,
        n="enhanced",
        max_iter_ssm=5,
        max_iter_nlc=500,
        lambda_reg1=0.001,
        lambda_reg2=0.0001,
        lambda_reg3=0.0,
        pruning_thresh=1e-3,
        conv_crit=1e-8,
    ):
        n_chans, n_dipoles = self.leadfield.shape

        # Stage 1: SSM detection
        candidates = self._ssm_detect(
            Y,
            n=n,
            max_iter=max_iter_ssm,
            lambda_reg1=lambda_reg1,
            lambda_reg2=lambda_reg2,
            lambda_reg3=lambda_reg3,
        )

        # Stage 2: NLChampagne refinement
        gamma, llambda = self._nlc_refine(
            Y, candidates, max_iter=max_iter_nlc, pruning_thresh=pruning_thresh, conv_crit=conv_crit
        )

        # Build final inverse operator
        L_sel = np.stack(
            [self.leadfields[order][:, dipole] for order, dipole in candidates], axis=1
        )
        gradients = np.stack(
            [self.gradients[order][dipole].toarray() for order, dipole in candidates],
            axis=1,
        )[0]

        # Use SBL-refined source covariance instead of identity
        Gamma = np.diag(gamma)
        Sigma_y = np.diag(llambda) + (L_sel * gamma) @ L_sel.T
        Sigma_y = 0.5 * (Sigma_y + Sigma_y.T)
        try:
            Sigma_y_inv = np.linalg.inv(Sigma_y)
        except np.linalg.LinAlgError:
            Sigma_y_inv = np.linalg.pinv(Sigma_y)

        inverse_operator = gradients.T @ Gamma @ L_sel.T @ Sigma_y_inv

        return inverse_operator

    @staticmethod
    def _robust_inv(M):
        try:
            return np.linalg.inv(M)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(M)


class SolverSubspaceSBLPlus(SolverSubspaceSBL):
    """SubspaceSBL+ with local support expansion around SSM candidates."""

    meta = SolverMeta(
        slug="subspace-sbl-plus",
        full_name="SubspaceSBL+ (SSM + NL-Champagne, expanded support)",
        category="Bayesian",
        description=(
            "Enhanced SubspaceSBL that expands the SSM-detected candidate set with "
            "local neighbors (graph hops) before running NL-Champagne refinement."
        ),
        references=[
            "Lukas Hecker 2025, unpublished",
        ],
    )

    def __init__(
        self,
        name: str = "SubspaceSBLPlus",
        *,
        neighbor_hops: int = 1,
        include_order0: bool = True,
        n_comp_offset_if_multi: int = 1,
        max_n_comp: int = 8,
        **kwargs,
    ):
        self.neighbor_hops = int(neighbor_hops)
        self.include_order0 = bool(include_order0)
        self.n_comp_offset_if_multi = int(n_comp_offset_if_multi)
        self.max_n_comp = int(max_n_comp)
        super().__init__(name=name, **kwargs)

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------
    def _ssm_detect(  # type: ignore[override]
        self,
        Y,
        n="enhanced",
        max_iter=5,
        lambda_reg1=0.001,
        lambda_reg2=0.0001,
        lambda_reg3=0.0,
    ):
        if isinstance(n, str):
            n_est = int(self.estimate_n_sources(Y, method=n))
            if n_est > 1:
                n_est = min(n_est + self.n_comp_offset_if_multi, self.max_n_comp)
            n = int(max(1, n_est))
        return super()._ssm_detect(
            Y,
            n=n,
            max_iter=max_iter,
            lambda_reg1=lambda_reg1,
            lambda_reg2=lambda_reg2,
            lambda_reg3=lambda_reg3,
        )

    def _ssm_nlc(  # type: ignore[override]
        self,
        Y,
        n="enhanced",
        max_iter_ssm=5,
        max_iter_nlc=500,
        lambda_reg1=0.001,
        lambda_reg2=0.0001,
        lambda_reg3=0.0,
        pruning_thresh=1e-3,
        conv_crit=1e-8,
    ):
        n_chans, n_dipoles = self.leadfield.shape

        # Stage 1: SSM detection
        base_candidates = self._ssm_detect(
            Y,
            n=n,
            max_iter=max_iter_ssm,
            lambda_reg1=lambda_reg1,
            lambda_reg2=lambda_reg2,
            lambda_reg3=lambda_reg3,
        )

        # Expand candidate set locally (helps recover near-misses, boosts precision/recall tradeoff)
        candidates = self._expand_candidates(base_candidates)
        if len(candidates) == 0:
            return np.zeros((n_dipoles, n_chans))

        # Stage 2: NLChampagne refinement
        gamma, llambda = self._nlc_refine(
            Y,
            candidates,
            max_iter=max_iter_nlc,
            pruning_thresh=pruning_thresh,
            conv_crit=conv_crit,
        )

        gamma = np.asarray(gamma, dtype=float)
        llambda = np.asarray(llambda, dtype=float)

        # Prune weak atoms after refinement for a cleaner inverse operator
        if gamma.size == 0 or not np.isfinite(gamma).any():
            return np.zeros((n_dipoles, n_chans))

        thresh = float(pruning_thresh) * float(np.nanmax(gamma))
        keep = np.where(gamma > thresh)[0]
        if keep.size == 0:
            keep = np.array([int(np.nanargmax(gamma))])

        candidates = [candidates[i] for i in keep.tolist()]
        gamma = gamma[keep]

        # Build final inverse operator
        L_sel = np.stack(
            [self.leadfields[order][:, dipole] for order, dipole in candidates], axis=1
        )
        gradients = np.stack(
            [self.gradients[order][dipole].toarray() for order, dipole in candidates],
            axis=1,
        )[0]

        Gamma = np.diag(gamma)
        Sigma_y = np.diag(llambda) + (L_sel * gamma) @ L_sel.T
        Sigma_y = 0.5 * (Sigma_y + Sigma_y.T)
        Sigma_y_inv = self._robust_inv(Sigma_y)

        inverse_operator = gradients.T @ Gamma @ L_sel.T @ Sigma_y_inv
        return inverse_operator

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _expand_candidates(self, base_candidates: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Expand (order, dipole) candidates by including local neighbors."""
        base = [(int(o), int(d)) for (o, d) in base_candidates]
        if self.neighbor_hops <= 0 or len(base) <= 1:
            if not self.include_order0:
                return base
            expanded = {(0, d) for (_o, d) in base}
            expanded.update(base)
            return sorted(expanded, key=lambda x: (x[0], x[1]))

        adjacency = csr_matrix(mne.spatial_src_adjacency(self.forward["src"], verbose=0))
        n_orders = len(self.leadfields)

        def hop_neighbors(seed: int) -> set[int]:
            visited: set[int] = {seed}
            frontier: set[int] = {seed}
            for _ in range(self.neighbor_hops):
                new_frontier: set[int] = set()
                for node in frontier:
                    for nb in adjacency[node].indices:
                        nb_i = int(nb)
                        if nb_i in visited:
                            continue
                        visited.add(nb_i)
                        new_frontier.add(nb_i)
                frontier = new_frontier
                if not frontier:
                    break
            return visited

        expanded: set[tuple[int, int]] = set()
        for order, dipole in base:
            order_i = int(order)
            if order_i < 0 or order_i >= n_orders:
                order_i = 0
            dipole_i = int(dipole)
            for d2 in hop_neighbors(dipole_i):
                if self.include_order0:
                    expanded.add((0, d2))
                expanded.add((order_i, d2))

        # Deterministic order for reproducibility
        return sorted(expanded, key=lambda x: (x[0], x[1]))
