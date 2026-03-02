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

from invert.util import build_source_adjacency

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
        mne_obj=None,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
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
        super().make_inverse_operator(forward, mne_obj, *args, alpha=alpha, **kwargs)
        wf = self.prepare_whitened_forward(noise_cov)
        self.is_prepared = False
        data = self.unpack_data_obj(mne_obj)
        data = wf.sensor_transform @ data

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
        self.inverse_operators = [
            InverseOperator(inverse_operator @ wf.sensor_transform, self.name)
        ]
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
            adjacency = build_source_adjacency(
                self.forward["src"],
                adjacency_type="spatial",
                adjacency_distance=self.adjacency_distance,
                verbose=0,
            )
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

        # Scale per channel type when channel metadata still matches rows in Y.
        # After whitening/projection, Y can live in an effective sensor basis
        # (n_eff) that no longer maps 1:1 to forward info channels.
        Y_work = deepcopy(Y)
        channel_types = self.forward["info"].get_channel_types()
        if len(channel_types) == Y_work.shape[0]:
            channel_types = np.asarray(channel_types)
            for ch_type in set(channel_types):
                sel = np.where(channel_types == ch_type)[0]
                if sel.size == 0:
                    continue
                C_ch = Y_work[sel] @ Y_work[sel].T
                scaler = np.sqrt(np.trace(C_ch)) / C_ch.shape[0]
                if np.isfinite(scaler) and scaler > 0:
                    Y_work[sel] /= scaler
        else:
            logger.debug(
                "SubspaceSBL: skipping channel-type scaling because channel metadata "
                "length (%d) != data rows (%d).",
                len(channel_types),
                Y_work.shape[0],
            )

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
                    P_A = self._compute_projection_matrix(
                        A_temp, lambda_reg=lambda_reg2
                    )
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

    def _nlc_refine_operator(
        self,
        Y: np.ndarray,
        candidates: list[tuple[int, int]],
        *,
        max_iter: int = 500,
        pruning_thresh: float = 1e-3,
        conv_crit: float = 1e-8,
    ) -> np.ndarray:
        """Run NL-Champagne on the detected support and return a (k, n_chans) kernel.

        This delegates to the library's NL-Champagne implementation to avoid
        duplicated (and historically unstable) math here.
        """
        # Build low-rank leadfield from detected sources
        L_sel = np.stack(
            [self.leadfields[int(order)][:, int(dipole)] for order, dipole in candidates],
            axis=1,
        )  # (n_chans, k)

        # Local import avoids a module-level dependency cycle.
        from .nl_champagne import SolverNLChampagne

        refiner = SolverNLChampagne(verbose=getattr(self, "verbose", 0))
        refiner.leadfield = np.asarray(L_sel, dtype=float)
        return refiner.nl_champagne(
            np.asarray(Y, dtype=float),
            alpha=0.01,
            max_iter=int(max_iter),
            prune=True,
            pruning_thresh=float(pruning_thresh),
            convergence_criterion=float(conv_crit),
        )

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
        K_low = self._nlc_refine_operator(
            Y,
            candidates,
            max_iter=max_iter_nlc,
            pruning_thresh=pruning_thresh,
            conv_crit=conv_crit,
        )

        # Build final inverse operator
        gradients = np.stack(
            [self.gradients[order][dipole].toarray() for order, dipole in candidates],
            axis=1,
        )[0]
        # Expand the k-support kernel to the full dictionary using the
        # diffusion gradients (patch extents).
        inverse_operator = gradients.T @ K_low

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

        # Stage 2: NL-Champagne refinement on the expanded support.
        # We use the same robust NL-Champagne implementation as SubspaceSBL.
        K_low = self._nlc_refine_operator(
            Y,
            candidates,
            max_iter=max_iter_nlc,
            pruning_thresh=pruning_thresh,
            conv_crit=conv_crit,
        )  # (k, n_chans)

        gradients = np.stack(
            [self.gradients[order][dipole].toarray() for order, dipole in candidates],
            axis=1,
        )[0]
        inverse_operator = gradients.T @ K_low
        return inverse_operator

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _expand_candidates(
        self, base_candidates: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Expand (order, dipole) candidates by including local neighbors."""
        base = [(int(o), int(d)) for (o, d) in base_candidates]
        if self.neighbor_hops <= 0 or len(base) <= 1:
            if not self.include_order0:
                return base
            expanded = {(0, d) for (_o, d) in base}
            expanded.update(base)
            return sorted(expanded, key=lambda x: (x[0], x[1]))

        adjacency = csr_matrix(
            build_source_adjacency(
                self.forward["src"],
                adjacency_type="spatial",
                adjacency_distance=self.adjacency_distance,
                verbose=0,
            )
        )
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

        expanded: set[tuple[int, int]] = set()  # type: ignore[no-redef]
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
