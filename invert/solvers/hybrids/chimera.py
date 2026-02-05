from __future__ import annotations

import logging
from dataclasses import dataclass

import mne
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian

from ..base import BaseSolver, SolverMeta
from ..beamformers.flex_esmv import SolverFlexESMV
from ..music.signal_subspace_matching import SolverSignalSubspaceMatching

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChimeraParams:
    """Heuristics for switching between solvers.

    The benchmark includes both focal (single/multi-dipole) and extended
    (multi-patch) sources. A single algorithm tends to be suboptimal across
    all regimes, so we use a tiny model-selection layer:

    - MDL (Wax–Kailath) estimates the signal subspace dimension.
    - A lightweight SSM order-score ratio distinguishes focal (order-0) from
      smooth (order>0) explanations when MDL suggests a single component.
    """

    # If the smooth-order dictionary does not improve the SSM score enough,
    # treat the sample as a focal single-source case.
    ratio_single_threshold: float = 0.995

    # Diffusion parameter matches the simulator default.
    diffusion_parameter: float = 0.1

    # Use up to order-2 for the detection ratio and for the SSM branch.
    ssm_n_orders: int = 2

    # Regularization used in SSM's data-subspace projector for detection.
    lambda_reg1: float = 0.001


class SolverChimera(BaseSolver):
    """Chimera: a small, data-driven switch between FlexESMV and SSM.

    Motivation
    ----------
    The benchmark mixes focal single-dipole data (where FlexESMV shines) with
    multi-source/patch data (where SSM/iterative subspace methods dominate).
    This solver uses a simple, fast decision rule per sample:

    - If MDL estimates 1 component and a smooth (order>0) SSM dictionary does
      not beat an order-0 explanation, run FlexESMV.
    - Otherwise, run SSM (with n_orders tuned to the simulator: orders 0..2).
    """

    meta = SolverMeta(
        acronym="CHIMERA",
        full_name="Chimera",
        category="Hybrid",
        description=(
            "Data-driven hybrid that switches between FlexESMV and Signal Subspace Matching "
            "(SSM) based on an MDL subspace-dimension estimate and an order-score ratio."
        ),
        references=[
            "Lukas Hecker 2025, unpublished",
        ],
    )

    def __init__(
        self, name: str = "Chimera", params: ChimeraParams | None = None, **kwargs
    ):
        self.name = name
        self._p = params or ChimeraParams()
        super().__init__(**kwargs)

        # Nonlinear / data-dependent ⇒ recompute per sample.
        self.require_recompute = True
        self.require_data = True

    def make_inverse_operator(self, forward, mne_obj, *args, alpha="auto", **kwargs):
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        _ = self.unpack_data_obj(mne_obj)

        # Precompute adjacency + diffusion operator for the detection ratio.
        n_dipoles = self.leadfield.shape[1]
        adjacency = mne.spatial_src_adjacency(self.forward["src"], verbose=0)
        LL = laplacian(adjacency)
        I = np.eye(n_dipoles)
        S = csr_matrix(I - self._p.diffusion_parameter * LL)

        # Detection dictionary: orders 0..2 (order-0 is raw leadfield).
        self._detect_leadfields = [
            self.leadfield,
            self.leadfield @ S,
            self.leadfield @ (S**2),
        ]
        return self

    def apply_inverse_operator(self, mne_obj):  # type: ignore[override]
        Y = self.unpack_data_obj(mne_obj).copy()

        is_single_focal = self._is_single_focal(Y)
        if is_single_focal:
            solver = SolverFlexESMV(verbose=self.verbose)
            solver.make_inverse_operator(self.forward, mne_obj, alpha="auto")
            return solver.apply_inverse_operator(mne_obj)

        solver = SolverSignalSubspaceMatching(verbose=self.verbose)
        solver.make_inverse_operator(
            self.forward,
            mne_obj,
            alpha="auto",
            n_orders=self._p.ssm_n_orders,
        )
        return solver.apply_inverse_operator(mne_obj)

    # ---------------------------------------------------------------------
    # Decision rule
    # ---------------------------------------------------------------------
    def _is_single_focal(self, Y: np.ndarray) -> bool:
        """Return True if this looks like a single focal (order-0) source."""
        k_hat = self._estimate_mdl_components(Y)
        if k_hat != 1:
            return False

        ratio = self._ssm_smooth_ratio(Y)
        return ratio < self._p.ratio_single_threshold

    @staticmethod
    def _estimate_mdl_components(Y: np.ndarray) -> int:
        """Estimate subspace dimension using Wax–Kailath MDL.

        Assumes temporally white sensor noise (reasonable for the benchmark).
        """
        n_chans, n_time = Y.shape
        if n_time < 2 or n_chans < 2:
            return 1

        C = (Y @ Y.T) / float(n_time)
        eigvals = np.linalg.eigvalsh(C)[::-1]
        eigvals = np.maximum(eigvals, 1e-30)

        m = len(eigvals)
        N = float(n_time)
        scores = np.empty(m)
        scores.fill(np.inf)

        for k in range(m):
            noise = eigvals[k:]
            if noise.size == 0:
                continue
            g = float(np.exp(np.mean(np.log(noise))))
            a = float(np.mean(noise))
            if g <= 0.0 or a <= 0.0:
                continue
            ll = -N * (m - k) * np.log(g / a)
            penalty = 0.5 * k * (2 * m - k) * np.log(N)
            scores[k] = ll + penalty

        k_hat = int(np.argmin(scores))
        # Keep within a sensible range; the solver only needs to detect k==1.
        return int(np.clip(k_hat, 1, m - 1))

    def _ssm_smooth_ratio(self, Y: np.ndarray) -> float:
        """Compute max score(order>0) / max score(order==0) with P_A=0."""
        # Mirror SSM's per-channel-type scaling for stability
        channel_types = self.forward["info"].get_channel_types()
        for ch_type in set(channel_types):
            selection = np.where(np.array(channel_types) == ch_type)[0]
            if selection.size == 0:
                continue
            C = Y[selection, :] @ Y[selection, :].T
            scaler = float(np.sqrt(np.trace(C)) / C.shape[0])
            if scaler > 0:
                Y[selection, :] /= scaler

        n_time = Y.shape[1]
        M_Y = Y.T @ Y
        YY = M_Y + self._p.lambda_reg1 * np.trace(M_Y) * np.eye(n_time)
        P_Y = (Y @ np.linalg.inv(YY)) @ Y.T
        C = P_Y.T @ P_Y

        # Order scores (P_A=0 ⇒ R=I)
        scores = []
        for lf in self._detect_leadfields:
            upper = np.einsum("ij,ij->j", lf, C @ lf)
            lower = np.einsum("ij,ij->j", lf, lf) + 1e-20
            scores.append(float(np.max(upper / lower)))

        s0 = scores[0]
        sp = max(scores[1:])
        if s0 <= 0:
            return float("inf")
        return float(sp / s0)
