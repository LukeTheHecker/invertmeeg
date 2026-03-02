import logging

import mne
import numpy as np
from scipy.stats import rankdata

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverMEM(BaseSolver):
    """Maximum Entropy on the Mean (MEM) source localization solver."""

    meta = SolverMeta(
        acronym="MEM",
        full_name="Maximum Entropy on the Mean",
        category="Bayesian",
        description=(
            "Maximum-entropy-on-the-mean approach without data-driven "
            "parcellation (source-wise formulation)."
        ),
        references=[
            "Amblard, C., Lapalme, E., & Bhatt, P. (2004). Biomagnetic source detection by maximum entropy and graphical models. IEEE Transactions on Biomedical Engineering, 51(3), 427-442.",
        ],
    )

    def __init__(
        self,
        name="MEM",
        clustered=False,
        num_parcels=200,
        max_iter=80,
        batch_size=100,
        n_outer=4,
        alpha_tol=2e-2,
        **kwargs,
    ):
        self.name = name
        self.clustered = clustered
        self.num_parcels = num_parcels
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.n_outer = n_outer
        self.alpha_tol = alpha_tol
        return super().__init__(**kwargs)

    def make_inverse_operator(
        self,
        forward,
        mne_obj=None,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
        adjacency=None,
        **kwargs,
    ):
        super().make_inverse_operator(forward, mne_obj, *args, alpha=alpha, **kwargs)
        wf = self.prepare_whitened_forward(noise_cov)
        data = self.unpack_data_obj(mne_obj)
        data = wf.sensor_transform @ data

        if self.clustered:
            from .cmem import _cmem

            if adjacency is None:
                try:
                    adjacency = mne.spatial_src_adjacency(self.forward["src"], verbose=0)
                except Exception:
                    adjacency = None
            self._adjacency = adjacency
            J, parcels = _cmem(
                data,
                self.leadfield,
                A=adjacency,
                num_parcels=self.num_parcels,
                max_iter=self.max_iter,
                batch_size=self.batch_size,
                n_outer=self.n_outer,
                alpha_tol=self.alpha_tol,
            )
            self.parcels = parcels
        else:
            J = _mem(
                data,
                self.leadfield,
                max_iter=self.max_iter,
                batch_size=self.batch_size,
                n_outer=self.n_outer,
                alpha_tol=self.alpha_tol,
            )

        self._mem_cached_obj = mne_obj
        self._mem_cached_J = J
        self.inverse_operators = [InverseOperator(J, self.name)]
        return self

    def apply_inverse_operator(self, mne_obj):
        J_cached = getattr(self, "_mem_cached_J", None)
        obj_cached = getattr(self, "_mem_cached_obj", None)
        if J_cached is not None and obj_cached is mne_obj:
            return self.source_to_object(J_cached)

        data = self.unpack_data_obj(mne_obj)
        data = self._sensor_transform @ data

        if self.clustered:
            from .cmem import _cmem

            A = getattr(self, "_adjacency", None)
            J, self.parcels = _cmem(
                data,
                self.leadfield,
                A=A,
                num_parcels=self.num_parcels,
                max_iter=self.max_iter,
                batch_size=self.batch_size,
                n_outer=self.n_outer,
                alpha_tol=self.alpha_tol,
            )
        else:
            J = _mem(
                data,
                self.leadfield,
                max_iter=self.max_iter,
                batch_size=self.batch_size,
                n_outer=self.n_outer,
                alpha_tol=self.alpha_tol,
            )
        self._mem_cached_obj = mne_obj
        self._mem_cached_J = J
        return self.source_to_object(J)


def _mem(
    Y,
    L,
    max_iter=80,
    batch_size=100,
    *,
    n_outer=4,
    alpha_tol=2e-2,
):
    """Source-wise MEM using vectorized fixed-point updates."""
    m, t = Y.shape
    m_l, n = L.shape
    assert m == m_l, "Dimension mismatch between Y and L"

    # Source-wise MNE initialization (sensor-space solve)
    fro2 = float(np.sum(L * L))
    reg = 0.1 * fro2 / n
    LLt = L @ L.T
    LLt.flat[:: m + 1] += reg
    J_mne = L.T @ np.linalg.solve(LLt, Y)
    J_mne_power = np.mean(J_mne**2, axis=1)

    # Per-source variance (diagonal Sigma): fast and robust for non-cluster MEM.
    sigma = np.maximum(J_mne_power, 1e-20)
    sigma /= np.median(sigma) + 1e-30

    # Robust scale calibration for sigma to keep E in a numerically stable range.
    n_ref = min(10, t)
    if n_ref > 1:
        time_power = np.sum(Y * Y, axis=0)
        ref_idx = np.argsort(time_power)[-n_ref:]
    else:
        ref_idx = np.array([0], dtype=int)
    Xi_ref = L.T @ Y[:, ref_idx]
    E_ref = 0.5 * np.median(sigma[:, np.newaxis] * (Xi_ref**2), axis=1)
    E_scale = float(np.percentile(E_ref, 95))
    if E_scale > 1e-30:
        sigma *= 8.0 / E_scale

    # Alpha initialization by source power rank.
    ranks = rankdata(J_mne_power, method="average").astype(float) - 1.0
    q = ranks / max(len(ranks) - 1, 1)
    alpha = np.clip(0.01 + 0.98 * q**2, 0.01, 0.99)

    lambda_prev = np.zeros((m, t))
    J = np.zeros((n, t))
    for outer in range(n_outer):
        num_batches = (t + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, t)
            Y_batch = Y[:, batch_start:batch_end]
            lambda_init = lambda_prev[:, batch_start:batch_end]

            lambda_batch = _optimize_lambda_batch_sourcewise(
                Y_batch,
                L,
                alpha,
                sigma,
                max_iter=max_iter,
                lambda_init=lambda_init,
            )
            lambda_prev[:, batch_start:batch_end] = lambda_batch
            J[:, batch_start:batch_end] = _compute_sources_batch_sourcewise(
                lambda_batch, L, alpha, sigma
            )

        if outer < n_outer - 1:
            max_alpha_change = _update_alpha_sourcewise(J, alpha)
            if max_alpha_change < float(alpha_tol):
                break
    return J


def _optimize_lambda_batch_sourcewise(
    Y_batch, L, alpha, sigma, max_iter=80, lambda_init=None
):
    """Vectorized damped fixed-point optimizer for source-wise MEM."""
    m, batch_size = Y_batch.shape
    n = int(L.shape[1])

    if lambda_init is None:
        lambda_batch = np.zeros((m, batch_size))
    else:
        lambda_batch = np.array(lambda_init, dtype=float, copy=True)

    y_norm = float(np.linalg.norm(Y_batch)) + 1e-30
    max_lambda_norm = 1e6 * y_norm
    omega = 0.2
    tol = 1e-4

    w = np.empty((n, batch_size))
    for _ in range(int(max_iter)):
        xi = L.T @ lambda_batch
        E = 0.5 * sigma[:, np.newaxis] * (xi**2)
        E = np.clip(E, -500, 500)

        exp_E = np.exp(E)
        p = (alpha[:, np.newaxis] * exp_E) / (
            (1.0 - alpha[:, np.newaxis]) + alpha[:, np.newaxis] * exp_E
        )
        w[:] = p * (sigma[:, np.newaxis] * xi)

        lambda_target = Y_batch - (L @ w)
        delta = lambda_target - lambda_batch
        res_norm = float(np.linalg.norm(delta))

        if not np.isfinite(res_norm):
            break

        omega_try = omega
        accepted = False
        for _ in range(6):
            candidate = lambda_batch + omega_try * delta
            cand_norm = float(np.linalg.norm(candidate))
            if np.isfinite(cand_norm) and cand_norm < max_lambda_norm:
                lambda_batch = candidate
                accepted = True
                break
            omega_try *= 0.5

        if not accepted or (res_norm / y_norm < tol):
            break

    return lambda_batch


def _compute_sources_batch_sourcewise(lambda_batch, L, alpha, sigma):
    xi = L.T @ lambda_batch
    E = 0.5 * sigma[:, np.newaxis] * (xi**2)
    E = np.clip(E, -500, 500)
    exp_E = np.exp(E)
    p = (alpha[:, np.newaxis] * exp_E) / (
        (1.0 - alpha[:, np.newaxis]) + alpha[:, np.newaxis] * exp_E
    )
    return p * (sigma[:, np.newaxis] * xi)


def _update_alpha_sourcewise(J, alpha):
    power = np.mean(J**2, axis=1)
    ranks = rankdata(power, method="average").astype(float) - 1.0
    q = ranks / max(len(ranks) - 1, 1)
    new_alpha = np.clip(0.01 + 0.98 * q**2, 0.01, 0.99)
    max_change = float(np.max(np.abs(new_alpha - alpha)))
    alpha[:] = new_alpha
    return max_change
