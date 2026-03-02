from __future__ import annotations

import logging

import mne
import numpy as np
from scipy.optimize import nnls
from scipy.sparse.csgraph import laplacian

from invert.util import build_source_adjacency

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)

_LAPLACIAN_EIG_CACHE: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}


class SolverGraphWienerSmoothSparseAdaptiveRobustEB(BaseSolver):
    """Low-SNR-robust Smooth+Sparse empirical Bayes Wiener.

    This solver is a variant of ``SolverGraphWienerSmoothSparseAdaptiveEB`` with
    extra regularization aimed at temporally correlated, very low-SNR regimes:

    - SNR-adaptive shrinkage of the sample sensor covariance.
    - Ridge-stabilized NNLS when fitting smooth covariance mixture weights.
    - Optional damping for the learned post-whitening isotropic noise scale σ².

    The goal is to reduce overfitting to sampling noise in the covariance
    estimates while keeping the model class (smooth background + diagonal
    residual) unchanged.
    """

    meta = SolverMeta(
        acronym="GW-SSAR",
        full_name="Graph Wiener Smooth+Sparse (Adaptive Robust EB)",
        category="Minimum Norm",
        description=(
            "Empirical-Bayes Wiener/MAP estimator with a low-rank smooth background "
            "plus diagonal residual covariance, using SNR-adaptive covariance shrinkage "
            "and ridge-stabilized smooth-mixture fitting for low-SNR robustness."
        ),
        internal=True,
    )

    def __init__(
        self,
        name: str = "Graph Wiener Smooth+Sparse (Adaptive Robust EB)",
        taus: tuple[float, ...] = (0.2, 0.8, 3.2),
        n_basis: float | int | None = 0.15,
        # Sensor covariance shrinkage is adapted based on an SNR proxy in whitened space.
        sensor_cov_shrinkage_min: float = 0.15,
        sensor_cov_shrinkage_max: float = 0.6,
        sensor_cov_shrinkage_snr_ref: float = 0.05,
        # Ridge stabilizes the NNLS mixture fit when the target covariance is noisy/negative.
        w_ridge_min: float = 1e-6,
        w_ridge_max: float = 5e-3,
        w_ridge_snr_ref: float = 0.05,
        w_sum_floor: float = 1e-4,
        time_whiten_cov: bool = False,
        time_cov_shrinkage: float = 0.2,
        time_cov_eps: float = 1e-9,
        n_outer_iters: int = 2,
        n_gamma_iters: int = 1,
        gamma_damping: float = 0.5,
        gamma_floor: float = 1e-10,
        learn_noise_scale: bool = True,
        noise_scale_floor: float = 1e-6,
        noise_scale_damping: float = 0.5,
        gamma_prior_a: float = 0.0,
        gamma_prior_b: float = 0.0,
        **kwargs,
    ):
        self.name = str(name)
        self.taus = tuple(float(t) for t in taus)
        self.n_basis = n_basis
        self.sensor_cov_shrinkage_min = float(sensor_cov_shrinkage_min)
        self.sensor_cov_shrinkage_max = float(sensor_cov_shrinkage_max)
        self.sensor_cov_shrinkage_snr_ref = float(sensor_cov_shrinkage_snr_ref)
        self.w_ridge_min = float(w_ridge_min)
        self.w_ridge_max = float(w_ridge_max)
        self.w_ridge_snr_ref = float(w_ridge_snr_ref)
        self.w_sum_floor = float(w_sum_floor)
        self.time_whiten_cov = bool(time_whiten_cov)
        self.time_cov_shrinkage = float(time_cov_shrinkage)
        self.time_cov_eps = float(time_cov_eps)
        self.n_outer_iters = int(n_outer_iters)
        self.n_gamma_iters = int(n_gamma_iters)
        self.gamma_damping = float(gamma_damping)
        self.gamma_floor = float(gamma_floor)
        self.learn_noise_scale = bool(learn_noise_scale)
        self.noise_scale_floor = float(noise_scale_floor)
        self.noise_scale_damping = float(noise_scale_damping)
        self.gamma_prior_a = float(gamma_prior_a)
        self.gamma_prior_b = float(gamma_prior_b)
        super().__init__(**kwargs)

        self.require_recompute = True
        self.require_data = True

    @staticmethod
    def _resolve_n_basis(n_vertices: int, n_basis):
        if n_basis is None:
            return n_vertices
        if isinstance(n_basis, float):
            if not 0 < n_basis <= 1:
                raise ValueError(f"n_basis as float must be in (0, 1], got {n_basis}")
            return max(2, int(np.ceil(n_basis * n_vertices)))
        n_basis_int = int(n_basis)
        if n_basis_int < 2:
            raise ValueError(f"n_basis must be >= 2, got {n_basis_int}")
        return min(n_vertices, n_basis_int)

    @staticmethod
    def _shrink_sensor_cov(C: np.ndarray, shrink: float) -> np.ndarray:
        shrink = float(shrink)
        if not np.isfinite(shrink) or shrink < 0 or shrink > 1:
            raise ValueError(f"shrink must be in [0, 1], got {shrink}")
        if shrink == 0:
            return C
        n = C.shape[0]
        mu = float(np.trace(C)) / float(n)
        return (1.0 - shrink) * C + shrink * mu * np.eye(n)

    @staticmethod
    def _snr_proxy_from_whitened_cov(C: np.ndarray) -> float:
        # In whitened space, pure noise would have E[trace(C)/n] ≈ 1.
        n = float(C.shape[0])
        return float(max(0.0, (np.trace(C) / n) - 1.0))

    def _adaptive_shrinkage(self, snr_proxy: float) -> float:
        lo = float(np.clip(self.sensor_cov_shrinkage_min, 0.0, 1.0))
        hi = float(np.clip(self.sensor_cov_shrinkage_max, lo, 1.0))
        sref = float(max(self.sensor_cov_shrinkage_snr_ref, 1e-12))
        # snr=0 -> hi; snr>>sref -> lo
        t = float(snr_proxy) / float(snr_proxy + sref)
        return (1.0 - t) * hi + t * lo

    def _adaptive_w_ridge(self, snr_proxy: float) -> float:
        lo = float(max(self.w_ridge_min, 0.0))
        hi = float(max(self.w_ridge_max, lo))
        sref = float(max(self.w_ridge_snr_ref, 1e-12))
        # snr=0 -> hi; snr>>sref -> lo
        t = float(snr_proxy) / float(snr_proxy + sref)
        return (1.0 - t) * hi + t * lo

    @staticmethod
    def _fit_nnls_ridge(
        target: np.ndarray, components: list[np.ndarray], ridge: float
    ) -> np.ndarray:
        n = int(target.shape[0])
        tri = np.triu_indices(n)
        b = np.asarray(target[tri], dtype=float)
        A = np.stack([np.asarray(C[tri], dtype=float) for C in components], axis=1)

        ridge = float(max(ridge, 0.0))
        if ridge > 0:
            # Solve min ||Aw-b||^2 + ridge*||w||^2  s.t. w>=0 via augmented NNLS.
            A = np.concatenate([A, np.sqrt(ridge) * np.eye(A.shape[1])], axis=0)
            b = np.concatenate([b, np.zeros(A.shape[1], dtype=float)], axis=0)

        w, _ = nnls(A, b)
        return np.asarray(w, dtype=float)

    def _laplacian_eigs(self, forward) -> tuple[np.ndarray, np.ndarray, float]:
        key = int(id(forward["src"]))
        cached = _LAPLACIAN_EIG_CACHE.get(key)
        if cached is not None:
            return cached
        adjacency = build_source_adjacency(forward["src"], verbose=0)
        graph_laplacian = laplacian(adjacency, normed=False).astype(float).toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(graph_laplacian)
        positive = eigenvalues[eigenvalues > 1e-12]
        eig_scale = float(np.median(positive)) if positive.size else 1.0
        _LAPLACIAN_EIG_CACHE[key] = (eigenvalues, eigenvectors, eig_scale)
        return eigenvalues, eigenvectors, eig_scale

    def make_inverse_operator(
        self,
        forward,
        mne_obj=None,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
        verbose=0,
        **kwargs,
    ):
        super().make_inverse_operator(forward, mne_obj, *args, alpha=alpha, **kwargs)
        wf = self.prepare_whitened_forward(noise_cov)

        Y = self.unpack_data_obj(mne_obj)
        self.validate_operator_data_compatibility(Y)
        Yw = wf.sensor_transform @ Y
        Yw = Yw - Yw.mean(axis=1, keepdims=True)
        n_eff, n_time = Yw.shape
        if n_time < 2:
            raise ValueError("Need at least 2 time points to estimate covariance.")

        if self.time_whiten_cov and n_time >= 3:
            # Right-whiten time to reduce the effective sample size loss under
            # temporally correlated noise. Only used for covariance estimation.
            T = (Yw.T @ Yw) / float(n_eff)
            T = 0.5 * (T + T.T)
            tw = float(np.clip(self.time_cov_shrinkage, 0.0, 1.0))
            if tw > 0:
                T = (1.0 - tw) * T + tw * np.diag(np.diag(T))
            diag_mean = float(np.trace(T)) / float(n_time)
            T.flat[:: n_time + 1] += float(self.time_cov_eps) * max(diag_mean, 1e-12)
            L_t = np.linalg.cholesky(T)
            Yw_cov = Yw @ np.linalg.solve(L_t, np.eye(n_time))
            C = (Yw_cov @ Yw_cov.T) / float(n_time)
        else:
            C = (Yw @ Yw.T) / float(n_time)

        snr_proxy = self._snr_proxy_from_whitened_cov(C)
        shrink = self._adaptive_shrinkage(snr_proxy)
        ridge = self._adaptive_w_ridge(snr_proxy)
        C = self._shrink_sensor_cov(C, shrink)

        eigenvalues, eigenvectors, eig_scale = self._laplacian_eigs(forward)
        n_vertices = int(eigenvectors.shape[0])
        n_basis_use = self._resolve_n_basis(n_vertices, self.n_basis)
        U = np.asarray(eigenvectors[:, :n_basis_use], dtype=float)
        lam = np.asarray(eigenvalues[:n_basis_use], dtype=float)
        lam_scaled = lam / float(max(eig_scale, 1e-12))

        G = wf.G_white  # (n_eff, n_src)
        G_basis = G @ U  # (n_eff, n_basis)

        smooth_components: list[np.ndarray] = []
        for tau in self.taus:
            k = np.exp(-float(tau) * lam_scaled)
            B = G_basis * np.sqrt(k[np.newaxis, :])
            smooth_components.append(B @ B.T)

        gamma = np.full(n_vertices, float(self.gamma_floor), dtype=float)
        sigma2 = 1.0
        w = np.zeros(len(self.taus), dtype=float)

        n_outer = max(self.n_outer_iters, 1)
        for _outer in range(n_outer):
            G_gamma = G * gamma[np.newaxis, :]
            C_gamma = G_gamma @ G.T
            target = C - sigma2 * np.eye(n_eff) - C_gamma
            target = 0.5 * (target + target.T)
            w = self._fit_nnls_ridge(target, smooth_components, ridge=ridge)

            if float(w.sum()) < float(self.w_sum_floor):
                w = w + float(self.w_sum_floor) / float(len(w))

            sigma_smooth = np.zeros_like(lam_scaled, dtype=float)
            for wk, tau in zip(w, self.taus, strict=True):
                wk = float(max(wk, 0.0))
                sigma_smooth += wk * np.exp(-float(tau) * lam_scaled)
            sigma_smooth = np.maximum(sigma_smooth, 1e-12)

            diag_smooth = (U * U) @ sigma_smooth  # (n_src,)
            GU = G_basis
            G_smooth = (GU * sigma_smooth[np.newaxis, :]) @ U.T  # (n_eff, n_src)

            for _ in range(max(self.n_gamma_iters, 1)):
                G_gamma = G * gamma[np.newaxis, :]
                G_Sy = G_smooth + G_gamma  # (n_eff, n_src)

                C_model = G_Sy @ G.T + sigma2 * np.eye(n_eff)
                C_model.flat[:: n_eff + 1] += 1e-9
                try:
                    L = np.linalg.cholesky(C_model)
                except np.linalg.LinAlgError:
                    logger.debug("Cholesky failed; breaking EB loop.")
                    break
                Cinv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n_eff)))

                D = G.T @ Cinv
                UD = U.T @ D
                A_smooth = U @ (sigma_smooth[:, np.newaxis] * UD)
                A = A_smooth + gamma[:, np.newaxis] * D

                mu = A @ Yw  # (n_src, n_time)

                diag_Sy = diag_smooth + gamma
                diag_term = np.sum(A * G_Sy.T, axis=1)
                diag_post = np.maximum(diag_Sy - diag_term, 0.0)

                gamma_raw = np.mean(mu * mu, axis=1) + diag_post
                a = max(self.gamma_prior_a, 0.0)
                b = max(self.gamma_prior_b, 0.0)
                if a > 0.0 or b > 0.0:
                    gamma_raw = (gamma_raw + 2.0 * b) / (1.0 + 2.0 * a)
                gamma_raw = np.maximum(gamma_raw, self.gamma_floor)

                damp = float(np.clip(self.gamma_damping, 0.0, 1.0))
                gamma = (1.0 - damp) * gamma + damp * gamma_raw

                if self.learn_noise_scale:
                    resid = Yw - (G @ mu)
                    rss = float(np.sum(resid * resid))
                    tr_term = float(np.trace(G @ (diag_post[:, np.newaxis] * G.T)))
                    sigma2_raw = (rss + float(n_time) * tr_term) / float(n_eff * n_time)
                    sigma2_raw = float(max(sigma2_raw, self.noise_scale_floor))
                    sd = float(np.clip(self.noise_scale_damping, 0.0, 1.0))
                    sigma2 = (1.0 - sd) * float(sigma2) + sd * sigma2_raw

        G_gamma = G * gamma[np.newaxis, :]
        G_Sy = G_smooth + G_gamma
        C_model = G_Sy @ G.T + sigma2 * np.eye(n_eff)
        C_model.flat[:: n_eff + 1] += 1e-9
        L = np.linalg.cholesky(C_model)
        Cinv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n_eff)))
        D = G.T @ Cinv
        UD = U.T @ D
        A_smooth = U @ (sigma_smooth[:, np.newaxis] * UD)
        A = A_smooth + gamma[:, np.newaxis] * D

        inverse_operator = A @ wf.sensor_transform
        self.alphas = np.asarray([1.0], dtype=float)
        self.inverse_operators = [InverseOperator(inverse_operator, self.name)]
        return self
