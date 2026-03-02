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


class SolverGraphWienerSmoothSparseAdaptiveEB(BaseSolver):
    """Smooth+Sparse empirical Bayes Wiener with adaptive noise/regularization.

    Compared to ``SolverGraphWienerSmoothSparseEB`` this adds:
    - optional learning of an isotropic noise scale σ² (post-whitening)
    - alternating updates of smooth weights w, diagonal residual γ, and σ²
    - optional Gamma/Inv-Gamma hyperprior regularization for γ

    Intended for low-SNR regimes where overfitting the sample covariance is a
    primary failure mode.
    """

    meta = SolverMeta(
        acronym="GW-SSA",
        full_name="Graph Wiener Smooth+Sparse (Adaptive EB)",
        category="Minimum Norm",
        description=(
            "Empirical-Bayes Wiener/MAP estimator with a low-rank smooth background "
            "plus diagonal residual covariance, optionally learning an isotropic noise scale "
            "and using a hyperprior on diagonal variances for low-SNR robustness."
        ),
        internal=True,
    )

    def __init__(
        self,
        name: str = "Graph Wiener Smooth+Sparse (Adaptive EB)",
        taus: tuple[float, ...] = (0.2, 0.8, 3.2),
        n_basis: float | int | None = 0.15,
        sensor_cov_shrinkage: float = 0.2,
        n_outer_iters: int = 2,
        n_gamma_iters: int = 1,
        gamma_damping: float = 0.5,
        gamma_floor: float = 1e-10,
        learn_noise_scale: bool = True,
        noise_scale_floor: float = 1e-6,
        gamma_prior_a: float = 0.0,
        gamma_prior_b: float = 0.0,
        **kwargs,
    ):
        self.name = str(name)
        self.taus = tuple(float(t) for t in taus)
        self.n_basis = n_basis
        self.sensor_cov_shrinkage = float(sensor_cov_shrinkage)
        self.n_outer_iters = int(n_outer_iters)
        self.n_gamma_iters = int(n_gamma_iters)
        self.gamma_damping = float(gamma_damping)
        self.gamma_floor = float(gamma_floor)
        self.learn_noise_scale = bool(learn_noise_scale)
        self.noise_scale_floor = float(noise_scale_floor)
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
            raise ValueError(f"sensor_cov_shrinkage must be in [0, 1], got {shrink}")
        if shrink == 0:
            return C
        n = C.shape[0]
        mu = float(np.trace(C)) / float(n)
        return (1.0 - shrink) * C + shrink * mu * np.eye(n)

    @staticmethod
    def _fit_nnls(target: np.ndarray, components: list[np.ndarray]) -> np.ndarray:
        n = int(target.shape[0])
        tri = np.triu_indices(n)
        b = np.asarray(target[tri], dtype=float)
        A = np.stack([np.asarray(C[tri], dtype=float) for C in components], axis=1)
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

        C = (Yw @ Yw.T) / float(n_time)
        C = self._shrink_sensor_cov(C, self.sensor_cov_shrinkage)

        eigenvalues, eigenvectors, eig_scale = self._laplacian_eigs(forward)
        n_vertices = int(eigenvectors.shape[0])
        n_basis_use = self._resolve_n_basis(n_vertices, self.n_basis)
        U = np.asarray(eigenvectors[:, :n_basis_use], dtype=float)
        lam = np.asarray(eigenvalues[:n_basis_use], dtype=float)
        lam_scaled = lam / float(max(eig_scale, 1e-12))

        G = wf.G_white  # (n_eff, n_src)
        G_basis = G @ U  # (n_eff, n_basis)

        # Precompute smooth covariance components on sensor level.
        smooth_components: list[np.ndarray] = []
        for tau in self.taus:
            k = np.exp(-float(tau) * lam_scaled)
            B = G_basis * np.sqrt(k[np.newaxis, :])
            smooth_components.append(B @ B.T)

        # Initialize parameters.
        gamma = np.full(n_vertices, float(self.gamma_floor), dtype=float)
        sigma2 = 1.0
        w = np.zeros(len(self.taus), dtype=float)

        # Alternate updates: smooth weights w, diagonal gamma, noise scale sigma2.
        n_outer = max(self.n_outer_iters, 1)
        for _outer in range(n_outer):
            # Update smooth weights given gamma, sigma2 by NNLS matching of residual covariance.
            G_gamma = G * gamma[np.newaxis, :]
            C_gamma = G_gamma @ G.T
            target = C - sigma2 * np.eye(n_eff) - C_gamma
            target = 0.5 * (target + target.T)
            w = self._fit_nnls(target, smooth_components)

            # Build sigma_smooth in Laplacian basis.
            sigma_smooth = np.zeros_like(lam_scaled, dtype=float)
            for wk, tau in zip(w, self.taus, strict=True):
                wk = float(max(wk, 0.0))
                sigma_smooth += wk * np.exp(-float(tau) * lam_scaled)
            sigma_smooth = np.maximum(sigma_smooth, 1e-12)

            diag_smooth = (U * U) @ sigma_smooth  # (n_src,)
            GU = G_basis  # (n_eff, n_basis)
            G_smooth = (GU * sigma_smooth[np.newaxis, :]) @ U.T  # (n_eff, n_src)

            # Update gamma (a few steps) and optionally sigma2.
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

                D = G.T @ Cinv  # (n_src, n_eff)
                UD = U.T @ D  # (n_basis, n_eff)
                A_smooth = U @ (sigma_smooth[:, np.newaxis] * UD)  # (n_src, n_eff)
                A = A_smooth + gamma[:, np.newaxis] * D  # (n_src, n_eff)

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
                    sigma2 = (rss + float(n_time) * tr_term) / float(n_eff * n_time)
                    sigma2 = float(max(sigma2, self.noise_scale_floor))

        # Final operator: W = Σ_y G^T C^{-1} = A from last iteration.
        # Ensure A exists even if we early-broke.
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
