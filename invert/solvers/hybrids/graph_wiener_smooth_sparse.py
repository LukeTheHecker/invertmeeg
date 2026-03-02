from __future__ import annotations

import logging

import mne
import numpy as np
from scipy.optimize import nnls
from scipy.sparse.csgraph import laplacian

from invert.util import build_source_adjacency

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)

# Module-level cache: forward objects are shared across BenchmarkRunner threads.
_LAPLACIAN_EIG_CACHE: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}


class SolverGraphWienerSmoothSparseEB(BaseSolver):
    """Empirical-Bayes Wiener filter with smooth background + sparse residual.

    Model (whitened sensor space)
    ----------------------------
    y_t = G x_t + n_t,   n_t ~ N(0, I)

    x_t ~ N(0, Σ_smooth + diag(γ))

    where Σ_smooth is a low-rank graph-stationary covariance (Laplacian
    eigenbasis, multi-scale heat kernels) and diag(γ) is a data-adaptive
    diagonal covariance to capture localized heavy-tailed/patchy residual
    activity.

    This matches the simulator's "mixture" regime (smooth background + patches)
    while remaining a linear-Gaussian MAP estimator (posterior mean).
    """

    meta = SolverMeta(
        acronym="GW-SS",
        full_name="Graph Wiener Smooth+Sparse (EB)",
        category="Minimum Norm",
        description=(
            "Empirical-Bayes Wiener/MAP estimator with a low-rank smooth background "
            "covariance plus a learned diagonal covariance for patchy residual activity."
        ),
        internal=True,
    )

    def __init__(
        self,
        name: str = "Graph Wiener Smooth+Sparse (EB)",
        taus: tuple[float, ...] = (0.2, 0.8, 3.2),
        n_basis: float | int | None = 0.2,
        sensor_cov_shrinkage: float = 0.2,
        temporal_whiten: bool = False,
        temporal_cov_shrinkage: float = 0.1,
        n_gamma_iters: int = 2,
        gamma_damping: float = 0.3,
        gamma_floor: float = 1e-10,
        **kwargs,
    ):
        self.name = str(name)
        self.taus = tuple(float(t) for t in taus)
        self.n_basis = n_basis
        self.sensor_cov_shrinkage = float(sensor_cov_shrinkage)
        self.temporal_whiten = bool(temporal_whiten)
        self.temporal_cov_shrinkage = float(temporal_cov_shrinkage)
        self.n_gamma_iters = int(n_gamma_iters)
        self.gamma_damping = float(gamma_damping)
        self.gamma_floor = float(gamma_floor)
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
    def _fit_smooth_weights_nnls(target: np.ndarray, components: list[np.ndarray]) -> np.ndarray:
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

        if self.temporal_whiten:
            # Rough separable-covariance prewhitening in time to reduce the
            # variance of spatial covariance estimates under temporally colored data.
            Ct = (Yw.T @ Yw) / float(max(n_eff, 1))
            shrink = float(self.temporal_cov_shrinkage)
            if not np.isfinite(shrink) or shrink < 0 or shrink > 1:
                raise ValueError(f"temporal_cov_shrinkage must be in [0,1], got {shrink}")
            if shrink:
                mu = float(np.trace(Ct)) / float(n_time)
                Ct = (1.0 - shrink) * Ct + shrink * mu * np.eye(n_time)
            d, V = np.linalg.eigh(Ct)
            d = np.maximum(d, 1e-12)
            invsqrt = (V * (1.0 / np.sqrt(d))[np.newaxis, :]) @ V.T
            Yw = Yw @ invsqrt

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

        # -----------------------------------------------------------------
        # Smooth background covariance: fit NNLS weights to C - I
        # (use only tau>0 components; sparse diag handles focal residuals).
        # -----------------------------------------------------------------
        target = C - np.eye(n_eff)
        components: list[np.ndarray] = []
        for tau in self.taus:
            k = np.exp(-float(tau) * lam_scaled)
            B = G_basis * np.sqrt(k[np.newaxis, :])
            components.append(B @ B.T)
        w = self._fit_smooth_weights_nnls(target, components)
        sigma_smooth = np.zeros_like(lam_scaled, dtype=float)
        for wk, tau in zip(w, self.taus, strict=True):
            wk = float(max(wk, 0.0))
            sigma_smooth += wk * np.exp(-float(tau) * lam_scaled)
        sigma_smooth = np.maximum(sigma_smooth, 1e-12)

        # Precompute helpers for low-rank Σ_smooth = U diag(sigma_smooth) U^T.
        diag_smooth = (U * U) @ sigma_smooth  # (n_src,)

        # -----------------------------------------------------------------
        # Learn diagonal residual covariance gamma via a few EB iterations.
        # -----------------------------------------------------------------
        gamma = np.full(n_vertices, float(self.gamma_floor), dtype=float)
        I = np.eye(n_eff)

        # Cache these across gamma iterations (depend only on smooth part).
        GU = G_basis  # (n_eff, n_basis)
        # (n_eff, n_src) for smooth contribution: G Σ_smooth = (G U) diag(sigma) U^T
        G_smooth = (GU * sigma_smooth[np.newaxis, :]) @ U.T

        for _ in range(max(self.n_gamma_iters, 0)):
            # G Σ_y = G Σ_smooth + G diag(gamma)
            G_gamma = G * gamma[np.newaxis, :]
            G_Sy = G_smooth + G_gamma  # (n_eff, n_src)

            C_model = G_Sy @ G.T + I
            C_model.flat[:: n_eff + 1] += 1e-9
            try:
                L = np.linalg.cholesky(C_model)
            except np.linalg.LinAlgError:
                logger.debug("Cholesky failed in gamma loop; breaking early.")
                break
            Cinv = np.linalg.solve(L.T, np.linalg.solve(L, I))

            # D = G^T C^{-1}
            D = G.T @ Cinv  # (n_src, n_eff)

            # A = Σ_y D = Σ_smooth D + diag(gamma) D
            UD = U.T @ D  # (n_basis, n_eff)
            A_smooth = U @ (sigma_smooth[:, np.newaxis] * UD)  # (n_src, n_eff)
            A = A_smooth + gamma[:, np.newaxis] * D  # (n_src, n_eff)

            mu = A @ Yw  # (n_src, n_time)

            # diag(Σ_post) = diag(Σ_y) - diag( A @ (G Σ_y) )
            diag_Sy = diag_smooth + gamma
            diag_term = np.sum(A * G_Sy.T, axis=1)
            diag_post = np.maximum(diag_Sy - diag_term, 0.0)

            gamma_new = np.mean(mu * mu, axis=1) + diag_post
            gamma_new = np.maximum(gamma_new, self.gamma_floor)
            damp = float(np.clip(self.gamma_damping, 0.0, 1.0))
            gamma = (1.0 - damp) * gamma + damp * gamma_new

        # Final operator: W = Σ_y G^T C^{-1} = A
        inverse_operator = A @ wf.sensor_transform
        self.alphas = np.asarray([1.0], dtype=float)
        self.inverse_operators = [InverseOperator(inverse_operator, self.name)]
        return self
