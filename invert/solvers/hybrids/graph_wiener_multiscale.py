from __future__ import annotations

import logging

import mne
import numpy as np
from scipy.optimize import minimize, nnls
from scipy.sparse.csgraph import laplacian

from invert.util import build_source_adjacency

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverGraphWienerMultiScale(BaseSolver):
    """Graph-spectral Wiener filter with an empirical-Bayes multi-scale prior.

    This solver targets "ongoing activity" regimes where the source covariance
    is not sparse and has mass across multiple spatial scales.

    In whitened space (noise ≈ I), we fit a small set of nonnegative weights
    {w_k} such that:

        C_y ≈ I + Σ_k w_k (G U diag(exp(-τ_k λ)) Uᵀ Gᵀ)

    where (U, λ) are graph Laplacian eigenvectors/values and τ_k are fixed
    spatial scales. We then build the corresponding Wiener/MAP inverse.
    """

    meta = SolverMeta(
        acronym="GW-MSEB",
        full_name="Graph Wiener Multi-Scale (EB)",
        category="Minimum Norm",
        description=(
            "Empirical-Bayes graph-spectral Wiener/MAP estimator that fits a "
            "multi-scale stationary source prior to the whitened sensor covariance "
            "via nonnegative least squares."
        ),
        internal=True,
    )

    def __init__(
        self,
        name: str = "Graph Wiener Multi-Scale (EB)",
        taus: tuple[float, ...] = (0.2, 0.8, 3.2),
        sensor_cov_shrinkage: float = 0.2,
        n_basis: float | int | None = 0.3,
        fit_noise_scale: bool = True,
        alpha_fixed: float | None = 1.0,
        weight_fit: str = "nnls",
        **kwargs,
    ):
        self.name = str(name)
        self.taus = tuple(float(t) for t in taus)
        self.sensor_cov_shrinkage = float(sensor_cov_shrinkage)
        self.n_basis = n_basis
        self.fit_noise_scale = bool(fit_noise_scale)
        self.alpha_fixed = None if alpha_fixed is None else float(alpha_fixed)
        self.weight_fit = str(weight_fit).lower()
        super().__init__(**kwargs)

        # Data-dependent covariance fit ⇒ recompute per sample.
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
    def _fit_nnls_weights(target: np.ndarray, components: list[np.ndarray]) -> np.ndarray:
        n = int(target.shape[0])
        tri = np.triu_indices(n)
        b = np.asarray(target[tri], dtype=float)
        A = np.stack([np.asarray(C[tri], dtype=float) for C in components], axis=1)
        w, _ = nnls(A, b)
        return np.asarray(w, dtype=float)

    @staticmethod
    def _fit_ml_weights(
        S: np.ndarray,
        components: list[np.ndarray],
        w0: np.ndarray,
        baseline_scale: float,
    ) -> np.ndarray:
        """Fit nonnegative weights by Gaussian covariance ML (convex in w>=0)."""

        comps = [np.asarray(C, dtype=float) for C in components]
        S = np.asarray(S, dtype=float)
        n = int(S.shape[0])
        I = np.eye(n)
        baseline_scale = float(baseline_scale)

        def obj(w: np.ndarray) -> float:
            C = baseline_scale * I
            for wi, Ci in zip(w, comps, strict=True):
                if wi != 0.0:
                    C += float(max(wi, 0.0)) * Ci
            # Jitter for numerical stability; objective remains well-defined.
            C.flat[:: n + 1] += 1e-9
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                return 1e30
            logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
            # Solve C^{-1} S via two triangular solves.
            X = np.linalg.solve(L, S)
            X = np.linalg.solve(L.T, X)
            tr_term = float(np.trace(X))
            return logdet + tr_term

        bounds = [(0.0, None) for _ in comps]
        res = minimize(
            obj,
            x0=np.asarray(w0, dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 50},
        )
        if not res.success or res.x is None:
            raise RuntimeError(f"ML weight fit failed: {res.message}")
        return np.asarray(res.x, dtype=float)

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

        adjacency = build_source_adjacency(forward["src"], verbose=0)
        graph_laplacian = laplacian(adjacency, normed=False).astype(float).toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(graph_laplacian)

        n_vertices = int(eigenvectors.shape[0])
        n_basis_use = self._resolve_n_basis(n_vertices, self.n_basis)
        U = np.asarray(eigenvectors[:, :n_basis_use], dtype=float)
        lam = np.asarray(eigenvalues[:n_basis_use], dtype=float)

        positive = lam[lam > 1e-12]
        eig_scale = float(np.median(positive)) if positive.size else 1.0
        lam_scaled = lam / float(max(eig_scale, 1e-12))

        G_white = wf.G_white
        G_basis = G_white @ U  # (n_eff, n_basis)

        # Build sensor covariance components for NNLS fit.
        components: list[np.ndarray] = []
        tau_list: list[float] = []

        if self.fit_noise_scale:
            components.append(np.eye(n_eff))
            tau_list.append(float("nan"))  # sentinel (noise scale)
            target = C
        else:
            target = C - np.eye(n_eff)

        for tau in (0.0, *self.taus):
            tau_f = float(tau)
            k = np.exp(-tau_f * lam_scaled)
            k = np.maximum(k, 0.0)
            B = G_basis * np.sqrt(k[np.newaxis, :])
            components.append(B @ B.T)
            tau_list.append(tau_f)

        w0 = self._fit_nnls_weights(target, components)
        if self.weight_fit == "ml":
            try:
                baseline_scale = 0.0 if self.fit_noise_scale else 1.0
                w = self._fit_ml_weights(C, components, w0=w0, baseline_scale=baseline_scale)
            except Exception as exc:
                logger.debug("ML weight fit failed; falling back to NNLS (%s).", exc)
                w = w0
        elif self.weight_fit == "nnls":
            w = w0
        else:
            raise ValueError("weight_fit must be 'ml' or 'nnls'")

        noise_scale = 1.0
        w_offset = 0
        if self.fit_noise_scale:
            noise_scale = float(max(w[0], 1e-12))
            w_offset = 1

        weights = w[w_offset:]
        if weights.size != len(tau_list) - w_offset:
            raise RuntimeError("Internal weight bookkeeping mismatch.")

        sigma_diag = np.zeros_like(lam_scaled, dtype=float)
        for wk, tau in zip(weights, tau_list[w_offset:], strict=True):
            wk = float(max(wk, 0.0))
            if tau == 0.0:
                sigma_diag += wk
            else:
                sigma_diag += wk * np.exp(-float(tau) * lam_scaled)
        sigma_diag = np.maximum(sigma_diag, 1e-12)

        # Regularization (noise scaling). For this EB Wiener estimator, classic
        # residual-based GCV can be brittle under temporally correlated samples;
        # by default we use a fixed α≈1 in whitened space.
        if self.alpha_fixed is None:
            self.get_alphas(reference=G_basis @ G_basis.T)
        else:
            self.alphas = np.asarray([max(self.alpha_fixed, 1e-12)], dtype=float)
        I = np.eye(n_eff)

        inverse_operators = []
        for alpha_value in self.alphas:
            alpha_safe = max(float(alpha_value), 1e-12)
            GS = G_basis * sigma_diag[np.newaxis, :]
            sensor_cov = GS @ G_basis.T + (noise_scale * alpha_safe) * I

            A = (sigma_diag[:, np.newaxis] * G_basis.T)  # (n_basis, n_eff)
            coef = np.linalg.solve(sensor_cov, A.T).T  # (n_basis, n_eff)
            op_white = U @ coef  # (n_sources, n_eff)
            inverse_operator = op_white @ wf.sensor_transform  # (n_sources, n_chans)
            inverse_operators.append(InverseOperator(inverse_operator, self.name))

        self.inverse_operators = inverse_operators
        return self
