import logging

import mne
import numpy as np
from scipy.sparse.csgraph import laplacian

from invert.util import build_source_adjacency

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverGraphWienerMixture(BaseSolver):
    """Graph-spectral Wiener filter with a smooth+unstructured mixture prior.

    This is a first-principles linear-Gaussian solver:

        x = G y + n,   n ~ N(0, I)   (after whitening)
        y ~ N(0, Σ_y), Σ_y = U diag(s(λ)) U^T

    where U, λ are Laplacian eigenvectors/values and s(λ) is a mixture:

        s(λ) = a / (λ + shift)^p  +  b

    The constant term b provides an unstructured component (helps retain
    higher graph frequencies / patch-like activity), while the first term
    favors smooth background activity.
    """

    meta = SolverMeta(
        acronym="GW-MIX",
        full_name="Graph Wiener Mixture",
        category="Minimum Norm",
        description=(
            "Graph-spectral Wiener/MAP estimator with a mixture prior combining "
            "a Laplacian-smooth component and an unstructured component."
        ),
        internal=True,
    )

    def __init__(
        self,
        name: str = "Graph Wiener Mixture",
        prior_shift: float = 0.1,
        smooth_exponent: float = 1.0,
        nugget: float = 0.05,
        a_scale: float = 1.0,
        n_basis: float | int | None = None,
        **kwargs,
    ):
        self.name = name
        self.prior_shift = float(prior_shift)
        self.smooth_exponent = float(smooth_exponent)
        self.nugget = float(nugget)
        self.a_scale = float(a_scale)
        self.n_basis = n_basis
        super().__init__(**kwargs)
        self.require_recompute = False
        self.require_data = False

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

    def make_inverse_operator(
        self,
        forward,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
        verbose=0,
        **kwargs,
    ):
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        wf = self.prepare_whitened_forward(noise_cov)

        adjacency = build_source_adjacency(forward["src"], verbose=0)
        graph_laplacian = laplacian(adjacency, normed=False).astype(float).toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(graph_laplacian)

        n_vertices = int(eigenvectors.shape[0])
        n_basis_use = self._resolve_n_basis(n_vertices, self.n_basis)
        U = np.asarray(eigenvectors[:, :n_basis_use], dtype=float)
        lam = np.asarray(eigenvalues[:n_basis_use], dtype=float)

        positive = lam[lam > 1e-12]
        eig_scale = float(np.median(positive)) if positive.size else 1.0
        shift = max(float(self.prior_shift) * eig_scale, 1e-12)

        p = float(max(self.smooth_exponent, 1e-6))
        a = float(max(self.a_scale, 0.0))
        b = float(max(self.nugget, 0.0))

        smooth = a / (lam + shift) ** p
        sigma_diag = smooth + b
        sigma_diag = np.maximum(sigma_diag, 1e-12)

        G_white = wf.G_white
        G_basis = G_white @ U  # (n_eff, n_basis)
        n_eff = int(G_basis.shape[0])
        I = np.eye(n_eff)

        # Regularization grid based on the effective sensor covariance scale.
        self.get_alphas(reference=G_basis @ G_basis.T)

        inverse_operators = []
        for alpha_value in self.alphas:
            alpha_safe = max(float(alpha_value), 1e-12)
            # Sensor covariance: G S G^T + alpha I, with S diagonal in basis.
            GS = G_basis * sigma_diag[np.newaxis, :]
            sensor_cov = GS @ G_basis.T + alpha_safe * I
            coef_operator = (sigma_diag[:, np.newaxis] * G_basis.T) @ np.linalg.inv(
                sensor_cov
            )  # (n_basis, n_eff)
            op_white = U @ coef_operator  # (n_sources, n_eff)
            inverse_operator = op_white @ wf.sensor_transform  # (n_sources, n_chans)
            inverse_operators.append(InverseOperator(inverse_operator, self.name))

        self.inverse_operators = inverse_operators
        return self

