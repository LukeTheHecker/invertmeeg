import mne
import numpy as np
from scipy.sparse.csgraph import laplacian

from ..base import BaseSolver, InverseOperator, SolverMeta


class SolverBasisFunctions(BaseSolver):
    """Class for the Minimum Norm Estimate (MNE) inverse solution [1] using
    basis functions. Gemoetric informed basis functions are based on [2].

    References
    ----------
    [1] Pascual-Marqui, R. D. (1999). Review of methods for solving the EEG
    inverse problem. International journal of bioelectromagnetism, 1(1), 75-86.

    [2] Wang, S., Wei, C., Lou, K., Gu, D., & Liu, Q. (2024). Advancing EEG/MEG
    Source Imaging with Geometric-Informed Basis Functions. arXiv preprint
    arXiv:2401.17939.

    """

    meta = SolverMeta(
        acronym="BF-MNE",
        full_name="MNE with Basis Functions",
        category="Minimum Norm",
        description=(
            "Minimum-norm inverse using a reduced basis (e.g., geometric-informed "
            "basis functions) to parameterize the source space."
        ),
        references=[
            "Hämäläinen, M. S., & Ilmoniemi, R. J. (1994). Interpreting magnetic fields of the brain: minimum norm estimates. Medical & Biological Engineering & Computing, 32(1), 35–42.",
            "Wang, S., Wei, C., Lou, K., Gu, D., & Liu, Q. (2024). Advancing EEG/MEG Source Imaging with Geometric-Informed Basis Functions. arXiv:2401.17939.",
        ],
    )

    def __init__(self, name="Minimum Norm Estimate with Basis Functions", **kwargs):
        self.name = name
        super().__init__(**kwargs)
        self.require_recompute = False
        self.require_data = False
        return None

    def make_inverse_operator(
        self,
        forward,
        *args,
        function="GBF",
        alpha="auto",
        n_basis=None,
        prior_shift=0.1,
        verbose=0,
        **kwargs,
    ):
        """Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        alpha : float
            The regularization parameter.

        Return
        ------
        self : object returns itself for convenience
        """
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        gbf_builder = self.create_basis_function(
            function=function, n_basis=n_basis, prior_shift=prior_shift
        )
        inverse_operators = [
            InverseOperator(gbf_builder(a), self.name) for a in self.alphas
        ]
        self.inverse_operators = inverse_operators
        return self

    def create_basis_function(self, function="GBF", n_basis=None, prior_shift=0.1):
        if function.lower() == "gbf":
            return self.create_gbf(n_basis=n_basis, prior_shift=prior_shift)
        raise ValueError(f"Function {function} not implemented.")

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

    def create_gbf(self, n_basis=None, prior_shift=0.1):
        """Create GBF inverse operators using graph-Laplacian eigenmodes.

        Source activity is represented as X = Phi * B where Phi are Laplacian
        eigenmodes and B are coefficients. MAP inference in basis space yields:
        B_hat = Sigma_b G^T (G Sigma_b G^T + alpha I)^-1 Y, with G = L Phi.
        """
        adjacency = mne.spatial_src_adjacency(self.forward["src"], verbose=0)
        graph_laplacian = laplacian(adjacency, normed=False).astype(float).toarray()

        eigenvalues, eigenvectors = np.linalg.eigh(graph_laplacian)
        n_vertices = eigenvectors.shape[0]
        n_basis_use = self._resolve_n_basis(n_vertices, n_basis)
        phi = eigenvectors[:, :n_basis_use]
        lam = eigenvalues[:n_basis_use]

        positive_eigs = lam[lam > 1e-12]
        eig_scale = float(np.median(positive_eigs)) if len(positive_eigs) else 1.0
        shift = max(float(prior_shift) * eig_scale, 1e-12)
        sigma_b_diag = 1.0 / (lam + shift)
        sigma_b = np.diag(sigma_b_diag)

        g_basis = self.leadfield @ phi
        n_chans = g_basis.shape[0]
        identity = np.eye(n_chans)

        self.phi = phi
        self.graph_laplacian = graph_laplacian
        self.eigenvalues = lam
        self.sigma_b_diag = sigma_b_diag

        def make_operator(alpha_value: float):
            alpha_safe = max(float(alpha_value), 1e-12)
            sensor_cov = g_basis @ sigma_b @ g_basis.T + alpha_safe * identity
            coef_operator = sigma_b @ g_basis.T @ np.linalg.inv(sensor_cov)
            return phi @ coef_operator

        return make_operator
