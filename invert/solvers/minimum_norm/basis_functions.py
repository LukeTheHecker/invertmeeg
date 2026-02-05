import numpy as np
from scipy.sparse import coo_matrix

from ...util import pos_from_forward
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
        return super().__init__(**kwargs)

    def make_inverse_operator(
        self, forward, *args, function="GBF", alpha="auto", verbose=0, **kwargs
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
        # n_chans, _ = leadfield.shape

        self.get_inverse_operator = self.create_basis_function(function)

        if alpha == "auto":
            r_grid = np.asarray(self.r_values, dtype=float)
        else:
            r_grid = np.asarray([float(alpha)], dtype=float)

        # No regularization leads to weird results with this approach
        if 0 in r_grid and len(r_grid) > 1:
            r_grid = r_grid[r_grid != 0]
        elif 0 in r_grid and len(r_grid) == 1:
            r_grid = np.asarray([0.01], dtype=float)
        self.alphas = list(r_grid)

        inverse_operators = []
        for alpha in self.alphas:
            inverse_operator = self.get_inverse_operator(alpha)
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        return self

    def create_basis_function(self, function="GBF"):
        if function.lower() == "gbf":
            return self.create_gbf()
        else:
            raise ValueError(f"Function {function} not implemented.")

    def create_gbf(self):
        """Create geometric informed basis functions."""
        n_vertices_left = self.forward["src"][0]["nuse"]
        self.faces = np.concatenate(
            [
                self.forward["src"][0]["use_tris"],
                n_vertices_left + self.forward["src"][1]["use_tris"],
            ],
            axis=0,
        )

        # self.pos = np.concatenate([
        #     self.forward['src'][0]['rr'],
        #     self.forward['src'][1]['rr'],
        # ], axis=0)
        self.pos = pos_from_forward(self.forward)

        A = self.compute_laplace_beltrami(self.pos.T, self.faces)
        _, eigenvalues, _ = np.linalg.svd(A.toarray(), full_matrices=False)
        Sigma = np.diag(1 / (eigenvalues + 0.1 * np.mean(eigenvalues)))
        Sigma_inv = np.linalg.inv(Sigma)
        L = self.leadfield @ A

        LTL = L.T @ L
        max_eig_LTL = float(np.linalg.svd(LTL, compute_uv=False).max())
        max_eig_penalty = float(np.linalg.svd(Sigma_inv, compute_uv=False).max())
        scale = max_eig_LTL / max(max_eig_penalty, 1e-15)

        return lambda alpha: np.linalg.inv(
            LTL + (float(alpha) * scale) * Sigma_inv
        ) @ L.T

    @staticmethod
    def cotangent_weight(v1, v2, v3):
        # Compute the cotangent weight of the edge opposite to v1
        edge1 = v2 - v1
        edge2 = v3 - v1
        cotangent = np.dot(edge1, edge2) / np.linalg.norm(np.cross(edge1, edge2))
        return cotangent

    def compute_laplace_beltrami(self, pos, faces):
        n = pos.shape[1]  # Number of vertices
        I = []
        J = []
        V = []

        for face in faces:
            for i in range(3):
                j = (i + 1) % 3
                k = (i + 2) % 3

                vi = pos[:, face[i]]
                vj = pos[:, face[j]]
                vk = pos[:, face[k]]

                # Compute cotangent weights for edges (vi, vj) and (vi, vk)
                cot_jk = self.cotangent_weight(vi, vj, vk)
                cot_kj = self.cotangent_weight(vi, vk, vj)

                # Update the entries for the Laplacian matrix
                I.append(face[i])
                J.append(face[j])
                V.append(-0.5 * (cot_jk + cot_kj))

                # Add the contribution to the diagonal element
                I.append(face[i])
                J.append(face[i])
                V.append(0.5 * (cot_jk + cot_kj))

        # Create the sparse Laplacian matrix
        L = coo_matrix((V, (I, J)), shape=(n, n))

        return L
