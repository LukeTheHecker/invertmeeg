import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta


class SolverMCMV(BaseSolver):
    """Class for the Multiple Constrained Minimum Variance (MCMV) Beamformer
    inverse solution [1].

    MCMV extends LCMV by applying multiple linear constraints simultaneously.
    This improves robustness in the presence of correlated sources.

    References
    ----------
    [1] Nunes, A. S., Moiseev, A., Kozhemiako, N., Cheung, T., Ribary, U., &
    Doesburg, S. M. (2020). Multiple constrained minimum variance beamformer
    (MCMV) performance in connectivity analyses. NeuroImage, 208, 116386.

    """

    meta = SolverMeta(
        slug="mcmv",
        full_name="Multiple Constrained Minimum Variance",
        category="Beamformers",
        description=(
            "Multiple-constrained extension of LCMV that imposes additional linear "
            "constraints (e.g., to improve robustness with correlated sources)."
        ),
        references=[
            "Nunes, A. S., Moiseev, A., Kozhemiako, N., Cheung, T., Ribary, U., "
            "& Doesburg, S. M. (2020). Multiple constrained minimum variance "
            "beamformer (MCMV) performance in connectivity analyses. "
            "NeuroImage, 208, 116386.",
        ],
    )

    def __init__(self, name="MCMV Beamformer", reduce_rank=True, rank="auto", **kwargs):
        self.name = name
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(
        self,
        forward,
        mne_obj,
        *args,
        weight_norm=True,
        noise_cov=None,
        alpha="auto",
        k_constraints=3,
        verbose=0,
        **kwargs,
    ):
        """Calculate inverse operator using MCMV formula.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        weight_norm : bool
            Normalize the filter weight matrix W to unit length of the columns.
        alpha : float
            The regularization parameter.
        k_constraints : int
            Number of constraints per source. When k=1, reduces to LCMV.
            For k>1, includes k-1 nearest neighbors in constraint matrix.

        Return
        ------
        self : object returns itself for convenience

        Notes
        -----
        Implements the MCMV formula: w = C_inv @ G @ inv(G.T @ C_inv @ G) @ f
        where G is the constraint matrix (m × k) and f is the constraint vector.
        """
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)

        leadfield = self.leadfield
        leadfield /= np.linalg.norm(leadfield, axis=0)
        n_chans, n_dipoles = leadfield.shape

        if noise_cov is None:
            noise_cov = np.identity(n_chans)

        self.weight_norm = weight_norm
        self.k_constraints = min(
            k_constraints, n_dipoles
        )  # Ensure k doesn't exceed n_dipoles

        y = data
        I = np.identity(n_chans)

        # Recompute regularization based on the max eigenvalue of the Covariance
        # Matrix (opposed to that of the leadfield)
        y -= y.mean(axis=1, keepdims=True)
        C = self.data_covariance(y, center=False, ddof=1)
        self.alphas = self.get_alphas(reference=C)

        # Precompute spatial distances for finding nearest neighbors
        if self.k_constraints > 1:
            # Compute pairwise correlations/similarities between leadfield columns
            leadfield_norm = leadfield / (np.linalg.norm(leadfield, axis=0) + 1e-10)
            similarity_matrix = leadfield_norm.T @ leadfield_norm

        inverse_operators = []
        for alpha in self.alphas:
            C_inv = self.robust_inverse(C + alpha * I)
            W = np.zeros((n_chans, n_dipoles))

            for i in range(n_dipoles):
                # Construct constraint matrix G for source i
                if self.k_constraints == 1:
                    # Single constraint (LCMV special case)
                    G = leadfield[:, i : i + 1]  # Keep as column vector
                    f = np.array([1.0])
                else:
                    # Multiple constraints: include k-1 nearest neighbors
                    # Find k-1 most similar sources (excluding self)
                    similarities = similarity_matrix[i, :].copy()
                    similarities[i] = -np.inf  # Exclude self
                    neighbor_indices = np.argsort(similarities)[::-1][
                        : self.k_constraints - 1
                    ]

                    # G = [target_source, neighbor_1, neighbor_2, ...]
                    constraint_indices = np.concatenate([[i], neighbor_indices])
                    G = leadfield[:, constraint_indices]

                    # f = [1, 0, 0, ...] - unit gain for target, zero for neighbors
                    f = np.zeros(self.k_constraints)
                    f[0] = 1.0

                # Apply MCMV formula: w = C_inv @ G @ inv(G.T @ C_inv @ G) @ f
                G_T_C_inv = G.T @ C_inv
                G_T_C_inv_G = G_T_C_inv @ G

                # Robust inverse of the k×k matrix
                try:
                    G_T_C_inv_G_inv = np.linalg.inv(G_T_C_inv_G)
                except np.linalg.LinAlgError:
                    # Fallback to pseudo-inverse if singular
                    G_T_C_inv_G_inv = np.linalg.pinv(G_T_C_inv_G)

                # w = C_inv @ G @ inv(G.T @ C_inv @ G) @ f
                w = C_inv @ G @ G_T_C_inv_G_inv @ f
                W[:, i] = w

            if self.weight_norm:
                W /= np.linalg.norm(W, axis=0)

            inverse_operator = W.T
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]

        return self
