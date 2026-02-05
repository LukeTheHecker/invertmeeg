import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta


class SolverMVAB(BaseSolver):
    """Class for the Minimum Variance Adaptive Beamformer (MVAB) inverse solution.

    References
    ----------
    [1] Vorobyov, S. A. (2013). Principles of minimum variance robust adaptive
        beamforming design. Signal Processing, 93(12), 3264-3277.
    """

    meta = SolverMeta(
        slug="mvab",
        full_name="Minimum Variance Adaptive Beamformer",
        category="Beamformers",
        description=(
            "Minimum-variance adaptive beamformer implementation, including "
            "regularization (as implemented here)."
        ),
        references=[
            "Vorobyov, S. A. (2013). Principles of minimum variance robust adaptive "
            "beamforming design. Signal Processing, 93(12), 3264-3277.",
        ],
    )

    def __init__(
        self,
        name="Minimum Variance Adaptive Beamformer",
        reduce_rank=True,
        rank="auto",
        **kwargs,
    ):
        self.name = name
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, alpha="auto", **kwargs):
        """Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        alpha : float
            The regularization parameter.

        Return
        ------
        self : object returns itself for convenience

        """

        # NOTE: For MVAB we treat `alpha` as a dimensionless ratio r and apply it
        # separately in sensor- and source-space matrices (which have different scales).
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)
        leadfield = self.leadfield
        leadfield /= np.linalg.norm(leadfield, axis=0)
        n_chans, n_dipoles = self.leadfield.shape

        y = data
        I = np.identity(n_chans)

        C = self.data_covariance(y, center=True, ddof=1)
        if alpha == "auto":
            r_grid = np.asarray(self.r_values, dtype=float)
        else:
            r_grid = np.asarray([float(alpha)], dtype=float)

        # For MVAB the "grid parameter" is r (dimensionless).
        self.alphas = list(r_grid)
        max_eig_C = float(np.linalg.svd(C, compute_uv=False).max())

        inverse_operators = []
        for r in r_grid:
            alpha_cov = float(r) * max_eig_C
            R_inv = self.robust_inverse(C + alpha_cov * I)
            G = leadfield.T @ R_inv @ leadfield
            max_eig_G = float(np.linalg.svd(G, compute_uv=False).max())
            alpha_src = float(r) * max_eig_G
            inverse_operator = np.linalg.solve(
                G + alpha_src * np.identity(n_dipoles),
                leadfield.T @ R_inv,
            )

            inverse_operators.append(inverse_operator)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        return self
