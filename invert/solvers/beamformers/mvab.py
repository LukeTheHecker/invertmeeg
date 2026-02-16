import mne
import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta
from .utils import build_covariance_candidates


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
        kwargs.setdefault("regularisation_method", "L")
        self.name = name
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(
        self,
        forward,
        mne_obj=None,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
        cov_reg: str = "oas",
        cov_reg_beta: float = 0.05,
        cov_reg_cond_target: float = 1e4,
        **kwargs,
    ):
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
        super().make_inverse_operator(forward, mne_obj, *args, alpha=alpha, **kwargs)
        wf = self.prepare_whitened_forward(noise_cov)
        data = self.unpack_data_obj(mne_obj)
        leadfield = wf.G_white
        leadfield /= np.linalg.norm(leadfield, axis=0)
        n_chans, n_dipoles = leadfield.shape

        y = wf.sensor_transform @ data
        I = np.identity(n_chans)
        eps = 1e-12

        C = self.data_covariance(y, center=True, ddof=1)
        cov_reg_l = str(cov_reg).strip().lower()
        if alpha == "auto" and cov_reg_l not in {"grid", "legacy", "maxeig"}:
            cov_mats, _alpha_eff, cov_meta = build_covariance_candidates(
                C=C,
                I=I,
                alpha="auto",
                get_alphas_fn=lambda reference: [0.0],
                n_samples=int(y.shape[1]),
                cov_reg=cov_reg,
                cov_reg_beta=float(cov_reg_beta),
                cov_reg_cond_target=float(cov_reg_cond_target),
            )
            if "oas_shrinkage" in cov_meta:
                self._cov_reg_oas_shrinkage = float(cov_meta["oas_shrinkage"])
            r_grid = np.zeros(len(cov_mats), dtype=float)
        else:
            if alpha == "auto":
                r_grid = np.asarray(self.r_values, dtype=float)
            else:
                r_grid = np.asarray([float(alpha)], dtype=float)
            cov_mats = [C for _ in range(len(r_grid))]

        # For MVAB the public "alpha" remains the dimensionless r.
        self.alphas = [float(r) for r in r_grid]

        inverse_operators = []
        for C_base, r in zip(cov_mats, r_grid, strict=False):
            if float(r) != 0.0:
                max_eig_C = float(np.linalg.svd(C_base, compute_uv=False).max())
                alpha_cov = float(r) * max_eig_C
                R_inv = self.robust_inverse(C_base + alpha_cov * I)
            else:
                R_inv = self.robust_inverse(C_base)
            G = leadfield.T @ R_inv @ leadfield
            max_eig_G = float(np.linalg.svd(G, compute_uv=False).max())
            alpha_src = max(float(r) * max_eig_G, eps)
            inverse_operator = np.linalg.solve(
                G + alpha_src * np.identity(n_dipoles),
                leadfield.T @ R_inv,
            )

            inverse_operators.append(inverse_operator @ wf.sensor_transform)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        return self
