import logging

import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverDSPM(BaseSolver):
    """Class for the Dynamic Statistical Parametric Mapping (dSPM) inverse
        solution [1,2].  The formulas provided by [3] were used for
        implementation.

    References
    ----------
    [1] Dale, A. M., Liu, A. K., Fischl, B. R., Buckner, R. L., Belliveau, J.
    W., Lewine, J. D., & Halgren, E. (2000). Dynamic statistical parametric
    mapping: combining fMRI and MEG for high-resolution imaging of cortical
    activity. neuron, 26(1), 55-67.

    [2] Dale, A. M., Fischl, B., & Sereno, M. I. (1999). Cortical surface-based
    analysis: I. Segmentation and surface reconstruction. Neuroimage, 9(2),
    179-194.

    [3] Grech, R., Cassar, T., Muscat, J., Camilleri, K. P., Fabri, S. G.,
    Zervakis, M., ... & Vanrumste, B. (2008). Review on solving the inverse
    problem in EEG source analysis. Journal of neuroengineering and
    rehabilitation, 5(1), 1-33.
    """

    meta = SolverMeta(
        acronym="dSPM",
        full_name="Dynamic Statistical Parametric Mapping",
        category="Minimum Norm",
        description=(
            "Noise-normalized minimum-norm inverse (MNE) that yields statistical "
            "maps by scaling the MNE estimate with an estimate of its variance."
        ),
        references=[
            "Dale, A. M., Liu, A. K., Fischl, B. R., Buckner, R. L., Belliveau, J. W., Lewine, J. D., & Halgren, E. (2000). Dynamic statistical parametric mapping: combining fMRI and MEG for high-resolution imaging of cortical activity. Neuron, 26(1), 55–67.",
        ],
    )

    def __init__(self, name="Dynamic Statistical Parametric Mapping", **kwargs):
        self.name = name
        kwargs.setdefault("use_depth_weighting", True)
        kwargs.setdefault("depth_weighting", 0.1)
        return super().__init__(**kwargs)

    def make_inverse_operator(
        self,
        forward,
        *args,
        alpha="auto",
        noise_cov=None,
        source_cov=None,
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
        noise_cov : numpy.ndarray
            The noise covariance matrix (channels x channels).
        source_cov : numpy.ndarray
            The source covariance matrix (dipoles x dipoles). This can be used if
            prior information, e.g., from fMRI images, is available.

        Return
        ------
        self : object returns itself for convenience
        """
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        n_chans, n_dipoles = self.leadfield.shape

        if noise_cov is None:
            noise_cov = np.identity(n_chans)
        if source_cov is None:
            source_cov = np.identity(n_dipoles)

        inverse_operators = []
        leadfield_source_cov = source_cov @ self.leadfield.T
        LLS = self.leadfield @ leadfield_source_cov
        if alpha == "auto":
            r_grid = np.asarray(self.r_values, dtype=float)
        else:
            r_grid = np.asarray([float(alpha)], dtype=float)
        max_eig_LLS = float(np.linalg.svd(LLS, compute_uv=False).max())
        max_eig_noise = float(np.linalg.svd(noise_cov, compute_uv=False).max())
        scale = max_eig_LLS / max(max_eig_noise, 1e-15)
        self.alphas = list(r_grid * scale)

        for alpha in self.alphas:
            K = leadfield_source_cov @ self.robust_inverse(LLS + alpha * noise_cov)
            W_dSPM = np.diag(np.sqrt(1 / np.diagonal(K @ noise_cov @ K.T)))
            inverse_operator = W_dSPM @ K
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        return self
