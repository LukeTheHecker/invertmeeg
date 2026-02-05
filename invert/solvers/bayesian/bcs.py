import mne
import numpy as np

from ..base import BaseSolver, SolverMeta


class SolverBCS(BaseSolver):
    """Class for the Bayesian Compressed Sensing (BCS) inverse solution [1].

    References
    ----------
    [1] Ji, S., Xue, Y., & Carin, L. (2008). Bayesian compressive sensing. IEEE
    Transactions on signal processing, 56(6), 2346-2356.

    """

    meta = SolverMeta(
        acronym="BCS",
        full_name="Bayesian Compressive Sensing",
        category="Bayesian",
        description=(
            "Sparse Bayesian inverse method based on Bayesian compressive sensing, "
            "using hierarchical priors to promote sparse source estimates."
        ),
        references=[
            "Ji, S., Xue, Y., & Carin, L. (2008). Bayesian compressive sensing. IEEE Transactions on Signal Processing, 56(6), 2346–2356.",
        ],
    )

    def __init__(self, name="Bayesian Compressed Sensing", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, *args, alpha="auto", **kwargs):
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
        self.leadfield_norm = self.leadfield

        return self

    def apply_inverse_operator(
        self, mne_obj, max_iter=100, alpha_0=0.01, eps=1e-16
    ) -> mne.SourceEstimate:
        """Apply the inverse operator.

        Parameters
        ----------
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        max_iter : int
            Maximum number of iterations
        alpha_0 : float
            Regularization parameter
        eps : float
            Epsilon, used to avoid division by zero.

        Return
        ------
        stc : mne.SourceEstimate
            The SourceEstimate data structure containing the inverse solution.

        """
        data = self.unpack_data_obj(mne_obj)
        source_mat = self.calc_bcs_solution(
            data, max_iter=max_iter, alpha_0=alpha_0, eps=eps
        )
        stc = self.source_to_object(source_mat)
        return stc

    def calc_bcs_solution(self, y, max_iter=100, alpha_0=0.01, eps=1e-16):
        """This function computes the BCS inverse solution.

        Parameters
        ----------
        y : numpy.ndarray
            The M/EEG data matrix (n_channels, n_timepoints)
        max_iter : int
            Maximum number of iterations
        alpha_0 : float
            Regularization parameter
        eps : float
            Epsilon, used to avoid division by zero.

        Return
        ------
        x_hat : numpy.ndarray
            The source estimate.
        """

        alpha_0 = np.clip(alpha_0, a_min=1e-6, a_max=None)
        n_chans, _ = y.shape
        n_dipoles = self.leadfield_norm.shape[1]

        # preprocessing
        y -= y.mean(axis=0)

        alphas = np.ones(n_dipoles)
        D = np.diag(alphas)

        LLT = self.leadfield_norm.T @ self.leadfield_norm
        sigma = np.linalg.inv(alpha_0 * LLT + D)
        mu = alpha_0 * sigma @ self.leadfield_norm.T @ y
        proj_norm = self.leadfield_norm.T @ y
        proj = self.leadfield.T @ y

        residual_norms = [1e99]
        x_hats = []
        for _i in range(max_iter):
            gammas = np.array(
                [1 - alphas[ii] * sigma[ii, ii] for ii in range(n_dipoles)]
            )
            gammas[np.isnan(gammas)] = 0

            alphas = gammas / np.linalg.norm(mu**2, axis=1)
            alpha_0 = 1 / (
                np.linalg.norm(y - self.leadfield_norm @ mu) / (n_chans - gammas.sum())
            )
            D = np.diag(alphas) + eps
            sigma = np.linalg.inv(alpha_0 * LLT + D)
            mu = alpha_0 * sigma @ proj_norm

            Gamma = np.diag(gammas)
            x_hat = Gamma @ proj
            residual_norm = np.linalg.norm(y - self.leadfield @ x_hat)
            if residual_norm > residual_norms[-1]:
                x_hat = x_hats[-1]
                break
            residual_norms.append(residual_norm)
            x_hats.append(x_hat)

        return x_hat
