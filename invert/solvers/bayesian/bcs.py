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

    def make_inverse_operator(
        self,
        forward,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
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
        wf = self.prepare_whitened_forward(noise_cov)
        self.leadfield = wf.G_white

        # Column-normalize for numerically stable SBL iteration
        col_norms = np.linalg.norm(wf.G_white, axis=0)
        col_norms = np.maximum(col_norms, 1e-15)
        self.leadfield_norm = wf.G_white / col_norms
        self._col_norms = col_norms

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
        self.validate_operator_data_compatibility(data)
        data = self._sensor_transform @ data
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
        n_chans, n_times = y.shape
        L = self.leadfield_norm
        n_dipoles = L.shape[1]

        # preprocessing
        y -= y.mean(axis=0)

        alphas = np.ones(n_dipoles)
        D = np.diag(alphas)

        LLT = L.T @ L
        sigma = np.linalg.inv(alpha_0 * LLT + D)
        mu = alpha_0 * sigma @ L.T @ y

        residual_norms = [1e99]
        mus = []
        for _i in range(max_iter):
            gammas = np.array(
                [1 - alphas[ii] * sigma[ii, ii] for ii in range(n_dipoles)]
            )
            gammas[np.isnan(gammas)] = 0

            mu_sq_norm = np.sum(mu**2, axis=1)
            alphas = gammas / np.maximum(mu_sq_norm, eps)
            # Noise precision: (N - sum(gamma)) / ||residual||^2
            # Use squared Frobenius norm averaged over timepoints (Eq. 11 of [1]).
            residual = y - L @ mu
            residual_sq = np.sum(residual**2) / n_times
            denom = max(residual_sq, eps)
            alpha_0 = max((n_chans - gammas.sum()), eps) / denom
            D = np.diag(alphas) + eps
            sigma = np.linalg.inv(alpha_0 * LLT + D)
            mu = alpha_0 * sigma @ L.T @ y

            # Convergence check: monitor residual in normalized space
            residual_norm = np.linalg.norm(y - L @ mu)
            if residual_norm > residual_norms[-1]:
                mu = mus[-1]
                break
            residual_norms.append(residual_norm)
            mus.append(mu.copy())

        # Convert from normalized source space to original amplitudes
        return mu / self._col_norms[:, np.newaxis]
