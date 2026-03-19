import logging

import mne
import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverNLChampagne(BaseSolver):
    """Class for the Noise Learning Champagne (Champagne-NL) inverse solution.

    References
    ----------
    [1] Cai, C., Hashemi, A., Diwakar, M., Haufe, S., Sekihara, K., & Nagarajan,
    S. S. (2021). Robust estimation of noise for electromagnetic brain imaging
    with the champagne algorithm. NeuroImage, 225, 117411.
    """

    meta = SolverMeta(
        acronym="Champagne-NL",
        full_name="Noise-Learning Champagne",
        category="Bayesian",
        description=(
            "Champagne sparse Bayesian learning variant that jointly estimates "
            "per-channel noise parameters/covariance while updating source variances."
        ),
        references=[
            "Wipf, D. P., Owen, J. P., Attias, H. T., Sekihara, K., & Nagarajan, S. S. (2010). Robust Bayesian estimation of the location, orientation, and time course of multiple correlated neural sources using MEG. NeuroImage, 49(1), 641–655.",
            "Cai, C., Hashemi, A., Diwakar, M., Haufe, S., Sekihara, K., & Nagarajan, S. S. (2021). Robust estimation of noise for electromagnetic brain imaging with the champagne algorithm. NeuroImage, 225, 117411.",
        ],
    )

    def __init__(self, name="Champagne-NL", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(
        self,
        forward,
        mne_obj=None,
        *args,
        alpha="auto",
        max_iter=1000,
        noise_cov: mne.Covariance | None = None,
        prune=True,
        pruning_thresh=1e-3,
        convergence_criterion=1e-8,
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
        max_iter : int
            Maximum number of iterations.
        noise_cov : [None, numpy.ndarray]
            The noise covariance matrix. Use "None" if not available.
        prune : bool
            If True, the algorithm sets small-activity dipoles to zero
            (pruning).
        pruning_thresh : float
            The threshold at which small gammas (dipole candidates) are set to
            zero.
        convergence_criterion : float
            Minimum change of loss function until convergence is assumed.

        Return
        ------
        self : object returns itself for convenience

        """
        super().make_inverse_operator(forward, mne_obj, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)

        # Use the standard whitening/projection pipeline if noise_cov is given.
        # On real MEG, the effective sensor rank can be far below n_channels;
        # projecting/whitening stabilizes NL-Champagne's noise learning.
        wf = self.prepare_whitened_forward(noise_cov)
        data_white = wf.sensor_transform @ np.asarray(data, dtype=float)

        original_leadfield = self.leadfield
        self.leadfield = wf.G_white
        try:
            inverse_operator_white = self.nl_champagne(
                data_white,
                alpha=0.01,
                max_iter=max_iter,
                prune=prune,
                pruning_thresh=pruning_thresh,
                convergence_criterion=convergence_criterion,
            )
        finally:
            self.leadfield = original_leadfield

        inverse_operator = inverse_operator_white @ wf.sensor_transform
        self.inverse_operators = [InverseOperator(inverse_operator, self.name)]
        return self

    def nl_champagne(
        self,
        Y,
        alpha=0.01,
        max_iter=1000,
        prune=True,
        pruning_thresh=1e-3,
        convergence_criterion=1e-8,
    ):
        """Noise Learning Champagne method.

        Parameters
        ----------
        Y : array, shape (n_sensors,)
            measurement vector, capturing sensor measurements
        alpha : float
            The regularization parameter.
        max_iter : int, optional
            The maximum number of inner loop iterations
        prune : bool
            If True, the algorithm sets small-activity dipoles to zero (pruning).
        pruning_thresh : float
            The threshold at which small gammas (dipole candidates) are set to
            zero.
        convergence_criterion : float
            Minimum change of loss function until convergence is assumed.
        Returns
        -------
        x : numpy.ndarray
            Parameter vector, e.g., source vector in the context of BSI (x in the cost
            function formula).

        """
        n_chans, n_dipoles = self.leadfield.shape
        _, n_times = Y.shape
        L = self.leadfield.copy()

        # Scaling of the data (necessary for convergence criterion and pruning
        # threshold)
        Y_scaled = Y.copy()
        Y_scaled /= abs(Y_scaled).mean() + 1e-12

        # Deterministic initialization improves stability on real data.
        gammas = np.ones(n_dipoles, dtype=float) * float(alpha)
        C_y = (Y_scaled @ Y_scaled.T) / max(float(n_times), 1.0)
        llambda = np.ones(n_chans, dtype=float) * float(np.trace(C_y) / (n_chans * 100))
        llambda = np.maximum(llambda, 1e-10)

        loss_list = []

        for i in range(max_iter):
            previous_gammas = gammas.copy()

            # Sigma_y = L @ diag(alpha) @ L.T + diag(llambda)
            # Using broadcasting: (L * alpha) @ L.T avoids d×d matrix
            Sigma_y = (L * gammas) @ L.T + np.diag(llambda)
            Sigma_y = 0.5 * (Sigma_y + Sigma_y.T)
            try:
                Sigma_y_inv = np.linalg.inv(Sigma_y)
            except np.linalg.LinAlgError:
                Sigma_y_inv = np.linalg.pinv(Sigma_y)
            Sigma_y_inv = np.asarray(Sigma_y_inv, dtype=float)
            Sigma_y_inv[~np.isfinite(Sigma_y_inv)] = 0.0

            # 1) Alpha update
            # s_bar = diag(alpha) @ L.T @ Sigma_y_inv @ Y_scaled
            SiL = Sigma_y_inv @ L  # (m, d), compute once
            SiY = Sigma_y_inv @ Y_scaled  # (m, t)
            s_bar = gammas[:, None] * (L.T @ SiY)  # (d, t)

            # z_hat = diag(L.T @ Sigma_y_inv @ L) via column-wise dot
            z_hat = np.sum(L * SiL, axis=0)  # (d,)
            C_s_bar = np.sum(s_bar**2, axis=1)  # (d,)

            z_hat = np.maximum(z_hat, 1e-20)
            C_s_bar = np.maximum(C_s_bar, 0.0)
            gammas = np.sqrt(C_s_bar / z_hat)
            gammas[~np.isfinite(gammas)] = 0.0
            gammas = np.maximum(gammas, 0.0)

            # 2) LLambda update
            Y_hat = L @ s_bar

            # Convex Bound update
            resid = np.sum((Y_scaled - Y_hat) ** 2, axis=1)
            resid = np.maximum(resid, 0.0)
            denom = np.maximum(np.diag(Sigma_y_inv), 1e-20)
            llambda = np.sqrt(resid / denom)
            llambda[~np.isfinite(llambda)] = 1.0
            llambda = np.maximum(llambda, 1e-10)

            # 3) Check convergence
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                sign, log_det_Sigma_y = np.linalg.slogdet(Sigma_y)
            if sign <= 0 or not np.isfinite(log_det_Sigma_y):
                log_det_Sigma_y = np.finfo(float).max / 2
            # tr(Sigma_y_inv @ Y @ Y.T) / n_times
            summation = float(np.sum(SiY * Y_scaled)) / n_times
            loss = log_det_Sigma_y + summation

            loss_list.append(loss)
            if self.verbose > 1:
                logger.debug(
                    f"iter {i}: loss {loss:.2f} ({log_det_Sigma_y:.2f} + {summation / n_times:.2f})"
                )

            if (
                loss == float("-inf")
                or loss == float("inf")
                or np.linalg.norm(gammas) == 0
            ):
                gammas = previous_gammas
                break

            # Check convergence only after first iteration
            if len(loss_list) > 1:
                change = abs(1 - (loss_list[-1] / loss_list[-2]))
                if change < convergence_criterion:
                    if self.verbose > 0:
                        logger.info("Converged!")
                    break

            if prune:
                prune_candidates = gammas < (pruning_thresh * gammas.max())
                gammas[prune_candidates] = 0

        # Final inverse: diag(alpha) @ L.T @ Sigma_y_inv
        Sigma_y = (L * gammas) @ L.T + np.diag(llambda)
        Sigma_y = 0.5 * (Sigma_y + Sigma_y.T)
        try:
            Sigma_y_inv = np.linalg.inv(Sigma_y)
        except np.linalg.LinAlgError:
            Sigma_y_inv = np.linalg.pinv(Sigma_y)
        inverse_operator = gammas[:, None] * (L.T @ Sigma_y_inv)

        return inverse_operator
