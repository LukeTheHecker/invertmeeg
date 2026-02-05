import logging
from copy import deepcopy

import numpy as np
from scipy.sparse import csr_matrix

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
            "Cai, C., Hashemi, A., Diwakar, M., Haufe, S., Sekihara, K., & Nagarajan, S. S. (2021). Robust estimation of noise for electromagnetic brain imaging with the champagne algorithm. NeuroImage, 225, 117411.",
        ],
    )

    def __init__(self, name="Champagne-NL", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(
        self,
        forward,
        mne_obj,
        *args,
        alpha="auto",
        max_iter=1000,
        noise_cov=None,
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
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)

        n_chans = self.leadfield.shape[0]
        if noise_cov is None:
            noise_cov = np.identity(n_chans)
        self.noise_cov = noise_cov

        inverse_operator = self.nl_champagne(
            data,
            alpha=0.01,
            max_iter=max_iter,
            prune=prune,
            pruning_thresh=pruning_thresh,
            convergence_criterion=convergence_criterion,
        )
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
        L = deepcopy(self.leadfield)

        # re-reference data
        # Y -= Y.mean(axis=0)

        # Scaling of the data (necessary for convergence criterion and pruning
        # threshold)
        Y_scaled = deepcopy(Y)
        Y_scaled /= abs(Y_scaled).mean()

        np.identity(n_chans)

        alpha = np.random.rand(n_dipoles)
        Alpha = csr_matrix(np.diag(alpha))

        llambda = np.random.rand(n_chans)
        LLambda = csr_matrix(np.diag(llambda))

        # Sigma_y = L @ Alpha @ L.T + LLambda
        # Sigma_y_inv = np.linalg.inv(Sigma_y)

        # Sigma_x = Gamma - Gamma @ L.T @ Sigma_y_inv @ L @ Gamma
        # z_0 = L.T @ Sigma_y_inv @ L
        # mu_x = Gamma @ L.T @ Sigma_y_inv @ Y_scaled

        loss_list = []

        for i in range(max_iter):
            previous_Alpha = deepcopy(Alpha)
            Sigma_y = L @ Alpha @ L.T + LLambda
            Sigma_y_inv = np.linalg.inv(Sigma_y)

            # 1) Alpha (formerly Gamma) update
            s_bar = np.squeeze(np.asarray(Alpha @ L.T @ Sigma_y_inv @ Y_scaled))
            z_hat = np.einsum("ij,ji->i", L.T @ Sigma_y_inv, L)
            C_s_bar = np.sum(s_bar**2, axis=1)

            alpha = np.sqrt(C_s_bar / z_hat)
            Alpha = csr_matrix(np.diag(alpha))

            # 2) LLambda update
            Y_hat = L @ s_bar

            ## Convex Bound update
            llambda = np.sqrt(
                np.sum((Y_scaled - Y_hat) ** 2, axis=1) / np.diag(Sigma_y_inv)
            )
            LLambda = csr_matrix(np.diag(llambda))

            # 3) Check convergence
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                sign, log_det_Sigma_y = np.linalg.slogdet(Sigma_y)
            if sign <= 0:
                log_det_Sigma_y = -np.inf
            summation = (1 / n_times) * np.sum(
                np.einsum("ti,ij,tj->t", Y_scaled.T, Sigma_y_inv, Y_scaled.T)
            )
            loss = log_det_Sigma_y + summation

            loss_list.append(loss)
            if self.verbose > 1:
                logger.debug(
                    f"iter {i}: loss {loss:.2f} ({log_det_Sigma_y:.2f} + {(1 / n_times) * summation:.2f})"
                )

            if (
                loss == float("-inf")
                or loss == float("inf")
                or np.linalg.norm(alpha) == 0
            ):
                Alpha = previous_Alpha
                break

            # Check convergence only after first iteration
            if len(loss_list) > 1:
                change = abs(1 - (loss_list[-1] / loss_list[-2]))
                if change < convergence_criterion:
                    if self.verbose > 0:
                        logger.info("Converged!")
                    break

            if prune:
                prune_candidates = alpha < (pruning_thresh * alpha.max())
                alpha[prune_candidates] = 0
                Alpha = csr_matrix(np.diag(alpha))

        # update rest
        Sigma_y = L @ Alpha @ L.T + LLambda
        Sigma_y_inv = np.linalg.inv(Sigma_y)
        inverse_operator = np.asarray(Alpha @ L.T @ Sigma_y_inv)

        return inverse_operator
