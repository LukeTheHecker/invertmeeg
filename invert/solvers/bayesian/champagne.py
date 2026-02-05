import logging
from copy import deepcopy

import numpy as np
from scipy.sparse import diags

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)
EPSILON = 1e-10


class SolverChampagne(BaseSolver):
    """Class for Champagne inverse solution (MacKay, Convexity Bound, LowSNR).

    References
    ----------
    [1] Cai, C., Kang, H., Hashemi, A., Chen, D., Diwakar, M., Haufe, S., ... &
    Nagarajan, S. S. (2023). Bayesian algorithms for joint estimation of brain
    activity and noise in electromagnetic imaging. IEEE Transactions on Medical
    Imaging.
    """

    meta = SolverMeta(
        acronym="Champagne",
        full_name="Champagne (Sparse Bayesian Learning)",
        category="Bayesian",
        description=(
            "Sparse Bayesian learning method for M/EEG/EEG source imaging using "
            "Type-II maximum likelihood / evidence maximization updates. Supports "
            "multiple update rules and optional noise learning variants."
        ),
        references=[
            "Wipf, D., & Nagarajan, S. (2009). A unified Bayesian framework for MEG/EEG source imaging. NeuroImage, 44(3), 947–966.",
            "Owen, J. P., Wipf, D. P., Attias, H. T., Sekihara, K., & Nagarajan, S. S. (2012). Performance evaluation of the Champagne source reconstruction algorithm on simulated and real M/EEG data. NeuroImage, 60(1), 305–323.",
            "Cai, C., Kang, H., Hashemi, A., Chen, D., Diwakar, M., Haufe, S., Sekihara, K., Wu, W., & Nagarajan, S. S. (2023). Bayesian algorithms for joint estimation of brain activity and noise in electromagnetic imaging. IEEE Transactions on Medical Imaging, 42(3), 762–773.",
        ],
    )

    def __init__(
        self,
        name="Champagne",
        update_rule="MacKay",
        beta_init=0.5,
        beta_lr=0.01,
        theta=0.01,
        noise_learning="diagonal",
        noise_learning_mode="fixed",
        **kwargs,
    ):
        """
        Parameters
        ----------
        update_rule : str
            Either of: "MacKay", "Convexity", "MM", "LowSNR", "EM", "AR-EM", "TEM"
        beta_init : float
            Initial AR(1) coefficient for AR-EM update rule (default: 0.5)
        beta_lr : float
            Learning rate for beta optimization in AR-EM (default: 0.01)
        theta : float
            Regularization parameter for TEM update rule (default: 0.01)
        noise_learning : str
            Noise learning strategy: "fixed", "learn", "FUN", "HSChampagne", "NLChampagne"
            - "fixed": Use provided noise covariance (standard Champagne)
            - "learn": Learn noise with specified mode
            - "FUN": Alias for noise_learning_mode="full"
            - "HSChampagne": Alias for noise_learning_mode="homoscedastic"
            - "NLChampagne": Alias for noise_learning_mode="precision"
        noise_learning_mode : str
            How to parameterize learned noise: "diagonal", "homoscedastic", "full", "precision"
            - "homoscedastic": Learn single scalar variance (like HSChampagne)
            - "diagonal": Learn diagonal elements independently (like NLChampagne diagonal mode)
            - "full": Learn full covariance matrix (like FUN)
            - "precision": Use precision-based updates (like NLChampagne)
        """
        self.name = update_rule + " " + name
        self.update_rule = update_rule
        self.beta_init = beta_init
        self.beta_lr = beta_lr
        self.theta = theta

        # Handle noise learning aliases
        if noise_learning.lower() == "fun":
            self.noise_learning = "learn"
            self.noise_learning_mode = "full"
        elif noise_learning.lower() == "hschampagne":
            self.noise_learning = "learn"
            self.noise_learning_mode = "homoscedastic"
        elif noise_learning.lower() == "nlchampagne":
            self.noise_learning = "learn"
            self.noise_learning_mode = "precision"
        else:
            self.noise_learning = noise_learning
            self.noise_learning_mode = noise_learning_mode

        if self.noise_learning == "learn":
            self.name = f"{update_rule} Champagne ({self.noise_learning_mode} noise)"

        return super().__init__(**kwargs)

    def make_inverse_operator(
        self,
        forward,
        mne_obj,
        *args,
        alpha="auto",
        max_iter=2000,
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

        # Store attributes
        n_chans = self.leadfield.shape[0]
        if noise_cov is None:
            noise_cov = np.identity(n_chans)
        self.noise_cov = noise_cov

        self.max_iter = max_iter
        self.prune = prune
        self.pruning_thresh = pruning_thresh
        self.convergence_criterion = convergence_criterion
        # Historically this solver used an additional alpha scaling factor to bring values
        # into a workable range. With consistent reference scaling (data covariance) this
        # should not be necessary.
        self.alpha_scaler = 1.0

        data = self.unpack_data_obj(mne_obj)

        data_cov = self.data_covariance(data, center=True, ddof=1)
        self.get_alphas(reference=data_cov)
        inverse_operators = []
        for alpha in self.alphas:
            inverse_operator = self.make_champagne(
                data, float(alpha) * float(self.alpha_scaler), pruning_thresh=pruning_thresh
            )
            inverse_operators.append(inverse_operator)
        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        return self

    def make_champagne(self, Y, alpha, pruning_thresh=1e-3):
        """Majority Maximization Champagne method.

        Parameters
        ----------
        Y : array, shape (n_sensors,)
            measurement vector, capturing sensor measurements
        alpha : float
            The regularization parameter.

        Returns
        -------
        x : numpy.ndarray
            Parameter vector, e.g., source vector in the context of BSI (x in the cost
            function formula).
        """
        n_chans, n_dipoles = self.leadfield.shape
        _, n_times = Y.shape
        L = deepcopy(self.leadfield)
        L /= np.linalg.norm(L, axis=0)

        # re-reference data for noise learning modes (FUN/HSChampagne requirement)
        if self.noise_learning == "learn":
            Y = Y - Y.mean(axis=0)

        # Scaling of the data (necessary for convergence criterion and pruning threshold)
        Y_scaled = deepcopy(Y)

        scaler = 1.0  # Keep track of scaler even if not used
        # scaler = abs(Y_scaled).mean()
        # Y_scaled /= scaler

        I = np.identity(n_chans)
        gammas = np.ones(n_dipoles)
        Gamma = diags(gammas, 0)

        # Initialize noise covariance based on learning mode
        base_noise_cov = getattr(self, "noise_cov", None)
        if base_noise_cov is None:
            base_noise_cov = I
        base_noise_cov = np.asarray(base_noise_cov, dtype=float)
        if base_noise_cov.shape != (n_chans, n_chans):
            base_noise_cov = I
        base_noise_cov = 0.5 * (base_noise_cov + base_noise_cov.T)

        if self.noise_learning == "learn":
            if self.noise_learning_mode in {"diagonal", "precision"}:
                base_noise_cov = np.diag(np.diag(base_noise_cov))
            elif self.noise_learning_mode == "homoscedastic":
                base_noise_cov = I
        noise_cov = float(alpha) * base_noise_cov

        Sigma_y = noise_cov + L @ Gamma @ L.T
        Sigma_y_inv = self.robust_inverse(Sigma_y)
        mu_x = Gamma @ L.T @ Sigma_y_inv @ Y_scaled

        # Initialize loss list with None to compute first loss properly
        loss_list = []
        active_set = np.arange(n_dipoles)

        # Initialize AR-EM specific variables if needed
        if self.update_rule.lower() == "ar-em":
            beta = np.clip(self.beta_init, 0, 0.99)

            # Helper function for AR(1) covariance
            def make_ar1_covariance(beta, n_times):
                if beta == 0:
                    return np.identity(n_times)
                indices = np.arange(n_times)
                B = beta ** np.abs(indices[:, None] - indices[None, :])
                B = B / (1 - beta**2)
                return B

            B = make_ar1_covariance(beta, n_times)

        # Initialize TEM specific variables if needed
        if self.update_rule.lower() == "tem":
            It = np.identity(n_times)
            # Initialize B matrix as identity
            B_hat = (
                np.stack(
                    [
                        (mu_x[nn, np.newaxis].T * mu_x[nn, np.newaxis]) / gammas[nn]
                        for nn in range(n_dipoles)
                    ],
                    axis=0,
                ).sum(axis=0)
                + self.theta * It
            )
            B = B_hat / self._frob(B_hat)

        for i_iter in range(self.max_iter):
            old_gammas = deepcopy(gammas)

            if self.update_rule.lower() == "em":
                # EM update: variance + mean-square (optimized)
                # Compute diagonal of Sigma_x efficiently (avoid full matrix)
                L_Sigma = Sigma_y_inv @ L  # (n_chans, n_dipoles)
                z_diag = np.sum(L * L_Sigma, axis=0)  # (n_dipoles,)
                diag_Sigma_x = gammas - gammas**2 * z_diag
                gammas = diag_Sigma_x + np.mean(mu_x**2, axis=1)

            elif self.update_rule.lower() == "mackay":
                # MacKay update (optimized with vectorized z_diag computation)
                upper_term = np.mean(mu_x**2, axis=1)
                # Vectorized: z_diag = diag(L.T @ Sigma_y_inv @ L)
                L_Sigma = Sigma_y_inv @ L  # (n_chans, n_dipoles)
                z_diag = np.sum(L * L_Sigma, axis=0)  # (n_dipoles,)
                lower_term = gammas * z_diag
                gammas = upper_term / lower_term

            elif (
                self.update_rule.lower() == "convexity"
                or self.update_rule.lower() == "mm"
            ):
                # Convexity/MM update: sqrt(mean(mu_x²) / z_n) (optimized)
                # Note: MM-Champagne is mathematically equivalent to Convexity bound
                upper_term = np.mean(mu_x**2, axis=1)
                # Vectorized: z_diag = diag(L.T @ Sigma_y_inv @ L)
                L_Sigma = Sigma_y_inv @ L  # (n_chans, n_dipoles)
                z_diag = np.sum(L * L_Sigma, axis=0)  # (n_dipoles,)
                gammas = np.sqrt(upper_term / z_diag)

            elif self.update_rule.lower() == "lowsnr":
                upper_term = np.mean(mu_x**2, axis=1)
                lower_term = np.sum(L**2, axis=0)
                # gammas = np.sqrt(upper_term / lower_term)
                gammas = np.sqrt(upper_term) / np.sqrt(lower_term)

            elif self.update_rule.lower() == "ar-em":
                # AR-EM update: Mahalanobis norm with AR(1) covariance (optimized)
                # Compute diagonal of Sigma_x efficiently (avoid full matrix)
                L_Sigma = Sigma_y_inv @ L  # (n_chans, n_dipoles)
                z_diag = np.sum(L * L_Sigma, axis=0)  # (n_dipoles,)
                diag_Sigma_x = gammas - gammas**2 * z_diag

                try:
                    B_inv = np.linalg.inv(B)
                except np.linalg.LinAlgError:
                    B_inv = np.linalg.pinv(B)

                # Vectorized gamma update: compute mu_x @ B_inv @ mu_x.T efficiently
                # Result is (n_dipoles,) array where each element is mu_x[n] @ B_inv @ mu_x[n].T
                mu_x_B_inv = mu_x @ B_inv  # (n_dipoles, n_times)
                mahalanobis_terms = np.sum(mu_x * mu_x_B_inv, axis=1)  # (n_dipoles,)
                gammas = diag_Sigma_x + mahalanobis_terms

                # Update beta based on autocorrelation (vectorized)
                # Only consider active dipoles (above pruning threshold)
                active_mask = gammas > self.pruning_thresh
                if np.any(active_mask):
                    mu_x_active = mu_x[active_mask]  # (n_active, n_times)
                    # Autocorrelation: sum over active sources of x(t) * x(t+1)
                    autocorr_sum = np.sum(mu_x_active[:, :-1] * mu_x_active[:, 1:])
                    norm_sum = np.sum(mu_x_active[:, :-1] ** 2)

                    if norm_sum > 1e-10:
                        beta_gradient = autocorr_sum / norm_sum
                        beta = beta + self.beta_lr * (beta_gradient - beta)
                        beta = np.clip(beta, 0, 0.99)
                        B = make_ar1_covariance(beta, n_times)

            elif self.update_rule.lower() == "tem":
                # TEM update: learns full temporal covariance matrix B (optimized)
                # Compute diagonal of Sigma_x efficiently (avoid full matrix)
                L_Sigma = Sigma_y_inv @ L  # (n_chans, n_dipoles)
                z_diag = np.sum(L * L_Sigma, axis=0)  # (n_dipoles,)
                diag_Sigma_x = gammas - gammas**2 * z_diag

                try:
                    B_inv = np.linalg.inv(B)
                except np.linalg.LinAlgError:
                    B_inv = np.linalg.pinv(B)

                # Vectorized gamma update: compute mu_x @ B_inv @ mu_x.T efficiently
                mu_x_B_inv = mu_x @ B_inv  # (n_dipoles, n_times)
                mahalanobis_terms = np.sum(mu_x * mu_x_B_inv, axis=1)  # (n_dipoles,)
                gammas = diag_Sigma_x + mahalanobis_terms

                # Update B matrix based on source estimates (vectorized)
                # B_hat = sum_n [mu_x[n].T @ mu_x[n] / gamma[n]] + theta * I
                # Vectorized: (mu_x.T / gammas) @ mu_x
                mu_x_scaled = mu_x / (
                    gammas[:, np.newaxis] + 1e-10
                )  # (n_dipoles, n_times)
                B_hat = mu_x_scaled.T @ mu_x + self.theta * It  # (n_times, n_times)
                B = B_hat / self._frob(B_hat)

            elif self.update_rule.lower() == "adaptive":
                # ai-composed update rule (optimized)
                upper_term = np.mean(mu_x**2, axis=1)
                L_Sigma = Sigma_y_inv @ L  # (n_chans, n_dipoles)
                lower_term = np.sum(L * L_Sigma, axis=0)  # (n_dipoles,)
                snr_estimate = upper_term / np.mean(np.diag(noise_cov))

                # Adaptive exponent based on estimated SNR
                adaptive_exponent = 0.5 + 0.5 / (1 + np.exp(-snr_estimate + 5))

                # Combine aspects of MacKay, Convexity, and LowSNR rules
                gammas = (upper_term / lower_term) ** adaptive_exponent
            elif self.update_rule.lower() == "dynamic_adaptive":
                # Dynamic adaptive update rule (optimized)
                upper_term = np.mean(mu_x**2, axis=1)
                L_Sigma = Sigma_y_inv @ L  # (n_chans, n_dipoles)
                lower_term = np.sum(L * L_Sigma, axis=0)  # (n_dipoles,)
                snr_estimate = upper_term / np.mean(np.diag(noise_cov))

                # Dynamic scaling factor based on iteration number and SNR
                iteration_factor = 1 - np.exp(
                    -i_iter / 10
                )  # Assumes 'i' is the current iteration number
                snr_factor = 1 / (1 + np.exp(-snr_estimate + 5))

                # Combine MacKay and Convexity rules with dynamic weighting
                mackay_update = upper_term / (gammas * lower_term)
                convexity_update = np.sqrt(upper_term / lower_term)

                # Apply dynamic weighting
                weighted_update = (
                    snr_factor * mackay_update + (1 - snr_factor) * convexity_update
                ) ** iteration_factor

                # Apply adaptive smoothing
                smoothing_factor = 0.1 * (1 - iteration_factor)
                gammas = (
                    1 - smoothing_factor
                ) * weighted_update + smoothing_factor * gammas

                # Apply soft thresholding for sparsity
                # threshold = np.percentile(gammas, 10)  # Adjust percentile as needed
                # gammas = np.maximum(gammas - threshold, 0)

            # Remove nans
            gammas[np.isnan(gammas)] = 0

            # Stop if gammas went to zero
            if np.linalg.norm(gammas) == 0:
                # print("breaking")
                gammas = old_gammas
                break

            if self.prune:
                # # Use relative threshold
                active_set_idc = np.where(gammas > (pruning_thresh * gammas.max()))[0]
                # print(f"Gammas: Max {gammas.max()}, Min {gammas.min()}, Mean {gammas.mean()}")
                # gammas_minmax_scaled = (gammas - gammas.min()) / (gammas.max() - gammas.min())
                # print(f"Gammas: Max {gammas_minmax_scaled.max()}, Min {gammas_minmax_scaled.min()}, Mean {gammas_minmax_scaled.mean()}")
                # active_set_idc = np.where(gammas_minmax_scaled>(pruning_thresh))[0]

                if len(active_set_idc) == 0:
                    # print("pruned too much")
                    gammas = old_gammas
                    break
                active_set = active_set[active_set_idc]
                # print(f"New set: {len(active_set)}")
                gammas = gammas[active_set_idc]
                L = L[:, active_set_idc]

            # Update noise covariance if learning is enabled
            if self.noise_learning == "learn":
                # Reconstruct full gamma vector for noise update
                gammas_full = np.zeros(n_dipoles)
                gammas_full[active_set] = gammas
                Gamma_full = diags(gammas_full, 0)
                L_full = deepcopy(self.leadfield)
                mu_x_full = Gamma_full @ L_full.T @ Sigma_y_inv @ Y_scaled
                noise_cov = self._update_noise_covariance(
                    Y_scaled, L_full, mu_x_full, gammas_full, noise_cov, n_times, scaler
                )

            # update rest
            Gamma = diags(gammas, 0)
            Sigma_y = noise_cov + L @ Gamma @ L.T
            Sigma_y_inv = self.robust_inverse(Sigma_y)
            # Sigma_x = Gamma - Gamma @ L.T @ Sigma_y_inv @ L @ Gamma
            mu_x = Gamma @ L.T @ Sigma_y_inv @ Y_scaled

            # Correct Champagne loss function: negative log-likelihood
            # loss = data_fit_term + log_det_term
            # Data fit: tr(Sigma_y_inv @ Y @ Y.T) / n_times
            # Model complexity: log(det(Sigma_y))
            data_fit = np.trace(Sigma_y_inv @ Y_scaled @ Y_scaled.T) / n_times
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                sign, log_det = np.linalg.slogdet(Sigma_y)
            if sign <= 0:
                log_det = -np.inf
            loss = data_fit + log_det

            # Compute the residuals
            loss_list.append(loss)

            # Check convergence only after first iteration
            if len(loss_list) > 1:
                relative_change = (loss_list[-2] - loss) / abs(loss_list[-2])

                if self.verbose > 1:
                    logger.debug(
                        f"Iteration {i_iter}: loss = {loss:.6f}, relative change = {relative_change:.6f}, Active set size = {len(active_set)}"
                    )

                # Only converge if loss is decreasing (positive relative change) and change is small
                if relative_change > 0 and relative_change < self.convergence_criterion:
                    if self.verbose > 0:
                        logger.info(
                            f"Converged because {relative_change:.6f} < {self.convergence_criterion:.6f}"
                        )
                    break
            else:
                # First iteration - just print the loss if verbose
                if self.verbose > 1:
                    logger.debug(
                        f"Iteration {i_iter}: loss = {loss:.6f}, Active set size = {len(active_set)}"
                    )

        # Final inverse operator construction
        L = deepcopy(self.leadfield)
        gammas_final = np.zeros(n_dipoles)
        gammas_final[active_set] = gammas
        Gamma = diags(gammas_final, 0)

        # Scale noise covariance back if learning was enabled
        if self.noise_learning == "learn":
            noise_cov_final = scaler * noise_cov
        else:
            noise_cov_final = noise_cov

        Sigma_y = noise_cov_final + L @ Gamma @ L.T
        Sigma_y_inv = self.robust_inverse(Sigma_y)
        inverse_operator = Gamma @ L.T @ Sigma_y_inv

        # Store learned noise covariance
        if self.noise_learning == "learn":
            self.learned_noise_cov = noise_cov_final
            if self.verbose > 0:
                if self.noise_learning_mode == "homoscedastic":
                    logger.info(f"Learned noise variance: {noise_cov_final[0, 0]:.6f}")
                elif self.noise_learning_mode == "diagonal":
                    logger.info(
                        f"Learned noise variance range: [{np.diag(noise_cov_final).min():.6f}, {np.diag(noise_cov_final).max():.6f}]"
                    )

        # Store learned beta for AR-EM
        if self.update_rule.lower() == "ar-em":
            self.learned_beta = beta
            if self.verbose > 0:
                logger.info(f"Learned AR(1) coefficient: {beta:.4f}")

        # This is how the final source estimate could be calculated:
        # mu_x = inverse_operator @ Y

        return inverse_operator

    @staticmethod
    def _frob(x):
        """Frobenius norm helper for TEM update rule"""
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        return np.sqrt(np.trace(x @ x.T))

    def _initialize_noise_covariance(self, Y_scaled, L, n_chans, n_times):
        """Initialize noise covariance based on learning mode.

        Parameters
        ----------
        Y_scaled : array, shape (n_chans, n_times)
            Scaled measurement data
        L : array, shape (n_chans, n_dipoles)
            Lead field matrix
        n_chans : int
            Number of channels
        n_times : int
            Number of time points

        Returns
        -------
        noise_cov : array, shape (n_chans, n_chans) or (n_chans,)
            Initial noise covariance estimate
        """
        if self.noise_learning_mode == "homoscedastic":
            # HSChampagne: scalar noise variance (identity scaling)
            # Initialize with small random value
            return np.identity(n_chans) * 0.01

        elif self.noise_learning_mode == "full":
            # FUN: full covariance matrix
            # Initialize with random positive definite matrix
            A = np.random.rand(n_chans, n_times)
            A = (A @ A.T) / n_times
            return A

        elif self.noise_learning_mode in ["diagonal", "precision"]:
            # NLChampagne: diagonal elements
            # Initialize with MNE-based estimate
            lin_lstq = np.linalg.pinv(L) @ Y_scaled
            residuals = Y_scaled - L @ lin_lstq
            diag_noise = np.mean(residuals**2, axis=1) + 1e-6
            return np.diag(diag_noise)

        else:
            # Default: identity
            return np.identity(n_chans) * 0.01

    def _update_noise_covariance(
        self, Y_scaled, L, mu_x, gammas, current_noise_cov, n_times, scaler
    ):
        """Update noise covariance based on learning mode.

        Parameters
        ----------
        Y_scaled : array, shape (n_chans, n_times)
            Scaled measurement data
        L : array, shape (n_chans, n_dipoles)
            Lead field matrix
        mu_x : array, shape (n_dipoles, n_times)
            Current source estimates
        gammas : array, shape (n_dipoles,)
            Current source variances
        current_noise_cov : array
            Current noise covariance estimate
        n_times : int
            Number of time points
        scaler : float
            Data scaling factor

        Returns
        -------
        noise_cov : array
            Updated noise covariance estimate
        """
        n_chans = L.shape[0]
        n_dipoles = L.shape[1]

        # Compute residuals
        residuals = Y_scaled - L @ mu_x

        if self.noise_learning_mode == "homoscedastic":
            # HSChampagne: scalar variance update (CORRECTED from original)
            # Original had double squaring error: np.sum(...**2)**2
            residual_power = np.sum(residuals**2) / n_times

            # Compute approximate degrees of freedom correction
            # This is a simplified version of the original complex calculation
            Sigma_y = current_noise_cov + L @ np.diag(gammas) @ L.T
            Sigma_y_inv = self.robust_inverse(Sigma_y)
            Sigma_y_inv_L = Sigma_y_inv @ L
            # Approximate posterior variance diagonal
            Sigma_X_diag = gammas * (1 - gammas * np.diag(L.T @ Sigma_y_inv_L))
            dof_correction = (
                n_chans - n_dipoles + np.sum(Sigma_X_diag / (gammas + 1e-10))
            )

            scalar_noise = residual_power / (dof_correction + 1e-10)
            return np.identity(n_chans) * scalar_noise

        elif self.noise_learning_mode == "full":
            # FUN: full covariance matrix update
            M_noise = (residuals @ residuals.T) / n_times
            C_noise = self.robust_inverse(current_noise_cov + 1e-8 * np.eye(n_chans))

            # Use corrected FUN learning (diagonal mode for stability)
            updated_noise = self._fun_learning_cov_est(
                C_noise, M_noise, update_mode="diagonal"
            )
            return updated_noise

        elif self.noise_learning_mode == "diagonal":
            # NLChampagne diagonal mode: learn each diagonal element independently
            diag_residual_power = np.sum(residuals**2, axis=1)

            # Compute current Sigma_y_inv diagonal for normalization
            Sigma_y = current_noise_cov + L @ np.diag(gammas) @ L.T
            Sigma_y_inv = self.robust_inverse(Sigma_y)
            normalization = np.diag(Sigma_y_inv) + 1e-10

            # Convex bound update from NLChampagne
            diag_noise = np.sqrt(diag_residual_power / (n_times * normalization))
            return np.diag(diag_noise)

        elif self.noise_learning_mode == "precision":
            # NLChampagne precision-based update
            # Update both source precision (Alpha) and noise precision (Lambda)
            # This uses the dual parameterization from NLChampagne

            # For simplicity, use diagonal update similar to diagonal mode
            # but with precision-based interpretation
            diag_residual_power = np.sum(residuals**2, axis=1)

            Sigma_y = current_noise_cov + L @ np.diag(gammas) @ L.T
            Sigma_y_inv = self.robust_inverse(Sigma_y)
            precision_update = np.sqrt(
                diag_residual_power / (n_times * np.diag(Sigma_y_inv) + 1e-10)
            )

            return np.diag(precision_update)

        else:
            # No update, return current
            return current_noise_cov

    @staticmethod
    def _fun_learning_cov_est(C, M, update_mode="diagonal"):
        """FUN covariance learning update (CORRECTED version).

        This is a corrected implementation of the FUN learning algorithm.
        Original had bugs in eigenvalue decomposition.

        Parameters
        ----------
        C : array, shape (n, n)
            Precision matrix (inverse covariance)
        M : array, shape (n, n)
            Sample covariance matrix
        update_mode : str
            Either "diagonal" or "geodesic"

        Returns
        -------
        S : array, shape (n, n)
            Updated covariance estimate
        """
        if update_mode == "diagonal":
            # Simple diagonal update
            h = np.diag(C)
            g = np.diag(M)
            p = np.sqrt(g / (h + 1e-10))
            S = np.diag(p)

        elif update_mode == "geodesic":
            # CORRECTED geodesic update (fixed eigenvalue decomposition)
            eps_default = 1e-8

            # Proper eigenvalue decomposition (FIXED)
            eigenvals, eigenvecs = np.linalg.eig(C)
            eigenvals = np.real(eigenvals)
            eigenvals_sqrt = np.sqrt(np.maximum(eigenvals, eps_default))

            # Build inverse square root of C
            inv_sqrt_eigenvals = np.zeros_like(eigenvals)
            valid_idx = eigenvals_sqrt >= eps_default
            inv_sqrt_eigenvals[valid_idx] = 1.0 / eigenvals_sqrt[valid_idx]

            # Reconstruct matrices
            sqrt_C = eigenvecs @ np.diag(eigenvals_sqrt) @ eigenvecs.T
            inv_sqrt_C = eigenvecs @ np.diag(inv_sqrt_eigenvals) @ eigenvecs.T

            # Inner eigenvalue decomposition
            inner_mat = inv_sqrt_C @ M @ inv_sqrt_C
            inner_eigenvals, inner_eigenvecs = np.linalg.eig(inner_mat)
            inner_eigenvals_sqrt = np.sqrt(np.maximum(np.real(inner_eigenvals), 0))
            A = inner_eigenvecs @ np.diag(inner_eigenvals_sqrt) @ inner_eigenvecs.T

            # Final result
            S = sqrt_C @ A @ sqrt_C

        else:
            raise ValueError(
                f"update_mode {update_mode} unknown. Use 'diagonal' or 'geodesic'."
            )

        return np.real(S)
