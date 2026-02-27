import logging

import mne
import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta
from .sbl_core import _cholesky_inv, sbl_iterate

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
        noise_learning="fixed",
        noise_learning_mode="fixed",
        rank_tol: float = 1e-12,
        eps: float = 1e-15,
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
        self.rank_tol = float(rank_tol)
        self.eps = float(eps)

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
        mne_obj=None,
        *args,
        alpha="auto",
        max_iter=2000,
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
        noise_cov : [None, mne.Covariance]
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

        n_chans = self.leadfield.shape[0]
        if noise_cov is None:
            noise_cov = self.make_identity_noise_cov(list(self.forward.ch_names))
        noise_cov, noise_cov_ch_names = self.coerce_noise_cov(noise_cov)
        forward_ch_names = list(self.forward.ch_names)
        if noise_cov_ch_names != forward_ch_names:
            noise_cov = self.reorder_covariance_to_channels(
                noise_cov, noise_cov_ch_names, forward_ch_names
            )
        if noise_cov.shape != (n_chans, n_chans):
            msg = (
                f"noise_cov has shape {noise_cov.shape}, expected {(n_chans, n_chans)}"
            )
            raise ValueError(msg)
        noise_cov = 0.5 * (noise_cov + noise_cov.T)

        # SSP projection via standard pipeline (no whitening — Champagne
        # handles noise_cov internally in its EM updates).
        wf = self.prepare_whitened_forward(None)
        self._sensor_projector = wf.projector

        self.max_iter = max_iter
        self.prune = prune
        self.pruning_thresh = pruning_thresh
        self.convergence_criterion = convergence_criterion
        self.alpha_scaler = 1.0

        data = self.unpack_data_obj(mne_obj)
        data_projected = wf.sensor_transform @ np.asarray(data, dtype=float)
        noise_cov_projected = wf.projector @ noise_cov @ wf.projector.T
        noise_cov_projected = 0.5 * (noise_cov_projected + noise_cov_projected.T)
        # Use a normalized shape matrix so alpha controls the noise scale.
        # This keeps noise_cov = alpha * shape_cov dimensionally coherent when
        # alpha is scaled from data covariance.
        noise_eigs = np.linalg.svd(noise_cov_projected, compute_uv=False)
        noise_scale = float(np.max(noise_eigs)) if noise_eigs.size else 1.0
        if not np.isfinite(noise_scale) or noise_scale <= self.eps:
            noise_scale = 1.0
        self.noise_cov_scale = noise_scale
        self.noise_cov_raw = noise_cov_projected
        self.noise_cov = noise_cov_projected / noise_scale

        data_cov = self.data_covariance(data_projected, center=True, ddof=1)
        # Noise-learning variants with Convexity-style updates converge to
        # the same fixed point regardless of the initial noise level (alpha),
        # because the update rule doesn't depend on the previous gammas and
        # the noise is re-estimated every iteration.  Searching over a grid
        # of alphas just wastes compute and triggers spurious edge-of-grid
        # warnings from GCV.  Use a single well-scaled initial noise level.
        if self.noise_learning == "learn" and self.alpha == "auto":
            self.alphas = [float(noise_scale)]
        else:
            self.get_alphas(reference=data_cov)
        inverse_operators = []
        original_leadfield = self.leadfield
        self.leadfield = wf.G_white
        try:
            for alpha in self.alphas:
                inverse_operator_projected = self.make_champagne(
                    data_projected,
                    float(alpha) * float(self.alpha_scaler),
                    pruning_thresh=pruning_thresh,
                )
                inverse_operators.append(
                    inverse_operator_projected @ wf.sensor_transform
                )
        finally:
            self.leadfield = original_leadfield

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
        L_orig = np.asarray(self.leadfield, dtype=float)
        L_norms = np.maximum(np.linalg.norm(L_orig, axis=0), self.eps)
        L_full_scaled = L_orig / L_norms
        L = L_full_scaled.copy()

        # re-reference data for noise learning modes (FUN/HSChampagne requirement)
        if self.noise_learning == "learn":
            Y = Y - Y.mean(axis=0)

        Y_scaled = Y.copy()

        scaler = 1.0  # Keep track of scaler even if not used

        gammas = np.ones(n_dipoles)

        # Initialize noise covariance based on learning mode
        I_chans = np.eye(n_chans)
        base_noise_cov = getattr(self, "noise_cov", None)
        if base_noise_cov is None:
            base_noise_cov = I_chans
        base_noise_cov = np.asarray(base_noise_cov, dtype=float)
        if base_noise_cov.shape != (n_chans, n_chans):
            base_noise_cov = I_chans
        base_noise_cov = 0.5 * (base_noise_cov + base_noise_cov.T)

        if self.noise_learning == "learn":
            if self.noise_learning_mode in {"diagonal", "precision"}:
                base_noise_cov = np.diag(np.diag(base_noise_cov))
            elif self.noise_learning_mode == "homoscedastic":
                base_noise_cov = I_chans
        noise_cov = float(alpha) * base_noise_cov

        # Fast path: delegate to shared SBL engine for standard rules
        rule_lower = self.update_rule.lower()
        _fast_rules = {
            "mackay",
            "convexity",
            "mm",
            "em",
            "lowsnr",
            "adaptive",
            "dynamic_adaptive",
        }
        if rule_lower in _fast_rules and self.noise_learning != "learn":
            if rule_lower == "lowsnr":
                sbl_rule = "convexity"
            elif rule_lower == "adaptive":
                noise_diag_mean = float(np.mean(np.diag(noise_cov)))

                def sbl_rule(mu_x, gammas, z_diag, n_times, _ndm=noise_diag_mean):
                    upper = np.mean(mu_x**2, axis=1)
                    snr = upper / _ndm
                    exp = 0.5 + 0.5 / (1 + np.exp(-snr + 5))
                    return (upper / (z_diag + 1e-20)) ** exp
            elif rule_lower == "dynamic_adaptive":
                noise_diag_mean = float(np.mean(np.diag(noise_cov)))
                _iter_count = [0]

                def sbl_rule(
                    mu_x, gammas, z_diag, n_times, _ndm=noise_diag_mean, _it=_iter_count
                ):
                    upper = np.mean(mu_x**2, axis=1)
                    snr = upper / _ndm
                    i = _it[0]
                    _it[0] += 1
                    ifact = 1 - np.exp(-i / 10)
                    sfact = 1 / (1 + np.exp(-snr + 5))
                    mackay = upper / (gammas * z_diag + 1e-20)
                    convex = np.sqrt(upper / (z_diag + 1e-20))
                    weighted = (sfact * mackay + (1 - sfact) * convex) ** ifact
                    smooth = 0.1 * (1 - ifact)
                    return (1 - smooth) * weighted + smooth * gammas
            else:
                sbl_rule = rule_lower
            result = sbl_iterate(
                L=L_full_scaled,
                Y=Y_scaled,
                noise_cov=noise_cov,
                update_rule=sbl_rule,
                max_iter=self.max_iter,
                prune=self.prune,
                pruning_thresh=pruning_thresh,
                conv_crit=self.convergence_criterion,
            )
            gammas_scaled = np.zeros(n_dipoles)
            gammas_scaled[result.active_set] = result.gammas
            gammas_final = gammas_scaled / (L_norms**2)
            Sigma_y_final = noise_cov + (L_orig * gammas_final) @ L_orig.T
            Sigma_y_final = 0.5 * (Sigma_y_final + Sigma_y_final.T)
            Sigma_y_final_inv = _cholesky_inv(Sigma_y_final)
            return (Sigma_y_final_inv @ L_orig).T * gammas_final[:, None]

        # Legacy path: complex update rules (AR-EM, TEM) or noise learning
        # Vectorized initial posterior
        Sigma_y = noise_cov + (L * gammas) @ L.T
        Sigma_y = 0.5 * (Sigma_y + Sigma_y.T)
        Sigma_y_inv = _cholesky_inv(Sigma_y)
        SiL = Sigma_y_inv @ L
        mu_x = (SiL.T @ Y_scaled) * gammas[:, None]

        loss_list = []
        active_set = np.arange(n_dipoles)

        # Initialize AR-EM specific variables if needed
        if rule_lower == "ar-em":
            beta = np.clip(self.beta_init, 0, 0.99)

            def make_ar1_covariance(beta, n_times):
                if beta == 0:
                    return np.identity(n_times)
                indices = np.arange(n_times)
                B = beta ** np.abs(indices[:, None] - indices[None, :])
                B = B / (1 - beta**2)
                return B

            B = make_ar1_covariance(beta, n_times)

        # Initialize TEM specific variables if needed
        if rule_lower == "tem":
            It = np.identity(n_times)
            mu_x_scaled_init = mu_x / (gammas[:, np.newaxis] + 1e-10)
            B_hat = mu_x_scaled_init.T @ mu_x + self.theta * It
            B = B_hat / self._frob(B_hat)

        for i_iter in range(self.max_iter):
            old_gammas = gammas.copy()

            # SiL cached from posterior recompute (or initial)
            z_diag = np.sum(L * SiL, axis=0)

            if rule_lower == "em":
                diag_Sigma_x = gammas - gammas**2 * z_diag
                gammas = diag_Sigma_x + np.mean(mu_x**2, axis=1)

            elif rule_lower == "mackay":
                upper_term = np.mean(mu_x**2, axis=1)
                gammas = upper_term / (gammas * z_diag + 1e-20)

            elif rule_lower in {"convexity", "mm", "lowsnr"}:
                upper_term = np.mean(mu_x**2, axis=1)
                gammas = np.sqrt(upper_term / (z_diag + 1e-20))

            elif rule_lower == "ar-em":
                diag_Sigma_x = gammas - gammas**2 * z_diag

                try:
                    B_inv = np.linalg.inv(B)
                except np.linalg.LinAlgError:
                    B_inv = np.linalg.pinv(B)

                mu_x_B_inv = mu_x @ B_inv
                mahalanobis_terms = np.sum(mu_x * mu_x_B_inv, axis=1)
                gammas = diag_Sigma_x + mahalanobis_terms

                active_mask = gammas > self.pruning_thresh
                if np.any(active_mask):
                    mu_x_active = mu_x[active_mask]
                    autocorr_sum = np.sum(mu_x_active[:, :-1] * mu_x_active[:, 1:])
                    norm_sum = np.sum(mu_x_active[:, :-1] ** 2)

                    if norm_sum > 1e-10:
                        beta_gradient = autocorr_sum / norm_sum
                        beta = beta + self.beta_lr * (beta_gradient - beta)
                        beta = np.clip(beta, 0, 0.99)
                        B = make_ar1_covariance(beta, n_times)

            elif rule_lower == "tem":
                diag_Sigma_x = gammas - gammas**2 * z_diag

                try:
                    B_inv = np.linalg.inv(B)
                except np.linalg.LinAlgError:
                    B_inv = np.linalg.pinv(B)

                mu_x_B_inv = mu_x @ B_inv
                mahalanobis_terms = np.sum(mu_x * mu_x_B_inv, axis=1)
                gammas = diag_Sigma_x + mahalanobis_terms

                mu_x_scaled_t = mu_x / (gammas[:, np.newaxis] + 1e-10)
                B_hat = mu_x_scaled_t.T @ mu_x + self.theta * It
                B = B_hat / self._frob(B_hat)

            elif rule_lower == "adaptive":
                upper_term = np.mean(mu_x**2, axis=1)
                snr_estimate = upper_term / np.mean(np.diag(noise_cov))
                adaptive_exponent = 0.5 + 0.5 / (1 + np.exp(-snr_estimate + 5))
                gammas = (upper_term / z_diag) ** adaptive_exponent

            elif rule_lower == "dynamic_adaptive":
                upper_term = np.mean(mu_x**2, axis=1)
                snr_estimate = upper_term / np.mean(np.diag(noise_cov))
                iteration_factor = 1 - np.exp(-i_iter / 10)
                snr_factor = 1 / (1 + np.exp(-snr_estimate + 5))
                mackay_update = upper_term / (gammas * z_diag + 1e-20)
                convexity_update = np.sqrt(upper_term / (z_diag + 1e-20))
                weighted_update = (
                    snr_factor * mackay_update + (1 - snr_factor) * convexity_update
                ) ** iteration_factor
                smoothing_factor = 0.1 * (1 - iteration_factor)
                gammas = (
                    1 - smoothing_factor
                ) * weighted_update + smoothing_factor * gammas

            # Remove nans
            gammas[np.isnan(gammas)] = 0

            # Stop if gammas went to zero
            if np.linalg.norm(gammas) == 0:
                gammas = old_gammas
                break

            if self.prune:
                active_set_idc = np.where(gammas > (pruning_thresh * gammas.max()))[0]

                if len(active_set_idc) == 0:
                    gammas = old_gammas
                    break
                active_set = active_set[active_set_idc]
                gammas = gammas[active_set_idc]
                L = L[:, active_set_idc]

            # Update noise covariance if learning is enabled
            if self.noise_learning == "learn":
                # Compute pruned posterior mean using only active columns
                L_act_full = L_full_scaled[:, active_set]
                SiL_act = Sigma_y_inv @ L_act_full
                mu_x_noise = (SiL_act.T @ Y_scaled) * gammas[:, None]
                residuals = Y_scaled - L_act_full @ mu_x_noise
                noise_cov = self._update_noise_covariance_pruned(
                    residuals,
                    L_act_full,
                    gammas,
                    Sigma_y_inv,
                    noise_cov,
                    n_times,
                )

            # Recompute posterior with Cholesky
            Sigma_y = noise_cov + (L * gammas) @ L.T
            Sigma_y = 0.5 * (Sigma_y + Sigma_y.T)
            Sigma_y_inv = _cholesky_inv(Sigma_y)
            SiL = Sigma_y_inv @ L
            mu_x = (SiL.T @ Y_scaled) * gammas[:, None]

            # Negative log-marginal-likelihood
            data_fit = np.trace(Sigma_y_inv @ Y_scaled @ Y_scaled.T) / n_times
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                sign, log_det = np.linalg.slogdet(Sigma_y)
            if sign <= 0 or not np.isfinite(log_det):
                log_det = np.finfo(float).max / 2
            loss = data_fit + log_det

            loss_list.append(loss)

            if len(loss_list) > 1:
                relative_change = (loss_list[-2] - loss) / abs(loss_list[-2])

                if self.verbose > 1:
                    logger.debug(
                        f"Iteration {i_iter}: loss = {loss:.6f}, relative change = {relative_change:.6f}, Active set size = {len(active_set)}"
                    )

                if relative_change > 0 and relative_change < self.convergence_criterion:
                    if self.verbose > 0:
                        logger.info(
                            f"Converged because {relative_change:.6f} < {self.convergence_criterion:.6f}"
                        )
                    break
            else:
                if self.verbose > 1:
                    logger.debug(
                        f"Iteration {i_iter}: loss = {loss:.6f}, Active set size = {len(active_set)}"
                    )

        # Final inverse operator construction in original source units.
        gammas_scaled = np.zeros(n_dipoles)
        gammas_scaled[active_set] = gammas
        gammas_final = gammas_scaled / (L_norms**2)

        # Scale noise covariance back if learning was enabled
        if self.noise_learning == "learn":
            noise_cov_final = scaler * noise_cov
        else:
            noise_cov_final = noise_cov

        Sigma_y = noise_cov_final + (L_orig * gammas_final) @ L_orig.T
        Sigma_y = 0.5 * (Sigma_y + Sigma_y.T)
        Sigma_y_inv = _cholesky_inv(Sigma_y)
        inverse_operator = (Sigma_y_inv @ L_orig).T * gammas_final[:, None]

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

    def _update_noise_covariance_pruned(
        self,
        residuals,
        L_act,
        gammas,
        Sigma_y_inv,
        current_noise_cov,
        n_times,
    ):
        """Update noise covariance using only pruned (active) arrays.

        Avoids reconstructing full-size gamma/leadfield arrays and recomputing
        Sigma_y_inv (already available from main loop).
        """
        n_chans = L_act.shape[0]

        if self.noise_learning_mode == "homoscedastic":
            residual_power = np.sum(residuals**2) / n_times
            Sigma_y_inv_L = Sigma_y_inv @ L_act
            Sigma_X_diag = gammas * (1 - gammas * np.sum(L_act * Sigma_y_inv_L, axis=0))
            n_act = len(gammas)
            dof = n_chans - n_act + np.sum(Sigma_X_diag / (gammas + 1e-10))
            return np.eye(n_chans) * (residual_power / (dof + 1e-10))

        elif self.noise_learning_mode == "full":
            M_noise = (residuals @ residuals.T) / n_times
            C_noise = _cholesky_inv(current_noise_cov + 1e-8 * np.eye(n_chans))
            return self._fun_learning_cov_est(C_noise, M_noise, update_mode="diagonal")

        elif self.noise_learning_mode == "diagonal":
            diag_residual_power = np.sum(residuals**2, axis=1)
            normalization = np.diag(Sigma_y_inv) + 1e-10
            return np.diag(np.sqrt(diag_residual_power / (n_times * normalization)))

        elif self.noise_learning_mode == "precision":
            diag_residual_power = np.sum(residuals**2, axis=1)
            return np.diag(
                np.sqrt(diag_residual_power / (n_times * np.diag(Sigma_y_inv) + 1e-10))
            )

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
