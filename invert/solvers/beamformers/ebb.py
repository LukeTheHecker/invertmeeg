import logging

import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverEBB(BaseSolver):
    """
    Empirical Bayesian Beamformer (EBB) solver for M/EEG inverse problem.
    """

    meta = SolverMeta(
        slug="ebb",
        full_name="Empirical Bayesian Beamformer",
        category="Beamformers",
        description=(
            "Iterative empirical-Bayes / ARD-style beamformer that updates per-source "
            "variances (hyperparameters) from the data and forms corresponding spatial "
            "filters."
        ),
        references=["tbd"],
    )

    def __init__(
        self,
        name="Empirical Bayesian Beamformer",
        reduce_rank=True,
        rank="auto",
        **kwargs,
    ):
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
        max_iter=100,
        tol=1e-6,
        **kwargs,
    ):
        """
        Solve the inverse problem using the Empirical Bayesian Beamformer method.

        This implements an iterative algorithm that estimates source variances (hyperparameters)
        from the data using an empirical Bayes approach, similar to Champagne/ARD methods.

        Parameters:
        -----------
        data : array, shape (n_channels, n_times)
            The sensor data.
        forward : array, shape (n_channels, n_sources)
            The forward solution.
        noise_cov : array, shape (n_channels, n_channels), optional
            The noise covariance matrix.
        max_iter : int
            Maximum number of iterations for the EM algorithm.
        tol : float
            Convergence tolerance for source variance updates.

        Returns:
        --------
        sources : array, shape (n_sources, n_times)
            The estimated source time series.
        """
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)

        data = self.unpack_data_obj(mne_obj)
        leadfield = self.leadfield
        n_channels, n_times = data.shape
        n_sources = leadfield.shape[1]

        # normalize leadfield
        leadfield /= np.linalg.norm(leadfield, axis=0)

        # Compute data covariance
        data_cov = self.data_covariance(data, center=True, ddof=1)

        # handle noise_cov - use scaled identity as default
        if noise_cov is None:
            noise_cov = np.identity(n_channels)

        inverse_operators = []
        self.alphas = self.get_alphas(reference=data_cov)

        for alpha in self.alphas:
            # Initialize source variances (diagonal of source covariance)
            # Start with uniform prior
            gamma = np.ones(n_sources)

            # Regularized noise covariance
            Cn = alpha * noise_cov

            # Iterative empirical Bayes updates
            for n_iter in range(max_iter):
                logger.debug(f"EBB iteration {n_iter + 1} of {max_iter}")
                # Build current model covariance: C = L @ diag(gamma) @ L.T + Cn
                # For efficiency, compute using woodbury: C^-1 = Cn^-1 - Cn^-1 @ L @ (I/gamma + L.T @ Cn^-1 @ L)^-1 @ L.T @ Cn^-1

                Cn_inv = self.robust_inverse(Cn)

                # Compute L.T @ Cn^-1 @ L efficiently
                Cn_inv_L = Cn_inv @ leadfield
                LT_Cn_inv_L = leadfield.T @ Cn_inv_L

                # Add diagonal loading: (I/gamma + L.T @ Cn^-1 @ L)
                middle = np.diag(1.0 / gamma) + LT_Cn_inv_L
                middle_inv = self.robust_inverse(middle)

                # Woodbury identity for C^-1
                Cn_inv - Cn_inv_L @ middle_inv @ Cn_inv_L.T

                # Compute posterior source covariance: Sigma_s = (I/gamma + L.T @ C^-1 @ L)^-1
                # This is actually just middle_inv from above!
                Sigma_s = middle_inv

                # Compute beamformer weights for this iteration
                W = Sigma_s @ leadfield.T @ Cn_inv  # shape: (n_sources, n_channels)

                # Estimate source activity
                source_estimates = W @ data  # shape: (n_sources, n_times)

                # Update source variances using empirical estimates
                # gamma_new[i] = trace(Sigma_s[i,i]) + (source_estimates[i,:]^2).mean()
                # Simplified: use source power + trace correction
                source_power = np.mean(
                    source_estimates**2, axis=1
                )  # shape: (n_sources,)
                gamma_new = source_power + np.diag(Sigma_s)

                # Check convergence
                rel_change = np.abs(gamma_new - gamma) / (np.abs(gamma) + 1e-10)
                max_change = np.max(rel_change)

                if max_change < tol:
                    logger.info(
                        f"EBB converged after {n_iter + 1} iterations (max change: {max_change:.2e})"
                    )
                    break

                # Update with damping for stability
                damping = 0.5
                gamma = damping * gamma_new + (1 - damping) * gamma

                # Prevent numerical issues
                gamma = np.maximum(gamma, 1e-12)

            else:
                logger.warning(
                    f"EBB reached max iterations ({max_iter}), max change: {max_change:.2e}"
                )

            # Final beamformer weights with converged hyperparameters
            Cn_inv = self.robust_inverse(Cn)
            Cn_inv_L = Cn_inv @ leadfield
            LT_Cn_inv_L = leadfield.T @ Cn_inv_L
            middle = np.diag(1.0 / gamma) + LT_Cn_inv_L
            middle_inv = self.robust_inverse(middle)
            Cn_inv - Cn_inv_L @ middle_inv @ Cn_inv_L.T

            W = middle_inv @ leadfield.T @ Cn_inv

            # Optional weight normalization
            if weight_norm:
                W /= np.linalg.norm(W, axis=1, keepdims=True)

            inverse_operators.append(W)  # Transpose to match expected shape

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]

        return self
