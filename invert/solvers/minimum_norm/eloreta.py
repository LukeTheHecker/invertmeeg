import logging

import numpy as np
from scipy.linalg import pinv

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverELORETA(BaseSolver):
    """Class for the exact Low Resolution Tomography (eLORETA) inverse
        solution [1].

    References
    ----------
    [1] Pascual-Marqui, R. D. (2007). Discrete, 3D distributed, linear imaging
    methods of electric neuronal activity. Part 1: exact, zero error
    localization. arXiv preprint arXiv:0710.3341.

    """

    meta = SolverMeta(
        acronym="eLORETA",
        full_name="Exact Low Resolution Electromagnetic Tomography",
        category="Minimum Norm",
        description=(
            "LORETA-family inverse that iteratively estimates source weights to "
            "achieve exact (zero-error) localization under idealized conditions."
        ),
        references=[
            "Pascual-Marqui, R. D. (2007). Discrete, 3D distributed, linear imaging methods of electric neuronal activity. Part 1: exact, zero error localization. arXiv:0710.3341.",
        ],
    )

    def __init__(
        self,
        name="Exact Low Resolution Tomography",
        use_noise_whitener: bool = True,
        use_trace_normalization: bool = True,
        rank_tol: float = 1e-12,
        eps: float = 1e-15,
        **kwargs,
    ):
        self.name = name
        self.use_noise_whitener = bool(use_noise_whitener)
        self.use_trace_normalization = bool(use_trace_normalization)
        self.rank_tol = float(rank_tol)
        self.eps = float(eps)
        super().__init__(**kwargs)
        self.require_recompute = False
        self.require_data = False

    def make_inverse_operator(
        self,
        forward,
        *args,
        alpha="auto",
        noise_cov=None,
        verbose=0,
        stop_crit=1e-3,
        max_iter=100,
        **kwargs,
    ):
        """Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        alpha : float
            The regularization parameter.
        stop_crit : float
            The convergence criterion to optimize the weight matrix.
        max_iter : int
            The stopping criterion to optimize the weight matrix.

        Return
        ------
        self : object returns itself for convenience
        """
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)

        if self.use_noise_whitener:
            n_chans = self.leadfield.shape[0]
            if noise_cov is None:
                noise_cov = np.eye(n_chans, dtype=float)
            wf = self.prepare_whitened_forward(
                noise_cov,
                trace_normalize=self.use_trace_normalization,
                rank_tol=self.rank_tol,
                eps=self.eps,
            )
        else:
            wf = self.prepare_whitened_forward(
                None,
                trace_normalize=self.use_trace_normalization,
                rank_tol=self.rank_tol,
                eps=self.eps,
            )
        if wf.whitener_mode not in ("projected", "none"):
            logger.warning("eLORETA whitener fallback used: %s", wf.whitener_mode)

        leadfield = wf.A if wf.A is not None else wf.G_white
        leadfield_scale = wf.trace_scale
        sensor_transform = wf.sensor_transform
        self._sensor_transform = sensor_transform
        # Keep alpha scaling consistent with the transformed leadfield.
        self.get_alphas(reference=leadfield @ leadfield.T)

        # Some pre-calculations
        I = np.identity(leadfield.shape[0])

        # No regularization leads to weird results with eLORETA
        if 0 in self.alphas and len(self.alphas) > 1:
            idx = self.alphas.index(0)
            self.alphas.pop(idx)
            self.r_values = np.delete(self.r_values, idx)
        elif 0 in self.alphas and len(self.alphas) == 1:
            idx = self.alphas.index(0)
            self.alphas = [0.01]

        inverse_operators = []
        original_leadfield = self.leadfield
        self.leadfield = leadfield
        try:
            for alpha in self.alphas:
                W = self.calc_W(alpha, max_iter=max_iter, stop_crit=stop_crit)

                # More efficient computation avoiding explicit W_inv matrix construction
                # Since W is diagonal, W_inv is also diagonal with reciprocal elements
                W_inv_diag = 1.0 / W.diagonal()

                # Compute leadfield @ W_inv more efficiently using broadcasting
                LW_inv = self.leadfield * W_inv_diag[np.newaxis, :]

                # Compute the final inverse operator
                inner_term = LW_inv @ self.leadfield.T + alpha * I
                inverse_operator_eff = (W_inv_diag[:, np.newaxis] * self.leadfield.T) @ pinv(
                    inner_term
                )
                inverse_operator = (
                    float(leadfield_scale) * inverse_operator_eff
                ) @ sensor_transform

                inverse_operators.append(inverse_operator)
        finally:
            self.leadfield = original_leadfield

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        return self

    def calc_W(self, alpha, max_iter=100, stop_crit=1e-3):
        K = self.leadfield
        n_chans, n_dipoles = K.shape

        # Input validation
        if alpha <= 0:
            raise ValueError("Alpha must be positive")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if stop_crit <= 0:
            raise ValueError("stop_crit must be positive")

        # Use dense matrices for better performance in this iterative context
        I = np.identity(n_chans)
        W_diag = np.ones(n_dipoles)  # Store only diagonal elements

        # Pre-allocate arrays to avoid repeated memory allocation
        KT = K.T  # Cache transpose

        # Refine W iteratively
        for iter in range(max_iter):
            W_diag_old = W_diag.copy()

            # Ensure numerical stability by avoiding division by very small numbers
            W_inv_diag = 1.0 / np.maximum(W_diag, self.eps)

            # Compute K @ W_inv @ K.T more efficiently
            # Since W is diagonal, K @ W_inv = K * W_inv_diag (broadcasting)
            KW_inv = K * W_inv_diag[np.newaxis, :]
            inner_matrix = KW_inv @ KT + alpha * I

            # Use more stable pseudo-inverse
            M = pinv(inner_matrix)

            # Compute diagonal elements more efficiently
            # diag(K.T @ M @ K) = sum(K.T * (M @ K), axis=0)
            MK = M @ K
            # KT is (n_dipoles, n_chans), MK is (n_chans, n_dipoles)
            # We need to compute the diagonal of KT @ MK
            diag_elements = np.sum(KT * MK.T, axis=1)

            # Ensure non-negative values before taking square root for numerical stability
            W_diag = np.sqrt(np.maximum(diag_elements, self.eps))

            # More efficient convergence check using relative change
            rel_change = np.mean(np.abs(W_diag - W_diag_old) / (W_diag_old + self.eps))

            if self.verbose > 1:
                logger.debug(f"iter {iter}: relative change = {rel_change:.6f}")

            if rel_change < stop_crit:
                if self.verbose > 0:
                    logger.info(f"eLORETA converged after {iter + 1} iterations")
                break
        else:
            if self.verbose > 0:
                logger.warning(
                    f"eLORETA reached max iterations ({max_iter}) without convergence"
                )

        # Return as sparse diagonal matrix for compatibility
        from scipy.sparse import diags

        return diags(W_diag, format="csr")
