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

    def __init__(self, name="Exact Low Resolution Tomography", **kwargs):
        self.name = name
        super().__init__(**kwargs)
        self.require_recompute = False
        self.require_data = False

    def make_inverse_operator(
        self,
        forward,
        *args,
        alpha="auto",
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
        leadfield = self.leadfield
        n_chans = leadfield.shape[0]
        # noise_cov = np.identity(n_chans)

        # Some pre-calculations
        I = np.identity(n_chans)

        # No regularization leads to weird results with eLORETA
        if 0 in self.alphas and len(self.alphas) > 1:
            idx = self.alphas.index(0)
            self.alphas.pop(idx)
            self.r_values = np.delete(self.r_values, idx)
        elif 0 in self.alphas and len(self.alphas) == 1:
            idx = self.alphas.index(0)
            self.alphas = [0.01]

        inverse_operators = []
        for alpha in self.alphas:
            W = self.calc_W(alpha, max_iter=max_iter, stop_crit=stop_crit)

            # More efficient computation avoiding explicit W_inv matrix construction
            # Since W is diagonal, W_inv is also diagonal with reciprocal elements
            W_inv_diag = 1.0 / W.diagonal()

            # Compute leadfield @ W_inv more efficiently using broadcasting
            LW_inv = leadfield * W_inv_diag[np.newaxis, :]

            # Compute the final inverse operator
            inner_term = LW_inv @ leadfield.T + alpha * I
            inverse_operator = (W_inv_diag[:, np.newaxis] * leadfield.T) @ pinv(
                inner_term
            )

            inverse_operators.append(inverse_operator)

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
            W_inv_diag = 1.0 / np.maximum(W_diag, 1e-12)

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
            W_diag = np.sqrt(np.maximum(diag_elements, 1e-12))

            # More efficient convergence check using relative change
            rel_change = np.mean(np.abs(W_diag - W_diag_old) / (W_diag_old + 1e-12))

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
