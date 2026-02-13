import logging
from copy import deepcopy

import mne
import numpy as np

from ...util import (
    thresholding,
)
from ..base import BaseSolver, SolverMeta

logger = logging.getLogger(__name__)


class SolverREMBO(BaseSolver):
    """Class for the Reduce Multi-Measurement-Vector and Boost (ReMBo) inverse
        solution [1]. The algorithm as describe in [2] was used for the
        imlpementation.

    References
    ----------
    [1] Mishali, M., & Eldar, Y. C. (2008). Reduce and boost: Recovering
    arbitrary sets of jointly sparse vectors. IEEE Transactions on Signal
    Processing, 56(10), 4692-4702.
    [2] Duarte, M. F., & Eldar, Y. C. (2011).
    Structured compressed sensing: From theory to applications. IEEE
    Transactions on signal processing, 59(9), 4053-4085.
    """

    meta = SolverMeta(
        acronym="ReMBo",
        full_name="Reduce Multi-Measurement-Vector and Boost",
        category="Matching Pursuit",
        description=(
            "Randomly reduces a multi-measurement problem to repeated single-measurement "
            "OMP-style recovery, then boosts by re-fitting on the recovered joint support."
        ),
        references=[
            "Mishali, M., & Eldar, Y. C. (2008). Reduce and boost: Recovering arbitrary sets of jointly sparse vectors. IEEE Transactions on Signal Processing, 56(10), 4692–4702.",
            "Duarte, M. F., & Eldar, Y. C. (2011). Structured compressed sensing: From theory to applications. IEEE Transactions on Signal Processing, 59(9), 4053–4085.",
        ],
    )

    def __init__(self, name="Reduce Multi-Measurement-Vector and Boost", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(
        self,
        forward,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
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

        Return
        ------
        self : object returns itself for convenience
        """
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        self.prepare_whitened_forward(noise_cov)
        self.leadfield_original = self.leadfield.copy()
        self.leadfield_normed = self.robust_normalize_leadfield(self.leadfield)

        self.inverse_operators = []
        return self

    def apply_inverse_operator(
        self, mne_obj, K="auto", max_boost_iter=None, fit_tol="auto"
    ) -> mne.SourceEstimate:  # type: ignore
        """Apply the REMBO inverse solution.

        Parameters
        ----------
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        K : ["auto", int]
            Maximum support size (sparsity budget) for the SMV solve.
            If "auto", uses a small adaptive budget based on channel count.
        max_boost_iter : [None, int]
            Maximum number of boost iterations. If None, defaults to 50.
        fit_tol : ["auto", float]
            Relative SMV fit tolerance used by the boosting acceptance check.
            If "auto", defaults to 1e-3.

        Return
        ------
        stc : mne.SourceEstimate
            The source estimate containing the inverse solution.
        """
        data = self.unpack_data_obj(mne_obj)
        self.validate_operator_data_compatibility(data)
        data = self._sensor_transform @ data
        source_mat = self.calc_rembo_solution(
            data, K=K, max_boost_iter=max_boost_iter, fit_tol=fit_tol
        )
        stc = self.source_to_object(source_mat)
        return stc

    def calc_rembo_solution(self, y, K="auto", max_boost_iter=None, fit_tol="auto"):
        """Calculate the REMBO inverse solution based on the measurement vector
            y.

        Parameters
        ----------
        y : numpy.ndarray
            The EEG matrix (channels, time)
        K : ["auto", int]
            Maximum support size (sparsity budget) for the SMV step.
            If "auto", set to min(8, max(2, n_chans // 10)).
        max_boost_iter : [None, int]
            Maximum number of random MMV->SMV reductions to try.
            If None, defaults to 50.
        fit_tol : ["auto", float]
            Relative SMV fit tolerance. Candidate supports satisfy
            ||y_vec - L x_hat||_2 / ||y_vec||_2 <= fit_tol.
            If "auto", defaults to 1e-3.

        Return
        ------
        x_hat : numpy.ndarray
            The source matrix (dipoles, time)
        """
        n_chans, n_time = y.shape
        if K == "auto":
            # Adaptive default: avoid single-source bias while keeping sparsity tight.
            K = min(8, max(2, n_chans // 10))
        K = int(K)
        if K <= 0:
            raise ValueError("K must be positive")

        if max_boost_iter is None:
            max_boost_iter = 50
        max_boost_iter = int(max_boost_iter)
        if max_boost_iter <= 0:
            raise ValueError("max_boost_iter must be positive")

        if fit_tol == "auto":
            fit_tol = 1e-3
        fit_tol = float(fit_tol)
        if fit_tol < 0:
            raise ValueError("fit_tol must be non-negative")

        n_dipoles = self.leadfield.shape[1]
        y_norm_eps = 1e-15
        best_support = np.array([], dtype=int)
        best_rel_resid = np.inf

        for _ in range(max_boost_iter):
            # Random merge vector from an absolutely continuous distribution.
            # Use NumPy's global RNG so benchmark seeds control reproducibility.
            a = np.random.standard_normal(n_time)
            y_vec = y @ a
            x_vec = self.calc_omp_solution(y_vec, K=K)
            S_hat = np.where(x_vec != 0)[0].astype(int)

            if len(S_hat) == 0 or len(S_hat) > K:
                continue

            rel_resid = np.linalg.norm(y_vec - self.leadfield @ x_vec) / max(
                np.linalg.norm(y_vec), y_norm_eps
            )
            if rel_resid < best_rel_resid:
                best_rel_resid = rel_resid
                best_support = S_hat

            if rel_resid <= fit_tol:
                break

        x_hat = np.zeros((n_dipoles, n_time))
        if len(best_support) == 0:
            return x_hat

        As_pinv = np.linalg.pinv(self.leadfield[:, best_support])
        x_hat[best_support, :] = As_pinv @ y

        return x_hat

    def calc_omp_solution(self, y, K="auto"):
        """Calculates the Orthogonal Matching Pursuit (OMP) inverse solution.
        (Used by REMBO algorithm)

        Parameters
        ----------
        y : numpy.ndarray
            The data matrix (channels,).
        K : ["auto", int]
            Maximum number of nonzero atoms.

        Return
        ------
        x_hat : numpy.ndarray
            The inverse solution (dipoles,)
        """
        n_chans = len(y)
        _, n_dipoles = self.leadfield.shape

        if K == "auto":
            K = 1
        K = int(K)
        if K <= 0:
            raise ValueError("K must be positive")

        x_hat = np.zeros(n_dipoles)
        x_hats = [deepcopy(x_hat)]

        omega = np.array([])
        r = deepcopy(y)
        residuals = np.array([np.linalg.norm(r)])

        max_iter = min(n_chans, K)
        for _ in range(max_iter):
            # Use normalized atoms for selection to avoid superficial/high-norm bias.
            b = self.leadfield_normed.T @ r
            b_thresh = thresholding(b, 1)
            new_atoms = np.where(b_thresh != 0)[0]
            omega = np.append(omega, new_atoms)
            omega = np.unique(omega.astype(int))

            # Use robust inverse from base class for coefficient estimation
            if len(omega) > 0:
                L_omega = self.leadfield_original[:, omega]
                x_hat[omega] = self.robust_inverse_solution(L_omega, y)

            r = y - self.leadfield_original @ x_hat

            residuals = np.append(residuals, np.linalg.norm(r))
            x_hats.append(deepcopy(x_hat))

            # Early stopping if residual starts increasing
            if len(residuals) > 1 and residuals[-1] > residuals[-2]:
                break

        x_hat = x_hats[int(np.argmin(residuals))]
        return x_hat
