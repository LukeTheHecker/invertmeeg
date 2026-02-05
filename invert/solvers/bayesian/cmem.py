import logging
import warnings

import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from tqdm import tqdm

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverCMEM(BaseSolver):
    """Coherent Maximum Entropy on the Mean (cMEM) source localization solver.

    Parameters
    ----------
    name : str
        Name of the solver.
    num_parcels : int
        Number of parcels for data-driven parcellation.
    max_iter : int
        Maximum optimization iterations per time point.
    batch_size : int
        Batch size for time-point processing.

    References
    ----------
    Amblard, C., Lapalme, E., & Bhatt, P. (2004). Biomagnetic source
    detection by maximum entropy and graphical models. IEEE
    Transactions on Biomedical Engineering, 51(3), 427-442.
    """

    meta = SolverMeta(
        acronym="cMEM",
        full_name="Coherent Maximum Entropy on the Mean",
        category="Bayesian",
        description=(
            "Maximum-entropy-on-the-mean approach using graphical models and "
            "parcel-wise optimization (data-driven parcellation) to estimate "
            "source activity."
        ),
        references=[
            "Amblard, C., Lapalme, E., & Bhatt, P. (2004). Biomagnetic source detection by maximum entropy and graphical models. IEEE Transactions on Biomedical Engineering, 51(3), 427–442.",
        ],
    )

    def __init__(
        self,
        name="cMEM",
        num_parcels=200,
        max_iter=100,
        batch_size=100,
        **kwargs,
    ):
        self.name = name
        self.num_parcels = num_parcels
        self.max_iter = max_iter
        self.batch_size = batch_size
        return super().__init__(**kwargs)

    def make_inverse_operator(
        self,
        forward,
        mne_obj,
        *args,
        alpha="auto",
        adjacency=None,
        positions=None,
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
        adjacency : numpy.ndarray, optional
            Source adjacency matrix (n, n).
        positions : numpy.ndarray, optional
            Source positions (n, 3).

        Return
        ------
        self : object returns itself for convenience
        """
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)

        J, parcels = _cmem(
            data,
            self.leadfield,
            A=adjacency,
            P=positions,
            num_parcels=self.num_parcels,
            max_iter=self.max_iter,
            batch_size=self.batch_size,
        )

        self.parcels = parcels
        # Store source estimate directly; wrap in InverseOperator for API compat
        self.inverse_operators = [
            InverseOperator(J, self.name),
        ]
        return self

    def apply_inverse_operator(self, mne_obj):
        """Apply the cMEM inverse operator.

        Since cMEM computes the full source time series during
        ``make_inverse_operator``, applying the operator re-runs the
        algorithm on the new data.

        Parameters
        ----------
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.

        Return
        ------
        stc : mne.SourceEstimate
            The source estimate.
        """
        data = self.unpack_data_obj(mne_obj)

        J, self.parcels = _cmem(
            data,
            self.leadfield,
            num_parcels=self.num_parcels,
            max_iter=self.max_iter,
            batch_size=self.batch_size,
        )

        stc = self.source_to_object(J)
        return stc


# ---------------------------------------------------------------------------
# Private helper functions
# ---------------------------------------------------------------------------


def _cmem(Y, L, A=None, P=None, num_parcels=200, max_iter=100, batch_size=100):
    """Coherent Maximum Entropy on the Mean (cMEM) source localization.

    Parameters
    ----------
    Y : array (m, t) - EEG/MEG data matrix
    L : array (m, n) - Lead field matrix
    A : array (n, n) - Source adjacency matrix (optional)
    P : array (n, 3) - Source positions (optional)
    num_parcels : int - Number of parcels for DDP
    max_iter : int - Maximum optimization iterations
    batch_size : int - Batch size for time processing

    Returns
    -------
    J : array (n, t) - Source time series
    parcels : array (n,) - Parcel assignment for each source
    """
    m, t = Y.shape
    m_l, n = L.shape

    assert m == m_l, "Dimension mismatch between Y and L"

    # Step 1: Data Driven Parcellation (DDP)
    logger.info("Performing Data Driven Parcellation...")
    parcels = _data_driven_parcellation(Y, L, num_parcels)

    # Step 2: Initialize parcel parameters
    logger.info("Initializing parcel parameters...")
    unique_parcels = np.unique(parcels)

    parcel_params = {}
    alpha_values = []
    for k, parcel_id in enumerate(unique_parcels):
        parcel_vertices = np.where(parcels == parcel_id)[0]
        parcel_size = len(parcel_vertices)

        mu_k = np.zeros(parcel_size)

        if A is not None:
            Sigma_k = _compute_spatial_covariance(parcel_vertices, A)
        else:
            Sigma_k = np.eye(parcel_size)

        L_k = L[:, parcel_vertices]
        msp_coeffs = _compute_msp_coefficients(Y, L_k)
        alpha_k = np.median(msp_coeffs)

        if k < 3:
            logger.debug(
                f"  Parcel {k}: MSP coeffs range [{np.min(msp_coeffs):.2e}, {np.max(msp_coeffs):.2e}], median={alpha_k:.2e}"
            )

        alpha_k = np.clip(alpha_k, 0.01, 0.99)
        alpha_values.append(alpha_k)

        parcel_params[parcel_id] = {
            "vertices": parcel_vertices,
            "mu": mu_k,
            "Sigma": Sigma_k,
            "alpha": alpha_k,
            "L": L_k,
        }

    logger.debug(
        f"Alpha values range: [{np.min(alpha_values):.4f}, {np.max(alpha_values):.4f}]"
    )

    # Step 3: Estimate noise covariance (diagonal)
    logger.info("Estimating noise covariance...")
    Sigma_e = np.diag(np.var(Y, axis=1))
    Sigma_e_inv = np.linalg.inv(Sigma_e)

    # Step 4: Precompute matrix operations
    logger.info("Precomputing matrix operations...")
    precomputed = _precompute_parcel_matrices(parcel_params, Sigma_e_inv)

    # Step 5: Vectorized MEM optimization with batch processing
    logger.info("Running vectorized MEM optimization...")
    J = np.zeros((n, t))

    num_batches = (t + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches)):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, t)

        Y_batch = Y[:, batch_start:batch_end]

        lambda_batch = _optimize_lambda_batch(
            Y_batch, parcel_params, precomputed, Sigma_e_inv, max_iter
        )

        J_batch = _compute_sources_batch(lambda_batch, parcel_params, precomputed, n)

        if batch_idx == 0:
            logger.debug(
                f"Lambda batch stats: min={np.min(lambda_batch):.2e}, max={np.max(lambda_batch):.2e}, mean={np.mean(lambda_batch):.2e}"
            )
            logger.debug(
                f"J_batch stats: min={np.min(J_batch):.2e}, max={np.max(J_batch):.2e}, mean={np.mean(J_batch):.2e}"
            )
            logger.debug(
                f"Non-zero sources in batch: {np.sum(np.abs(J_batch) > 1e-10)}"
            )

        J[:, batch_start:batch_end] = J_batch

        if (batch_idx + 1) % max(1, num_batches // 10) == 0:
            logger.info(
                f"Processed batch {batch_idx + 1}/{num_batches} ({batch_end}/{t} time samples)"
            )

    return J, parcels


def _data_driven_parcellation(Y, L, num_parcels):
    """Perform data-driven parcellation using MSP and clustering."""
    msp_coeffs = _compute_msp_coefficients(Y, L)

    kmeans = KMeans(n_clusters=num_parcels, random_state=42)
    parcels = kmeans.fit_predict(msp_coeffs.reshape(-1, 1))

    return parcels


def _compute_msp_coefficients(Y, L):
    """Compute Multivariate Source Prelocalization coefficients."""
    U, s, Vt = np.linalg.svd(L, full_matrices=False)
    L_pinv = Vt.T @ np.diag(1 / s) @ U.T

    J = L_pinv @ Y

    coeffs = np.var(J, axis=1)

    return coeffs


def _compute_spatial_covariance(vertices, A, alpha=0.2):
    """Compute spatial covariance matrix using adjacency."""
    n_v = len(vertices)

    A_sub = A[np.ix_(vertices, vertices)]

    D = np.diag(np.sum(A_sub, axis=1))
    L_graph = D - A_sub

    L_reg = L_graph + alpha * np.eye(n_v)

    try:
        Sigma = np.linalg.inv(L_reg)
    except np.linalg.LinAlgError:
        Sigma = np.eye(n_v)

    return Sigma


def _precompute_parcel_matrices(parcel_params, Sigma_e_inv):
    """Precompute expensive matrix operations for vectorized cMEM."""
    precomputed = {}

    for parcel_id, params in parcel_params.items():
        L_k = params["L"]
        Sigma_k = params["Sigma"]
        mu_k = params["mu"]

        try:
            Sigma_k_inv = np.linalg.inv(Sigma_k)
            Sigma_k_logdet = np.log(np.linalg.det(2 * np.pi * Sigma_k))
        except np.linalg.LinAlgError:
            Sigma_k_reg = Sigma_k + 1e-6 * np.eye(Sigma_k.shape[0])
            Sigma_k_inv = np.linalg.inv(Sigma_k_reg)
            Sigma_k_logdet = np.log(np.linalg.det(2 * np.pi * Sigma_k_reg))

        LT_Sigma_e_inv = L_k.T @ Sigma_e_inv
        LT_Sigma_e_inv_L = LT_Sigma_e_inv @ L_k
        mu_Sigma_inv_mu = mu_k.T @ Sigma_k_inv @ mu_k

        precomputed[parcel_id] = {
            "L_k": L_k,
            "Sigma_k": Sigma_k,
            "Sigma_k_inv": Sigma_k_inv,
            "Sigma_k_logdet": Sigma_k_logdet,
            "mu_k": mu_k,
            "alpha_k": params["alpha"],
            "vertices": params["vertices"],
            "LT_Sigma_e_inv": LT_Sigma_e_inv,
            "LT_Sigma_e_inv_L": LT_Sigma_e_inv_L,
            "mu_Sigma_inv_mu": mu_Sigma_inv_mu,
        }

    return precomputed


def _optimize_lambda_batch(
    Y_batch, parcel_params, precomputed, Sigma_e_inv, max_iter=100
):
    """Vectorized optimization of Lagrange multipliers for batch of time points."""
    m, batch_size = Y_batch.shape
    lambda_batch = np.zeros((m, batch_size))

    for i in range(batch_size):
        y_t = Y_batch[:, i]

        def objective(lambda_vec, y_t=y_t):
            return _mem_dual_function_vectorized(
                lambda_vec, y_t, precomputed, Sigma_e_inv
            )

        lambda_init = np.zeros(m)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                objective, lambda_init, method="L-BFGS-B", options={"maxiter": max_iter}
            )

        lambda_batch[:, i] = result.x

        if i < 3:
            logger.debug(
                f"  Time {i}: optimization success={result.success}, final objective={result.fun:.2e}"
            )
            logger.debug(
                f"  Lambda stats: min={np.min(result.x):.2e}, max={np.max(result.x):.2e}"
            )

    return lambda_batch


def _compute_sources_batch(lambda_batch, parcel_params, precomputed, n):
    """Vectorized computation of source estimates for batch of time points."""
    m, batch_size = lambda_batch.shape

    J_batch = np.zeros((n, batch_size))

    for parcel_id, precomp in precomputed.items():
        vertices = precomp["vertices"]
        L_k = precomp["L_k"]
        Sigma_k = precomp["Sigma_k"]
        mu_k = precomp["mu_k"]
        alpha_k = precomp["alpha_k"]
        mu_Sigma_inv_mu = precomp["mu_Sigma_inv_mu"]
        Sigma_k_logdet = precomp["Sigma_k_logdet"]

        xi_k_batch = L_k.T @ lambda_batch  # (parcel_size, batch_size)

        xi_Sigma_xi_batch = np.sum(
            xi_k_batch * (Sigma_k @ xi_k_batch), axis=0
        )
        F_nu_k_batch = 0.5 * (
            mu_Sigma_inv_mu + xi_Sigma_xi_batch + Sigma_k_logdet
        )

        exp_neg_F = np.exp(-F_nu_k_batch)
        alpha_k_updated_batch = alpha_k / (
            alpha_k + (1 - alpha_k) * exp_neg_F
        )

        Sigma_xi_batch = Sigma_k @ xi_k_batch
        mu_expanded = mu_k.reshape(-1, 1)

        j_k_batch = alpha_k_updated_batch.reshape(1, -1) * (
            mu_expanded + Sigma_xi_batch
        )

        J_batch[vertices, :] = j_k_batch

        if parcel_id == list(precomputed.keys())[0]:
            logger.debug(
                f"  Parcel {parcel_id}: alpha={alpha_k:.4f}, vertices={len(vertices)}"
            )
            logger.debug(
                f"  xi_k range: [{np.min(xi_k_batch):.2e}, {np.max(xi_k_batch):.2e}]"
            )
            logger.debug(
                f"  F_nu_k range: [{np.min(F_nu_k_batch):.2e}, {np.max(F_nu_k_batch):.2e}]"
            )
            logger.debug(
                f"  alpha_updated range: [{np.min(alpha_k_updated_batch):.4f}, {np.max(alpha_k_updated_batch):.4f}]"
            )
            logger.debug(
                f"  j_k range: [{np.min(j_k_batch):.2e}, {np.max(j_k_batch):.2e}]"
            )

    return J_batch


def _mem_dual_function_vectorized(lambda_vec, y_t, precomputed, Sigma_e_inv):
    """Vectorized MEM dual function computation using precomputed matrices."""
    data_term = np.dot(lambda_vec, y_t) + 0.5 * np.dot(
        lambda_vec, Sigma_e_inv @ lambda_vec
    )

    free_energy = 0.0

    for _parcel_id, precomp in precomputed.items():
        L_k = precomp["L_k"]
        Sigma_k = precomp["Sigma_k"]
        mu_Sigma_inv_mu = precomp["mu_Sigma_inv_mu"]
        alpha_k = precomp["alpha_k"]
        Sigma_k_logdet = precomp["Sigma_k_logdet"]

        xi_k = L_k.T @ lambda_vec

        xi_Sigma_xi = np.dot(xi_k, Sigma_k @ xi_k)
        F_nu_k = 0.5 * (mu_Sigma_inv_mu + xi_Sigma_xi + Sigma_k_logdet)

        exp_neg_F = np.exp(-F_nu_k)
        if exp_neg_F > 1e-300:
            parcel_free_energy = -np.log(alpha_k * exp_neg_F + (1 - alpha_k))
        else:
            parcel_free_energy = F_nu_k - np.log(alpha_k)

        free_energy += parcel_free_energy

    return -(data_term - free_energy)
