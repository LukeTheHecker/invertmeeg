import logging
import warnings
from collections import deque

import mne
import numpy as np
from scipy.optimize import minimize
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
            "Amblard, C., Lapalme, E., & Bhatt, P. (2004). Biomagnetic source detection by maximum entropy and graphical models. IEEE Transactions on Biomedical Engineering, 51(3), 427-442.",
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
        mne_obj=None,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
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
        adjacency : scipy.sparse matrix, optional
            Source adjacency matrix. Computed from forward if not provided.
        positions : numpy.ndarray, optional
            Source positions (n, 3).

        Return
        ------
        self : object returns itself for convenience
        """
        super().make_inverse_operator(forward, mne_obj, *args, alpha=alpha, **kwargs)
        wf = self.prepare_whitened_forward(noise_cov)
        data = self.unpack_data_obj(mne_obj)
        data = wf.sensor_transform @ data

        # Compute adjacency from forward model if not provided
        if adjacency is None:
            adjacency = mne.spatial_src_adjacency(self.forward["src"], verbose=0)
        self._adjacency = adjacency

        J, parcels = _cmem(
            data,
            self.leadfield,
            A=adjacency,
            num_parcels=self.num_parcels,
            max_iter=self.max_iter,
            batch_size=self.batch_size,
        )

        self.parcels = parcels
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
        # Skip validate_operator_data_compatibility: cMEM stores the full
        # solution in InverseOperator (not an operator matrix), so the
        # dimension check would fail.  Re-running _cmem below is the
        # intended application path.
        data = self._sensor_transform @ data

        A = getattr(self, "_adjacency", None)

        J, self.parcels = _cmem(
            data,
            self.leadfield,
            A=A,
            num_parcels=self.num_parcels,
            max_iter=self.max_iter,
            batch_size=self.batch_size,
        )

        stc = self.source_to_object(J)
        return stc


# ---------------------------------------------------------------------------
# Private helper functions
# ---------------------------------------------------------------------------


def _cmem(Y, L, A=None, num_parcels=200, max_iter=100, batch_size=100):
    """Coherent Maximum Entropy on the Mean (cMEM) source localization.

    Assumes whitened data (noise cov = identity).

    Parameters
    ----------
    Y : array (m, t) - whitened EEG/MEG data
    L : array (m, n) - whitened lead field matrix
    A : sparse matrix (n, n) - source adjacency matrix (optional)
    num_parcels : int - number of parcels for DDP
    max_iter : int - maximum optimization iterations
    batch_size : int - batch size for time processing

    Returns
    -------
    J : array (n, t) - source time series
    parcels : array (n,) - parcel assignment for each source
    """
    m, t = Y.shape
    m_l, n = L.shape
    assert m == m_l, "Dimension mismatch between Y and L"

    # Ensure parcels have at least ~10 sources for spatial coherence
    num_parcels = min(num_parcels, max(20, n // 10))
    num_parcels = max(num_parcels, 1)

    # Step 1: Data Driven Parcellation using MSP + region growing
    logger.info("Performing Data Driven Parcellation...")
    msp_all = _compute_msp_coefficients(Y, L)
    parcels = _data_driven_parcellation(Y, L, num_parcels, A=A, msp_scores=msp_all)

    # Step 2: Regularized MNE estimate for energy scaling and alpha init
    LtL = L.T @ L
    reg = 0.1 * np.trace(LtL) / n
    J_mne = np.linalg.solve(LtL + reg * np.eye(n), L.T @ Y)
    J_mne_power = np.mean(J_mne**2, axis=1)  # per-source RMS power

    # Step 3: Initialize parcel parameters
    logger.info("Initializing parcel parameters...")
    unique_parcels = np.unique(parcels)

    # Rank parcels by mean MNE power for alpha initialization
    parcel_powers = {}
    for parcel_id in unique_parcels:
        verts = np.where(parcels == parcel_id)[0]
        parcel_powers[parcel_id] = np.mean(J_mne_power[verts])
    power_vals = np.array([parcel_powers[p] for p in unique_parcels])
    power_ranks = np.argsort(np.argsort(power_vals)).astype(float)
    power_quantiles = power_ranks / max(len(power_ranks) - 1, 1)
    parcel_quantile = dict(zip(unique_parcels, power_quantiles, strict=True))

    precomputed = {}
    max_E_k = 0.0  # track for calibration
    for parcel_id in unique_parcels:
        verts = np.where(parcels == parcel_id)[0]
        n_v = len(verts)

        mu_k = np.zeros(n_v)

        # Spatial smoothness prior from graph Laplacian
        if A is not None:
            W_k = _compute_spatial_covariance(verts, A)
        else:
            W_k = np.eye(n_v)
        # Normalize W_k to have per-source trace = 1
        W_k *= n_v / (np.trace(W_k) + 1e-30)

        # Per-source prior variance from MNE power, with spatial structure
        parcel_power = parcel_powers[parcel_id]
        Sigma_k = parcel_power * W_k

        L_k = L[:, verts]

        # Estimate E_k scale: E_k = 0.5 * xi^T Sigma xi for reference lambda
        # Use y as reference lambda direction (near-optimal for small Sigma)
        xi_ref = L_k.T @ Y[:, 0]
        E_ref = 0.5 * np.dot(xi_ref, Sigma_k @ xi_ref)
        max_E_k = max(max_E_k, E_ref)

        # Alpha from power rank: quadratic mapping for sparsity bias
        q = parcel_quantile[parcel_id]
        alpha_k = np.clip(0.01 + 0.98 * q**2, 0.01, 0.99)

        precomputed[parcel_id] = {
            "L_k": L_k,
            "Sigma_k": Sigma_k,
            "mu_k": mu_k,
            "alpha_k": alpha_k,
            "vertices": verts,
        }

    # Calibrate Sigma scale so max E_k ≈ target at the optimum.
    # Without this, E_k >> 1 saturates exp(E_k) and disables alpha.
    target_E = 10.0
    if max_E_k > 1e-30:
        sigma_scale = target_E / max_E_k
        for precomp in precomputed.values():
            precomp["Sigma_k"] *= sigma_scale

    # Step 4: Iterative MEM optimization with alpha/eta updates
    # Outer EM loop: optimize lambda (E-step), update alpha/eta (M-step)
    n_outer = 5
    logger.info("Running MEM optimization (%d outer iterations)...", n_outer)

    for outer in range(n_outer):
        J = np.zeros((n, t))
        num_batches = (t + batch_size - 1) // batch_size

        for batch_idx in tqdm(
            range(num_batches), desc=f"MEM iter {outer + 1}/{n_outer}", disable=True
        ):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, t)
            Y_batch = Y[:, batch_start:batch_end]

            lambda_batch = _optimize_lambda_batch(Y_batch, precomputed, max_iter)
            J_batch = _compute_sources_batch(lambda_batch, precomputed, n)
            J[:, batch_start:batch_end] = J_batch

        if outer < n_outer - 1:
            _update_parcel_params(J, precomputed, L)

    return J, parcels


def _data_driven_parcellation(Y, L, num_parcels, A=None, msp_scores=None):
    """Data-driven parcellation using MSP scores and region growing on mesh.

    Seeds parcels from highest-MSP source, grows along mesh edges to form
    spatially contiguous regions. Falls back to K-Means when no adjacency
    is available.
    """
    if msp_scores is None:
        msp_scores = _compute_msp_coefficients(Y, L)
    n = len(msp_scores)
    num_parcels = min(num_parcels, n)

    if A is None:
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=num_parcels, random_state=42, n_init=10)
        return kmeans.fit_predict(msp_scores.reshape(-1, 1))

    # Region growing on mesh, seeded by descending MSP score
    if hasattr(A, "tocsr"):
        A_csr = A.tocsr()
    else:
        from scipy.sparse import csr_matrix

        A_csr = csr_matrix(A)

    parcels = np.full(n, -1, dtype=int)
    target_size = max(1, n // num_parcels)
    seeds = np.argsort(msp_scores)[::-1]

    parcel_id = 0
    for seed in seeds:
        if parcels[seed] >= 0:
            continue

        queue = deque([seed])
        parcels[seed] = parcel_id
        count = 1

        while queue and count < target_size:
            node = queue.popleft()
            start, end = A_csr.indptr[node], A_csr.indptr[node + 1]
            for nb in A_csr.indices[start:end]:
                if parcels[nb] < 0:
                    parcels[nb] = parcel_id
                    queue.append(nb)
                    count += 1
                    if count >= target_size:
                        break

        parcel_id += 1

    # Assign any remaining unassigned vertices to nearest assigned neighbor
    unassigned = np.where(parcels < 0)[0]
    for v in unassigned:
        start, end = A_csr.indptr[v], A_csr.indptr[v + 1]
        neighbors = A_csr.indices[start:end]
        assigned = neighbors[parcels[neighbors] >= 0]
        if len(assigned) > 0:
            parcels[v] = parcels[assigned[np.argmax(msp_scores[assigned])]]
        else:
            parcels[v] = 0

    return parcels


def _compute_msp_coefficients(Y, L):
    """Multivariate Source Prelocalization (Mattout et al. 2006).

    MSP(i) = ||U_bar^T g_hat_i||^2 where U_bar is the signal subspace
    from SVD of Y and g_hat_i is the normalized leadfield column i.
    Returns values in [0, 1].
    """
    U, s, _ = np.linalg.svd(Y, full_matrices=False)
    cumvar = np.cumsum(s**2) / np.sum(s**2)
    r = max(1, np.searchsorted(cumvar, 0.99) + 1)
    U_bar = U[:, :r]

    norms = np.linalg.norm(L, axis=0, keepdims=True)
    norms = np.maximum(norms, 1e-30)
    L_bar = L / norms

    proj = U_bar.T @ L_bar  # (r, n_sources)
    return np.sum(proj**2, axis=0)


def _compute_spatial_covariance(vertices, A, alpha=0.2):
    """Compute spatial covariance from graph Laplacian: inv(L_graph + alpha*I)."""
    n_v = len(vertices)

    A_sub = A[np.ix_(vertices, vertices)]
    if hasattr(A_sub, "toarray"):
        A_sub = A_sub.toarray()

    D = np.diag(np.sum(A_sub, axis=1))
    L_graph = D - A_sub
    L_reg = L_graph + alpha * np.eye(n_v)

    try:
        Sigma = np.linalg.inv(L_reg)
    except np.linalg.LinAlgError:
        Sigma = np.eye(n_v)

    return Sigma


def _update_parcel_params(J, precomputed, L):
    """Update alpha from posterior source estimates (M-step).

    After solving MEM with current parameters, use the estimated source
    power to sharpen activation probabilities.  Sigma is kept fixed
    (already calibrated in the initialization).
    """
    J_power = np.mean(J**2, axis=1)

    # Rank parcels by posterior power
    parcel_powers = []
    for precomp in precomputed.values():
        verts = precomp["vertices"]
        parcel_powers.append(np.mean(J_power[verts]))
    power_arr = np.array(parcel_powers)
    ranks = np.argsort(np.argsort(power_arr)).astype(float)
    quantiles = ranks / max(len(ranks) - 1, 1)

    for i, precomp in enumerate(precomputed.values()):
        q = quantiles[i]
        precomp["alpha_k"] = np.clip(0.01 + 0.98 * q**2, 0.01, 0.99)


def _mem_dual_obj_grad(lambda_vec, y_t, precomputed):
    """MEM dual objective and gradient (minimization form).

    Minimize: D(lambda) = -lambda^T y + 0.5*||lambda||^2
              + sum_k log[(1-alpha_k) + alpha_k*exp(E_k)]
    where E_k = mu_k^T xi_k + 0.5*xi_k^T Sigma_k xi_k
    and xi_k = L_k^T lambda.
    """
    obj = -np.dot(lambda_vec, y_t) + 0.5 * np.dot(lambda_vec, lambda_vec)
    grad = -y_t + lambda_vec.copy()

    for precomp in precomputed.values():
        L_k = precomp["L_k"]
        Sigma_k = precomp["Sigma_k"]
        mu_k = precomp["mu_k"]
        alpha_k = precomp["alpha_k"]

        xi_k = L_k.T @ lambda_vec
        Sigma_xi = Sigma_k @ xi_k
        E_k = np.dot(mu_k, xi_k) + 0.5 * np.dot(xi_k, Sigma_xi)

        # Numerically stable computation
        E_k_safe = min(float(E_k), 500.0)
        exp_E = np.exp(E_k_safe)
        denom = (1.0 - alpha_k) + alpha_k * exp_E
        p_k = alpha_k * exp_E / denom

        obj += np.log(denom)
        grad += L_k @ (p_k * (mu_k + Sigma_xi))

    return obj, grad


def _optimize_lambda_batch(Y_batch, precomputed, max_iter=100):
    """Optimize Lagrange multipliers for a batch of time points."""
    m, batch_size = Y_batch.shape
    lambda_batch = np.zeros((m, batch_size))

    for i in range(batch_size):
        y_t = Y_batch[:, i]

        def obj_grad(lam, y_t=y_t):
            return _mem_dual_obj_grad(lam, y_t, precomputed)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                obj_grad,
                np.zeros(m),
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": max_iter},
            )

        lambda_batch[:, i] = result.x

    return lambda_batch


def _compute_sources_batch(lambda_batch, precomputed, n):
    """Compute source estimates from optimized Lagrange multipliers.

    J_k = p_k * (mu_k + Sigma_k @ xi_k)
    where p_k = alpha_k*exp(E_k) / [(1-alpha_k) + alpha_k*exp(E_k)]
    """
    _, batch_size = lambda_batch.shape
    J_batch = np.zeros((n, batch_size))

    for precomp in precomputed.values():
        vertices = precomp["vertices"]
        L_k = precomp["L_k"]
        Sigma_k = precomp["Sigma_k"]
        mu_k = precomp["mu_k"]
        alpha_k = precomp["alpha_k"]

        xi_k = L_k.T @ lambda_batch  # (n_v, batch)
        Sigma_xi = Sigma_k @ xi_k  # (n_v, batch)

        # E_k per time point
        E_k = mu_k @ xi_k + 0.5 * np.sum(xi_k * Sigma_xi, axis=0)
        E_k = np.clip(E_k, -500, 500)

        # Posterior activation probability
        exp_E = np.exp(E_k)
        p_k = alpha_k * exp_E / ((1.0 - alpha_k) + alpha_k * exp_E)

        # Source estimate
        J_batch[vertices, :] = p_k[np.newaxis, :] * (mu_k[:, np.newaxis] + Sigma_xi)

    return J_batch
