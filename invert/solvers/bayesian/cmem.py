import logging
import os
import time
import warnings
from collections import deque

import mne
import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata
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
        max_iter=50,
        batch_size=100,
        n_outer=3,
        alpha_tol=5e-2,
        **kwargs,
    ):
        self.name = name
        self.num_parcels = num_parcels
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.n_outer = n_outer
        self.alpha_tol = alpha_tol
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
            try:
                adjacency = mne.spatial_src_adjacency(self.forward["src"], verbose=0)
            except Exception:
                adjacency = None  # triggers KMeans fallback in _data_driven_parcellation
        self._adjacency = adjacency

        J, parcels = _cmem(
            data,
            self.leadfield,
            A=adjacency,
            num_parcels=self.num_parcels,
            max_iter=self.max_iter,
            batch_size=self.batch_size,
            n_outer=self.n_outer,
            alpha_tol=self.alpha_tol,
        )

        # Cache the result for immediate apply() on the same object.
        # cMEM is iterative and expensive; notebooks often do:
        #   make_inverse_operator(fwd, evoked); apply_inverse_operator(evoked)
        # so returning the cached solution avoids re-running the full solver.
        self._cmem_cached_obj = mne_obj
        self._cmem_cached_J = J

        self.parcels = parcels
        self.inverse_operators = [
            InverseOperator(J, self.name),
        ]
        return self

    def apply_inverse_operator(self, mne_obj):
        """Apply the cMEM inverse operator.

        Since cMEM computes the full source time series during
        ``make_inverse_operator``, applying the operator re-runs the
        algorithm on the new data unless the same object was used during
        ``make_inverse_operator`` (cached).

        Parameters
        ----------
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.

        Return
        ------
        stc : mne.SourceEstimate
            The source estimate.
        """
        J_cached = getattr(self, "_cmem_cached_J", None)
        obj_cached = getattr(self, "_cmem_cached_obj", None)
        if J_cached is not None and obj_cached is mne_obj:
            return self.source_to_object(J_cached)

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
            n_outer=self.n_outer,
            alpha_tol=self.alpha_tol,
        )

        self._cmem_cached_obj = mne_obj
        self._cmem_cached_J = J

        stc = self.source_to_object(J)
        return stc


# ---------------------------------------------------------------------------
# Private helper functions
# ---------------------------------------------------------------------------


def _cmem(
    Y,
    L,
    A=None,
    num_parcels=200,
    max_iter=100,
    batch_size=100,
    *,
    n_outer=5,
    alpha_tol=1e-2,
):
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

    timing = os.getenv("INVERT_CMEM_TIMING", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    t_total = time.perf_counter()
    if timing:
        # Also print: notebooks/scripts often don't configure logging handlers.
        print("cMEM timing enabled (set logging level to see logger output).")

        def _tlog(msg, *args):
            logger.info(msg, *args)
            try:
                print(msg % args)
            except Exception:
                # Fallback if formatting fails for any reason.
                print(msg, *args)

    else:

        def _tlog(msg, *args):
            return

    # If no explicit noise covariance whitening was provided upstream, cMEM's
    # whitened-noise assumption can be badly violated on colored-noise data
    # (including our SimulationGenerator). As a pragmatic default, apply a mild
    # shrinkage whitening based on the sample sensor covariance.
    auto_whiten = os.getenv("INVERT_CMEM_AUTO_WHITEN", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if auto_whiten and t >= 2:
        t_step = time.perf_counter()
        cov = (Y @ Y.T) / float(t)
        cov = 0.5 * (cov + cov.T)
        shrink = 0.05
        cov = (1.0 - shrink) * cov + shrink * np.diag(np.diag(cov))
        evals, evecs = np.linalg.eigh(cov)
        evals = np.maximum(evals, 1e-12 * float(np.max(evals)))
        W = (evecs / np.sqrt(evals)) @ evecs.T  # invsqrt(cov)
        Y = W @ Y
        L = W @ L
        if timing:
            _tlog("cMEM timing: auto-whiten %.3fs", time.perf_counter() - t_step)

    # Optional: project to a low-dimensional sensor subspace to reduce the
    # dual optimization cost (λ lives in sensor space). This is particularly
    # important for large source spaces where repeated L.T@λ and L@w dominate.
    # Sensor-subspace projection is an *approximation* (restricts λ to a
    # low-dimensional subspace). Keep it opt-in to preserve cMEM accuracy
    # by default; enable via INVERT_CMEM_SENSOR_RANK="auto" or an int.
    sensor_rank = os.getenv("INVERT_CMEM_SENSOR_RANK", "full").strip().lower()
    if sensor_rank not in {"0", "none", "false", "off", "full"} and m > 1:
        if sensor_rank == "auto":
            # Heuristic: keep singular vectors above a noise-floor proxy,
            # with conservative bounds to avoid over-aggressive truncation.
            U_s, s_s, _ = np.linalg.svd(Y, full_matrices=False)
            noise = float(np.median(s_s)) if s_s.size else 0.0
            r0 = int(np.sum(s_s > (1.1 * noise)))
            r = int(np.clip(r0, 20, 80))
            r = min(r, m)
        else:
            try:
                r = int(sensor_rank)
            except Exception as err:
                raise ValueError(
                    "INVERT_CMEM_SENSOR_RANK must be 'auto', 'full', or an int, "
                    f"got {sensor_rank!r}"
                ) from err
            r = int(np.clip(r, 1, m))

        # Ensure r is valid for the reduced SVD size k=min(m, t).
        r = min(r, min(m, t))
        if 1 <= r < m:
            t_step = time.perf_counter()
            # U_s are left singular vectors of Y: use them as an orthonormal basis.
            # If auto mode did not compute them (manual rank), compute now.
            if sensor_rank != "auto":
                U_s, _s, _ = np.linalg.svd(Y, full_matrices=False)
            basis = U_s[:, :r]
            Y = basis.T @ Y  # (r, t)
            L = basis.T @ L  # (r, n)
            m = r
            if timing:
                _tlog(
                    "cMEM timing: sensor subspace m->%d in %.3fs",
                    r,
                    time.perf_counter() - t_step,
                )

    # Ensure parcels are not too small.
    # For smooth/patch-like sources (e.g. SimulationGenerator order>=1), parcels of
    # ~10 vertices tend to fragment true sources, hurting accuracy and increasing
    # the number of parcel terms in the MEM dual. Prefer larger parcels by default.
    min_vertices_per_parcel = 60
    num_parcels = min(num_parcels, max(5, n // min_vertices_per_parcel))
    num_parcels = max(num_parcels, 1)

    # Step 1: Regularized MNE estimate for robust initialization
    t_step = time.perf_counter()
    # Avoid forming (n, n) matrices (LtL) which is prohibitive when n_sources >> n_sensors.
    # Use the ridge/MNE identity:
    #   (L^T L + reg I)^{-1} L^T Y = L^T (L L^T + reg I)^{-1} Y
    # This preserves the estimate (up to numerical precision) while reducing the solve to (m, m).
    fro2 = float(np.sum(L * L))  # trace(L^T L)
    reg = 0.1 * fro2 / n
    LLt = L @ L.T
    LLt.flat[:: m + 1] += reg
    J_mne = L.T @ np.linalg.solve(LLt, Y)
    J_mne_power = np.mean(J_mne**2, axis=1)  # per-source RMS power
    if timing:
        _tlog("cMEM timing: MNE init %.3fs", time.perf_counter() - t_step)

    # Step 2: Data Driven Parcellation using MSP + region growing
    logger.info("Performing Data Driven Parcellation...")
    t_step = time.perf_counter()
    msp_all = _compute_msp_coefficients(Y, L)
    parcels = _data_driven_parcellation(Y, L, num_parcels, A=A, msp_scores=msp_all)
    if timing:
        _tlog(
            "cMEM timing: parcellation %.3fs (m=%d n=%d t=%d parcels=%d)",
            time.perf_counter() - t_step,
            m,
            n,
            t,
            int(np.unique(parcels).size),
        )

    # Step 3: Initialize parcel parameters
    logger.info("Initializing parcel parameters...")
    t_step = time.perf_counter()
    unique_parcels = np.unique(parcels)

    # Rank parcels by mean MNE power for alpha initialization
    parcel_powers = {}
    for parcel_id in unique_parcels:
        verts = np.where(parcels == parcel_id)[0]
        parcel_powers[parcel_id] = np.mean(J_mne_power[verts])
    power_vals = np.array([parcel_powers[p] for p in unique_parcels])
    # Use tie-aware ranks: at low SNR many parcels can have ~equal power.
    # A plain argsort(argsort(.)) breaks ties arbitrarily, which leads to
    # essentially random alpha assignments and can hurt accuracy.
    power_ranks = rankdata(power_vals, method="average").astype(float) - 1.0
    power_quantiles = power_ranks / max(len(power_ranks) - 1, 1)
    parcel_quantile = dict(zip(unique_parcels, power_quantiles, strict=True))

    # Robust Sigma calibration reference.
    #
    # Using a single time point here makes the Sigma scaling extremely sensitive
    # to noise/outliers (especially when the chosen time point happens to be
    # noise-dominated). Instead, use a small set of high-energy samples and a
    # robust aggregation (median), then scale based on a high percentile across
    # parcels.
    n_ref = min(10, t)
    if n_ref == 1:
        ref_idx = np.array([0], dtype=int)
    else:
        time_power = np.sum(Y * Y, axis=0)
        ref_idx = np.argsort(time_power)[-n_ref:]
    Y_ref = np.asarray(Y[:, ref_idx], dtype=float, order="F")
    # Scale out amplitude: Sigma calibration should reflect geometry, not the
    # absolute momentary signal/noise magnitude at a particular time sample.
    y_norm = np.linalg.norm(Y_ref, axis=0, keepdims=True)
    Y_ref = Y_ref / np.maximum(y_norm, 1e-30)
    Xi_ref = L.T @ Y_ref  # (n_sources, n_ref)

    precomputed = []
    E_refs = []  # per-parcel reference energies for calibration
    sigma_rank_max = int(os.getenv("INVERT_CMEM_SIGMA_RANK_MAX", "0") or "0")
    sigma_rank_max = max(0, sigma_rank_max)
    sigma_energy = float(os.getenv("INVERT_CMEM_SIGMA_ENERGY", "0.995") or "0.995")
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
        Sigma_k, Sigma_basis = _compress_sigma_matrix(
            Sigma_k, rank_max=sigma_rank_max, energy=sigma_energy
        )

        # Estimate E_k scale: E_k = 0.5 * xi^T Sigma xi for reference lambda
        # Use a few high-energy samples and a robust aggregation.
        xi_ref = Xi_ref[verts, :]  # (n_v, n_ref)
        Sigma_xi_ref = _apply_sigma(Sigma_k, Sigma_basis, xi_ref)
        E_ref = 0.5 * np.median(np.sum(xi_ref * Sigma_xi_ref, axis=0))
        E_refs.append(float(E_ref))

        # Alpha from power rank: quadratic mapping for sparsity bias
        q = parcel_quantile[parcel_id]
        alpha_k = np.clip(0.01 + 0.98 * q**2, 0.01, 0.99)

        precomputed.append(
            {
                "parcel_id": int(parcel_id),
                "Sigma_k": Sigma_k,
                "Sigma_basis": Sigma_basis,
                "mu_k": mu_k,
                "alpha_k": alpha_k,
                "vertices": verts,
            }
        )

    # Calibrate Sigma scale so E_k hits a reasonable dynamic range.
    # Without this, E_k >> 1 saturates exp(E_k) and disables alpha.
    # Adapt the target based on a simple SNR proxy: at low SNR we want smaller
    # E_k so the mixture weights stay closer to alpha (avoid noise-driven
    # saturation), while at high SNR we allow more decisive activations.
    svals = np.linalg.svd(Y, full_matrices=False, compute_uv=False)
    noise_s = float(np.median(svals)) if svals.size else 0.0
    snr_proxy = float(svals[0] / max(noise_s, 1e-30)) if svals.size else 1.0
    if snr_proxy < 2.0:
        target_E = 3.0
    elif snr_proxy < 4.0:
        target_E = 6.0
    else:
        target_E = 10.0
    # Use a high percentile rather than max to reduce sensitivity to outliers.
    E_scale = float(np.percentile(E_refs, 95)) if len(E_refs) else 0.0
    if E_scale > 1e-30:
        sigma_scale = target_E / E_scale
        for precomp in precomputed:
            precomp["Sigma_k"] *= sigma_scale
            if precomp["Sigma_basis"] is not None:
                precomp["Sigma_basis"] *= np.sqrt(sigma_scale)
    if timing:
        median_parcel_size = (
            int(np.median([len(p["vertices"]) for p in precomputed]))
            if precomputed
            else 0
        )
        _tlog(
            "cMEM timing: precompute %.3fs (n_parcels=%d, median parcel size=%d)",
            time.perf_counter() - t_step,
            len(precomputed),
            median_parcel_size,
        )

    # Step 4: Iterative MEM optimization with alpha/eta updates
    # Outer EM loop: optimize lambda (E-step), update alpha/eta (M-step)
    logger.info("Running MEM optimization (%d outer iterations)...", n_outer)

    # Warm-start across time points and outer iterations.
    lambda_prev = np.zeros((m, t))

    for outer in range(n_outer):
        J = np.zeros((n, t))
        num_batches = (t + batch_size - 1) // batch_size

        t_outer = time.perf_counter()
        for batch_idx in tqdm(
            range(num_batches), desc=f"MEM iter {outer + 1}/{n_outer}", disable=True
        ):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, t)
            Y_batch = Y[:, batch_start:batch_end]

            lambda_init = lambda_prev[:, batch_start:batch_end]
            t_opt = time.perf_counter()
            lambda_batch = _optimize_lambda_batch(
                Y_batch, L, precomputed, max_iter=max_iter, lambda_init=lambda_init
            )
            lambda_prev[:, batch_start:batch_end] = lambda_batch
            J_batch = _compute_sources_batch(lambda_batch, L, precomputed, n)
            J[:, batch_start:batch_end] = J_batch
            if timing:
                logger.debug(
                    "cMEM timing: outer=%d batch=%d/%d optimize+J %.3fs",
                    outer + 1,
                    batch_idx + 1,
                    num_batches,
                    time.perf_counter() - t_opt,
                )

        if outer < n_outer - 1:
            max_alpha_change = _update_parcel_params(J, precomputed)
            # Early stop if alpha stabilizes: saves expensive re-optimizations.
            if max_alpha_change < float(alpha_tol):
                if timing:
                    _tlog(
                        "cMEM timing: early stop at outer %d (max Δalpha=%.3g)",
                        outer + 1,
                        max_alpha_change,
                    )
                break
        if timing:
            _tlog(
                "cMEM timing: outer %d total %.3fs",
                outer + 1,
                time.perf_counter() - t_outer,
            )

    if timing:
        _tlog("cMEM timing: total %.3fs", time.perf_counter() - t_total)
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

    # Region growing on mesh, seeded by descending MSP score.
    #
    # Important: grow *exactly* `num_parcels` spatially contiguous parcels.
    # A previous greedy "fill target_size then start a new parcel" approach can
    # fragment the mesh when seeds are scattered, producing far more parcels
    # than requested (many tiny parcels), which slows MEM optimization and
    # hurts spatial coherence.
    if hasattr(A, "tocsr"):
        A_csr = A.tocsr()
    else:
        from scipy.sparse import csr_matrix

        A_csr = csr_matrix(A)

    parcels = np.full(n, -1, dtype=int)
    seeds = np.argsort(msp_scores)[::-1][:num_parcels]

    # Multi-source BFS: simultaneous growth from all seeds.
    queue = deque()
    for parcel_id, seed in enumerate(seeds):
        parcels[seed] = parcel_id
        queue.append(seed)

    while queue:
        node = queue.popleft()
        parcel_id = parcels[node]
        start, end = A_csr.indptr[node], A_csr.indptr[node + 1]
        for nb in A_csr.indices[start:end]:
            if parcels[nb] < 0:
                parcels[nb] = parcel_id
                queue.append(nb)

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
    # Subspace dimension selection matters a lot at low SNR.
    # Using a near-1 cumulative-variance threshold can include mostly-noise
    # singular vectors, making MSP scores nearly flat (and parcellation unstable).
    #
    # Use a simple, noise-robust rule: keep components clearly above the median
    # singular value (noise floor proxy), capped to keep MSP discriminative.
    noise = float(np.median(s)) if s.size else 0.0
    r = int(np.sum(s > (1.2 * noise)))
    r = int(np.clip(r, 1, min(int(Y.shape[0]), 10)))
    U_bar = U[:, :r]

    norms = np.linalg.norm(L, axis=0, keepdims=True)
    norms = np.maximum(norms, 1e-30)
    L_bar = L / norms

    proj = U_bar.T @ L_bar  # (r, n_sources)
    return np.sum(proj**2, axis=0)


def _compute_spatial_covariance(vertices, A, alpha=0.05):
    """Compute spatial covariance from graph Laplacian: inv(L_graph + alpha*I)."""
    n_v = len(vertices)

    # Convert to CSR for fancy indexing (COO doesn't support it in older scipy)
    A_csr = A.tocsr() if hasattr(A, "tocsr") else A
    A_sub = A_csr[np.ix_(vertices, vertices)]
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


def _compress_sigma_matrix(
    Sigma,
    *,
    energy=0.995,
    rank_max=24,
    min_size=40,
):
    """Optionally compress parcel covariance with a low-rank PSD factor."""
    Sigma = np.asarray(Sigma, dtype=float)
    n_v = int(Sigma.shape[0])
    if int(rank_max) <= 0 or n_v < int(min_size):
        return Sigma, None

    Sigma = 0.5 * (Sigma + Sigma.T)
    evals, evecs = np.linalg.eigh(Sigma)
    evals = np.maximum(evals, 0.0)
    total = float(np.sum(evals))
    if not np.isfinite(total) or total <= 1e-30:
        return Sigma, None

    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    target = float(energy) * total
    r = int(np.searchsorted(np.cumsum(evals), target) + 1)
    r = int(np.clip(r, 1, min(int(rank_max), n_v)))
    if r >= int(0.8 * n_v):
        return Sigma, None

    basis = evecs[:, :r] * np.sqrt(evals[:r])[np.newaxis, :]
    return Sigma, basis


def _apply_sigma(Sigma_k, Sigma_basis, x):
    """Apply parcel covariance to vector/matrix x."""
    if Sigma_basis is None:
        return Sigma_k @ x
    return Sigma_basis @ (Sigma_basis.T @ x)


def _update_parcel_params(J, precomputed):
    """Update alpha from posterior source estimates (M-step).

    After solving MEM with current parameters, use the estimated source
    power to sharpen activation probabilities.  Sigma is kept fixed
    (already calibrated in the initialization).
    """
    J_power = np.mean(J**2, axis=1)

    # Rank parcels by posterior power
    parcel_powers = []
    for precomp in precomputed:
        verts = precomp["vertices"]
        parcel_powers.append(np.mean(J_power[verts]))
    power_arr = np.array(parcel_powers)
    ranks = rankdata(power_arr, method="average").astype(float) - 1.0
    quantiles = ranks / max(len(ranks) - 1, 1)

    max_change = 0.0
    for i, precomp in enumerate(precomputed):
        q = quantiles[i]
        new_alpha = float(np.clip(0.01 + 0.98 * q**2, 0.01, 0.99))
        max_change = max(max_change, abs(new_alpha - float(precomp["alpha_k"])))
        precomp["alpha_k"] = new_alpha

    return float(max_change)


def _mem_dual_obj_grad(lambda_vec, y_t, L, precomputed, w_full):
    """MEM dual objective and gradient (minimization form).

    Minimize: D(lambda) = -lambda^T y + 0.5*||lambda||^2
              + sum_k log[(1-alpha_k) + alpha_k*exp(E_k)]
    where E_k = mu_k^T xi_k + 0.5*xi_k^T Sigma_k xi_k
    and xi_k = (L[:, vertices_k]^T) lambda.
    """
    obj = -np.dot(lambda_vec, y_t) + 0.5 * np.dot(lambda_vec, lambda_vec)
    grad = -y_t + lambda_vec.copy()

    # Parcels form a partition of the source space, so summing L_k @ v_k over parcels
    # is equivalent to a single L @ v_full. Computing xi_full = L.T @ lambda once and
    # accumulating v_full avoids many small BLAS calls inside the optimizer.
    xi_full = L.T @ lambda_vec
    w_full.fill(0.0)

    for precomp in precomputed:
        Sigma_k = precomp["Sigma_k"]
        Sigma_basis = precomp.get("Sigma_basis")
        mu_k = precomp["mu_k"]
        alpha_k = precomp["alpha_k"]
        vertices = precomp["vertices"]

        xi_k = xi_full[vertices]
        Sigma_xi = _apply_sigma(Sigma_k, Sigma_basis, xi_k)
        E_k = np.dot(mu_k, xi_k) + 0.5 * np.dot(xi_k, Sigma_xi)

        # Numerically stable computation
        E_k_safe = min(float(E_k), 500.0)
        exp_E = np.exp(E_k_safe)
        denom = (1.0 - alpha_k) + alpha_k * exp_E
        p_k = alpha_k * exp_E / denom

        obj += np.log(denom)
        w_full[vertices] = p_k * (mu_k + Sigma_xi)

    grad += L @ w_full

    return obj, grad


def _optimize_lambda_batch(Y_batch, L, precomputed, max_iter=100, lambda_init=None):
    """Optimize Lagrange multipliers for a batch of time points.

    cMEM's dual decouples over time points, but calling SciPy's optimizer once
    per column can be very slow (Python overhead dominates). As a fast default,
    use a damped fixed-point iteration derived from the stationarity condition
    ∇D(λ)=0:

        λ = y - L @ w(λ)

    where w(λ) is the parcel-wise posterior mean term.

    Set `INVERT_CMEM_LAMBDA_OPT=fixed_point` to force the fast fixed-point solver.
    """
    method = os.getenv("INVERT_CMEM_LAMBDA_OPT", "auto").strip().lower()
    if method == "auto":
        # Use the stable optimizer on tiny problems (fixed-point can diverge
        # when the dual is poorly conditioned); use fixed-point otherwise.
        m = int(L.shape[0])
        batch = int(Y_batch.shape[1])
        method = "lbfgs" if (m <= 32 and batch <= 32) else "fixed_point"

    if method in {"fixed_point", "fp"}:
        return _optimize_lambda_batch_fixed_point(
            Y_batch, L, precomputed, max_iter=max_iter, lambda_init=lambda_init
        )
    return _optimize_lambda_batch_lbfgs(
        Y_batch, L, precomputed, max_iter=max_iter, lambda_init=lambda_init
    )


def _optimize_lambda_batch_fixed_point(
    Y_batch, L, precomputed, max_iter=100, lambda_init=None
):
    """Vectorized damped fixed-point solver for λ over a batch of time points."""
    m, batch_size = Y_batch.shape
    n = L.shape[1]

    if lambda_init is None:
        lambda_batch = np.zeros((m, batch_size))
    else:
        lambda_batch = np.array(lambda_init, dtype=float, copy=True)

    # Damping and tolerance: conservative defaults; we fall back to L-BFGS-B
    # if the fixed-point iteration becomes unstable.
    omega = 0.2
    tol = 1e-4
    anderson_m = int(os.getenv("INVERT_CMEM_ANDERSON_M", "5") or "0")
    anderson_m = int(np.clip(anderson_m, 0, 10))
    if n >= 3000:
        anderson_m = min(anderson_m, 3)

    xi_full = np.empty((n, batch_size))
    w_full = np.empty((n, batch_size))

    y_norm = float(np.linalg.norm(Y_batch)) + 1e-30

    max_lambda_norm = 1e6 * y_norm

    # Anderson acceleration history (store x and residual r=F(x)-x).
    # We keep m_hist+1 entries to build up to m_hist differences.
    x_hist: list[np.ndarray] = []
    r_hist: list[np.ndarray] = []
    dim = int(m * batch_size)

    for _ in range(int(max_iter)):
        # xi_full = L.T @ lambda_batch
        xi_full[:] = L.T @ lambda_batch
        w_full.fill(0.0)

        for precomp in precomputed:
            vertices = precomp["vertices"]
            Sigma_k = precomp["Sigma_k"]
            Sigma_basis = precomp.get("Sigma_basis")
            mu_k = precomp["mu_k"]
            alpha_k = float(precomp["alpha_k"])

            xi_k = xi_full[vertices, :]  # (n_v, batch)
            Sigma_xi = _apply_sigma(Sigma_k, Sigma_basis, xi_k)  # (n_v, batch)
            E_k = mu_k @ xi_k + 0.5 * np.sum(xi_k * Sigma_xi, axis=0)  # (batch,)
            E_k = np.clip(E_k, -500, 500)
            exp_E = np.exp(E_k)
            p_k = alpha_k * exp_E / ((1.0 - alpha_k) + alpha_k * exp_E)  # (batch,)

            w_full[vertices, :] = p_k[np.newaxis, :] * (
                mu_k[:, np.newaxis] + Sigma_xi
            )

        lambda_target = Y_batch - (L @ w_full)  # stationarity condition
        residual = lambda_target - lambda_batch

        # Simple stability check + backtracking on omega.
        res_norm = float(np.linalg.norm(residual))
        if not np.isfinite(res_norm):
            return _optimize_lambda_batch_lbfgs(
                Y_batch, L, precomputed, max_iter=max_iter, lambda_init=lambda_batch
            )

        # Anderson acceleration candidate (optional).
        candidate_target = lambda_target
        if anderson_m > 0:
            x_hist.append(lambda_batch.copy())
            r_hist.append(residual.copy())
            if len(x_hist) > anderson_m + 1:
                x_hist.pop(0)
                r_hist.pop(0)

            k = min(anderson_m, len(x_hist) - 1)
            if k >= 1:
                try:
                    dR = np.empty((dim, k))
                    dX = np.empty((dim, k))
                    for j in range(k):
                        dR[:, j] = (r_hist[-(j + 1)] - r_hist[-(j + 2)]).ravel()
                        dX[:, j] = (x_hist[-(j + 1)] - x_hist[-(j + 2)]).ravel()

                    gamma, *_ = np.linalg.lstsq(dR, r_hist[-1].ravel(), rcond=None)
                    if np.isfinite(gamma).all():
                        candidate = lambda_target.ravel() - dX @ gamma
                        candidate = candidate.reshape(m, batch_size)
                        if np.isfinite(candidate).all():
                            candidate_target = candidate
                except Exception:
                    candidate_target = lambda_target

        delta = candidate_target - lambda_batch

        omega_try = omega
        accepted = False
        for _ls in range(6):
            candidate = lambda_batch + omega_try * delta
            cand_norm = float(np.linalg.norm(candidate))
            if np.isfinite(cand_norm) and cand_norm < max_lambda_norm:
                lambda_batch = candidate
                accepted = True
                break
            omega_try *= 0.5

        if not accepted:
            return _optimize_lambda_batch_lbfgs(
                Y_batch, L, precomputed, max_iter=max_iter, lambda_init=lambda_batch
            )

        if res_norm / y_norm < tol:
            break

    return lambda_batch


def _optimize_lambda_batch_lbfgs(Y_batch, L, precomputed, max_iter=100, lambda_init=None):
    """Original SciPy L-BFGS-B optimizer (slow, but accurate)."""
    m, batch_size = Y_batch.shape
    lambda_batch = np.zeros((m, batch_size))
    n = L.shape[1]

    for i in range(batch_size):
        y_t = Y_batch[:, i]
        if lambda_init is not None:
            x0 = lambda_init[:, i]
            if not np.any(x0):
                # Better default than zeros: the quadratic part alone is minimized at λ=y.
                x0 = y_t
        elif i > 0:
            x0 = lambda_batch[:, i - 1]
        else:
            x0 = y_t

        w_full = np.empty(n)

        def obj_grad(lam, y_t=y_t, w_full=w_full):
            return _mem_dual_obj_grad(lam, y_t, L, precomputed, w_full)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                obj_grad,
                x0,
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": max_iter, "maxfun": int(max_iter) * 25},
            )

        lambda_batch[:, i] = result.x

    return lambda_batch


def _compute_sources_batch(lambda_batch, L, precomputed, n):
    """Compute source estimates from optimized Lagrange multipliers.

    J_k = p_k * (mu_k + Sigma_k @ xi_k)
    where p_k = alpha_k*exp(E_k) / [(1-alpha_k) + alpha_k*exp(E_k)]
    """
    _, batch_size = lambda_batch.shape
    J_batch = np.zeros((n, batch_size))

    xi_full = L.T @ lambda_batch  # (n, batch)

    for precomp in precomputed:
        vertices = precomp["vertices"]
        Sigma_k = precomp["Sigma_k"]
        Sigma_basis = precomp.get("Sigma_basis")
        mu_k = precomp["mu_k"]
        alpha_k = precomp["alpha_k"]

        xi_k = xi_full[vertices, :]  # (n_v, batch)
        Sigma_xi = _apply_sigma(Sigma_k, Sigma_basis, xi_k)  # (n_v, batch)

        # E_k per time point
        E_k = mu_k @ xi_k + 0.5 * np.sum(xi_k * Sigma_xi, axis=0)
        E_k = np.clip(E_k, -500, 500)

        # Posterior activation probability
        exp_E = np.exp(E_k)
        p_k = alpha_k * exp_E / ((1.0 - alpha_k) + alpha_k * exp_E)

        # Source estimate
        J_batch[vertices, :] = p_k[np.newaxis, :] * (mu_k[:, np.newaxis] + Sigma_xi)

    return J_batch
