"""Mixed-Norm Estimate (MxNE) via Block Coordinate Descent with Active Set.

Implements the L21 mixed-norm inverse solver:

    min_X  0.5 * ||M - G X||_F^2  +  alpha * ||X||_{L21}

where ||X||_{L21} groups orientations per source position. The BCD solver
with active-set screening provides large speedups over naive FISTA when the
solution is sparse.

References
----------
Gramfort, A., Kowalski, M., & Hämäläinen, M. (2012). Mixed-norm estimates
for the M/EEG inverse problem using accelerated gradient methods. Physics
in Medicine and Biology, 57(7), 1937-1961.
"""

import logging

import mne
import numpy as np

from ..base import BaseSolver, SolverMeta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# L21 norm helpers
# ---------------------------------------------------------------------------


def _groups_norm2(X, n_orient):
    """Squared Frobenius norm per source position.

    Parameters
    ----------
    X : ndarray, shape (n_sources, ...) or (n_sources,)
    n_orient : int

    Returns
    -------
    gn2 : ndarray, shape (n_positions,)
    """
    if X.ndim == 1:
        X = X[:, np.newaxis]
    if n_orient == 1:
        return np.sum(X**2, axis=1)
    n_pos = X.shape[0] // n_orient
    return np.sum(X.reshape(n_pos, n_orient, -1) ** 2, axis=(1, 2))


def _norm_l2inf(X, n_orient):
    """L2-infinity norm (dual of L21): max block Frobenius norm."""
    gn2 = _groups_norm2(X, n_orient)
    return np.sqrt(np.max(gn2)) if gn2.size > 0 else 0.0


def _compute_lipschitz(G, n_orient):
    """Lipschitz constant per source position.

    n_orient=1: lc[j] = ||g_j||^2
    n_orient>1: lc[j] = largest singular value of G_j squared
    """
    if n_orient == 1:
        return np.sum(G**2, axis=0)
    n_positions = G.shape[1] // n_orient
    lc = np.empty(n_positions)
    for j in range(n_positions):
        s = j * n_orient
        lc[j] = np.linalg.norm(G[:, s : s + n_orient], ord=2) ** 2
    return lc


def _compute_depth_weights(G, n_orient, depth, depth_limit=10.0):
    """Depth weights from whitened leadfield column norms.

    Returns per-source weights w such that G_scaled = G / w equalizes
    sensitivity across source depths.

    Parameters
    ----------
    G : (n_sensors, n_sources)
    n_orient : int
    depth : float
        Exponent (0 = no weighting, 1 = full column normalization).
    depth_limit : float
        Maximum ratio between largest and smallest weight.

    Returns
    -------
    w : (n_sources,)  – multiplicative weights (>= 1/depth_limit).
    """
    n_sources = G.shape[1]
    n_positions = n_sources // n_orient

    if n_orient == 1:
        col_norms = np.sqrt(np.sum(G**2, axis=0))
    else:
        col_norms = np.empty(n_positions)
        for j in range(n_positions):
            s = j * n_orient
            col_norms[j] = np.linalg.norm(G[:, s : s + n_orient])

    w_pos = col_norms**depth
    w_max = w_pos.max()
    if w_max > 0:
        w_pos /= w_max
    w_pos = np.maximum(w_pos, 1.0 / depth_limit)

    if n_orient == 1:
        return w_pos
    return np.repeat(w_pos, n_orient)


# ---------------------------------------------------------------------------
# Core BCD routines
# ---------------------------------------------------------------------------


def _bcd_pass(G, X, R, positions, lipschitz, n_orient, alpha):
    """One BCD pass with L21 block soft-thresholding.

    Modifies *X* and *R* in-place.

    Parameters
    ----------
    G : (n_sensors, n_sources)
    X : (n_sources, n_times)
    R : (n_sensors, n_times) – residual M - G @ X
    positions : iterable of int – position indices to process
    lipschitz : (n_positions,)
    n_orient : int
    alpha : float – effective regularization

    Returns
    -------
    nonzero : set of int – positions that survived thresholding
    """
    nonzero = set()
    for j in positions:
        s = j * n_orient
        e = s + n_orient
        G_j = G[:, s:e]
        lc_j = lipschitz[j]

        # Gradient step: X_j_new = X_j + (1/lc_j) * G_j^T R
        X_j_new = (G_j.T @ R) / lc_j

        X_j_old = X[s:e]
        has_old = np.any(X_j_old != 0)
        if has_old:
            R += G_j @ X_j_old  # restore: R becomes R_without_j
            X_j_new += X_j_old

        # Block soft-thresholding (L21 proximal)
        block_norm = np.sqrt(np.sum(X_j_new**2))
        threshold = alpha / lc_j

        if block_norm <= threshold:
            X[s:e] = 0.0
            # R already equals R_without_j
        else:
            X_j_new *= 1.0 - threshold / block_norm
            X[s:e] = X_j_new
            R -= G_j @ X_j_new
            nonzero.add(j)

    return nonzero


def _dgap_l21(G, X, R, alpha, n_orient):
    """Duality gap for the L21 problem.

    Returns (gap, primal_objective).
    """
    resid_sq = np.sum(R**2)
    l21 = np.sum(np.sqrt(_groups_norm2(X, n_orient)))
    p_obj = 0.5 * resid_sq + alpha * l21

    dual_norm = _norm_l2inf(G.T @ R, n_orient)
    scaling = min(1.0, alpha / max(dual_norm, 1e-30))

    d_obj = (scaling - 0.5 * scaling**2) * resid_sq
    if l21 > 0:
        d_obj += scaling * np.sum(R * (G @ X))

    return p_obj - d_obj, p_obj


def _debias(G, M, X, n_orient, max_iter=200, tol=1e-6):
    """FISTA debiasing: correct L1 amplitude shrinkage.

    Solves  min_D  0.5 ||M - G diag(D) X||_F^2   s.t. D >= 1
    """
    gn2 = _groups_norm2(X, n_orient)
    active_pos = np.where(gn2 > 0)[0]
    if len(active_pos) == 0:
        return X

    active_idx = np.concatenate(
        [np.arange(j * n_orient, (j + 1) * n_orient) for j in active_pos]
    )
    G_act = G[:, active_idx]
    X_act = X[active_idx]
    n_act = len(active_idx)

    # Lipschitz constant for the diagonal-scaling sub-problem
    x_norms = np.sqrt(np.sum(X_act**2, axis=1))
    lip = 1.1 * np.linalg.norm(G_act * x_norms, ord=2) ** 2
    lip = max(lip, 1e-30)

    D = np.ones(n_act)
    Y_d = D.copy()
    t = 1.0

    for _ in range(max_iter):
        D_old = D.copy()
        Res = M - G_act @ (Y_d[:, np.newaxis] * X_act)
        grad = -np.sum((G_act.T @ Res) * X_act, axis=1)
        D = Y_d - grad / lip

        # Project D >= 1, shared across orientations within each position
        if n_orient > 1:
            n_pos = n_act // n_orient
            D_avg = D.reshape(n_pos, n_orient).mean(axis=1)
            D = np.repeat(np.maximum(D_avg, 1.0), n_orient)
        else:
            D = np.maximum(D, 1.0)

        # FISTA acceleration
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t**2))
        Y_d = D + (t - 1.0) / t_new * (D - D_old)
        t = t_new

        if np.max(np.abs(D - D_old)) < tol:
            break

    X_out = X.copy()
    X_out[active_idx] *= D[:, np.newaxis]
    return X_out


# ---------------------------------------------------------------------------
# Solver class
# ---------------------------------------------------------------------------


class SolverMxNEBCD(BaseSolver):
    """Mixed-Norm Estimate via Block Coordinate Descent with Active Set.

    Solves the L21 mixed-norm problem::

        min_X  0.5 ||M - G X||_F^2  +  alpha * sum_j ||X_j||_F

    The L21 penalty enforces spatial sparsity (few active source positions)
    while each active source retains a free time course.  The BCD solver
    with active-set screening avoids processing the full source space,
    giving large speedups when the true source configuration is sparse.

    References
    ----------
    Gramfort, A., Kowalski, M., & Hämäläinen, M. (2012). Mixed-norm
    estimates for the M/EEG inverse problem using accelerated gradient
    methods. Physics in Medicine and Biology, 57(7), 1937-1961.
    """

    meta = SolverMeta(
        acronym="MxNE-BCD",
        full_name="Mixed-Norm Estimate (BCD with Active Set)",
        category="Sparse / Mixed Norm",
        description=(
            "L21 mixed-norm inverse via block coordinate descent with "
            "active-set screening and optional amplitude debiasing."
        ),
        references=[
            "Gramfort, A., Kowalski, M., & Hämäläinen, M. (2012). Mixed-norm "
            "estimates for the M/EEG inverse problem using accelerated gradient "
            "methods. Physics in Medicine and Biology, 57(7), 1937-1961.",
        ],
    )

    SUPPORTS_VECTOR_ORIENTATION = True

    def __init__(self, name="MxNE-BCD", **kwargs):
        self.name = name
        super().__init__(**kwargs)

    def make_inverse_operator(
        self,
        forward,
        *args,
        alpha=0.01,
        noise_cov: mne.Covariance | None = None,
        **kwargs,
    ):
        """Prepare whitened forward model.

        Parameters
        ----------
        forward : mne.Forward
        alpha : float
            Unused here (regularization is set in apply_inverse_operator).
        noise_cov : mne.Covariance | None
            Noise covariance for sensor whitening.
        """
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        self.prepare_whitened_forward(noise_cov)
        return self

    def apply_inverse_operator(
        self,
        mne_obj,
        alpha=0.2,
        maxit=3000,
        tol=1e-4,
        active_set_size=10,
        dgap_freq=10,
        bcd_maxit=200,
        debias=True,
        time_pca=True,
        center_data=True,
        depth=0.8,
        depth_limit=10.0,
    ) -> mne.SourceEstimate:
        """Apply the MxNE-BCD solver.

        Parameters
        ----------
        mne_obj : mne.Evoked | mne.Epochs | mne.io.Raw
        alpha : float
            Regularization as a fraction of alpha_max (0 = no penalty,
            1 = all-zero solution).  Typical range 0.05 – 0.5.
        maxit : int
            Maximum outer (active-set expansion) iterations.
        tol : float
            Relative duality-gap tolerance for convergence.
        active_set_size : int
            Number of source positions added per expansion step.
        dgap_freq : int
            Evaluate duality gap every *dgap_freq* BCD iterations.
        bcd_maxit : int
            Maximum BCD iterations per active-set solve.
        debias : bool
            Apply FISTA debiasing to correct L1 amplitude shrinkage.
        time_pca : bool
            Reduce temporal dimension via SVD when n_times > n_sensors.
        center_data : bool
            Subtract per-channel temporal mean before solving.
        depth : float
            Depth weighting exponent applied to leadfield column norms.
            Equalizes sensitivity across source depths (0 = off, 1 = full
            column normalization, 0.8 = MNE default).
        depth_limit : float
            Maximum ratio between largest and smallest depth weight.
        """
        data = self.unpack_data_obj(mne_obj)
        self.validate_operator_data_compatibility(data)
        data = self._sensor_transform @ data

        G = self.leadfield.copy()
        n_orient = int(getattr(self, "_n_orient", 1))
        n_sensors, n_sources = G.shape
        n_positions = n_sources // n_orient

        M = data.copy()
        if center_data:
            M -= M.mean(axis=0, keepdims=True)

        # ------------------------------------------------------------------
        # Depth weighting: normalize leadfield columns to equalize
        # sensitivity across source depths.
        # ------------------------------------------------------------------
        depth_w = None
        if depth > 0:
            depth_w = _compute_depth_weights(
                G,
                n_orient,
                depth,
                depth_limit,
            )
            G = G / depth_w[np.newaxis, :]

        # Time PCA: compress temporal dimension for efficiency
        Vh = None
        if time_pca and M.shape[1] > n_sensors:
            U, s, Vh = np.linalg.svd(M, full_matrices=False)
            rank = max(1, int(np.sum(s > s[0] * 1e-6)))
            M = U[:, :rank] * s[:rank]
            Vh = Vh[:rank]

        n_times = M.shape[1]
        if n_times == 0 or np.allclose(M, 0):
            return self.source_to_object(np.zeros((n_sources, data.shape[1])))

        # ------------------------------------------------------------------
        # Regularization: alpha as fraction of alpha_max
        # ------------------------------------------------------------------
        GtM = G.T @ M
        alpha_max = _norm_l2inf(GtM, n_orient)
        if alpha_max == 0:
            return self.source_to_object(np.zeros((n_sources, data.shape[1])))
        alpha_eff = float(alpha) * alpha_max

        # ------------------------------------------------------------------
        # Lipschitz constants (precomputed, reused across iterations)
        # ------------------------------------------------------------------
        lipschitz = _compute_lipschitz(G, n_orient)

        # ------------------------------------------------------------------
        # Initialise solution and active set
        # ------------------------------------------------------------------
        X = np.zeros((n_sources, n_times))
        R = M.copy()

        gn2 = _groups_norm2(GtM, n_orient)
        n_init = min(active_set_size, n_positions)
        active = set(np.argsort(gn2)[-n_init:].tolist())

        # ------------------------------------------------------------------
        # Outer loop: BCD solve → expand active set → repeat
        # ------------------------------------------------------------------
        converged = False
        for _outer in range(maxit):
            # --- BCD iterations on current active set ---
            for bcd_it in range(bcd_maxit):
                active = _bcd_pass(
                    G,
                    X,
                    R,
                    sorted(active),
                    lipschitz,
                    n_orient,
                    alpha_eff,
                )

                if (bcd_it + 1) % dgap_freq == 0:
                    gap, p_obj = _dgap_l21(G, X, R, alpha_eff, n_orient)
                    if gap < tol * max(p_obj, 1e-30):
                        converged = True
                        break

            if converged:
                break

            # --- Expand active set ---
            GtR = G.T @ R
            gn2_resid = _groups_norm2(GtR, n_orient)
            for j in active:
                gn2_resid[j] = 0.0

            remaining = n_positions - len(active)
            n_add = min(active_set_size, remaining)
            if n_add <= 0:
                break

            new_pos = np.argsort(gn2_resid)[-n_add:]
            active |= set(new_pos.tolist())

        # ------------------------------------------------------------------
        # Optional debiasing
        # ------------------------------------------------------------------
        if debias:
            X = _debias(G, M, X, n_orient)

        # ------------------------------------------------------------------
        # Undo depth weighting and time PCA
        # ------------------------------------------------------------------
        if depth_w is not None:
            X = X / depth_w[:, np.newaxis]

        if Vh is not None:
            X = X @ Vh

        return self.source_to_object(X)
