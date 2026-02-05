from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import PCA

try:
    from scipy.linalg import cho_factor, cho_solve, eigh, svd
    from scipy.sparse.linalg import LinearOperator, eigsh

    SCIPY_OK = True
except ImportError:
    from numpy.linalg import eigh, svd

    SCIPY_OK = False

logger = logging.getLogger(__name__)


def bayesian_pca_covariance(Y, rank=None, var_threshold=0.95):
    """
    Compute Bayesian PCA-regularized covariance for beamforming.

    Parameters
    ----------
    Y : ndarray (m, t)
        EEG/MEG data (channels × time)
    rank : int or None
        Latent dimensionality. If None, chosen by explained variance threshold.
    var_threshold : float
        Variance ratio cutoff used if rank=None.

    Returns
    -------
    Sigma_bayes : ndarray (m, m)
        Regularized covariance estimate.
    W : ndarray (m, k)
        Estimated loading matrix.
    sigma2 : float
        Estimated noise variance.
    """
    m, t = Y.shape

    # Center data
    Yc = Y - Y.mean(axis=1, keepdims=True)

    # Fit probabilistic PCA (Bayesian-like) model
    pca = PCA(svd_solver="full")
    pca.fit(Yc.T)

    # Select rank by cumulative variance if not given
    if rank is None:
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        rank = np.searchsorted(cum_var, var_threshold) + 1

    # Loadings (W = U * sqrt(S))
    W = pca.components_[:rank].T * np.sqrt(pca.explained_variance_[:rank])

    # Estimate isotropic noise variance σ²
    sigma2 = np.mean(pca.explained_variance_[rank:])

    # Bayesian covariance estimate
    Sigma_bayes = W @ W.T + sigma2 * np.eye(m)
    return Sigma_bayes, W, sigma2


def apply_rest(eeg_data, leadfield, ref_idx=None, eps=1e-6):
    """
    Compute REST (Reference Electrode Standardization Technique) re-reference.

    Parameters
    ----------
    eeg_data : ndarray, shape (n_channels, n_samples)
        EEG potentials under the current reference.
    leadfield : ndarray, shape (n_channels, n_sources)
        Forward model matrix under the same reference as eeg_data.
    ref_idx : int or None
        Index of the current reference electrode in the montage.
        If None, assumes data are already average-referenced.
    eps : float
        Regularization term for pseudoinverse stability.

    Returns
    -------
    eeg_rest : ndarray, shape (n_channels, n_samples)
        EEG data re-referenced to infinity using REST.
    """

    n_ch, n_src = leadfield.shape

    # remove current reference
    if ref_idx is not None:
        L_cur = leadfield - leadfield[ref_idx, :][None, :]
        V_cur = eeg_data - eeg_data[ref_idx, :][None, :]
    else:
        L_cur = leadfield.copy()
        V_cur = eeg_data.copy()

    # compute "infinity" leadfield (mean of sensors = 0)
    L_inf = leadfield - leadfield.mean(axis=0, keepdims=True)

    # compute transformation matrix T = L_inf * pinv(L_cur)
    L_cur_pinv = np.linalg.pinv(L_cur, rcond=eps)
    T = L_inf @ L_cur_pinv

    # apply REST transformation
    eeg_rest = T @ V_cur

    return eeg_rest


def _estimate_rank_mdl(
    eigvals: np.ndarray, n_samples: int, max_rank: int | None = None
) -> int:
    """Estimate signal subspace rank via Wax & Kailath MDL."""
    eigvals = np.asarray(eigvals, dtype=float)
    m = eigvals.size
    if m == 0:
        return 1

    n_samples = int(max(2, n_samples))
    eigvals = np.clip(eigvals, 1e-18, None)

    mdls = np.empty(m, dtype=float)
    for k in range(m):
        tail = eigvals[k:]
        if tail.size == 0:
            mdls[k] = np.inf
            continue
        am = float(np.mean(tail))
        gm = float(np.exp(np.mean(np.log(tail))))
        if am <= 0 or gm <= 0:
            mdls[k] = np.inf
            continue
        mdls[k] = -n_samples * (m - k) * np.log(gm / am) + 0.5 * k * (
            2 * m - k
        ) * np.log(n_samples)

    k_hat = int(np.argmin(mdls))
    if max_rank is not None:
        k_hat = min(k_hat, int(max_rank))
    return int(np.clip(k_hat, 1, m))


# ---------------------------
# ReciPSIICOS Utilities
# ---------------------------


def _cov(Y: np.ndarray) -> np.ndarray:
    """Columnwise covariance: Y (m, t) -> (m, m)."""
    return (Y @ Y.T) / max(Y.shape[1] - 1, 1)


def _vec(M: np.ndarray) -> np.ndarray:
    """Column-major vec (Fortran order) for consistency with kron identities."""
    return M.reshape(-1, order="F")


def _unvec(v: np.ndarray, m: int) -> np.ndarray:
    """Inverse of _vec for square m x m matrices."""
    return v.reshape((m, m), order="F")


def _psd_spectral_flip(C: np.ndarray) -> np.ndarray:
    """Make symmetric matrix PSD by 'spectral flip' (abs on eigenvalues)."""
    w, V = eigh(C)
    return (V * np.abs(w)) @ V.T


def _regularize(C: np.ndarray, lam: float) -> np.ndarray:
    """Add Tikhonov diagonal regularization."""
    if lam <= 0:
        return C
    return C + lam * np.trace(C) / C.shape[0] * np.eye(C.shape[0], dtype=C.dtype)


def _whiten(
    Y: np.ndarray, L: np.ndarray, noise_cov: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Prewhiten Y and L by noise covariance (if provided)."""
    if noise_cov is None:
        return Y, L
    w, V = eigh(noise_cov)
    w = np.clip(w, 1e-12, None)
    W = (V * (1.0 / np.sqrt(w))) @ V.T
    return W @ Y, W @ L


def _virtual_sensors(
    L: np.ndarray, keep_energy: float = 0.99
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reduce sensors via SVD of leadfield: L (m, n) -> (r, n), with r chosen by energy.
    Returns (Ur^T, Lr) where Y_r = Ur^T @ Y reduces data consistently.
    """
    U, s, _ = svd(L, full_matrices=False)
    s2 = s**2
    cum = np.cumsum(s2) / np.sum(s2)
    r = int(np.searchsorted(cum, keep_energy) + 1)
    Ur = U[:, :r]
    return Ur.T, (Ur.T @ L)


def _lcmv_inverse_operator(
    Lr: np.ndarray, Cr_tilde: np.ndarray, reg: float = 1e-3
) -> np.ndarray:
    """
    Compute LCMV inverse operator on reduced space.
    """
    Creg = _regularize(Cr_tilde, reg)
    if SCIPY_OK:
        cfac = cho_factor(Creg, lower=True, check_finite=False)

        def Cinv_matvec(v):
            return cho_solve(cfac, v, check_finite=False)
    else:
        Cinv = np.linalg.pinv(Creg)

        def Cinv_matvec(v):
            return Cinv @ v

    n = Lr.shape[1]
    Lr.shape[0]

    CinvL = np.empty_like(Lr)
    den = np.empty(n)
    for i in range(n):
        gi = Lr[:, i]
        Cinv_gi = Cinv_matvec(gi)
        CinvL[:, i] = Cinv_gi
        den[i] = float(gi.T @ Cinv_gi) + 1e-20

    W = CinvL.T / den[:, None]
    return W


def _Q_power(Lr: np.ndarray) -> np.ndarray:
    """Build power subspace columns."""
    r, n = Lr.shape
    Q = np.empty((r * r, n), dtype=Lr.dtype)
    for i in range(n):
        gi = Lr[:, i]
        Q[:, i] = np.kron(gi, gi)
    return Q


def _svd_rank_from_energy(s: np.ndarray, keep_energy: float) -> int:
    s2 = s**2
    cum = np.cumsum(s2) / np.sum(s2) if np.sum(s2) > 0 else np.ones_like(s2)
    return int(np.searchsorted(cum, keep_energy) + 1)


def _iter_pairs(n: int, max_pairs: int | None, rng: np.random.Generator):
    """Yield pairs (i<j) either all or a random subset limited by max_pairs."""
    total = n * (n - 1) // 2
    if max_pairs is None or max_pairs >= total:
        for i in range(n):
            for j in range(i + 1, n):
                yield i, j
    else:
        choices = rng.choice(total, size=max_pairs, replace=False)

        def k_to_ij(k):
            lo, hi = 0, n - 1
            while lo < hi:
                mid = (lo + hi) // 2
                c = mid * (2 * n - mid - 1) // 2
                if k < c:
                    hi = mid
                else:
                    lo = mid + 1
            i = lo - 1
            if i < 0:
                i = 0
                base = 0
            else:
                base = i * (2 * n - i - 1) // 2
            j = (k - base) + i + 1
            return i, j

        for k in choices:
            yield k_to_ij(int(k))


def _linop_Ccor(Lr: np.ndarray, max_pairs: int | None, seed: int) -> LinearOperator:
    """Create a LinearOperator representing C_cor = Q_cor Q_cor^T."""
    r, n = Lr.shape
    rng = np.random.default_rng(seed)
    pairs = list(_iter_pairs(n, max_pairs, rng))

    def matvec(x: np.ndarray) -> np.ndarray:
        x = x.reshape(-1)
        y = np.zeros(r * r, dtype=Lr.dtype)
        for i, j in pairs:
            gi = Lr[:, i]
            gj = Lr[:, j]
            q1 = np.kron(gj, gi)
            q2 = np.kron(gi, gj)
            alpha = np.dot(q1, x) + np.dot(q2, x)
            y += alpha * (q1 + q2)
        return y

    return LinearOperator(
        dtype=Lr.dtype, shape=(r * r, r * r), matvec=matvec, rmatvec=matvec
    )


def _project_covariance_whitened(
    Cr: np.ndarray,
    Lr: np.ndarray,
    pwr_energy: float = 0.99,
    pwr_rank: int | None = None,
    cor_rank: int | None = None,
    max_pairs: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Whitened ReciPSIICOS covariance projection."""
    r, n = Lr.shape

    Qp = _Q_power(Lr)
    Up, sp, _ = svd(Qp, full_matrices=False)

    if pwr_rank is None:
        kp = _svd_rank_from_energy(sp, pwr_energy)
    else:
        kp = min(pwr_rank, Up.shape[1])
    Uk = Up[:, :kp]
    sk = sp[:kp]

    def to_uk(v):
        return Uk.T @ v

    def from_uk(c):
        return Uk @ c

    inv_sp = 1.0 / np.clip(sk, 1e-12, None)

    if SCIPY_OK:
        Ccor_op_full = _linop_Ccor(Lr, max_pairs=max_pairs, seed=seed)

        def uk_matvec(c: np.ndarray) -> np.ndarray:
            v_full = from_uk(c * inv_sp)
            Cv_full = Ccor_op_full @ v_full
            return (to_uk(Cv_full)) * inv_sp

        Ccor_w_uk = LinearOperator(
            dtype=Lr.dtype, shape=(kp, kp), matvec=uk_matvec, rmatvec=uk_matvec
        )

        kc = min(kp - 1, cor_rank) if cor_rank is not None else max(1, min(20, kp // 4))
        logger.info(f"Correlation subspace rank: {kc} (out of {kp})")
        vals, vecs = eigsh(Ccor_w_uk, k=kc, which="LM")
        Ec_w = vecs
    else:
        rng = np.random.default_rng(seed)
        pairs = list(_iter_pairs(n, max_pairs=max(500, 2 * n), rng=rng))
        r2 = r * r
        Qc = np.empty((r2, len(pairs)), dtype=Lr.dtype)
        for k, (i, j) in enumerate(pairs):
            gi = Lr[:, i]
            gj = Lr[:, j]
            Qc[:, k] = np.kron(gj, gi) + np.kron(gi, gj)
        Qc_uk = (Uk.T @ Qc) * inv_sp[:, None]
        Cw = Qc_uk @ Qc_uk.T
        vals, Ec_w = eigh(Cw)
        order = np.argsort(vals)[::-1]
        Ec_w = Ec_w[:, order]
        kc = (
            min(Ec_w.shape[1] - 1, cor_rank)
            if cor_rank is not None
            else max(1, min(20, Ec_w.shape[1] // 4))
        )
        Ec_w = Ec_w[:, :kc]

    P_w = np.eye(kp, dtype=Lr.dtype) - Ec_w @ Ec_w.T

    v = _vec(Cr)
    v_uk = to_uk(v)
    v_w = v_uk * inv_sp
    v_w_proj = P_w @ v_w
    v_tilde = from_uk(v_w_proj * sk)
    Ct = _unvec(v_tilde, Cr.shape[0])
    return Ct
