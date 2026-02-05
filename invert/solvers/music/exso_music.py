import logging

import numpy as np
from scipy.linalg import eigh

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverExSoMUSIC(BaseSolver):
    """4-ExSo-MUSIC source localization solver.

    Extended Signal Subspace MUSIC using fourth-order statistics
    (quadricovariance) for improved source localization, especially
    with correlated sources.

    Parameters
    ----------
    name : str
        Name of the solver.
    max_disk_size : int
        Maximum disk size for source patches.

    References
    ----------
    Albera, L., Ferréol, A., Cosandier-Rimélé, D., Merlet, I., &
    Wendling, F. (2008). Brain source localization using a
    fourth-order deflation scheme. IEEE Transactions on Biomedical
    Engineering, 55(2), 490-501.
    """

    meta = SolverMeta(
        acronym="4-ExSo-MUSIC",
        full_name="Fourth-Order Extended Signal Subspace MUSIC",
        category="Subspace Methods",
        description=(
            "Fourth-order statistics (quadricovariance) variant of ExSo-MUSIC for "
            "localizing sources using higher-order cumulants, useful for correlated "
            "sources."
        ),
        references=[
            "Albera, L., Ferréol, A., Cosandier-Rimélé, D., Merlet, I., & Wendling, F. (2008). Brain source localization using a fourth-order deflation scheme. IEEE Transactions on Biomedical Engineering, 55(2), 490–501.",
        ],
    )

    def __init__(self, name="4-ExSo-MUSIC", max_disk_size=500, **kwargs):
        self.name = name
        self.max_disk_size = max_disk_size
        return super().__init__(**kwargs)

    def make_inverse_operator(
        self,
        forward,
        mne_obj,
        *args,
        alpha="auto",
        n="auto",
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
        n : int or str
            Number of sources to estimate, or "auto".
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

        if not isinstance(n, int):
            num_sources = self.estimate_n_sources(data, method=n)
        else:
            num_sources = n

        source_map, metric_map = _exso_music(
            data,
            self.leadfield,
            A=adjacency,
            P=positions,
            num_sources=num_sources,
            max_disk_size=self.max_disk_size,
        )

        self.source_map = source_map
        self.metric_map = metric_map

        # Build a WMNE-based inverse operator on the selected dipoles
        dipole_idc = np.where(source_map > 0)[0]
        n_dipoles, n_chans = self.leadfield.shape[1], self.leadfield.shape[0]
        inverse_operator = np.zeros((n_dipoles, n_chans))

        if len(dipole_idc) > 0:
            L_sel = self.leadfield[:, dipole_idc]
            W = np.diag(np.linalg.norm(L_sel, axis=0))
            inverse_operator[dipole_idc, :] = (
                np.linalg.inv(L_sel.T @ L_sel + W.T @ W) @ L_sel.T
            )

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name),
        ]
        return self


# ---------------------------------------------------------------------------
# Private helper functions
# ---------------------------------------------------------------------------


def _exso_music(Y, L, A=None, P=None, num_sources=1, max_disk_size=500):
    """4-ExSo-MUSIC source localization (fourth-order cumulant MUSIC).

    Parameters
    ----------
    Y : array (m, t) - EEG/MEG data matrix
    L : array (m, n) - Lead field matrix
    A : array (n, n) - Source adjacency matrix (optional)
    P : array (n, 3) - Source positions (optional)
    num_sources : int - Number of sources to estimate
    max_disk_size : int - Maximum disk size for source patches

    Returns
    -------
    source_map : array (n,) - Binary source map
    metric_map : array (n,) - 4-ExSo-MUSIC metric values
    """

    m, t = Y.shape
    m_l, n = L.shape

    assert m == m_l, "Dimension mismatch between Y and L"

    if num_sources < 1:
        raise ValueError("num_sources must be >= 1")

    # ---------------------------------------------------------------------
    # Step 1: Dimension reduction + whitening (standard for 4th-order methods)
    # ---------------------------------------------------------------------
    # Working in sensor-space (m^2 x m^2) is not feasible for typical MEG.
    # Reduce to a data-driven sensor subspace, then compute the 4th-order
    # cumulant in that reduced space.
    Y_centered = Y - np.mean(Y, axis=1, keepdims=True)
    Y_r, L_r = _reduce_and_whiten(Y_centered, L, num_sources=num_sources)

    # ---------------------------------------------------------------------
    # Step 2: Fourth-order cumulant ("quadricovariance") and signal subspace
    # ---------------------------------------------------------------------
    logger.info("Computing fourth-order cumulant matrix...")
    C4 = _fourth_order_cumulant_matrix(Y_r)
    E4 = _signal_subspace(C4, num_sources=num_sources)
    logger.info(f"Signal subspace dimension: {E4.shape[1]}")

    # ---------------------------------------------------------------------
    # Step 3: Generate disk candidates
    # ---------------------------------------------------------------------
    logger.info("Generating disk candidates...")
    disks = _generate_disks(n, A, max_disk_size)

    # ---------------------------------------------------------------------
    # Step 4: Sequential selection with fourth-order deflation
    # ---------------------------------------------------------------------
    # Use a subspace criterion for each disk D:
    #   metric(D) = 1 - max_{q in span(Q_D), ||q||=1} ||E4^T q||^2
    #            = 1 - ||E4^T Q_D||_2^2
    # where Q_D spans the symmetric Kronecker subspace induced by the disk
    # leadfield columns (extended-source model).
    metric_map, best_disks = _compute_metric_map(E4, L_r, disks)
    metric_map_initial = metric_map.copy()

    selected_vertices = []
    excluded = np.zeros(n, dtype=bool)
    C4_work = C4

    for k in range(num_sources):
        # Recompute subspace after deflation (k>0)
        if k > 0:
            E4 = _signal_subspace(C4_work, num_sources=max(1, num_sources - k))
            metric_map, best_disks = _compute_metric_map(E4, L_r, disks, excluded=excluded)

        candidates = np.where(~excluded)[0]
        if candidates.size == 0:
            break

        best_idx = candidates[np.argmin(metric_map[candidates])]
        if not np.isfinite(metric_map[best_idx]):
            break

        disk = best_disks.get(best_idx)
        if disk is None or len(disk) == 0:
            excluded[best_idx] = True
            continue

        selected_vertices.append(best_idx)
        excluded[np.asarray(disk, dtype=int)] = True

        Q_disk = _disk_fourth_order_basis(L_r, disk)
        if Q_disk is not None and Q_disk.size > 0:
            C4_work = _deflate_cumulant(C4_work, Q_disk)

    source_map = np.zeros(n)
    if selected_vertices:
        source_map[np.asarray(selected_vertices, dtype=int)] = 1.0

    return source_map, metric_map_initial


def _generate_disks(n, A=None, max_disk_size=500):
    """Generate disk candidates for 4-ExSo-MUSIC."""
    disks = {}

    if A is None:
        for i in range(n):
            disks[i] = []
            for size in [1, 5, 10, 20, 50, min(max_disk_size, n // 10)]:
                start_idx = max(0, i - size // 2)
                end_idx = min(n, i + size // 2 + 1)
                disk = list(range(start_idx, end_idx))
                if len(disk) <= max_disk_size:
                    disks[i].append(disk)
    else:
        A_arr = A
        for i in range(n):
            disks[i] = []
            current_disk = {i}
            disks[i].append([i])

            # Grow disk by graph distance; store intermediate radii until max size
            for _radius in range(1, 20):
                new_vertices = set()
                for v in current_disk:
                    if hasattr(A_arr, "getrow"):
                        neighbors = A_arr.getrow(v).nonzero()[1]
                    else:
                        neighbors = np.where(A_arr[v, :] > 0)[0]
                    new_vertices.update(neighbors.tolist())

                if not new_vertices:
                    break

                current_disk.update(new_vertices)
                if len(current_disk) <= max_disk_size:
                    disks[i].append(list(current_disk))
                else:
                    break

    return disks


def _reduce_and_whiten(Y, L, num_sources=1, max_rank=None):
    """Reduce sensor space and whiten.

    Uses PCA on the sample covariance and keeps a modest number of components
    relative to the expected number of sources. This makes the fourth-order
    cumulant computation feasible (otherwise it scales as O(m^4)).
    """
    m, t = Y.shape
    C2 = (Y @ Y.T) / max(1, t)

    # Conservative default: enough components to represent the signal subspace.
    # (Using too many makes the 4th-order matrix explode in size.)
    if max_rank is None:
        max_rank = min(m, max(8, 4 * num_sources))

    evals, evecs = eigh(C2)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Keep components with meaningful energy, but cap at max_rank
    tol = max(1e-12, evals[0] * 1e-10) if evals.size else 1e-12
    keep = np.where(evals > tol)[0]
    r = int(min(max_rank, max(1, keep.size if keep.size else 1)))

    U = evecs[:, :r]
    Y_r = U.T @ Y
    L_r = U.T @ L

    # Whitening in reduced space
    C2_r = (Y_r @ Y_r.T) / max(1, t)
    evals_r, evecs_r = eigh(C2_r)
    evals_r = np.clip(evals_r, 1e-15, None)
    W = evecs_r @ np.diag(1.0 / np.sqrt(evals_r)) @ evecs_r.T

    return W @ Y_r, W @ L_r


def _fourth_order_cumulant_matrix(Y, chunk_size=512):
    """Estimate the 4th-order cumulant matrix of a zero-mean signal.

    Let y(t) be m-dimensional, zero-mean. Define z(t) = kron(y(t), y(t)) using
    row-major indexing of the quadratic terms. The 4th-order moment is
    M4 = E[z z^T] with entries E[y_i y_j y_k y_l]. The 4th-order cumulant is:

        cum(y_i,y_j,y_k,y_l) = E[y_i y_j y_k y_l]
                              - E[y_i y_j]E[y_k y_l]
                              - E[y_i y_k]E[y_j y_l]
                              - E[y_i y_l]E[y_j y_k]

    and we arrange these cumulants into an (m^2 x m^2) symmetric matrix.
    """
    m, t = Y.shape
    if t < 2:
        raise ValueError("Need at least 2 time samples to estimate cumulants")

    # Second-order covariance
    R = (Y @ Y.T) / t

    # Fourth-order moment M4 = E[(y⊗y)(y⊗y)^T]
    N = m * m
    M4 = np.zeros((N, N), dtype=float)
    Yt = Y.T  # (t, m)
    for start in range(0, t, chunk_size):
        stop = min(t, start + chunk_size)
        Yc = Yt[start:stop]  # (c, m)
        # z_k = vec_row(y y^T) for each time sample
        Z = (Yc[:, :, None] * Yc[:, None, :]).reshape(-1, N)  # (c, m^2)
        M4 += Z.T @ Z
    M4 /= t

    # Cumulant correction terms (in the same indexing as Z)
    term1 = np.outer(R.reshape(-1, order="C"), R.reshape(-1, order="C"))
    term2 = np.einsum("ik,jl->ijkl", R, R).reshape(N, N, order="C")
    term3 = np.einsum("il,jk->ijkl", R, R).reshape(N, N, order="C")

    C4 = M4 - term1 - term2 - term3
    C4 = 0.5 * (C4 + C4.T)
    return C4


def _signal_subspace(C4, num_sources=1):
    """Extract an orthonormal basis for the fourth-order signal subspace."""
    evals, evecs = eigh(C4)
    if evals.size == 0:
        return np.zeros((C4.shape[0], 0))

    # Cumulant matrices can be indefinite; keep directions by magnitude.
    idx = np.argsort(np.abs(evals))[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    tol = max(1e-12, np.abs(evals[0]) * 1e-10)
    keep = np.where(np.abs(evals) > tol)[0]
    if keep.size == 0:
        return np.zeros((C4.shape[0], 0))

    # For correlated sources, the 4th-order subspace can exceed num_sources.
    target = max(1, num_sources * (num_sources + 1) // 2)
    d = int(min(target, keep.size))
    return evecs[:, :d]


def _disk_fourth_order_basis(L, disk, max_rank=3, svd_tol=1e-8):
    """Build an orthonormal basis for an extended-source disk in 4th-order space.

    For a disk with leadfield block H (m x p), the fourth-order mixing subspace
    is spanned (in the symmetric quadratic space) by vectors of the form:
        vec_sym(u_a u_b^T),  a <= b
    where u_a are basis vectors spanning col(H). We approximate col(H) by a
    truncated SVD with rank <= max_rank, then form the symmetric Kronecker
    basis and orthonormalize it.
    """
    disk = np.asarray(disk, dtype=int)
    if disk.size == 0:
        return None

    H = L[:, disk]
    if H.ndim != 2 or H.shape[1] == 0:
        return None

    # Orthonormal basis of the disk leadfield subspace
    U, s, _Vt = np.linalg.svd(H, full_matrices=False)
    if s.size == 0:
        return None

    rel = s / max(s[0], 1e-12)
    r0 = int(min(max_rank, np.sum(rel > svd_tol)))
    r0 = max(1, r0)
    U = U[:, :r0]  # (m, r0)

    # Symmetric Kronecker basis in m^2 space
    cols = []
    for a in range(r0):
        ua = U[:, a]
        cols.append(np.kron(ua, ua))
        for b in range(a + 1, r0):
            ub = U[:, b]
            cols.append((np.kron(ua, ub) + np.kron(ub, ua)) / np.sqrt(2.0))

    K = np.column_stack(cols)  # (m^2, r0(r0+1)/2)
    Q, R = np.linalg.qr(K, mode="reduced")
    diag = np.abs(np.diag(R))
    keep = diag > (diag.max() * 1e-12 if diag.size else 0.0)
    return Q[:, keep] if np.any(keep) else Q


def _compute_metric_map(E4, L, disks, excluded=None):
    """Compute the per-vertex ExSo-MUSIC metric and the best disk per vertex."""
    n = L.shape[1]
    metric_map = np.full(n, np.inf, dtype=float)
    best_disks = {}
    if E4.shape[1] == 0:
        return metric_map, best_disks

    excluded_mask = np.zeros(n, dtype=bool) if excluded is None else excluded

    # Evaluate each vertex's family of disks and keep the best (minimum metric)
    for v, v_disks in disks.items():
        if excluded_mask[v]:
            continue
        best_metric = np.inf
        best_disk = None

        for disk in v_disks:
            Q = _disk_fourth_order_basis(L, disk)
            if Q is None or Q.size == 0:
                continue

            M = E4.T @ Q  # (d, r)
            G = M @ M.T  # (d, d) symmetric PSD
            lam_max = float(np.max(np.linalg.eigvalsh(G)))
            lam_max = min(max(lam_max, 0.0), 1.0)
            metric = 1.0 - lam_max

            if metric < best_metric:
                best_metric = metric
                best_disk = disk

        metric_map[v] = best_metric
        if best_disk is not None:
            best_disks[v] = best_disk

    # Any remaining inf metrics are set to worst-case
    inf = ~np.isfinite(metric_map)
    if np.any(inf):
        metric_map[inf] = 1.0
    return metric_map, best_disks


def _deflate_cumulant(C4, Q):
    """Deflate cumulant matrix by removing the selected disk subspace."""
    P = Q @ Q.T  # (N, N)
    C4_new = C4 - P @ C4 - C4 @ P + P @ C4 @ P
    return 0.5 * (C4_new + C4_new.T)
