import heapq
import logging

import mne
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix

from invert.util import build_source_adjacency

from ..base import BaseSolver, InverseOperator, SolverMeta
from ..orientation import estimate_orientation_pca, reduce_free_to_scalar

logger = logging.getLogger(__name__)


class SolverExSoMUSIC(BaseSolver):
    """ExSo-MUSIC extended-source localization (q=1 or q=2).

    Scans pseudo-disks on the source-space mesh, evaluating how well the
    compound steering vector ``h(theta)^{⊗q}`` (sum of leadfield columns for
    all vertices in the disk) projects onto the signal subspace estimated from
    the (2q)-order cumulant matrix.  Disks with lowest metric are selected as
    active sources and reconstructed via weighted minimum-norm.

    By default uses 2nd-order statistics (q=1, covariance-based).  Set q=2
    for the 4th-order (quadricovariance) variant, which can handle correlated
    sources but needs substantially more time samples for a reliable estimate.

    Notes
    -----
    - This solver is scalar-orientation. For free-orientation forwards it
      reduces to scalar per-location leadfields (default: ``orientation='pca'``).
    - If ``noise_cov`` is provided, the data/leadfield are whitened first. For
      q=1 this means subtracting an identity noise covariance in the whitened
      space.
    """

    meta = SolverMeta(
        acronym="ExSo-MUSIC",
        full_name="Extended Signal Subspace MUSIC",
        category="Subspace Methods",
        description=(
            "ExSo-MUSIC extended-source localization via pseudo-disk scanning.  "
            "Supports 2nd-order (q=1, covariance-based, default) and 4th-order "
            "(q=2, quadricovariance) subspace estimation.  q=2 can separate "
            "correlated sources but needs many more time samples."
        ),
        references=[
            "Albera, L., Ferréol, A., Cosandier-Rimélé, D., Merlet, I., & Wendling, F. (2008). Brain source localization using a fourth-order deflation scheme. IEEE Transactions on Biomedical Engineering, 55(2), 490–501.",
        ],
    )

    def __init__(
        self,
        name="ExSo-MUSIC",
        *,
        q: int | str = "auto",
        max_disk_size: int | str = "auto",
        lambda_threshold: float | str = "auto",
        disk_hops: list[int] | str = "auto",
        sensor_rank: int | str | None = "auto",
        **kwargs,
    ):
        """Initialise ExSo-MUSIC solver.

        Parameters
        ----------
        q : int (1 or 2), or "auto"
            Order of the cumulant tensor used for subspace estimation.
            q=1 uses the ordinary (2nd-order) covariance matrix C₂ and works
            well when sources are uncorrelated.  q=2 builds the 4th-order
            cumulant matrix C₄ (quadricovariance), which can separate
            correlated sources because Gaussian noise vanishes in 4th-order
            cumulants.  The trade-off is that C₄ has dimension m² × m² and
            requires substantially more time samples for a reliable estimate.
            ``"auto"`` (default) selects q based on the number of time samples
            and sensors: uses q=2 when ``t >= 8*m`` (enough data for a stable
            4th-order cumulant estimate), otherwise q=1.
        max_disk_size : int or "auto"
            Maximum number of vertices a pseudo-disk may contain during the
            graph-expansion scan.  At each hop the disk grows by one ring of
            neighbours on the source-space mesh; expansion stops early if the
            vertex count exceeds this limit.  Prevents run-away growth on dense
            meshes (e.g. ico-5 with ~10k vertices per hemisphere).
            ``"auto"`` (default) uses 15, which captures exactly 1 hop of
            neighbours on typical icosahedral meshes (~6 neighbours per
            vertex).  This is mesh-resolution independent.
        lambda_threshold : float or "auto"
            Acceptance threshold for the ExSo-MUSIC metric Υ(E, h^{⊗q}).
            The metric ranges from 0 (perfect match to signal subspace) to 1
            (orthogonal).  When a float, every disk whose best metric across
            hop levels is ≤ λ is marked active.  When ``"auto"`` (default),
            thresholding is skipped and the ``n`` disks with lowest metric are
            selected globally via a min-heap.
        disk_hops : list of int, or "auto"
            Graph-hop distances at which the metric is evaluated during disk
            expansion.  At hop 0 the disk is just the centre vertex; at hop k
            it includes all vertices reachable within k edges on the
            source-space mesh.  The compound steering vector h is the sum of
            leadfield columns for all vertices in the current disk, so larger
            hops model broader extended sources.
            ``"auto"`` defaults to ``[0, 1, 2, 3, 4, 5, 7, 10, 15, 20]``.
            Note that ``max_disk_size`` caps the actual expansion regardless
            of the hop schedule.
        sensor_rank : int, "auto", or None
            Dimensionality reduction applied to the sensor space *before*
            computing 4th-order cumulants (q=2 only).  Reduces the C₄ matrix
            from m² × m² to r² × r² where r < m.  ``"auto"`` sets
            r = min(m, max(16, 4·n_sources)).  ``None`` disables reduction.
            Ignored when q=1.
        """
        self.name = name
        self.q = q
        self.max_disk_size = max_disk_size
        self.lambda_threshold = lambda_threshold
        self.disk_hops = disk_hops
        self.sensor_rank = sensor_rank
        # ExSo steering vectors are scalar; use PCA reduction for free-orientation
        # volume/discrete forwards unless the user explicitly overrides.
        kwargs.setdefault("orientation", "pca")
        return super().__init__(**kwargs)

    def make_inverse_operator(
        self,
        forward,
        mne_obj=None,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
        n="auto",
        q: int | None = None,
        lambda_threshold: float | str | None = None,
        adjacency=None,
        positions=None,
        disk_hops: list[int] | str | None = None,
        sensor_rank: int | str | None = None,
        source_weights=None,
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
        q : int
            ExSo-MUSIC order parameter (q=1 for 2nd-order, q=2 for 4th-order).
        lambda_threshold : float or "auto"
            Threshold λ in Eq. (7). If "auto", selects the best ``n`` disks globally.
        adjacency : numpy.ndarray, optional
            Source adjacency matrix (n, n).

        Return
        ------
        self : object returns itself for convenience
        """
        super().make_inverse_operator(forward, mne_obj, *args, alpha=alpha, **kwargs)
        wf = self.prepare_whitened_forward(noise_cov)
        data = self.unpack_data_obj(mne_obj)
        data = wf.sensor_transform @ data

        m, t = data.shape
        q_raw = self.q if q is None else q
        q_eff = _coerce_q(q_raw, m=m, t=t)
        if q_eff not in (1, 2):
            raise ValueError(f"q must be 1 or 2, got {q_eff}")

        if not isinstance(n, int):
            num_sources = self.estimate_n_sources(data, method=n)
        else:
            num_sources = n

        if adjacency is None:
            try:
                adjacency = build_source_adjacency(forward["src"])
            except Exception as exc:
                logger.warning(
                    "Could not build source adjacency from forward['src'] (%s). "
                    "Falling back to index-neighborhood disks.",
                    exc,
                )
                adjacency = None
        else:
            # If a free-orientation leadfield slips through (e.g. volume forward with
            # orientation='auto'), reduce to scalar so adjacency/disk scanning work.
            try:
                n_adj = int(getattr(adjacency, "shape", (0, 0))[0])
            except Exception:
                n_adj = 0
            if n_adj > 0 and self.leadfield.shape[1] == 3 * n_adj:
                logger.info(
                    "%s: reducing free-orientation leadfield (m x 3n) -> (m x n) "
                    "using PCA orientations for ExSo-MUSIC compatibility.",
                    self.name,
                )
                q_loc = estimate_orientation_pca(
                    self.leadfield,
                    data,
                    reg=float(self.orientation_pca_reg),
                    deterministic_sign=bool(self.orientation_pca_deterministic_sign),
                )
                self.leadfield = reduce_free_to_scalar(self.leadfield, q_loc)

        lam_eff = self.lambda_threshold if lambda_threshold is None else lambda_threshold
        hops_eff = self.disk_hops if disk_hops is None else disk_hops
        rank_eff = self.sensor_rank if sensor_rank is None else sensor_rank

        L_scan = self.leadfield
        if source_weights is None:
            source_weights = _vertex_area_weights_from_forward(forward)
        if source_weights is not None:
            w = np.asarray(source_weights, dtype=float).reshape(-1)
            if w.shape[0] == L_scan.shape[1]:
                L_scan = L_scan * w[np.newaxis, :]
            else:
                logger.warning(
                    "%s: source_weights has shape %s, expected (%d,); ignoring.",
                    self.name,
                    w.shape,
                    int(L_scan.shape[1]),
                )

        max_disk_eff = _coerce_max_disk_size(self.max_disk_size)

        source_map, metric_map = _exso_music(
            data,
            L_scan,
            q=q_eff,
            num_sources=num_sources,
            max_disk_size=max_disk_eff,
            adjacency=adjacency,
            lambda_threshold=lam_eff,
            disk_hops=hops_eff,
            sensor_rank=rank_eff,
            noise_cov_identity=bool(noise_cov is not None),
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
            InverseOperator(inverse_operator @ wf.sensor_transform, self.name),
        ]
        return self


# ---------------------------------------------------------------------------
# Private helper functions
# ---------------------------------------------------------------------------


def _exso_music(
    Y,
    L,
    *,
    q: int = 2,
    num_sources: int = 1,
    max_disk_size: int = 500,
    adjacency=None,
    lambda_threshold: float | str = "auto",
    disk_hops: list[int] | str = "auto",
    sensor_rank: int | str | None = "auto",
    noise_cov_identity: bool = False,
):
    """(2q)-ExSo-MUSIC region estimate via pseudo-disk scanning.

    See Eq. (5)–(7) in the paper: build ``C_{2q,x}`` (subtract noise term for q=1),
    estimate the signal subspace projector, then scan pseudo-disks with
    ``Upsilon(E, h(theta)^{⊗q})``.
    """

    Y = np.asarray(Y, dtype=float)
    L = np.asarray(L, dtype=float)
    if Y.ndim != 2 or L.ndim != 2:
        raise ValueError("Y and L must be 2D arrays")
    m, t = Y.shape
    m_l, n = L.shape
    if m != m_l:
        raise ValueError(f"Dimension mismatch: Y is {Y.shape}, L is {L.shape}")
    if num_sources < 1:
        raise ValueError("num_sources must be >= 1")

    q = int(q)
    if q not in (1, 2):
        raise ValueError(f"q must be 1 or 2, got {q}")

    # Zero-mean across time.
    Y = Y - np.mean(Y, axis=1, keepdims=True)

    # Optional sensor subspace projection for q=2 to keep m^2 manageable.
    if q == 2:
        r = _coerce_sensor_rank(sensor_rank, m=m, num_sources=num_sources)
        if r is not None and r < m:
            U = _sensor_pca_basis(Y, r=r)
            Y = U.T @ Y
            L = U.T @ L
            m = int(Y.shape[0])

    # Eq. (5): build C_{2q,x} (and subtract noise term for q=1 if requested).
    if q == 1:
        C2 = (Y @ Y.T) / max(1, t)
        if noise_cov_identity:
            C2 = C2 - np.eye(m, dtype=float)
        C = 0.5 * (C2 + C2.T)
    else:
        C = _fourth_order_cumulant_matrix(Y)

    E = _signal_subspace(C, q=q, num_sources=num_sources)
    if E.shape[1] == 0:
        return np.zeros(n, dtype=float), np.ones(n, dtype=float)

    E3 = None
    if q == 2:
        E3 = E.reshape(m, m, -1, order="C")

    hop_schedule = _coerce_disk_hops(disk_hops)
    max_hop = int(max(hop_schedule)) if hop_schedule else 0

    source_map = np.zeros(n, dtype=float)
    metric_map = np.full(n, np.inf, dtype=float)

    auto = isinstance(lambda_threshold, str) and str(lambda_threshold).lower() == "auto"
    if not auto:
        lam = float(lambda_threshold)

    # Heap stores (-metric, center, disk_vertices) to keep the best num_sources.
    top_disks: list[tuple[float, int, list[int]]] = []

    neighbors = _coerce_neighbors(adjacency, n=n)

    logger.info(
        "ExSo-MUSIC scan: q=%d m=%d n=%d t=%d hops=%s max_disk_size=%d lambda=%s",
        q,
        m,
        n,
        t,
        hop_schedule,
        int(max_disk_size),
        lambda_threshold,
    )

    for center in range(n):
        disk_vertices = {center}
        frontier = {center}
        h = L[:, center].copy()
        best = np.inf

        for hop in range(max_hop + 1):
            if hop in hop_schedule:
                metric = _exso_music_metric(E, h, q=q, E3=E3)
                best = min(best, metric)

                if auto:
                    if len(top_disks) < num_sources:
                        heapq.heappush(top_disks, (-metric, center, list(disk_vertices)))
                    else:
                        worst = -top_disks[0][0]
                        if metric < worst:
                            heapq.heapreplace(
                                top_disks, (-metric, center, list(disk_vertices))
                            )
                else:
                    if metric <= lam:
                        source_map[np.fromiter(disk_vertices, dtype=int)] = 1.0

            if hop == max_hop:
                break
            if len(disk_vertices) >= max_disk_size:
                break
            if neighbors is None:
                break

            new_frontier: set[int] = set()
            for v in frontier:
                for nb in neighbors[v]:
                    if nb not in disk_vertices:
                        new_frontier.add(int(nb))
            if not new_frontier:
                break

            disk_vertices.update(new_frontier)
            if len(disk_vertices) > max_disk_size:
                break
            new_arr = np.fromiter(new_frontier, dtype=int)
            h = h + np.sum(L[:, new_arr], axis=1)
            frontier = new_frontier

        metric_map[center] = best if np.isfinite(best) else 1.0

    if auto and top_disks:
        for _neg_metric, _center, verts in top_disks:
            source_map[np.asarray(verts, dtype=int)] = 1.0

    inf = ~np.isfinite(metric_map)
    if np.any(inf):
        metric_map[inf] = 1.0
    metric_map = np.clip(metric_map, 0.0, 1.0)

    return source_map, metric_map


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


def _signal_subspace(C, *, q: int, num_sources: int) -> np.ndarray:
    """Signal subspace basis E_{2q,s} from EVD of C_{2q,x} (or noise-subtracted)."""
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"C must be square, got {C.shape}")
    if C.shape[0] == 0:
        return np.zeros((0, 0), dtype=float)

    evals, evecs = eigh(C)
    if evals.size == 0:
        return np.zeros((C.shape[0], 0), dtype=float)

    if q == 1:
        idx = np.argsort(evals)[::-1]
    else:
        idx = np.argsort(np.abs(evals))[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    tol = max(1e-12, float(np.max(np.abs(evals))) * 1e-10)
    keep = np.where(np.abs(evals) > tol)[0]
    if keep.size == 0:
        return np.zeros((C.shape[0], 0), dtype=float)

    if q == 1:
        target = int(num_sources)
    else:
        target = int(num_sources) * (int(num_sources) + 1) // 2
    target = max(1, target)
    d = int(min(target, keep.size))
    return evecs[:, :d]


def _exso_music_metric(E: np.ndarray, h: np.ndarray, *, q: int, E3=None) -> float:
    """Compute Upsilon(E, h^{⊗q}) = 1 - ||E^T y||^2 / ||y||^2."""
    h = np.asarray(h, dtype=float)
    if h.ndim != 1:
        h = np.ravel(h)

    if q == 1:
        denom = float(h @ h)
        if denom <= 0:
            return 1.0
        z = E.T @ h
        proj = float(z @ z) / denom
    else:
        hh = float(h @ h)
        denom = hh * hh
        if denom <= 0:
            return 1.0
        if E3 is None:
            m = int(h.shape[0])
            E3 = E.reshape(m, m, -1, order="C")
        z = np.einsum("i,ijk,j->k", h, E3, h, optimize=True)
        proj = float(z @ z) / denom

    proj = min(max(proj, 0.0), 1.0)
    return 1.0 - proj


def _coerce_q(q, *, m: int, t: int) -> int:
    """Resolve q parameter.  'auto' → q=2 if t >= 8*m, else q=1."""
    if isinstance(q, str) and q.lower() == "auto":
        return 2 if t >= 8 * m else 1
    return int(q)


def _coerce_max_disk_size(max_disk_size) -> int:
    """Resolve max_disk_size.  'auto' → 15 (captures 1 hop on any ico mesh)."""
    if isinstance(max_disk_size, str) and str(max_disk_size).lower() == "auto":
        return 15
    return int(max_disk_size)


def _coerce_disk_hops(disk_hops: list[int] | str) -> list[int]:
    if isinstance(disk_hops, str) and disk_hops.lower() == "auto":
        return [0, 1, 2, 3, 4, 5, 7, 10, 15, 20]
    hops = [int(h) for h in disk_hops]
    hops = sorted(set([h for h in hops if h >= 0]))
    return hops if hops else [0]


def _coerce_sensor_rank(
    sensor_rank: int | str | None, *, m: int, num_sources: int
) -> int | None:
    if sensor_rank is None:
        return None
    if isinstance(sensor_rank, str) and sensor_rank.lower() == "auto":
        return int(min(m, max(16, 4 * int(num_sources))))
    r = int(sensor_rank)
    if r <= 0:
        return None
    return int(min(m, r))


def _sensor_pca_basis(Y: np.ndarray, *, r: int) -> np.ndarray:
    """Orthonormal sensor basis U (m x r) from sample covariance EVD."""
    Y = np.asarray(Y, dtype=float)
    m, t = Y.shape
    C2 = (Y @ Y.T) / max(1, t)
    evals, evecs = eigh(0.5 * (C2 + C2.T))
    idx = np.argsort(evals)[::-1]
    return evecs[:, idx[:r]]


def _coerce_neighbors(adjacency, *, n: int) -> list[np.ndarray] | None:
    if adjacency is None:
        return None
    if isinstance(adjacency, csr_matrix):
        A = adjacency
    else:
        A = csr_matrix(adjacency)
    if A.shape != (n, n):
        raise ValueError(f"adjacency must have shape {(n, n)}, got {A.shape}")
    A = A.tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    indptr = A.indptr
    indices = A.indices
    return [indices[indptr[i] : indptr[i + 1]] for i in range(n)]


def _vertex_area_weights_from_forward(forward) -> np.ndarray | None:
    """Compute per-source surface area weights (Δ_m) for surface source spaces.

    Returns weights aligned to the forward solution column order (lh then rh),
    or None if not available.
    """
    try:
        src = forward["src"]
    except Exception:
        return None

    try:
        src_types = [str(s.get("type", "")) for s in src]
    except Exception:
        return None

    if not (len(src_types) == 2 and all(t == "surf" for t in src_types)):
        return None

    weights = []
    for hemi in src:
        use_tris = hemi.get("use_tris", None)
        rr = hemi.get("rr", None)
        vertno = hemi.get("vertno", None)
        if use_tris is None or rr is None or vertno is None:
            return None

        tris = np.asarray(use_tris, dtype=int)
        rr = np.asarray(rr, dtype=float)
        vertno = np.asarray(vertno, dtype=int)
        if tris.ndim != 2 or tris.shape[1] != 3:
            return None
        if rr.ndim != 2 or rr.shape[1] != 3:
            return None
        if vertno.ndim != 1:
            return None

        p0 = rr[tris[:, 0]]
        p1 = rr[tris[:, 1]]
        p2 = rr[tris[:, 2]]
        tri_area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)

        area_per_vert = np.zeros(rr.shape[0], dtype=float)
        contrib = tri_area / 3.0
        np.add.at(area_per_vert, tris[:, 0], contrib)
        np.add.at(area_per_vert, tris[:, 1], contrib)
        np.add.at(area_per_vert, tris[:, 2], contrib)

        w = area_per_vert[vertno]
        w = np.maximum(w, 0.0)
        weights.append(w)

    out = np.concatenate(weights) if weights else None
    if out is None or out.size == 0:
        return None
    if not np.all(np.isfinite(out)):
        return None
    return out
