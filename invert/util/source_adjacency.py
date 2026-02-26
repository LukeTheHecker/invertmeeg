from __future__ import annotations

import hashlib
import logging

import mne
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)
_ADJ_CACHE: dict[tuple[str, str, float], csr_matrix] = {}


def _hash_array(arr: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(arr)
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    hasher.update(str(arr.dtype).encode("utf-8"))
    hasher.update(arr.view(np.uint8).data)
    return hasher.digest()


def source_space_signature(src) -> str:
    """Stable content signature for source space geometry."""
    hasher = hashlib.blake2b(digest_size=16)
    for space in src:
        hasher.update(str(space.get("type", "")).encode("utf-8"))
        vertno = np.asarray(space["vertno"], dtype=np.int32)
        hasher.update(_hash_array(vertno))
        rr = np.asarray(space["rr"], dtype=np.float32)[vertno]
        hasher.update(_hash_array(rr))
    return hasher.hexdigest()


def _points_from_src(src) -> np.ndarray:
    points = []
    for space in src:
        vertno = np.asarray(space.get("vertno", []), dtype=int)
        if vertno.size == 0:
            continue
        rr = np.asarray(space["rr"], dtype=float)
        points.append(rr[vertno])
    if not points:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(points)


def _adjacency_from_points(
    points: np.ndarray,
    *,
    adjacency_distance: float = 3e-3,
    min_k_neighbors: int = 6,
) -> csr_matrix:
    """Build adjacency from point coordinates using robust radius/kNN fallback."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (n_vertices, 3), got {points.shape}")

    n_vertices = int(points.shape[0])
    if n_vertices == 0:
        return csr_matrix((0, 0), dtype=float)
    if n_vertices == 1:
        return csr_matrix((1, 1), dtype=float)

    tree = cKDTree(points)

    rows: list[int] = []
    cols: list[int] = []

    radius = float(adjacency_distance)
    if radius > 0:
        pairs = tree.query_pairs(radius, output_type="ndarray")
        if pairs.size > 0:
            rows.extend(pairs[:, 0].tolist())
            cols.extend(pairs[:, 1].tolist())

    if not rows:
        # If the configured radius is too small (e.g., 3 mm on an 8 mm grid),
        # adapt to the nearest-neighbor spacing to keep adjacency connected.
        nn_dists, _ = tree.query(points, k=2)
        nearest = nn_dists[:, 1]
        nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
        if nearest.size > 0:
            auto_radius = 1.05 * float(np.median(nearest))
            pairs = tree.query_pairs(auto_radius, output_type="ndarray")
            if pairs.size > 0:
                rows.extend(pairs[:, 0].tolist())
                cols.extend(pairs[:, 1].tolist())

    if not rows:
        # Final fallback: connect each vertex to a few nearest neighbors.
        k = max(1, min(int(min_k_neighbors), n_vertices - 1))
        dists, idxs = tree.query(points, k=k + 1)
        for i in range(n_vertices):
            for j in np.asarray(idxs[i, 1:], dtype=int):
                if j != i:
                    a, b = (i, int(j)) if i < int(j) else (int(j), i)
                    rows.append(a)
                    cols.append(b)

    if not rows:
        return csr_matrix((n_vertices, n_vertices), dtype=float)

    data = np.ones(len(rows), dtype=float)
    adjacency = csr_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices))
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    return adjacency


def _adjacency_from_discrete_src(
    src,
    *,
    adjacency_distance: float = 3e-3,
) -> csr_matrix:
    """Build adjacency for discrete source spaces.

    Prefers ``neighbor_vert`` when available and falls back to coordinate-based
    adjacency when it is missing.
    """
    if len(src) != 1 or src[0].get("type") != "discrete":
        msg = "Discrete fallback only supports single discrete source spaces."
        raise RuntimeError(msg)

    space = src[0]
    vertno = np.asarray(space["vertno"], dtype=int)
    n_vertices = int(vertno.size)

    neighbor_vert = space.get("neighbor_vert")
    if neighbor_vert is None:
        rr = np.asarray(space["rr"], dtype=float)[vertno]
        return _adjacency_from_points(rr, adjacency_distance=adjacency_distance)

    global_to_local = {int(v): idx for idx, v in enumerate(vertno)}

    rows: list[int] = []
    cols: list[int] = []
    for local_i, vertex in enumerate(vertno):
        neighbors = neighbor_vert[int(vertex)]
        for neighbor in neighbors:
            local_j = global_to_local.get(int(neighbor))
            if local_j is not None and local_j != local_i:
                rows.append(local_i)
                cols.append(local_j)

    if not rows:
        rr = np.asarray(space["rr"], dtype=float)[vertno]
        return _adjacency_from_points(rr, adjacency_distance=adjacency_distance)

    data = np.ones(len(rows), dtype=float)
    adjacency = csr_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices))
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    return adjacency


def build_source_adjacency(
    src,
    *,
    adjacency_type: str = "spatial",
    adjacency_distance: float = 3e-3,
    verbose: int | None = 0,
) -> csr_matrix:
    """Build a robust source-space adjacency matrix for arbitrary MNE source spaces.

    For ``adjacency_type='spatial'``, this first tries mesh-based adjacency
    (`mne.spatial_src_adjacency`). If that fails (e.g., non-ico/discrete spaces),
    it falls back to distance-based adjacency and finally to ``neighbor_vert`` for
    discrete spaces.
    """
    mode = str(adjacency_type).lower()
    if mode not in {"spatial", "distance"}:
        msg = f"Unknown adjacency_type: {adjacency_type!r}"
        raise ValueError(msg)

    cache_key = (source_space_signature(src), mode, float(adjacency_distance))
    cached = _ADJ_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()

    if mode == "distance":
        try:
            adjacency = mne.spatial_dist_adjacency(
                src, adjacency_distance, verbose=verbose
            )
            adjacency = csr_matrix(adjacency)
            _ADJ_CACHE[cache_key] = adjacency
            return adjacency.copy()
        except Exception as distance_exc:
            logger.warning(
                "distance adjacency failed (%s). Trying discrete source fallback.",
                distance_exc,
            )
            try:
                adjacency = _adjacency_from_discrete_src(
                    src, adjacency_distance=adjacency_distance
                )
            except Exception as discrete_exc:
                logger.warning(
                    "discrete fallback failed (%s). Falling back to coordinate "
                    "adjacency.",
                    discrete_exc,
                )
                adjacency = _adjacency_from_points(
                    _points_from_src(src), adjacency_distance=adjacency_distance
                )
            _ADJ_CACHE[cache_key] = adjacency
            return adjacency.copy()

    try:
        adjacency = mne.spatial_src_adjacency(src, verbose=verbose)
        adjacency = csr_matrix(adjacency)
        _ADJ_CACHE[cache_key] = adjacency
        return adjacency.copy()
    except Exception as spatial_exc:
        logger.warning(
            "spatial_src_adjacency failed (%s). Falling back to distance adjacency "
            "(dist=%s m).",
            spatial_exc,
            adjacency_distance,
        )

    try:
        adjacency = mne.spatial_dist_adjacency(src, adjacency_distance, verbose=verbose)
        adjacency = csr_matrix(adjacency)
        _ADJ_CACHE[cache_key] = adjacency
        return adjacency.copy()
    except Exception as distance_exc:
        logger.warning(
            "spatial_dist_adjacency also failed (%s). Trying discrete source fallback.",
            distance_exc,
        )
    try:
        adjacency = _adjacency_from_discrete_src(
            src, adjacency_distance=adjacency_distance
        )
    except Exception as discrete_exc:
        logger.warning(
            "discrete fallback failed (%s). Falling back to coordinate adjacency.",
            discrete_exc,
        )
        adjacency = _adjacency_from_points(
            _points_from_src(src), adjacency_distance=adjacency_distance
        )
    _ADJ_CACHE[cache_key] = adjacency
    return adjacency.copy()
