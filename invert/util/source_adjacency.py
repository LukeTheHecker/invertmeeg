from __future__ import annotations

import logging

import mne
import numpy as np
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


def _adjacency_from_discrete_src(src) -> csr_matrix:
    """Build adjacency from ``neighbor_vert`` for discrete source spaces."""
    if len(src) != 1 or src[0].get("type") != "discrete":
        msg = "Discrete fallback only supports single discrete source spaces."
        raise RuntimeError(msg)

    space = src[0]
    neighbor_vert = space.get("neighbor_vert")
    if neighbor_vert is None:
        msg = "Discrete source space is missing 'neighbor_vert'."
        raise RuntimeError(msg)

    vertno = np.asarray(space["vertno"], dtype=int)
    n_vertices = int(vertno.size)
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
        return csr_matrix((n_vertices, n_vertices), dtype=float)

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

    if mode == "distance":
        adjacency = mne.spatial_dist_adjacency(src, adjacency_distance, verbose=verbose)
        return csr_matrix(adjacency)

    try:
        adjacency = mne.spatial_src_adjacency(src, verbose=verbose)
        return csr_matrix(adjacency)
    except Exception as spatial_exc:
        logger.warning(
            "spatial_src_adjacency failed (%s). Falling back to distance adjacency "
            "(dist=%s m).",
            spatial_exc,
            adjacency_distance,
        )

    try:
        adjacency = mne.spatial_dist_adjacency(src, adjacency_distance, verbose=verbose)
        return csr_matrix(adjacency)
    except Exception as distance_exc:
        logger.warning(
            "spatial_dist_adjacency also failed (%s). Trying discrete neighbor_vert "
            "fallback.",
            distance_exc,
        )

    return _adjacency_from_discrete_src(src)
