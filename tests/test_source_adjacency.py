from copy import deepcopy

import numpy as np

from invert.util import build_source_adjacency


def _assert_valid_undirected_adjacency(adjacency, n_vertices: int) -> None:
    assert adjacency.shape == (n_vertices, n_vertices)
    assert adjacency.nnz > 0
    assert (adjacency != adjacency.T).nnz == 0
    assert np.allclose(adjacency.diagonal(), 0.0)


def test_discrete_fallback_without_neighbor_vert(forward_model_discrete_free):
    src = deepcopy(forward_model_discrete_free["src"])
    src[0].pop("neighbor_vert", None)

    adjacency = build_source_adjacency(
        src,
        adjacency_type="spatial",
        adjacency_distance=3e-3,
        verbose=0,
    )

    n_vertices = int(np.asarray(src[0]["vertno"]).size)
    _assert_valid_undirected_adjacency(adjacency, n_vertices)


def test_distance_mode_discrete_fallback_without_neighbor_vert(
    forward_model_discrete_free,
):
    src = deepcopy(forward_model_discrete_free["src"])
    src[0].pop("neighbor_vert", None)

    adjacency = build_source_adjacency(
        src,
        adjacency_type="distance",
        adjacency_distance=3e-3,
        verbose=0,
    )

    n_vertices = int(np.asarray(src[0]["vertno"]).size)
    _assert_valid_undirected_adjacency(adjacency, n_vertices)
