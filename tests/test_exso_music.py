import numpy as np


def test_fourth_order_cumulant_near_zero_for_gaussian():
    """For zero-mean Gaussian data the 4th-order cumulant is (in expectation) 0."""
    from invert.solvers.music.exso_music import _fourth_order_cumulant_matrix

    rng = np.random.RandomState(0)
    m, t = 5, 10_000
    Y = rng.randn(m, t)
    Y -= Y.mean(axis=1, keepdims=True)

    C4 = _fourth_order_cumulant_matrix(Y, chunk_size=1024)

    # Compare to the 4th-order moment magnitude to avoid scale sensitivity.
    Z = (Y.T[:, :, None] * Y.T[:, None, :]).reshape(t, m * m)
    M4 = (Z.T @ Z) / t

    ratio = np.linalg.norm(C4, ord="fro") / (np.linalg.norm(M4, ord="fro") + 1e-12)
    assert ratio < 0.08


def test_exso_music_recovers_two_point_sources():
    """With max_disk_size=1, ExSo-MUSIC reduces to a point-source FO-MUSIC variant."""
    from invert.solvers.music.exso_music import _exso_music

    rng = np.random.RandomState(1)
    m, n, t = 8, 30, 2000

    L = rng.randn(m, n)
    L /= np.linalg.norm(L, axis=0, keepdims=True) + 1e-12

    true_idx = np.array([5, 20])
    S = rng.laplace(size=(true_idx.size, t))
    Y = L[:, true_idx] @ S + 0.01 * rng.randn(m, t)

    source_map, metric_map = _exso_music(Y, L, num_sources=2, max_disk_size=1)
    selected = np.where(source_map > 0)[0]

    assert metric_map.shape == (n,)
    assert set(selected.tolist()) == set(true_idx.tolist())

