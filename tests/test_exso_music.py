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


def _chain_adjacency(n: int):
    from scipy.sparse import csr_matrix

    rows = []
    cols = []
    for i in range(n - 1):
        rows.extend([i, i + 1])
        cols.extend([i + 1, i])
    data = np.ones(len(rows), dtype=float)
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def test_exso_music_q1_recovers_single_disk_with_noise_cov():
    """q=1: best pseudo-disk matches the true extended source (with noise subtraction)."""
    from invert.solvers.music.exso_music import _exso_music

    rng = np.random.RandomState(1)
    m, n, t = 8, 30, 2000

    L = rng.randn(m, n)
    L /= np.linalg.norm(L, axis=0, keepdims=True) + 1e-12

    center = 12
    true_disk = np.array([center - 1, center, center + 1], dtype=int)
    s = rng.laplace(size=t)
    h = np.sum(L[:, true_disk], axis=1)
    Y = h[:, np.newaxis] @ s[np.newaxis, :] + 0.2 * rng.randn(m, t)

    A = _chain_adjacency(n)
    source_map, metric_map = _exso_music(
        Y,
        L,
        q=1,
        num_sources=1,
        adjacency=A,
        disk_hops=[1],
        lambda_threshold="auto",
        max_disk_size=10,
        noise_cov_identity=True,
        sensor_rank=None,
    )
    selected = np.where(source_map > 0)[0]

    assert metric_map.shape == (n,)
    assert set(selected.tolist()) == set(true_disk.tolist())


def test_exso_music_q2_recovers_single_disk():
    """q=2: 4th-order subspace + ExSo steering recovers the true extended source."""
    from invert.solvers.music.exso_music import _exso_music

    rng = np.random.RandomState(2)
    m, n, t = 8, 30, 4000

    L = rng.randn(m, n)
    L /= np.linalg.norm(L, axis=0, keepdims=True) + 1e-12

    center = 10
    true_disk = np.array([center - 1, center, center + 1], dtype=int)
    s = rng.laplace(size=t)
    h = np.sum(L[:, true_disk], axis=1)
    Y = h[:, np.newaxis] @ s[np.newaxis, :] + 0.02 * rng.randn(m, t)

    A = _chain_adjacency(n)
    source_map, metric_map = _exso_music(
        Y,
        L,
        q=2,
        num_sources=1,
        adjacency=A,
        disk_hops=[1],
        lambda_threshold="auto",
        max_disk_size=10,
        noise_cov_identity=False,
        sensor_rank=None,
    )
    selected = np.where(source_map > 0)[0]

    assert metric_map.shape == (n,)
    assert set(selected.tolist()) == set(true_disk.tolist())
