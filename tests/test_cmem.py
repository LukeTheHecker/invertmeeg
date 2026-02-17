import numpy as np
from scipy.sparse import csr_matrix

from invert.solvers.bayesian.cmem import _cmem


def test_cmem_mne_identity_small():
    """cMEM uses a Woodbury-form ridge/MNE solve; it should match the direct form."""
    rng = np.random.default_rng(0)
    m, n, t = 9, 25, 7
    L = rng.standard_normal((m, n))
    Y = rng.standard_normal((m, t))

    fro2 = float(np.sum(L * L))  # trace(L^T L)
    reg = 0.1 * fro2 / n

    direct = np.linalg.solve(L.T @ L + reg * np.eye(n), L.T @ Y)
    woodbury = L.T @ np.linalg.solve(L @ L.T + reg * np.eye(m), Y)

    np.testing.assert_allclose(direct, woodbury, rtol=1e-10, atol=1e-10)


def test_cmem_runs_small():
    rng = np.random.default_rng(0)
    m, n, t = 10, 40, 6
    L = rng.standard_normal((m, n)) * 0.1
    Y = rng.standard_normal((m, t))

    rows, cols = [], []
    for i in range(n):
        if i - 1 >= 0:
            rows.append(i)
            cols.append(i - 1)
        if i + 1 < n:
            rows.append(i)
            cols.append(i + 1)
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))

    J, parcels = _cmem(Y, L, A=A, num_parcels=8, max_iter=3, batch_size=3)
    assert J.shape == (n, t)
    assert parcels.shape == (n,)
    assert np.isfinite(J).all()

