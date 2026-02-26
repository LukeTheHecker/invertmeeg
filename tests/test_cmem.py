import numpy as np
from scipy.sparse import csr_matrix

from invert.solvers.bayesian.cmem import (
    _cmem,
    _compute_msp_coefficients,
    _data_driven_parcellation,
)


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


def test_cmem_parcellation_respects_num_parcels_with_adjacency():
    rng = np.random.default_rng(0)
    m, n, t = 8, 500, 10
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

    msp = _compute_msp_coefficients(Y, L)
    num_parcels = 40
    parcels = _data_driven_parcellation(Y, L, num_parcels=num_parcels, A=A, msp_scores=msp)
    assert np.unique(parcels).size == num_parcels


def test_cmem_does_not_collapse_under_noise():
    """Regression: Sigma calibration must not drive the solution to ~zero."""
    rng = np.random.default_rng(0)
    m, n, t = 16, 200, 30
    L = rng.standard_normal((m, n)) * 0.1

    active = int(rng.integers(0, n))
    S = np.zeros((n, t))
    S[active] = rng.standard_normal(t)

    Y_clean = L @ S
    noise_rel = 0.30
    Y = Y_clean + rng.standard_normal(Y_clean.shape) * (
        noise_rel * float(np.std(Y_clean))
    )

    rows, cols = [], []
    for i in range(n):
        if i - 1 >= 0:
            rows.append(i)
            cols.append(i - 1)
        if i + 1 < n:
            rows.append(i)
            cols.append(i + 1)
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))

    J, _ = _cmem(Y, L, A=A, num_parcels=20, max_iter=15, batch_size=10)
    residual = Y - L @ J
    r2 = 1.0 - float(np.sum(residual**2)) / float(np.sum(Y**2))
    assert r2 > 0.9
