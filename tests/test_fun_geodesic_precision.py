"""The FUN helper takes total precision C: its positive solution obeys S C S=M."""

import numpy as np
import pytest
from scipy.linalg import sqrtm

from invert.solvers.bayesian.champagne import SolverChampagne


def test_diagonal_precision_contract():
    C = np.diag([4.0, 9.0])
    M = np.diag([9.0, 16.0])
    actual = SolverChampagne._fun_learning_cov_est(C, M, "geodesic")
    np.testing.assert_allclose(actual, np.diag([1.5, 4 / 3]), rtol=1e-14, atol=0)


@pytest.mark.parametrize("rank", [3, 6])
@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_noncommuting_psd_moment(rank, scale):
    rng = np.random.default_rng(107)
    A = rng.normal(size=(6, 6))
    C = A @ A.T + 0.2 * np.eye(6)
    A = rng.normal(size=(6, rank))
    M = scale * (A @ A.T)
    actual = SolverChampagne._fun_learning_cov_est(C, M, "geodesic")
    # Alternative Cholesky factorization, independent of implementation's C eigenbasis.
    K = np.linalg.cholesky(C)
    R = np.real(sqrtm(K.T @ M @ K))
    reference = np.linalg.solve(K.T, np.linalg.solve(K.T, R.T).T)
    # At zero moment eigenvalues, matrix sqrt is only Holder-1/2 continuous;
    # independent decompositions differ by O(sqrt(machine epsilon)), not O(eps).
    tolerance = 1e-7 if rank < 6 else 1e-11
    assert np.linalg.norm(actual - reference) / np.linalg.norm(reference) < tolerance
    np.testing.assert_allclose(actual @ C @ actual, M, rtol=2e-11, atol=2e-12 * scale)
    np.testing.assert_allclose(actual, actual.T, rtol=0, atol=1e-12 * np.sqrt(scale))
    assert np.linalg.eigvalsh(actual).min() >= -1e-12 * np.sqrt(scale)
