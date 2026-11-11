"""Reference-equation and zero-entry regressions for robust irMxNE."""

import numpy as np
import pytest
from mne.inverse_sparse.mxne_optim import iterative_mixed_norm_solver

from invert.solvers.minimum_norm.irmxne_optim import iterative_mixed_norm_solver_safe


def _expand(fitted, active, shape):
    result = np.zeros(shape)
    result[active] = fitted
    return result


@pytest.mark.parametrize("n_orient", [1, 3])
def test_zero_first_sample_and_orientation_convex_solution(n_orient):
    n_columns = 2 * n_orient
    data = np.zeros((n_columns, 16))
    data[0 if n_orient == 1 else 1, 1:] = np.linspace(0.2, 1, 15)
    data[n_orient if n_orient == 1 else 4, 1:] = np.linspace(1, 0.2, 15)
    data /= np.linalg.norm(data)
    expected = (
        data
        * np.repeat(
            1 - 0.1 / np.linalg.norm(data.reshape(2, n_orient, 16), axis=(1, 2)),
            n_orient,
        )[:, None]
    )
    fitted, active, _, info = iterative_mixed_norm_solver_safe(
        data,
        np.eye(n_columns),
        0.1,
        1,
        n_orient=n_orient,
        debias=False,
        tol=1e-10,
    )
    np.testing.assert_allclose(
        _expand(fitted, active, data.shape), expected, atol=1e-12
    )
    assert info["inner_converged"].all()
    assert info["inner_duality_gaps"].max() <= 1e-10


@pytest.mark.parametrize("n_orient", [1, 3])
def test_nonconvex_stationary_equation_with_zero_entries(n_orient):
    data = np.zeros((2 * n_orient, 4))
    data[0 if n_orient == 1 else 1, 1:] = [1, 2, 3]
    data[n_orient if n_orient == 1 else 4, 1:] = [0.005, 0.005, 0.005]
    fitted, active, history, info = iterative_mixed_norm_solver_safe(
        data,
        np.eye(2 * n_orient),
        0.2,
        100,
        n_orient=n_orient,
        debias=False,
        tol=1e-11,
    )
    source = _expand(fitted, active, data.shape)
    x, y = source[:n_orient], data[:n_orient]
    gradient = x - y + 0.2 * x / (2 * np.linalg.norm(x) ** 1.5)
    assert np.linalg.norm(gradient) < 1e-9
    assert not np.any(source[n_orient:])
    assert np.max(np.diff(history)) < 1e-10
    assert info["outer_converged"]


@pytest.mark.parametrize("n_orient", [1, 3])
def test_ordinary_case_matches_mne_reference(n_orient):
    rng = np.random.default_rng(202)
    gain, _ = np.linalg.qr(rng.normal(size=(18, 6)))
    data = rng.normal(size=(18, 15))
    data /= np.linalg.norm(data)
    expected, expected_active, expected_history = iterative_mixed_norm_solver(
        data,
        gain,
        0.025,
        6,
        n_orient=n_orient,
        debias=False,
        tol=1e-10,
        maxit=1000,
        active_set_size=2,
        solver="bcd",
        verbose=False,
    )
    result, active, history, info = iterative_mixed_norm_solver_safe(
        data,
        gain,
        0.025,
        6,
        n_orient=n_orient,
        debias=False,
        tol=1e-10,
        maxit=1000,
        active_set_size=2,
    )
    np.testing.assert_array_equal(active, expected_active)
    np.testing.assert_allclose(result, expected, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(history, expected_history, rtol=1e-8, atol=1e-10)
    assert info["inner_converged"].all()


def test_vector_rotation_equivariance():
    rng = np.random.default_rng(203)
    gain, _ = np.linalg.qr(rng.normal(size=(16, 6)))
    data = rng.normal(size=(16, 10))
    rotation = np.zeros((6, 6))
    for j in range(2):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        rotation[j * 3 : (j + 1) * 3, j * 3 : (j + 1) * 3] = q
    x, active, _, _ = iterative_mixed_norm_solver_safe(
        data, gain, 0.1, 8, n_orient=3, debias=False, tol=1e-11
    )
    z, active_z, _, _ = iterative_mixed_norm_solver_safe(
        data, gain @ rotation, 0.1, 8, n_orient=3, debias=False, tol=1e-11
    )
    np.testing.assert_allclose(
        _expand(x, active, (6, 10)),
        rotation @ _expand(z, active_z, (6, 10)),
        atol=1e-10,
    )


def test_inner_failure_is_reported_and_stops_majorization():
    rng = np.random.default_rng(204)
    gain = rng.normal(size=(8, 30))
    data = rng.normal(size=(8, 20))
    _, _, history, info = iterative_mixed_norm_solver_safe(
        data,
        gain,
        0.1,
        10,
        n_orient=3,
        maxit=1,
        tol=1e-14,
        debias=False,
    )
    assert len(history) == 1
    assert not info["inner_converged"].all()
    assert not info["outer_converged"]
    assert info["outer_stop_reason"] == "inner_iteration_limit"


def test_amplitude_debias_does_not_overwrite_objective_history():
    rng = np.random.default_rng(205)
    data = rng.normal(size=(6, 12))
    raw, active, history, info = iterative_mixed_norm_solver_safe(
        data, np.eye(6), 0.3, 6, debias=False, n_orient=3
    )
    debiased, active_db, history_db, info_db = iterative_mixed_norm_solver_safe(
        data, np.eye(6), 0.3, 6, debias=True, n_orient=3
    )
    np.testing.assert_array_equal(active, active_db)
    np.testing.assert_allclose(history, history_db)
    assert np.linalg.norm(data[active] - debiased) < np.linalg.norm(data[active] - raw)
    np.testing.assert_allclose(
        info["inner_duality_gaps"], info_db["inner_duality_gaps"]
    )


def test_zero_penalty_is_least_squares_with_unexplained_residual():
    rng = np.random.default_rng(206)
    gain, data = rng.normal(size=(12, 6)), rng.normal(size=(12, 10))
    fitted, active, _, info = iterative_mixed_norm_solver_safe(
        data, gain, 0, 10, n_orient=3
    )
    np.testing.assert_allclose(
        _expand(fitted, active, (6, 10)),
        np.linalg.lstsq(gain, data, rcond=None)[0],
        atol=1e-12,
    )
    assert info["outer_stop_reason"] == "unregularized_least_squares"
