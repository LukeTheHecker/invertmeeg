import numpy as np
import pytest

from invert.solvers.base import BaseSolver
from invert.solvers.minimum_norm.dspm_mne import SolverDSPMMNE


def test_coerce_diag_source_prior_accepts_scalar_vector_and_diag():
    n_sources = 3

    scalar = BaseSolver.coerce_diag_source_prior(2.5, n_sources)
    np.testing.assert_allclose(scalar, [2.5, 2.5, 2.5])

    vector = BaseSolver.coerce_diag_source_prior(np.array([1.0, 2.0, 3.0]), n_sources)
    np.testing.assert_allclose(vector, [1.0, 2.0, 3.0])

    matrix = BaseSolver.coerce_diag_source_prior(np.diag([4.0, 5.0, 6.0]), n_sources)
    np.testing.assert_allclose(matrix, [4.0, 5.0, 6.0])


def test_coerce_diag_source_prior_rejects_full_matrix():
    cov = np.eye(3)
    cov[0, 1] = 0.1
    cov[1, 0] = 0.1
    with pytest.raises(ValueError, match="non-diagonal"):
        BaseSolver.coerce_diag_source_prior(cov, n_sources=3)


def test_compute_sensor_projector_identity_without_projs():
    solver = BaseSolver()
    info_like = {
        "projs": [],
        "bads": [],
        "ch_names": ["EEG001", "EEG002", "EEG003"],
    }
    projector = solver.compute_sensor_projector(forward_or_info=info_like, n_chans=3)
    np.testing.assert_allclose(projector, np.eye(3))


def test_compute_sensor_whitener_whitens_retained_subspace():
    noise_cov = np.diag([4.0, 1.0, 0.0])
    whitener = BaseSolver.compute_sensor_whitener(noise_cov, rank_tol=1e-12, eps=1e-15)
    assert whitener.shape == (2, 3)
    whitened_cov = whitener @ noise_cov @ whitener.T
    np.testing.assert_allclose(whitened_cov, np.eye(2), atol=1e-12)


def test_compute_depth_prior_whitened_depth_zero_returns_inverse_sensitivity():
    G_white = np.array([[1.0, 0.5, 2.0], [0.0, 0.5, 0.0]])
    prior = BaseSolver.compute_depth_prior_whitened(
        G_white, depth=0.0, depth_limit=10.0, eps=1e-15
    )
    expected = 1.0 / np.sum(G_white * G_white, axis=0)
    np.testing.assert_allclose(prior, expected)


def test_compute_depth_prior_whitened_clipping_with_limit_one():
    G_white = np.array([[1e-3, 1.0, 2.0]])
    prior = BaseSolver.compute_depth_prior_whitened(
        G_white, depth=1.0, depth_limit=1.0, eps=1e-15
    )
    np.testing.assert_allclose(prior, np.full(3, prior[0]))


def test_trace_normalize_operator_sets_target_trace():
    A = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, -0.5]])
    A_norm, scale = BaseSolver.trace_normalize_operator(A, target_rank=2, eps=1e-15)
    assert scale > 0
    np.testing.assert_allclose(np.sum(A_norm * A_norm), 2.0, atol=1e-12)


def test_solve_tikhonov_svd_matches_closed_form():
    rng = np.random.RandomState(0)
    A = rng.randn(4, 6)
    lambda2 = 0.2
    left_scale = np.abs(rng.randn(6)) + 0.1

    K = BaseSolver.solve_tikhonov_svd(A, lambda2, left_scale=left_scale, eps=1e-15)
    K_expected = np.diag(left_scale) @ A.T @ np.linalg.inv(
        A @ A.T + lambda2 * np.eye(A.shape[0])
    )
    np.testing.assert_allclose(K, K_expected, atol=1e-10)


def test_noise_normalize_rows_returns_finite_unit_norm_rows():
    K_white = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
    K_norm, noise_std = BaseSolver.noise_normalize_rows(K_white, eps=1e-12)
    np.testing.assert_allclose(noise_std[0], 5.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(K_norm[0]), 1.0, atol=1e-12)
    np.testing.assert_allclose(K_norm[1], 0.0, atol=1e-12)
    assert np.all(np.isfinite(K_norm))


def test_dspm_mne_uses_base_tikhonov_solver(monkeypatch, forward_model):
    calls = {"n": 0}
    original = BaseSolver.solve_tikhonov_svd

    def _wrapped(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(SolverDSPMMNE, "solve_tikhonov_svd", staticmethod(_wrapped))

    solver = SolverDSPMMNE(n_reg_params=3)
    solver.r_values = np.asarray([1e-3, 1e-2, 1e-1], dtype=float)
    n_chans = int(forward_model["sol"]["data"].shape[0])
    solver.make_inverse_operator(
        forward_model,
        alpha="auto",
        noise_cov=np.eye(n_chans, dtype=float),
    )

    assert calls["n"] == len(solver.alphas)
    kernel = solver.inverse_operators[0].data[0]
    assert kernel.shape == (solver.leadfield.shape[1], solver.leadfield.shape[0])
    assert np.all(np.isfinite(kernel))
