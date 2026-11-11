"""Equation checks and BaseSolver adapters for irMxNE and TF-MxNE."""

import mne
import numpy as np
import pytest
from mne.inverse_sparse.mxne_optim import iterative_mixed_norm_solver

from invert import Solver, list_solvers
from invert.solvers.minimum_norm.mixed_norm import SolverIRMxNE, SolverTFMxNE


def test_irmxne_orthogonal_stationary_equation():
    """An active scalar block satisfies x-y+lambda/(2 sqrt(x))=0."""
    Y = np.array([[2.0], [0.01]])
    X, active, history = iterative_mixed_norm_solver(
        Y,
        np.eye(2),
        0.2,
        100,
        tol=1e-11,
        maxit=2000,
        debias=False,
        solver="cd",
        verbose=False,
    )
    assert active.tolist() == [True, False]
    x = float(X[0, 0])
    assert abs(x - 2 + 0.2 / (2 * np.sqrt(x))) < 1e-8
    assert np.max(np.diff(history)) < 1e-9


@pytest.mark.parametrize(
    "solver_id,cls", [("ir-mxne", SolverIRMxNE), ("tfmxne", SolverTFMxNE)]
)
def test_factory(solver_id, cls):
    assert solver_id in list_solvers()
    assert isinstance(Solver(solver_id), cls)


@pytest.mark.parametrize("cls", [SolverIRMxNE, SolverTFMxNE])
def test_mixed_norm_applies_eeg_average_reference(cls, forward_model, simulated_evoked):
    n_channels = len(forward_model.ch_names)
    solver = cls()
    solver.make_inverse_operator(forward_model, simulated_evoked)

    projector = np.eye(n_channels) - np.ones((n_channels, n_channels)) / n_channels
    np.testing.assert_allclose(solver._sensor_transform, projector)


@pytest.mark.parametrize("cls", [SolverIRMxNE, SolverTFMxNE])
def test_whitening_scaling_zero_and_channels(cls, forward_model, simulated_evoked):
    fwd, evoked = forward_model, simulated_evoked
    cov = mne.Covariance(
        np.diag(np.linspace(1, 2, len(fwd.ch_names))), fwd.ch_names, [], [], 100
    )
    solver = cls(depth=0, debias=False, max_iter=1500, tol=1e-7)
    solver.make_inverse_operator(fwd, evoked, alpha=0.08, noise_cov=cov)
    original = solver.apply_inverse_operator(evoked).data
    assert np.isfinite(original).all() and np.linalg.norm(original) > 0
    scaled = evoked.copy()
    scaled.data *= 1e-12
    np.testing.assert_allclose(
        solver.apply_inverse_operator(scaled).data / 1e-12,
        original,
        rtol=1e-7,
        atol=1e-10,
    )
    reordered = evoked.copy().reorder_channels(list(reversed(evoked.ch_names)))
    np.testing.assert_allclose(
        solver.apply_inverse_operator(reordered).data, original, rtol=1e-7, atol=1e-10
    )
    zero = evoked.copy()
    zero.data[:] = 0
    assert not np.any(solver.apply_inverse_operator(zero).data)
    assert not np.any(solver.apply_inverse_operator(evoked, alpha=1).data)


def test_irmxne_adapter_matches_reference(forward_model, simulated_evoked):
    solver = SolverIRMxNE(depth=0, debias=False, n_mxne_iter=4)
    solver.make_inverse_operator(forward_model, simulated_evoked, alpha=0.1)
    estimate = solver.apply_inverse_operator(simulated_evoked).data
    G = solver.leadfield / solver.last_solver_info["gain_scale"]
    M = (
        solver._sensor_transform
        @ simulated_evoked.data
        / solver.last_solver_info["data_scale"]
    )
    X, active, _ = iterative_mixed_norm_solver(
        M,
        G,
        solver.last_solver_info["lambda"],
        4,
        maxit=2000,
        tol=1e-6,
        active_set_size=10,
        debias=False,
        solver="cd",
        verbose=False,
    )
    expected = np.zeros_like(estimate)
    expected[active] = (
        X
        * solver.last_solver_info["data_scale"]
        / solver.last_solver_info["gain_scale"]
    )
    np.testing.assert_allclose(estimate, expected, rtol=1e-10, atol=1e-12)


def test_tf_without_temporal_penalty_equals_convex_mxne(
    forward_model, simulated_evoked
):
    """The tight STFT synthesis formulation reduces to group Lasso at l1_ratio=0."""
    irm = SolverIRMxNE(depth=0, debias=False, n_mxne_iter=1, tol=1e-9, max_iter=4000)
    tf = SolverTFMxNE(
        depth=0, debias=False, l1_ratio=0, tol=1e-9, max_iter=4000, wsize=16, tstep=4
    )
    for s in (irm, tf):
        s.make_inverse_operator(forward_model, simulated_evoked, alpha=0.1)
    x = irm.apply_inverse_operator(simulated_evoked).data
    z = tf.apply_inverse_operator(simulated_evoked).data
    np.testing.assert_allclose(z, x, rtol=2e-3, atol=2e-4 * np.max(np.abs(x)))
    assert tf.last_solver_info["duality_gap"] < 1e-7


@pytest.mark.parametrize("cls", [SolverIRMxNE, SolverTFMxNE])
def test_free_orientation(
    cls, forward_model_free_surface, simulated_evoked_free_surface
):
    solver = cls(orientation="free", depth=0, debias=False, max_iter=200)
    solver.make_inverse_operator(
        forward_model_free_surface, simulated_evoked_free_surface, alpha=0.3
    )
    stc = solver.apply_inverse_operator_vector(simulated_evoked_free_surface)
    assert isinstance(stc, mne.VectorSourceEstimate)
    assert stc.data.shape[1] == 3
    assert np.isfinite(stc.data).all()


@pytest.mark.parametrize("cls", [SolverIRMxNE, SolverTFMxNE])
def test_pca_orientation(
    cls, forward_model_free_surface, simulated_evoked_free_surface
):
    solver = cls(orientation="pca", depth=0, debias=False, max_iter=300)
    solver.make_inverse_operator(
        forward_model_free_surface, simulated_evoked_free_surface, alpha=0.3
    )
    stc = solver.apply_inverse_operator(simulated_evoked_free_surface)
    assert stc.data.shape == (
        forward_model_free_surface["nsource"],
        len(simulated_evoked_free_surface.times),
    )
    assert np.isfinite(stc.data).all()
    assert np.linalg.norm(stc.data) > 0


def test_tf_reweighted_and_validation(forward_model, simulated_evoked):
    solver = SolverTFMxNE(
        n_tfmxne_iter=2, max_iter=300, wsize=16, tstep=4, debias=False
    )
    solver.make_inverse_operator(forward_model, simulated_evoked, alpha=0.1)
    stc = solver.apply_inverse_operator(simulated_evoked)
    assert np.isfinite(stc.data).all() and np.linalg.norm(stc.data) > 0
    with pytest.raises(ValueError, match="wsize"):
        solver.apply_inverse_operator(simulated_evoked, wsize=15)
    with pytest.raises(ValueError, match="alpha"):
        solver.apply_inverse_operator(simulated_evoked, alpha=-0.1)
