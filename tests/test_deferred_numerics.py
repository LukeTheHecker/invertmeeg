"""Independent invariants for projector, rank and depth corrections."""

import mne
import numpy as np
import pytest

from invert import Solver
from invert.solvers.base import BaseSolver, InverseOperator


def test_average_reference_projector():
    info = mne.create_info(["Fp1", "Fp2", "Cz"], 100, "eeg")
    evoked = mne.EvokedArray(np.eye(3), info, verbose=False)
    evoked.set_eeg_reference("average", projection=True, verbose=False)
    actual = BaseSolver().compute_sensor_projector(evoked.info, n_chans=3)
    np.testing.assert_allclose(actual, np.eye(3) - np.ones((3, 3)) / 3)


@pytest.mark.parametrize("method", ["auto", "L", "drop", "mean", "enhanced"])
def test_rank_invariant_to_units(method):
    rng = np.random.default_rng(4)
    data = rng.normal(size=(8, 100))
    data -= data.mean(axis=0)
    solver = BaseSolver()
    expected = solver.estimate_n_sources(data, method)
    for scale in (1e-100, 1e-9, 0.1, 10, 1e100):
        assert solver.estimate_n_sources(scale * data, method) == expected


@pytest.mark.parametrize("orientation", ["fixed", "free"])
@pytest.mark.parametrize("depth", [None, 0, 0.8])
def test_depth_kernel_matches_gaussian_posterior(
    forward_model_free_surface, simulated_evoked_free_surface, orientation, depth
):
    fwd = forward_model_free_surface
    data = simulated_evoked_free_surface
    names = fwd.ch_names[::2]
    covariance = mne.Covariance(
        np.diag(np.linspace(1, 3, len(names))), names, [], [], 100
    )
    solver = Solver("mne", depth=depth, orientation=orientation)
    solver.make_inverse_operator(fwd, data, noise_cov=covariance, alpha=0.2)
    gain = solver.leadfield
    n_orient = solver._n_orient
    prior = np.ones(gain.shape[1])
    if depth:
        powers = np.array(
            [
                np.linalg.svd(gain[:, j : j + n_orient], compute_uv=False)[0] ** 2
                for j in range(0, gain.shape[1], n_orient)
            ]
        )
        prior = np.repeat(np.maximum(powers / powers.max(), 0.01) ** -depth, n_orient)
    covariance_sensor = (gain * prior) @ gain.T + solver.alphas[0] * np.eye(len(gain))
    expected = (prior[:, None] * gain.T) @ np.linalg.solve(
        covariance_sensor, solver._sensor_transform
    )
    np.testing.assert_allclose(
        solver.inverse_operators[0].data[0], expected, rtol=1e-10, atol=1e-12
    )
    selected = Solver("mne", depth=depth, orientation=orientation)
    selected.make_inverse_operator(
        fwd.copy().pick_channels(names, ordered=True),
        data.copy().pick(names),
        noise_cov=covariance,
        alpha=0.2,
    )
    np.testing.assert_allclose(
        selected.inverse_operators[0].data[0], expected, rtol=1e-10, atol=1e-12
    )


def test_rgcv_uses_trace_square_under_sensor_similarity():
    solver = BaseSolver()
    gain = np.eye(3)
    transform = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]])
    influence = np.diag([0.2, 0.5, 0.7])
    kernel = np.linalg.solve(transform, influence @ transform)
    solver._leadfield_orig = gain
    solver.leadfield = transform
    solver._sensor_transform = transform
    solver.inverse_operators = [InverseOperator(kernel, "test")]
    solver.alphas = [0.2]
    data = np.arange(12.0).reshape(3, 4)
    residual = transform @ data - transform @ kernel @ data
    expected = np.sum(residual**2) / (3 - np.trace(influence)) ** 2
    expected *= 0.5 + 0.5 * np.trace(influence @ influence) / 3
    np.testing.assert_allclose(
        solver.compute_gcv_scores(data, method="rgcv"), [expected]
    )


@pytest.mark.parametrize(
    "name", ["mne", "dspm", "sloreta", "eloreta", "champagne", "lcmv", "esmv"]
)
@pytest.mark.parametrize("with_covariance", [False, True])
def test_public_solver_rejects_projected_sensor_component(
    name, with_covariance, forward_model, simulated_evoked
):
    n = len(forward_model.ch_names)
    solver = Solver(name, n_reg_params=1)
    covariance = (
        mne.Covariance(np.eye(n), forward_model.ch_names, [], [], 100)
        if with_covariance
        else None
    )
    solver.make_inverse_operator(
        forward_model, simulated_evoked, noise_cov=covariance, alpha=0.1
    )
    kernel = solver.inverse_operators[0].data[0]
    assert np.isfinite(kernel).all()
    np.testing.assert_allclose(
        kernel @ np.ones(n), 0, atol=1e-9 * max(np.linalg.norm(kernel), 1e-20)
    )


@pytest.mark.parametrize("name", ["mne", "dspm", "sloreta", "eloreta"])
@pytest.mark.parametrize("method", ["RGCV", "R1GCV"])
def test_regularization_public_entry(name, method, forward_model, simulated_evoked):
    solver = Solver(name, regularisation_method=method, n_reg_params=5)
    solver.make_inverse_operator(forward_model, simulated_evoked, alpha="auto")
    result = solver.apply_inverse_operator(simulated_evoked)
    assert np.isfinite(result.data).all()
    assert 0 <= solver.last_reg_idx < 5


def test_trace_normalized_kernel_preserves_physical_amplitude(
    forward_model, simulated_evoked
):
    solver = BaseSolver()
    solver.make_inverse_operator(forward_model, simulated_evoked)
    wf = solver.prepare_whitened_forward(None, trace_normalize=True)
    alpha = 0.1 * np.linalg.norm(wf.G_white, ord=2) ** 2
    actual = solver.solve_tikhonov_svd(
        wf.A, alpha * wf.trace_scale**2, left_scale=wf.R_sqrt
    )
    expected = wf.G_white.T @ np.linalg.inv(
        wf.G_white @ wf.G_white.T + alpha * np.eye(len(wf.G_white))
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize(
    "name", ["esmv", "ir-esmv", "ssp-esmv", "ssp-ir-esmv", "esmv-mvpure"]
)
def test_ssp_before_rank_selection(name, forward_model, simulated_evoked):
    clean = simulated_evoked.copy().apply_proj(verbose=False)
    contaminated = clean.copy()
    rng = np.random.default_rng(29)
    amplitude = np.max(np.abs(clean.data))
    contaminated.data[:] += 100 * amplitude * rng.normal(size=(1, clean.data.shape[1]))
    outputs = []
    for data in (clean, contaminated):
        solver = Solver(name, rank=3, n_reg_params=1)
        solver.make_inverse_operator(forward_model, data, alpha=0.1)
        outputs.append(solver.apply_inverse_operator(data).data)
    np.testing.assert_allclose(
        outputs[1], outputs[0], rtol=1e-7, atol=1e-8 * np.max(np.abs(outputs[0]))
    )


def test_invalid_projector_does_not_silently_disable_ssp(monkeypatch):
    info = mne.create_info(["Fp1", "Fp2", "Cz"], 100, "eeg")
    data = mne.EvokedArray(np.eye(3), info, verbose=False)
    data.set_eeg_reference("average", projection=True, verbose=False)

    def broken(*args, **kwargs):
        raise ValueError("invalid projector")

    monkeypatch.setattr(BaseSolver, "_make_sensor_projector", broken)
    with pytest.raises(ValueError, match="invalid projector"):
        BaseSolver().compute_sensor_projector(data.info, n_chans=3)


@pytest.mark.parametrize(
    "name", ["esmv", "ir-esmv", "ssp-esmv", "ssp-ir-esmv", "esmv-mvpure"]
)
@pytest.mark.parametrize("scale", [1e-9, 1e3])
def test_public_esmv_units(name, scale, forward_model, simulated_evoked):
    outputs = []
    for factor in (1.0, scale):
        data = simulated_evoked.copy()
        data.data[:] *= factor
        solver = Solver(name, n_reg_params=1)
        solver.make_inverse_operator(forward_model, data, alpha=0.1)
        outputs.append(solver.apply_inverse_operator(data).data / factor)
    np.testing.assert_allclose(
        outputs[1], outputs[0], rtol=1e-7, atol=1e-8 * np.max(np.abs(outputs[0]))
    )


@pytest.mark.parametrize("rank", [0, -1, True, 100])
def test_invalid_signal_rank_is_rejected(rank):
    with pytest.raises(ValueError, match="rank"):
        BaseSolver().select_signal_subspace(np.eye(5), rank=rank)


def test_numpy_integer_rank():
    result = BaseSolver().select_signal_subspace(
        np.diag([4.0, 3.0, 2.0]), rank=np.int64(2)
    )
    np.testing.assert_allclose(result, np.diag([4.0, 3.0, 0.0]))


def test_free_depth_prior_is_orientation_rotation_invariant(
    forward_model_free_surface, simulated_evoked_free_surface
):
    fwd = mne.convert_forward_solution(
        forward_model_free_surface, surf_ori=True, force_fixed=False, verbose=False
    )
    rng = np.random.default_rng(42)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    kernels = []
    for rotate in (False, True):
        current = fwd.copy()
        if rotate:
            gain = current["sol"]["data"]
            for j in range(0, gain.shape[1], 3):
                gain[:, j : j + 3] = gain[:, j : j + 3] @ rotation
        solver = Solver("mne", orientation="free", depth=0.8)
        solver.make_inverse_operator(current, simulated_evoked_free_surface, alpha=0.1)
        kernels.append(solver.inverse_operators[0].data[0])
    expected = kernels[0].copy()
    for j in range(0, len(expected), 3):
        expected[j : j + 3] = rotation.T @ expected[j : j + 3]
    np.testing.assert_allclose(kernels[1], expected, rtol=1e-9, atol=1e-12)
