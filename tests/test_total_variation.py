import numpy as np
import mne

from invert.benchmark import BenchmarkRunner
from invert.benchmark.datasets import BENCHMARK_DATASETS
from invert.benchmark.runner import evaluate_all
from invert.forward import create_forward_model, get_info
from invert.solvers.minimum_norm.total_variation import SolverTotalVariation
from invert.util import build_source_adjacency
from invert.util.util import pos_from_forward


def _reference_apply_total_variation_dense(
    solver,
    data,
    *,
    tv_weight,
    n_irls,
    eps,
    ridge,
    auto_scale_hyperparams,
    cg_tol=None,
    cg_max_iter=None,
):
    leadfield = solver.leadfield
    n_chans, n_sources = leadfield.shape
    if data.ndim == 1:
        data = data[:, np.newaxis]

    if auto_scale_hyperparams:
        lam, ridge_eff = solver._auto_scaled_hyperparams(
            leadfield,
            solver._edges_i,
            solver._edges_j,
            tv_weight=float(tv_weight),
            ridge=ridge,
        )
    else:
        lam = max(float(tv_weight), 0.0)
        ridge_eff = 1e-6 if ridge is None else max(float(ridge), 1e-15)

    ltl = leadfield.T @ leadfield
    sensor_system = leadfield @ leadfield.T + ridge_eff * np.eye(n_chans)
    x = leadfield.T @ np.linalg.solve(sensor_system, data)
    b = leadfield.T @ data

    edges_i = solver._edges_i
    edges_j = solver._edges_j

    base_system = ltl + ridge_eff * np.eye(n_sources)

    for _ in range(int(n_irls)):
        diff = x[edges_i] - x[edges_j]
        grad_sq = np.sum(diff * diff, axis=1)
        w = 1.0 / np.sqrt(grad_sq + float(eps) ** 2)

        system = base_system.copy()
        scaled = lam * w
        np.add.at(system, (edges_i, edges_i), scaled)
        np.add.at(system, (edges_j, edges_j), scaled)
        np.add.at(system, (edges_i, edges_j), -scaled)
        np.add.at(system, (edges_j, edges_i), -scaled)

        x_new = np.linalg.solve(system, b)

        rel = float(np.linalg.norm(x_new - x)) / max(float(np.linalg.norm(x)), 1e-15)
        x = x_new
        if rel < 1e-3:
            break

    return x


def test_total_variation_weighted_laplacian_matches_dense_reference():
    solver = SolverTotalVariation(n_reg_params=1)
    solver._edges_i = np.array([0, 0, 1, 2], dtype=int)
    solver._edges_j = np.array([1, 2, 3, 3], dtype=int)
    solver.leadfield = np.zeros((1, 4), dtype=float)

    weights = np.array([0.5, 1.0, 1.5, 2.0], dtype=float)

    actual = solver._build_weighted_laplacian(weights).toarray()

    expected = np.zeros((4, 4), dtype=float)
    for ii, jj, ww in zip(solver._edges_i, solver._edges_j, weights, strict=False):
        expected[ii, ii] += ww
        expected[jj, jj] += ww
        expected[ii, jj] -= ww
        expected[jj, ii] -= ww

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_total_variation_does_not_require_recompute():
    solver = SolverTotalVariation(n_reg_params=1)
    assert solver.require_recompute is False


def test_total_variation_matches_dense_reference(forward_model, simulated_evoked):
    solver = SolverTotalVariation(n_reg_params=1)
    solver.make_inverse_operator(forward_model, simulated_evoked, alpha=0.1)

    whitened = solver._sensor_transform @ simulated_evoked.data
    kwargs = dict(
        tv_weight=0.02,
        n_irls=4,
        eps=1e-3,
        ridge=1e-4,
        auto_scale_hyperparams=False,
        cg_tol=1e-12,
        cg_max_iter=2000,
    )

    actual = solver.apply_inverse_operator(simulated_evoked, **kwargs).data
    expected = _reference_apply_total_variation_dense(solver, whitened, **kwargs)

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-4)


def test_total_variation_default_tol_preserves_benchmark_metrics():
    info = get_info(kind="biosemi32")
    forward = create_forward_model(sampling="ico3", info=info)
    runner = BenchmarkRunner(
        forward,
        info,
        n_samples=1,
        categories=["minimum_norm"],
        solvers=["tv"],
        n_jobs=1,
        random_seed=42,
    )
    dataset = BENCHMARK_DATASETS["focal_source"]
    x_batch, y_batch, _sim_info = runner._generate_default_batch(dataset)
    evoked = mne.EvokedArray(x_batch[0], info, tmin=0.0, verbose=0)

    solver = SolverTotalVariation(n_reg_params=1)
    solver.make_inverse_operator(forward, evoked, alpha=0.1)

    stc_default = solver.apply_inverse_operator(evoked)
    stc_reference = solver.apply_inverse_operator(evoked, cg_tol=1e-4)

    adjacency = build_source_adjacency(forward["src"], verbose=0)
    pos = pos_from_forward(forward)
    metrics_default = evaluate_all(
        y_batch[0], stc_default.data, adjacency, adjacency, pos, pos
    )
    metrics_reference = evaluate_all(
        y_batch[0], stc_reference.data, adjacency, adjacency, pos, pos
    )

    assert metrics_default["EMD"] <= metrics_reference["EMD"] + 1e-3
    assert (
        metrics_default["Mean_Localization_Error"]
        <= metrics_reference["Mean_Localization_Error"] + 1e-3
    )
    assert (
        metrics_default["average_precision"]
        >= metrics_reference["average_precision"] - 1e-3
    )
    assert metrics_default["correlation"] >= metrics_reference["correlation"] - 1e-3
