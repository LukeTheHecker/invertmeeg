import gzip
import json

import mne
import numpy as np
import pytest

from invert.benchmark import runner as mod
from invert.benchmark.datasets import BENCHMARK_DATASETS


@pytest.mark.parametrize("solver_id,n_jobs", [("dspm", 1), ("lcmv", 2)])
def test_benchmark_matches_public_solver_per_recording(
    monkeypatch, forward_model, sensor_info, solver_id, n_jobs
):
    ds = BENCHMARK_DATASETS["multi_source"]
    runner = mod.BenchmarkRunner(
        forward_model,
        sensor_info,
        solvers=[solver_id],
        datasets={"multi": ds},
        n_samples=2,
        n_jobs=n_jobs,
        random_seed=37,
        depth_by_category={},
        depth_by_solver={},
    )
    batch = runner._generate_default_batch(ds)
    monkeypatch.setattr(runner, "_generate_default_batch", lambda _: batch)
    predictions = {}
    original = mod.evaluate_all

    def capture(truth, prediction, *geometry):
        predictions[truth.tobytes()] = prediction.copy()
        return original(truth, prediction, *geometry)

    monkeypatch.setattr(mod, "evaluate_all", capture)
    (result,) = runner.run()
    spec = mod.get_solver_spec(solver_id)
    for i, sample in enumerate(result.samples):
        assert sample.error is None
        solver = mod.get_solver_class(solver_id)(**spec.default_kwargs)
        data = mne.EvokedArray(batch.x[i], sensor_info, verbose=False)
        covariance = mod._make_mne_covariance(
            batch.info.iloc[i]["noise_cov_est_scaled"], sensor_info
        )
        args = (data,) if solver.require_data else ()
        solver.make_inverse_operator(
            forward_model,
            *args,
            alpha="auto",
            noise_cov=covariance,
        )
        expected = solver.apply_inverse_operator(data).data
        np.testing.assert_allclose(
            predictions[batch.y[i].tobytes()], expected, rtol=1e-10, atol=0
        )


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_failed_case_survives_save_and_cannot_rank(
    monkeypatch, tmp_path, forward_model, sensor_info, n_jobs
):
    runner = mod.BenchmarkRunner(
        forward_model,
        sensor_info,
        solvers=["mne"],
        datasets={"focal": BENCHMARK_DATASETS["focal_source"]},
        n_samples=2,
        random_seed=37,
        n_jobs=n_jobs,
    )
    worker = mod._compute_and_apply_worker

    def fail_second(idx, *args):
        if idx == 1:
            raise ValueError("second recording failed")
        return worker(idx, *args)

    monkeypatch.setattr(mod, "_compute_and_apply_worker", fail_second)
    runner.run()
    path = tmp_path / "results.json.gz"
    runner.save(path, compact=True)
    data = json.loads(
        gzip.decompress(path.read_bytes()),
        parse_constant=lambda value: pytest.fail(value),
    )
    assert data["global_ranks"] == {}
    assert data["ranks"] == {"focal": {}}
    (result,) = runner.load(path)
    assert result.case_count == 2
    assert result.metrics["average_precision"].valid_count == 1
    assert result.samples[1].case_id == 1
    assert "second recording failed" in result.samples[1].error


def test_neural_mass_solver_receives_simulated_clock(
    monkeypatch, forward_model, sensor_info
):
    original_rate = sensor_info["sfreq"]
    scenario = BENCHMARK_DATASETS["nmm"].model_copy(update={"n_pool": 5})
    runner = mod.BenchmarkRunner(
        forward_model,
        sensor_info,
        solvers=["dics"],
        datasets={"nmm": scenario},
        n_samples=1,
        n_jobs=1,
        random_seed=42,
    )
    original_worker = mod._compute_and_apply_worker
    recording_rates = []

    def capture(idx, module, cls, fwd, info, x, *args):
        recording_rates.append((info["sfreq"], x.shape[-1]))
        return original_worker(idx, module, cls, fwd, info, x, *args)

    monkeypatch.setattr(mod, "_compute_and_apply_worker", capture)
    (row,) = runner.run()
    assert row.samples[0].error is None
    assert recording_rates == [
        (scenario.sfreq, int(scenario.sfreq * scenario.duration_s))
    ]
    assert sensor_info["sfreq"] == original_rate
