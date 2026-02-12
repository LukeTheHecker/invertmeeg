"""Evaluate dSPM vs dSPM-MNE under realistic correlated noise conditions.

This script uses the upgraded SimulationGenerator with realistic sensor noise:
- correlated low-rank spatial covariance
- temporal 1/f coloring
- projector-induced rank deficiency
- per-sample estimated noise covariance made available to solvers

It evaluates both solvers on the standard benchmark datasets with 50 samples each.
"""

from __future__ import annotations

import importlib.util as _ilu
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import mne
import numpy as np

from invert import Solver
from invert.benchmark.datasets import BENCHMARK_DATASETS
from invert.benchmark.runner import _make_mne_covariance
from invert.forward import create_forward_model, get_info
from invert.simulate import SimulationConfig, SimulationGenerator
from invert.util.util import pos_from_forward

SOLVERS = ["dSPM", "dSPM-MNE"]
N_SAMPLES = 50
RANDOM_SEED = 42

METRIC_DIRECTIONS = {
    "Mean_Localization_Error": "min",
    "EMD": "min",
    "sd": "min",
    "average_precision": "max",
    "correlation": "max",
}


def _load_evaluate_all():
    eval_path = (
        Path(__file__).resolve().parents[1] / "invert" / "evaluate" / "evaluate.py"
    )
    spec = _ilu.spec_from_file_location("invert.evaluate.evaluate", str(eval_path))
    assert spec is not None and spec.loader is not None
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.evaluate_all


def _aggregate(sample_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    arrs: dict[str, list[float]] = defaultdict(list)
    for metric_dict in sample_metrics:
        for key, value in metric_dict.items():
            arrs[key].append(float(value))

    out: dict[str, dict[str, float]] = {}
    for key, vals in arrs.items():
        vec = np.asarray(vals, dtype=float)
        out[key] = {
            "mean": float(np.nanmean(vec)),
            "std": float(np.nanstd(vec)),
            "median": float(np.nanmedian(vec)),
        }
    return out


def _solver_rank(values: dict[str, float], metric: str) -> dict[str, float]:
    direction = METRIC_DIRECTIONS[metric]
    items = list(values.items())
    if direction == "min":
        items.sort(key=lambda item: item[1])
    else:
        items.sort(key=lambda item: item[1], reverse=True)

    ranks: dict[str, float] = {}
    for idx, (solver_name, _value) in enumerate(items, start=1):
        ranks[solver_name] = float(idx)
    return ranks


def _compute_ranks(
    dataset_results: dict[str, dict[str, dict[str, dict[str, float]]]],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    per_dataset: dict[str, dict[str, float]] = {}
    solver_dataset_ranks: dict[str, list[float]] = defaultdict(list)

    for ds_name, solver_data in dataset_results.items():
        metric_ranks_per_solver: dict[str, list[float]] = defaultdict(list)
        for metric in METRIC_DIRECTIONS:
            values = {
                solver_name: solver_data[solver_name][metric]["mean"]
                for solver_name in solver_data
            }
            ranks = _solver_rank(values, metric)
            for solver_name, rank_val in ranks.items():
                metric_ranks_per_solver[solver_name].append(rank_val)

        per_dataset[ds_name] = {}
        for solver_name, rank_vals in metric_ranks_per_solver.items():
            mean_rank = float(np.mean(rank_vals))
            per_dataset[ds_name][solver_name] = mean_rank
            solver_dataset_ranks[solver_name].append(mean_rank)

    global_ranks = {
        solver_name: float(np.mean(rank_vals))
        for solver_name, rank_vals in solver_dataset_ranks.items()
    }
    return per_dataset, global_ranks


def main() -> Path:
    evaluate_all = _load_evaluate_all()

    info = get_info(kind="biosemi32")
    forward = create_forward_model(sampling="ico2", info=info)
    adjacency = mne.spatial_src_adjacency(forward["src"], verbose=0)
    pos = pos_from_forward(forward)

    realistic_noise_cfg = {
        "correlation_mode": "low_rank",
        "noise_color_coeff": (0.4, 0.9),
        "noise_temporal_beta": (0.5, 1.5),
        "noise_rank_deficiency": (1, 3),
        "noise_low_rank_dim": (3, 8),
        "apply_sensor_projector": True,
        "return_noise_cov": True,
        "estimate_noise_cov": True,
        "noise_cov_n_baseline": 300,
        "noise_cov_shrinkage": 0.05,
    }

    dataset_results: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    dataset_noise_stats: dict[str, dict[str, float]] = {}

    for ds_idx, (ds_name, ds_cfg) in enumerate(BENCHMARK_DATASETS.items()):
        sim_config = SimulationConfig(
            batch_size=N_SAMPLES,
            batch_repetitions=1,
            n_sources=ds_cfg.n_sources,
            n_orders=ds_cfg.n_orders,
            snr_range=ds_cfg.snr_range,
            n_timepoints=ds_cfg.n_timepoints,
            random_seed=RANDOM_SEED + ds_idx,
            **realistic_noise_cfg,
        )
        generator = SimulationGenerator(forward, config=sim_config)
        x_batch, y_batch, sim_info = next(generator.generate())

        dataset_noise_stats[ds_name] = {
            "snr_target_mean": float(np.mean(sim_info["snr"])),
            "snr_realized_mean": float(np.mean(sim_info["snr_realized"])),
            "projector_rank_mean": float(np.mean(sim_info["projector_rank"])),
            "noise_cov_rank_true_mean": float(np.mean(sim_info["noise_cov_rank_true"])),
            "noise_cov_rank_est_mean": float(np.mean(sim_info["noise_cov_rank_est"])),
        }

        dataset_results[ds_name] = {}
        for solver_name in SOLVERS:
            sample_metrics: list[dict[str, float]] = []
            for sample_idx in range(N_SAMPLES):
                evoked = mne.EvokedArray(
                    x_batch[sample_idx], info.copy(), tmin=0.0, verbose=0
                )
                noise_cov_est = sim_info.iloc[sample_idx]["noise_cov_est"]
                noise_cov = _make_mne_covariance(
                    noise_cov_est, info, nfree=sim_config.noise_cov_n_baseline
                )

                solver = Solver(solver_name)
                solver.make_inverse_operator(forward, alpha="auto", noise_cov=noise_cov)
                stc = solver.apply_inverse_operator(evoked)

                metrics = evaluate_all(
                    y_batch[sample_idx], stc.data, adjacency, adjacency, pos, pos
                )
                sample_metrics.append({k: float(v) for k, v in metrics.items()})

            dataset_results[ds_name][solver_name] = _aggregate(sample_metrics)

    ranks, global_ranks = _compute_ranks(dataset_results)

    payload = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_samples": N_SAMPLES,
            "solvers": SOLVERS,
            "random_seed": RANDOM_SEED,
            "forward_sampling": "ico2",
        },
        "realistic_noise_config": realistic_noise_cfg,
        "dataset_noise_stats": dataset_noise_stats,
        "dataset_results": dataset_results,
        "ranks": ranks,
        "global_ranks": global_ranks,
    }

    out_path = Path("results/compare_dspm_realistic_noise.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    print(f"Saved results to {out_path}")
    print("Global ranks (lower is better):")
    for solver_name in SOLVERS:
        print(f"  {solver_name}: {global_ranks[solver_name]:.3f}")

    return out_path


if __name__ == "__main__":
    main()
