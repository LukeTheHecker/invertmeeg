"""Evaluate beamformer regularization range behavior across all datasets.

Focuses on beamformer solvers that search over multiple regularization
candidates. For each dataset and solver, runs two conditions:
  (a) noise_cov = None
  (b) noise_cov = estimated (mean over generated samples)

Primary diagnostics:
  - whether regularization edge warning was emitted
  - selected regularization index / alpha
  - whether selected index is on tested-range edge

Configuration:
  - sensor: biosemi32
  - sampling: ico2
  - n_samples: 20
  - n_timepoints: 200
"""

from __future__ import annotations

import importlib.util as _ilu
import json
import logging
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import mne
import numpy as np

from invert.benchmark.datasets import BENCHMARK_DATASETS
from invert.benchmark.runner import (
    _make_inverse_operator_with_covariance,
    _make_mne_covariance,
    get_solver_class,
    resolve_solvers,
)
from invert.forward import create_forward_model, get_info
from invert.simulate import SimulationConfig, SimulationGenerator
from invert.solvers.base import BaseSolver
from invert.util.util import pos_from_forward

_spec = _ilu.spec_from_file_location(
    "invert.evaluate.evaluate",
    str(Path(__file__).resolve().parents[1] / "invert" / "evaluate" / "evaluate.py"),
)
assert _spec is not None and _spec.loader is not None
_eval_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_eval_mod)
evaluate_all = _eval_mod.evaluate_all

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Evaluate all beamformer solvers while excluding known slow methods.
EXCLUDED_SOLVERS = [
    "ReciPSIICOS-Plain",
    "ReciPSIICOS-Whitened",
    "EBB",
    "SESAME",
]
SOLVERS = resolve_solvers(categories=["beamformer"], exclude=EXCLUDED_SOLVERS)
CONDITIONS = ["none", "estimated"]

N_SAMPLES = 20
N_TIMEPOINTS = 200
RANDOM_SEED = 42
SAMPLING = "ico2"
SENSOR_KIND = "biosemi32"

METRICS = ["mle", "emd", "sd", "ap", "correlation"]
METRIC_KEYS = {
    "mle": "Mean_Localization_Error",
    "emd": "EMD",
    "sd": "sd",
    "ap": "average_precision",
    "correlation": "correlation",
}

EDGE_WARNING_SNIPPET = "regularization parameter in the search range"


class _LogCollector(logging.Handler):
    def __init__(self, *, level: int = logging.WARNING) -> None:
        super().__init__(level=level)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(self.format(record))


@contextmanager
def capture_base_warnings() -> Any:
    base_logger = logging.getLogger("invert.solvers.base")
    collector = _LogCollector(level=logging.WARNING)
    collector.setFormatter(logging.Formatter("%(message)s"))
    base_logger.addHandler(collector)
    old_level = base_logger.level
    old_propagate = base_logger.propagate
    base_logger.propagate = False
    if old_level > logging.WARNING:
        base_logger.setLevel(logging.WARNING)
    try:
        yield collector.messages
    finally:
        base_logger.removeHandler(collector)
        base_logger.propagate = old_propagate
        base_logger.setLevel(old_level)


def _extract_selection_diagnostics(
    solver: BaseSolver,
    warning_messages: list[str],
) -> dict[str, Any]:
    n_inverse_operators = len(getattr(solver, "inverse_operators", []) or [])
    n_alphas = len(getattr(solver, "alphas", []) or [])
    n_tested = n_inverse_operators if n_inverse_operators > 0 else n_alphas
    selected_idx = getattr(solver, "last_reg_idx", None)
    selected_alpha = None
    if selected_idx is not None and n_alphas > 0 and 0 <= int(selected_idx) < n_alphas:
        selected_alpha = float(solver.alphas[int(selected_idx)])
    elif n_alphas == 1:
        selected_alpha = float(solver.alphas[0])

    edge_selected = False
    if selected_idx is not None and n_tested > 1:
        idx_int = int(selected_idx)
        edge_selected = idx_int in (0, n_tested - 1)

    edge_warning_messages = [
        msg for msg in warning_messages if EDGE_WARNING_SNIPPET in msg
    ]
    warning_emitted = len(edge_warning_messages) > 0

    return {
        "selected_idx": int(selected_idx) if selected_idx is not None else None,
        "selected_alpha": selected_alpha,
        "n_alphas": int(n_alphas),
        "n_inverse_operators": int(n_inverse_operators),
        "n_tested": int(n_tested),
        "edge_selected": bool(edge_selected),
        "edge_warning_emitted": bool(warning_emitted),
        "edge_warning_messages": edge_warning_messages,
    }


def _nan_metrics() -> dict[str, float]:
    return {m: float("nan") for m in METRICS}


def aggregate_metrics(sample_records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for m in METRICS:
        vals = np.array([float(s["metrics"][m]) for s in sample_records], dtype=float)
        out[m] = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals)),
            "median": float(np.nanmedian(vals)),
        }
    return out


def aggregate_diagnostics(sample_records: list[dict[str, Any]]) -> dict[str, Any]:
    n_total = len(sample_records)
    n_failed = sum(1 for s in sample_records if s.get("error") is not None)
    n_success = n_total - n_failed

    warn_count = sum(
        1
        for s in sample_records
        if s["diagnostics"].get("edge_warning_emitted", False)
    )
    edge_count = sum(
        1 for s in sample_records if s["diagnostics"].get("edge_selected", False)
    )
    low_edge_count = sum(
        1 for s in sample_records if s["diagnostics"].get("selected_idx", None) == 0
    )
    high_edge_count = sum(
        1
        for s in sample_records
        if (
            s["diagnostics"].get("selected_idx", None) is not None
            and s["diagnostics"].get("n_tested", 0) > 1
            and s["diagnostics"]["selected_idx"] == s["diagnostics"]["n_tested"] - 1
        )
    )

    selected_positions: list[float] = []
    for s in sample_records:
        idx = s["diagnostics"].get("selected_idx", None)
        n_tested = int(s["diagnostics"].get("n_tested", 0))
        if idx is None or n_tested <= 1:
            continue
        selected_positions.append(float(idx) / float(max(n_tested - 1, 1)))

    return {
        "n_total": int(n_total),
        "n_success": int(n_success),
        "n_failed": int(n_failed),
        "edge_warning_count": int(warn_count),
        "edge_warning_rate": float(warn_count / n_total) if n_total else float("nan"),
        "edge_selected_count": int(edge_count),
        "edge_selected_rate": float(edge_count / n_total) if n_total else float("nan"),
        "low_edge_count": int(low_edge_count),
        "high_edge_count": int(high_edge_count),
        "median_selected_position": (
            float(np.median(np.array(selected_positions, dtype=float)))
            if selected_positions
            else None
        ),
    }


def evaluate_solver_condition(
    solver_name: str,
    forward: mne.Forward,
    info: mne.Info,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    adjacency,
    pos: np.ndarray,
    noise_cov: mne.Covariance | None,
) -> list[dict[str, Any]]:
    solver_cls = get_solver_class(solver_name)
    probe = solver_cls()
    require_data = bool(getattr(probe, "require_data", True))

    sample_records: list[dict[str, Any]] = []
    solver_shared: BaseSolver = solver_cls()
    build_error: str | None = None
    build_warning_messages: list[str] = []

    evoked_build = (
        mne.EvokedArray(x_batch[0], info, tmin=0.0, verbose=0) if require_data else None
    )
    try:
        with capture_base_warnings() as warnings_buffer:
            _make_inverse_operator_with_covariance(
                solver=solver_shared,
                forward=forward,
                require_data=require_data,
                evoked=evoked_build,
                alpha="auto",
                noise_cov=noise_cov,
            )
            build_warning_messages = list(warnings_buffer)
    except Exception as exc:
        build_error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    for i in range(x_batch.shape[0]):
        evoked = mne.EvokedArray(x_batch[i], info, tmin=0.0, verbose=0)

        if build_error is not None:
            sample_records.append(
                {
                    "sample_idx": int(i),
                    "metrics": _nan_metrics(),
                    "diagnostics": {
                        "selected_idx": None,
                        "selected_alpha": None,
                        "n_alphas": 0,
                        "n_inverse_operators": 0,
                        "n_tested": 0,
                        "edge_selected": False,
                        "edge_warning_emitted": False,
                        "edge_warning_messages": [],
                    },
                    "error": build_error,
                }
            )
            continue

        solver = solver_shared
        warning_messages: list[str] = []
        if i == 0 and build_warning_messages:
            warning_messages.extend(build_warning_messages)

        try:
            with capture_base_warnings() as warnings_buffer:
                stc = solver.apply_inverse_operator(evoked)
                warning_messages.extend(list(warnings_buffer))

            y_pred = stc.data
            raw = evaluate_all(y_batch[i], y_pred, adjacency, adjacency, pos, pos)
            metrics = {m: float(raw[METRIC_KEYS[m]]) for m in METRICS}
            diagnostics = _extract_selection_diagnostics(solver, warning_messages)

            sample_records.append(
                {
                    "sample_idx": int(i),
                    "metrics": metrics,
                    "diagnostics": diagnostics,
                    "error": None,
                }
            )
        except Exception as exc:
            traceback.print_exc()
            diagnostics = _extract_selection_diagnostics(solver, warning_messages)
            sample_records.append(
                {
                    "sample_idx": int(i),
                    "metrics": _nan_metrics(),
                    "diagnostics": diagnostics,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    return sample_records


def main() -> None:
    print("=" * 80)
    print("Beamformer Regularization Range Evaluation")
    print("=" * 80)
    print(
        f"Config: {SENSOR_KIND}/{SAMPLING}, n_samples={N_SAMPLES}, n_timepoints={N_TIMEPOINTS}, seed={RANDOM_SEED}"
    )
    print(f"Solvers ({len(SOLVERS)}): {', '.join(SOLVERS)}")
    print(f"Conditions: {', '.join(CONDITIONS)}")

    info = get_info(kind=SENSOR_KIND)
    forward = create_forward_model(sampling=SAMPLING, info=info)
    pos = pos_from_forward(forward)
    adjacency = mne.spatial_src_adjacency(forward["src"], verbose=0)
    n_sources = int(forward["sol"]["data"].shape[1])

    print(f"Forward model sources: {n_sources}")
    print(f"Datasets: {', '.join(BENCHMARK_DATASETS.keys())}")

    all_results: dict[str, dict[str, dict[str, Any]]] = {}

    for dataset_name, dataset_cfg in BENCHMARK_DATASETS.items():
        print(f"\nDataset: {dataset_name}")
        sim_config = SimulationConfig(
            batch_size=N_SAMPLES,
            n_sources=dataset_cfg.n_sources,
            n_orders=dataset_cfg.n_orders,
            snr_range=dataset_cfg.snr_range,
            n_timepoints=N_TIMEPOINTS,
            random_seed=RANDOM_SEED,
            estimate_noise_cov=True,
            return_noise_cov=True,
        )
        generator = SimulationGenerator(forward, config=sim_config)
        x_batch, y_batch, sim_info = next(generator.generate())

        noise_cov_est_array = np.mean(
            [sim_info.iloc[j]["noise_cov_est"] for j in range(N_SAMPLES)],
            axis=0,
        )
        noise_covs = {
            "none": None,
            "estimated": _make_mne_covariance(noise_cov_est_array, info, nfree=N_SAMPLES),
        }

        all_results[dataset_name] = {}
        for solver_name in SOLVERS:
            all_results[dataset_name][solver_name] = {}
            for condition in CONDITIONS:
                print(f"  {solver_name:<10s} | {condition:<9s} ...", end="", flush=True)
                samples = evaluate_solver_condition(
                    solver_name=solver_name,
                    forward=forward,
                    info=info,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    adjacency=adjacency,
                    pos=pos,
                    noise_cov=noise_covs[condition],
                )
                metric_agg = aggregate_metrics(samples)
                diag_agg = aggregate_diagnostics(samples)
                all_results[dataset_name][solver_name][condition] = {
                    "aggregate_metrics": metric_agg,
                    "aggregate_diagnostics": diag_agg,
                    "samples": samples,
                }
                print(
                    f" done (warn_rate={diag_agg['edge_warning_rate']:.2f}, edge_rate={diag_agg['edge_selected_rate']:.2f})"
                )

    out_dir = Path("results/release")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "beamformer_reg_range_eval.json"

    payload = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "sensor_kind": SENSOR_KIND,
            "sampling": SAMPLING,
            "n_samples": N_SAMPLES,
            "n_timepoints": N_TIMEPOINTS,
            "random_seed": RANDOM_SEED,
            "solvers": SOLVERS,
            "conditions": CONDITIONS,
            "datasets": {
                name: {
                    "n_sources": cfg.n_sources,
                    "n_orders": cfg.n_orders,
                    "snr_range": cfg.snr_range,
                    "n_timepoints": N_TIMEPOINTS,
                }
                for name, cfg in BENCHMARK_DATASETS.items()
            },
        },
        "results": all_results,
    }

    out_path.write_text(json.dumps(payload, indent=2))
    print("\nSaved:", out_path)
    print("\nSummary (edge-warning rate by dataset/solver/condition):")
    for dataset_name in BENCHMARK_DATASETS.keys():
        print(f"\n[{dataset_name}]")
        for solver_name in SOLVERS:
            parts = []
            for condition in CONDITIONS:
                rate = all_results[dataset_name][solver_name][condition][
                    "aggregate_diagnostics"
                ]["edge_warning_rate"]
                parts.append(f"{condition}={rate:.2f}")
            print(f"  {solver_name:<10s} " + " | ".join(parts))


if __name__ == "__main__":
    main()
