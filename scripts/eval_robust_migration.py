"""Robust-migration evaluation harness.

Purpose
-------
Run deterministic pre/post migration evaluations with realistic simulation
metadata and emit:
- machine-readable artifacts (raw + summary + optional comparison)
- an easy-to-read terminal summary for fast solo-dev iteration.

Fast defaults are intentional. Use CLI flags for broader/full sweeps.

Examples
--------
Quick smoke run:
    python scripts/eval_robust_migration.py

Run all beamformers + bayesian on 20 samples:
    python scripts/eval_robust_migration.py \
        --categories beamformer,bayesian \
        --n-samples 20 \
        --cov-source estimated \
        --cov-mode dataset_mean

Rigorous per-sample covariance + projector run:
    python scripts/eval_robust_migration.py \
        --categories minimum_norm,beamformer,bayesian,music,matching_pursuit,other \
        --n-samples 50 \
        --cov-source estimated \
        --cov-mode per_sample \
        --projector-mode per_sample
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

from invert.benchmark.datasets import BENCHMARK_DATASETS, DatasetConfig
from invert.benchmark.runner import (
    SOLVER_CATEGORIES,
    get_solver_category,
    get_solver_class,
    resolve_solvers,
)
from invert.forward import create_forward_model, get_info
from invert.simulate import SimulationConfig, SimulationGenerator
from invert.util import pos_from_forward

LOGGER = logging.getLogger("eval_robust_migration")


FAST_DEFAULT_SOLVERS = [
    "dSPM",
    "dSPM-MNE",
    "eLORETA",
    "LCMV",
    "ESMV",
    "Champagne",
]


def _load_evaluate_all():
    eval_path = Path(__file__).resolve().parents[1] / "invert" / "evaluate" / "evaluate.py"
    spec = spec_from_file_location("invert.evaluate.evaluate", str(eval_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load evaluate module from {eval_path}")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.evaluate_all


EVALUATE_ALL = _load_evaluate_all()


@dataclass
class SolverCapabilities:
    expects_simulation_config: bool
    supports_noise_cov: bool


def _parse_csv_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",")]
    return [item for item in items if item]


def _parse_solver_overrides_json(value: str | None) -> dict[str, dict[str, Any]]:
    if value is None:
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for --solver-overrides-json: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("--solver-overrides-json must decode to an object/dict.")
    overrides: dict[str, dict[str, Any]] = {}
    for solver_name, solver_kwargs in payload.items():
        if not isinstance(solver_name, str):
            raise ValueError("Solver override keys must be solver-name strings.")
        if not isinstance(solver_kwargs, dict):
            raise ValueError(
                f"Override for {solver_name!r} must be a dict of constructor kwargs."
            )
        overrides[solver_name] = dict(solver_kwargs)
    return overrides


def _instantiate_solver(
    solver_cls: type[Any],
    solver_name: str,
    solver_overrides: dict[str, dict[str, Any]],
) -> Any:
    kwargs = solver_overrides.get(solver_name, {})
    try:
        return solver_cls(**kwargs)
    except TypeError as exc:
        raise TypeError(
            f"Failed to instantiate solver {solver_name!r} with overrides {kwargs}: {exc}"
        ) from exc


def _get_solver_capabilities(solver_cls: type[Any]) -> SolverCapabilities:
    try:
        sig = inspect.signature(solver_cls.make_inverse_operator)
    except (TypeError, ValueError):
        return SolverCapabilities(
            expects_simulation_config=False,
            supports_noise_cov=False,
        )

    params = list(sig.parameters.values())
    expects_simulation_config = False
    supports_noise_cov = "noise_cov" in sig.parameters

    # First user-facing arg after (self, forward)
    for p in params[2:]:
        if p.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        expects_simulation_config = p.name == "simulation_config"
        break

    return SolverCapabilities(
        expects_simulation_config=expects_simulation_config,
        supports_noise_cov=supports_noise_cov,
    )


def _robust_mean_matrix(mats: list[np.ndarray]) -> np.ndarray | None:
    if not mats:
        return None
    stack = np.stack(mats, axis=0)
    return np.mean(stack, axis=0)


def _extract_matrix_column(info_df: pd.DataFrame, column: str) -> list[np.ndarray]:
    if column not in info_df.columns:
        return []
    values = []
    for v in info_df[column].tolist():
        if isinstance(v, np.ndarray) and v.ndim == 2:
            values.append(v)
    return values


def _project_forward(forward: mne.Forward, projector: np.ndarray | None) -> mne.Forward:
    if projector is None:
        return forward
    fwd = deepcopy(forward)
    fwd["sol"]["data"] = projector @ fwd["sol"]["data"]
    return fwd


def _build_simulation_config(
    ds_cfg: DatasetConfig,
    *,
    n_samples: int,
    seed: int,
    realism: str,
) -> SimulationConfig:
    kwargs: dict[str, Any] = {
        "batch_size": n_samples,
        "batch_repetitions": 1,
        "n_sources": ds_cfg.n_sources,
        "n_orders": ds_cfg.n_orders,
        "snr_range": ds_cfg.snr_range,
        "n_timepoints": ds_cfg.n_timepoints,
        "random_seed": seed,
        "return_noise_cov": True,
        "estimate_noise_cov": True,
        "apply_sensor_projector": True,
        "correlation_mode": "auto",
    }
    if realism == "enhanced":
        kwargs.update(
            {
                "noise_rank_deficiency": (0, 3),
                "noise_temporal_beta": (0.0, 2.0),
                "noise_low_rank_dim": (2, 8),
            }
        )
    return SimulationConfig(**kwargs)


def _select_covariance(
    *,
    cov_source: str,
    cov_mode: str,
    sample_idx: int,
    info_df: pd.DataFrame,
    dataset_cov: np.ndarray | None,
) -> np.ndarray | None:
    if cov_source == "none" or cov_mode == "none":
        return None

    if cov_source == "estimated":
        column = "noise_cov_est"
    elif cov_source == "estimated_scaled":
        column = "noise_cov_est_scaled"
    elif cov_source == "true":
        column = "noise_cov_true"
    elif cov_source == "model":
        column = "noise_cov_model"
    else:
        raise ValueError(f"Unknown cov_source: {cov_source!r}")

    if cov_mode == "dataset_mean":
        return dataset_cov

    if column not in info_df.columns:
        return None
    cov = info_df.iloc[sample_idx][column]
    if isinstance(cov, np.ndarray) and cov.ndim == 2:
        return cov
    return None


def _select_projector(
    *,
    projector_mode: str,
    sample_idx: int,
    info_df: pd.DataFrame,
    dataset_projector: np.ndarray | None,
) -> np.ndarray | None:
    if projector_mode == "none":
        return None

    if projector_mode == "dataset_mean":
        return dataset_projector

    if "projector" not in info_df.columns:
        return None
    projector = info_df.iloc[sample_idx]["projector"]
    if isinstance(projector, np.ndarray) and projector.ndim == 2:
        return projector
    return None


def _call_make_inverse_operator(
    solver: Any,
    forward: mne.Forward,
    evoked: mne.EvokedArray,
    *,
    make_kwargs: dict[str, Any],
) -> None:
    if getattr(solver, "require_data", True):
        solver.make_inverse_operator(forward, evoked, alpha="auto", **make_kwargs)
    else:
        solver.make_inverse_operator(forward, alpha="auto", **make_kwargs)


def _to_metric_row(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "mean_localization_error": float(metrics["Mean_Localization_Error"]),
        "emd": float(metrics["EMD"]),
        "spatial_dispersion": float(metrics["sd"]),
        "average_precision": float(metrics["average_precision"]),
        "correlation": float(metrics["correlation"]),
    }


def _summarize(
    raw_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> pd.DataFrame:
    raw_df = pd.DataFrame(raw_rows)
    if raw_df.empty:
        return pd.DataFrame()

    grouped = raw_df.groupby(["solver", "dataset"], as_index=False)
    summary = grouped.agg(
        n_samples=("sample_idx", "count"),
        mean_localization_error=("mean_localization_error", "mean"),
        emd=("emd", "mean"),
        spatial_dispersion=("spatial_dispersion", "mean"),
        average_precision=("average_precision", "mean"),
        correlation=("correlation", "mean"),
        fit_time_ms=("fit_time_ms", "mean"),
        apply_time_ms=("apply_time_ms", "mean"),
        total_time_ms=("total_time_ms", "mean"),
    )

    fail_df = pd.DataFrame(failures)
    if fail_df.empty:
        summary["n_failures"] = 0
        summary["failure_rate"] = 0.0
        return summary

    fail_counts = (
        fail_df.groupby(["solver", "dataset"]).size().reset_index(name="n_failures")
    )
    summary = summary.merge(fail_counts, on=["solver", "dataset"], how="left")
    summary["n_failures"] = summary["n_failures"].fillna(0).astype(int)
    summary["failure_rate"] = summary["n_failures"] / np.maximum(summary["n_samples"], 1)
    return summary


def _print_summary_table(summary_df: pd.DataFrame, title: str) -> None:
    print(f"\n=== {title} ===")
    if summary_df.empty:
        print("No summary rows to display.")
        return

    for dataset in sorted(summary_df["dataset"].unique()):
        block = summary_df[summary_df["dataset"] == dataset].copy()
        block = block.sort_values("mean_localization_error", ascending=True)
        print(f"\nDataset: {dataset}")
        cols = [
            "solver",
            "mean_localization_error",
            "emd",
            "spatial_dispersion",
            "average_precision",
            "correlation",
            "failure_rate",
            "total_time_ms",
        ]
        with pd.option_context("display.max_rows", None, "display.width", 220):
            print(
                block[cols].to_string(
                    index=False,
                    float_format=lambda v: f"{v:8.4f}",
                )
            )


def _print_run_report(summary_df: pd.DataFrame, failures: list[dict[str, Any]]) -> None:
    if summary_df.empty:
        print("\nRun report: no successful metrics.")
        return

    metric_cols = [
        "mean_localization_error",
        "emd",
        "spatial_dispersion",
        "average_precision",
        "correlation",
    ]
    print("\n=== Run Report ===")
    print(f"Rows: {len(summary_df)} (solver-dataset combinations)")
    print(f"Failures: {len(failures)}")

    overall = summary_df[metric_cols + ["failure_rate", "total_time_ms"]].mean(
        numeric_only=True
    )
    print("Overall means:")
    for k, v in overall.items():
        print(f"- {k}: {float(v):.4f}")


def _compare_with_baseline(
    *,
    summary_df: pd.DataFrame,
    baseline_summary_path: Path,
) -> pd.DataFrame:
    base_df = pd.read_csv(baseline_summary_path)
    keys = ["solver", "dataset"]
    merged = summary_df.merge(base_df, on=keys, suffixes=("_post", "_pre"))
    metric_cols = [
        "mean_localization_error",
        "emd",
        "spatial_dispersion",
        "average_precision",
        "correlation",
        "failure_rate",
        "total_time_ms",
    ]
    for col in metric_cols:
        merged[f"delta_{col}"] = merged[f"{col}_post"] - merged[f"{col}_pre"]
    return merged


def _print_comparison_report(comparison_df: pd.DataFrame, tolerance: float) -> None:
    print("\n=== Pre/Post Comparison Report ===")
    if comparison_df.empty:
        print("No overlapping solver-dataset rows with baseline.")
        return

    # Regressions: lower-is-better metrics worsen by > tolerance*|pre|.
    lower_is_better = ["mean_localization_error", "emd", "spatial_dispersion"]
    higher_is_better = ["average_precision", "correlation"]

    alerts = []
    for _, row in comparison_df.iterrows():
        solver = row["solver"]
        dataset = row["dataset"]
        for metric in lower_is_better:
            pre = float(row[f"{metric}_pre"])
            delta = float(row[f"delta_{metric}"])
            thresh = tolerance * max(abs(pre), 1e-12)
            if delta > thresh:
                alerts.append((solver, dataset, metric, delta, thresh))
        for metric in higher_is_better:
            pre = float(row[f"{metric}_pre"])
            delta = float(row[f"delta_{metric}"])
            thresh = tolerance * max(abs(pre), 1e-12)
            if -delta > thresh:
                alerts.append((solver, dataset, metric, delta, -thresh))

    print(f"Compared rows: {len(comparison_df)}")
    print(f"Regression alerts (tolerance={tolerance:.3f}): {len(alerts)}")
    if alerts:
        for solver, dataset, metric, delta, thresh in alerts[:40]:
            print(
                f"- {solver} | {dataset} | {metric}: delta={delta:+.4f} (threshold {thresh:+.4f})"
            )
        if len(alerts) > 40:
            print(f"... and {len(alerts) - 40} more")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robust migration evaluation harness")
    parser.add_argument(
        "--solvers",
        type=str,
        default=",".join(FAST_DEFAULT_SOLVERS),
        help="Comma-separated solver names. Default is a fast representative set.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated categories (used in addition to --solvers).",
    )
    parser.add_argument(
        "--exclude-solvers",
        type=str,
        default=None,
        help="Comma-separated solver names to exclude.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="focal_source,multi_source,extended_source,noisy",
        help="Comma-separated dataset keys from BENCHMARK_DATASETS.",
    )
    parser.add_argument("--n-samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sampling", type=str, default="ico2")
    parser.add_argument("--montage", type=str, default="biosemi32")
    parser.add_argument(
        "--cov-source",
        choices=["none", "estimated", "estimated_scaled", "true", "model"],
        default="estimated",
    )
    parser.add_argument(
        "--cov-mode",
        choices=["none", "dataset_mean", "per_sample"],
        default="dataset_mean",
    )
    parser.add_argument(
        "--projector-mode",
        choices=["none", "dataset_mean", "per_sample"],
        default="none",
    )
    parser.add_argument(
        "--realism",
        choices=["basic", "enhanced"],
        default="basic",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/lukas/projects/invert/results/robust_migration"),
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=None,
        help="Path to baseline summary.csv for pre/post comparison.",
    )
    parser.add_argument(
        "--regression-tolerance",
        type=float,
        default=0.02,
        help="Relative tolerance for regression alerts in comparison report.",
    )
    parser.add_argument(
        "--solver-overrides-json",
        type=str,
        default=None,
        help=(
            "JSON mapping solver name -> constructor kwargs, "
            'e.g. \'{"ESMV":{"use_robust_covariance":true}}\''
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    solvers = _parse_csv_arg(args.solvers) or []
    categories = _parse_csv_arg(args.categories)
    exclude = _parse_csv_arg(args.exclude_solvers)
    datasets = _parse_csv_arg(args.datasets) or []
    solver_overrides = _parse_solver_overrides_json(args.solver_overrides_json)

    if args.cov_source == "none":
        args.cov_mode = "none"

    if args.run_id is None:
        args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = args.output_dir / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_map: dict[str, DatasetConfig] = {}
    for ds in datasets:
        if ds not in BENCHMARK_DATASETS:
            raise ValueError(
                f"Unknown dataset {ds!r}. Available: {sorted(BENCHMARK_DATASETS)}"
            )
        dataset_map[ds] = BENCHMARK_DATASETS[ds]

    solver_names = resolve_solvers(
        solvers=solvers,
        categories=categories,
        exclude=exclude,
    )
    if not solver_names:
        raise ValueError("No solvers selected. Provide --solvers and/or --categories.")

    info = get_info(kind=args.montage)
    forward = create_forward_model(sampling=args.sampling, info=info)

    pos = pos_from_forward(forward)
    adjacency = mne.spatial_src_adjacency(forward["src"], verbose=0)

    raw_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for ds_idx, (ds_name, ds_cfg) in enumerate(dataset_map.items()):
        sim_seed = int(args.seed + ds_idx)
        sim_cfg = _build_simulation_config(
            ds_cfg,
            n_samples=int(args.n_samples),
            seed=sim_seed,
            realism=args.realism,
        )
        sim_gen = SimulationGenerator(forward, config=sim_cfg)
        x_batch, y_batch, info_df = next(sim_gen.generate())

        if args.cov_source == "estimated":
            cov_column = "noise_cov_est"
        elif args.cov_source == "estimated_scaled":
            cov_column = "noise_cov_est_scaled"
        elif args.cov_source == "true":
            cov_column = "noise_cov_true"
        elif args.cov_source == "model":
            cov_column = "noise_cov_model"
        else:
            cov_column = ""
        cov_values = _extract_matrix_column(info_df, cov_column)
        proj_values = _extract_matrix_column(info_df, "projector")
        dataset_cov = _robust_mean_matrix(cov_values)
        dataset_projector = _robust_mean_matrix(proj_values)

        for solver_name in solver_names:
            solver_cls = get_solver_class(solver_name)
            caps = _get_solver_capabilities(solver_cls)

            if caps.expects_simulation_config:
                skipped.append(
                    {
                        "solver": solver_name,
                        "dataset": ds_name,
                        "reason": "expects_simulation_config (ANN-like), skipped in this harness",
                    }
                )
                continue

            # Reuse fast path is only valid when per-sample inputs are not required.
            dynamic_inputs = args.cov_mode == "per_sample" or args.projector_mode == "per_sample"

            # Use a temporary instance to query recomputation behavior.
            probe_solver = _instantiate_solver(
                solver_cls, solver_name, solver_overrides
            )
            can_reuse = (not dynamic_inputs) and (not getattr(probe_solver, "require_recompute", True))

            if can_reuse:
                solver = _instantiate_solver(solver_cls, solver_name, solver_overrides)
                cov0 = _select_covariance(
                    cov_source=args.cov_source,
                    cov_mode=args.cov_mode,
                    sample_idx=0,
                    info_df=info_df,
                    dataset_cov=dataset_cov,
                )
                proj0 = _select_projector(
                    projector_mode=args.projector_mode,
                    sample_idx=0,
                    info_df=info_df,
                    dataset_projector=dataset_projector,
                )
                fwd0 = _project_forward(forward, proj0)
                evoked0 = mne.EvokedArray(x_batch[0], info, tmin=0.0, verbose=0)
                make_kwargs = {}
                if caps.supports_noise_cov and cov0 is not None:
                    make_kwargs["noise_cov"] = cov0

                try:
                    fit_t0 = time.perf_counter()
                    _call_make_inverse_operator(
                        solver,
                        fwd0,
                        evoked0,
                        make_kwargs=make_kwargs,
                    )
                    fit_time_ms = (time.perf_counter() - fit_t0) * 1000.0
                except Exception as exc:
                    failures.append(
                        {
                            "solver": solver_name,
                            "dataset": ds_name,
                            "sample_idx": -1,
                            "stage": "fit_once",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue

                for i in range(len(x_batch)):
                    evoked_i = mne.EvokedArray(x_batch[i], info, tmin=0.0, verbose=0)
                    try:
                        apply_t0 = time.perf_counter()
                        stc = solver.apply_inverse_operator(evoked_i)
                        apply_time_ms = (time.perf_counter() - apply_t0) * 1000.0
                        y_pred = np.asarray(stc.data)
                        metrics = EVALUATE_ALL(
                            y_batch[i],
                            y_pred,
                            adjacency,
                            adjacency,
                            pos,
                            pos,
                        )
                        metric_row = _to_metric_row(metrics)
                        raw_rows.append(
                            {
                                "solver": solver_name,
                                "dataset": ds_name,
                                "sample_idx": i,
                                "fit_time_ms": fit_time_ms,
                                "apply_time_ms": apply_time_ms,
                                "total_time_ms": fit_time_ms + apply_time_ms,
                                "category": get_solver_category(solver_name),
                                **metric_row,
                            }
                        )
                    except Exception as exc:
                        failures.append(
                            {
                                "solver": solver_name,
                                "dataset": ds_name,
                                "sample_idx": i,
                                "stage": "apply",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                continue

            # Per-sample compute path (for dynamic inputs or recompute-required solvers)
            for i in range(len(x_batch)):
                solver = _instantiate_solver(solver_cls, solver_name, solver_overrides)
                cov_i = _select_covariance(
                    cov_source=args.cov_source,
                    cov_mode=args.cov_mode,
                    sample_idx=i,
                    info_df=info_df,
                    dataset_cov=dataset_cov,
                )
                proj_i = _select_projector(
                    projector_mode=args.projector_mode,
                    sample_idx=i,
                    info_df=info_df,
                    dataset_projector=dataset_projector,
                )
                fwd_i = _project_forward(forward, proj_i)
                evoked_i = mne.EvokedArray(x_batch[i], info, tmin=0.0, verbose=0)
                make_kwargs = {}
                if caps.supports_noise_cov and cov_i is not None:
                    make_kwargs["noise_cov"] = cov_i

                try:
                    fit_t0 = time.perf_counter()
                    _call_make_inverse_operator(
                        solver,
                        fwd_i,
                        evoked_i,
                        make_kwargs=make_kwargs,
                    )
                    fit_time_ms = (time.perf_counter() - fit_t0) * 1000.0

                    apply_t0 = time.perf_counter()
                    stc = solver.apply_inverse_operator(evoked_i)
                    apply_time_ms = (time.perf_counter() - apply_t0) * 1000.0

                    y_pred = np.asarray(stc.data)
                    metrics = EVALUATE_ALL(
                        y_batch[i],
                        y_pred,
                        adjacency,
                        adjacency,
                        pos,
                        pos,
                    )
                    metric_row = _to_metric_row(metrics)
                    raw_rows.append(
                        {
                            "solver": solver_name,
                            "dataset": ds_name,
                            "sample_idx": i,
                            "fit_time_ms": fit_time_ms,
                            "apply_time_ms": apply_time_ms,
                            "total_time_ms": fit_time_ms + apply_time_ms,
                            "category": get_solver_category(solver_name),
                            **metric_row,
                        }
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "solver": solver_name,
                            "dataset": ds_name,
                            "sample_idx": i,
                            "stage": "fit_or_apply",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

    summary_df = _summarize(raw_rows, failures)

    run_meta = {
        "timestamp": datetime.now().isoformat(),
        "run_id": args.run_id,
        "solvers_requested": solver_names,
        "datasets": datasets,
        "n_samples": int(args.n_samples),
        "seed": int(args.seed),
        "sampling": args.sampling,
        "montage": args.montage,
        "cov_source": args.cov_source,
        "cov_mode": args.cov_mode,
        "projector_mode": args.projector_mode,
        "solver_overrides": solver_overrides,
        "realism": args.realism,
        "solver_categories_available": sorted(SOLVER_CATEGORIES.keys()),
    }

    (out_dir / "config.json").write_text(json.dumps(run_meta, indent=2))

    raw_df = pd.DataFrame(raw_rows)
    fail_df = pd.DataFrame(failures)
    skipped_df = pd.DataFrame(skipped)

    raw_df.to_json(out_dir / "metrics_raw.jsonl", orient="records", lines=True)
    summary_df.to_csv(out_dir / "metrics_aggregate.csv", index=False)
    fail_df.to_csv(out_dir / "failures.csv", index=False)
    skipped_df.to_csv(out_dir / "skipped.csv", index=False)
    if not raw_df.empty:
        timing_cols = ["solver", "dataset", "sample_idx", "fit_time_ms", "apply_time_ms", "total_time_ms"]
        raw_df[timing_cols].to_csv(out_dir / "timing.csv", index=False)
    else:
        pd.DataFrame(
            columns=["solver", "dataset", "sample_idx", "fit_time_ms", "apply_time_ms", "total_time_ms"]
        ).to_csv(out_dir / "timing.csv", index=False)

    # A concise report file that can be copied into journal updates.
    report_lines = [
        "Robust Migration Eval Summary",
        f"Run ID: {args.run_id}",
        f"Rows: {len(summary_df)}",
        f"Failures: {len(failures)}",
        f"Skipped: {len(skipped)}",
    ]
    (out_dir / "summary_report.txt").write_text("\n".join(report_lines) + "\n")

    _print_summary_table(summary_df, "Robust Migration Evaluation Summary")
    _print_run_report(summary_df, failures)

    if args.baseline_summary is not None:
        comparison_df = _compare_with_baseline(
            summary_df=summary_df,
            baseline_summary_path=args.baseline_summary,
        )
        comparison_df.to_csv(out_dir / "comparison_pre_post.csv", index=False)
        _print_comparison_report(comparison_df, tolerance=float(args.regression_tolerance))

    print(f"\nArtifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
