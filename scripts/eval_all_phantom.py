#!/usr/bin/env python
"""Evaluate all invert solvers on the 4D BTi phantom (with/without whitening).

This script mirrors the phantom preprocessing used in scripts/phantom_eval.py
and stores runner-compatible JSON results (same schema as BenchmarkRunner.save).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os.path as op
import signal
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np
from mne.datasets import phantom_4dbti
from scipy import sparse
from scipy.spatial.distance import cdist

from invert.benchmark.runner import BenchmarkRunner, SampleMetrics, TimingInfo
from invert.solver_registry import get_solver_class, get_solver_spec, iter_solver_specs
from invert.util.util import pos_from_forward

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

_EVAL_SPEC = importlib.util.spec_from_file_location(
    "invert.evaluate.evaluate",
    str(Path(__file__).resolve().parents[1] / "invert" / "evaluate" / "evaluate.py"),
)
if _EVAL_SPEC is None or _EVAL_SPEC.loader is None:
    raise RuntimeError("Could not load invert/evaluate/evaluate.py")
_EVAL_MOD = importlib.util.module_from_spec(_EVAL_SPEC)
_EVAL_SPEC.loader.exec_module(_EVAL_MOD)
evaluate_all = _EVAL_MOD.evaluate_all

RUN_IDS = [1, 2, 3, 4]
BAD_CHANNELS = ["A173", "A213", "A232"]
PEAK_TIME = 0.07
PEAK_WINDOW = 0.02
MODE_NAMES = {"no_whitening": "phantom-no-whitening", "whitening": "phantom-whitening"}
USE_EPOCHS_FOR_COV = {"lcmv", "esmv"}
DEFAULT_EXCLUDE = ["champagne-tem", "champagne-ar-em"]

ACTUAL_POS = 0.01 * np.array(
    [
        [0.16, 1.61, 5.13],
        [0.17, 1.35, 4.15],
        [0.16, 1.05, 3.19],
        [0.13, 0.80, 2.26],
    ]
) @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])


@dataclass
class PhantomRun:
    epochs: mne.Epochs
    evoked: mne.Evoked
    cov: mne.Covariance
    forward: mne.Forward
    pos: np.ndarray
    adjacency: Any
    y_true: np.ndarray


def load_run(run_id: int, data_path: str) -> tuple[mne.Epochs, mne.Evoked, mne.Covariance]:
    raw = mne.io.read_raw_bti(
        op.join(data_path, f"{run_id}/e,rfhp1.0Hz"),
        rename_channels=False,
        preload=True,
        verbose=False,
    )
    raw.info["bads"] = BAD_CHANNELS
    events = mne.find_events(
        raw,
        stim_channel="TRIGGER",
        mask=4350,
        mask_type="not_and",
        verbose=False,
    )
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=8192,
        tmin=-0.2,
        tmax=0.4,
        preload=True,
        baseline=(None, 0.0),
        verbose=False,
    )
    evoked = epochs.average()
    evoked.drop_channels(evoked.info["bads"])
    cov = mne.compute_covariance(epochs, tmax=0.0, verbose=False)
    return epochs, evoked, cov


def make_fwd(info: mne.Info) -> mne.Forward:
    sphere = mne.make_sphere_model(
        r0=(0.0, 0.0, 0.0),
        head_radius=0.080,
        verbose=False,
    )
    src_grid = mne.setup_volume_source_space(
        subject=None,
        pos=8.0,
        sphere=(0.0, 0.0, 0.0, 0.072),
        mindist=5.0,
        exclude=0.0,
        verbose=False,
    )
    rr_used = src_grid[0]["rr"][src_grid[0]["vertno"]]
    rr = np.vstack([rr_used, ACTUAL_POS])
    rr = np.unique(np.round(rr, 6), axis=0)
    norms = np.linalg.norm(rr, axis=1, keepdims=True)
    nn = np.zeros_like(rr)
    mask = norms.squeeze() > 0
    nn[mask] = rr[mask] / norms[mask]
    nn[~mask] = np.array([1.0, 0.0, 0.0])
    src = mne.setup_volume_source_space(
        subject=None,
        pos={"rr": rr, "nn": nn},
        sphere=(0.0, 0.0, 0.0, 0.072),
        mindist=0.0,
        exclude=0.0,
        verbose=False,
    )
    return mne.make_forward_solution(
        info,
        trans=None,
        src=src,
        bem=sphere,
        meg=True,
        eeg=False,
        verbose=False,
    )


def _collapse_orientation(source_mat: np.ndarray, n_sources: int) -> np.ndarray:
    data = np.asarray(source_mat, dtype=float)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    if data.shape[0] == n_sources:
        return data
    if data.shape[0] % n_sources == 0:
        n_orient = data.shape[0] // n_sources
        data = data.reshape(n_sources, n_orient, data.shape[1])
        return np.linalg.norm(data, axis=1)
    raise ValueError(
        f"Predicted source matrix has incompatible shape {data.shape}; "
        f"expected first dim {n_sources} or a multiple of it."
    )


def _make_ground_truth(pos: np.ndarray, true_pos: np.ndarray) -> np.ndarray:
    idx = int(np.argmin(np.linalg.norm(pos - true_pos, axis=1)))
    y_true = np.zeros((pos.shape[0], 1), dtype=float)
    y_true[idx, 0] = 1.0
    return y_true


def _crop_peak_window(evoked: mne.Evoked, peak_time: float, window: float) -> mne.Evoked:
    half = float(max(window, 0.0)) / 2.0
    tmin = float(max(evoked.times[0], peak_time - half))
    tmax = float(min(evoked.times[-1], peak_time + half))
    if tmax <= tmin:
        tmin = tmax = float(peak_time)
    return evoked.copy().crop(tmin, tmax)


def _build_adjacency(src) -> Any:
    try:
        return mne.spatial_src_adjacency(src, verbose=0)
    except RuntimeError:
        try:
            return mne.spatial_dist_adjacency(src, dist=0.02, verbose=0)
        except RuntimeError:
            pos = np.concatenate([s["rr"][s["vertno"]] for s in src], axis=0)
            dist = cdist(pos, pos)
            adjacency = (dist > 0.0) & (dist <= 0.02)
            return sparse.csr_matrix(adjacency.astype(float))


def _sample_metrics_from_eval(metrics: dict[str, float]) -> SampleMetrics:
    return SampleMetrics(
        mle=float(metrics["Mean_Localization_Error"]),
        emd=float(metrics["EMD"]),
        sd=float(metrics["sd"]),
        ap=float(metrics["average_precision"]),
        correlation=float(metrics["correlation"]),
    )


def _nan_metrics() -> SampleMetrics:
    return SampleMetrics(
        mle=float("nan"),
        emd=float("nan"),
        sd=float("nan"),
        ap=float("nan"),
        correlation=float("nan"),
    )


def _resolve_solver_ids(
    solvers: list[str] | None,
    exclude: list[str],
    include_internal: bool,
) -> list[str]:
    if solvers:
        solver_ids = [get_solver_spec(name).solver_id for name in solvers]
    else:
        solver_ids = [
            spec.solver_id
            for spec in iter_solver_specs(
                include_internal=include_internal,
                include_unavailable=False,
            )
        ]
    exclude_ids = {get_solver_spec(name).solver_id for name in exclude}
    return [sid for sid in solver_ids if sid not in exclude_ids]


def _is_neural_network_solver(solver_id: str) -> bool:
    spec = get_solver_spec(solver_id)
    return ("torch" in spec.requires) or (".neural_networks." in spec.module_path)


def _evaluate_solver_mode(
    solver_id: str,
    mode: str,
    runs: dict[int, PhantomRun],
    skip_reason: str | None = None,
    per_run_timeout: float | None = None,
    peak_time: float = PEAK_TIME,
    peak_window: float = PEAK_WINDOW,
    orientation: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    spec = get_solver_spec(solver_id)
    solver_cls = get_solver_class(solver_id)
    mode_dataset = MODE_NAMES[mode]
    whitening = mode == "whitening"

    sample_metrics: list[SampleMetrics] = []
    peak_errors_mm: list[float] = []
    failures: list[dict[str, Any]] = []
    make_seconds = 0.0
    apply_seconds = 0.0
    t_total = time.perf_counter()

    if skip_reason is not None:
        for rid in RUN_IDS:
            sample_metrics.append(_nan_metrics())
            peak_errors_mm.append(float("nan"))
            failures.append({"run_id": rid, "error": skip_reason, "traceback": ""})
    else:
        for rid in RUN_IDS:
            run = runs[rid]
            solver_kwargs = dict(spec.default_kwargs)
            if orientation is not None:
                solver_kwargs["orientation"] = orientation
            solver = solver_cls(**solver_kwargs)
            mne_obj = run.epochs if solver_id in USE_EPOCHS_FOR_COV else run.evoked
            try:
                timed_out = False
                if per_run_timeout and per_run_timeout > 0:
                    old_handler = signal.getsignal(signal.SIGALRM)

                    def _alarm_handler(_signum, _frame):
                        raise TimeoutError(
                            f"Timeout after {per_run_timeout:.1f}s in run {rid}"
                        )

                    signal.signal(signal.SIGALRM, _alarm_handler)
                    signal.setitimer(signal.ITIMER_REAL, per_run_timeout)
                else:
                    old_handler = None

                try:
                    t_make = time.perf_counter()
                    solver.make_inverse_operator(
                        run.forward,
                        mne_obj,
                        alpha="auto",
                        noise_cov=run.cov if whitening else None,
                    )
                    make_seconds += time.perf_counter() - t_make

                    t_apply = time.perf_counter()
                    stc = solver.apply_inverse_operator(
                        _crop_peak_window(run.evoked, peak_time, peak_window)
                    )
                    apply_seconds += time.perf_counter() - t_apply

                    source_mat = _collapse_orientation(stc.data, run.pos.shape[0])
                    metrics = evaluate_all(
                        run.y_true,
                        source_mat,
                        run.adjacency,
                        run.adjacency,
                        run.pos,
                        run.pos,
                    )
                    sample_metrics.append(_sample_metrics_from_eval(metrics))
                    peak_idx = int(np.argmax(np.max(np.abs(source_mat), axis=1)))
                    peak_pos = run.pos[peak_idx]
                    peak_error = float(
                        1e3 * np.linalg.norm(peak_pos - ACTUAL_POS[rid - 1])
                    )
                    peak_errors_mm.append(peak_error)
                finally:
                    if per_run_timeout and per_run_timeout > 0:
                        signal.setitimer(signal.ITIMER_REAL, 0)
                        if old_handler is not None:
                            signal.signal(signal.SIGALRM, old_handler)
            except Exception as err:
                sample_metrics.append(_nan_metrics())
                peak_errors_mm.append(float("nan"))
                failures.append(
                    {
                        "run_id": rid,
                        "error": str(err),
                        "traceback": traceback.format_exc(),
                    }
                )

    result = BenchmarkRunner._aggregate(
        solver_name=solver_id,
        dataset_name=mode_dataset,
        samples=sample_metrics,
    )
    result.timing = TimingInfo(
        total_seconds=float(time.perf_counter() - t_total),
        make_inverse_operator_seconds=float(make_seconds),
        apply_inverse_seconds=float(apply_seconds),
    )

    mode_summary = {
        "solver_name": solver_id,
        "dataset_name": mode_dataset,
        "mode": mode,
        "mean_peak_error_mm": float(np.nanmean(peak_errors_mm)),
        "median_peak_error_mm": float(np.nanmedian(peak_errors_mm)),
        "run_peak_errors_mm": peak_errors_mm,
        "n_failed_runs": len(failures),
        "failed_runs": [f["run_id"] for f in failures],
        "peak_time_s": float(peak_time),
        "peak_window_s": float(peak_window),
        "orientation": orientation,
        "failures": failures,
    }
    return result, mode_summary


def _save_runner_like(
    results: list[Any],
    solvers: list[str],
    forward: mne.Forward,
    out_path: Path,
    compact: bool,
) -> None:
    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner.forward = forward
    runner.solvers = solvers
    runner.datasets = {
        MODE_NAMES["no_whitening"]: {
            "kind": "phantom_4dbti",
            "whitening": False,
            "n_runs": len(RUN_IDS),
            "n_sources": 1,
        },
        MODE_NAMES["whitening"]: {
            "kind": "phantom_4dbti",
            "whitening": True,
            "n_runs": len(RUN_IDS),
            "n_sources": 1,
        },
    }
    runner.n_samples = len(RUN_IDS)
    runner.random_seed = None
    runner._results = results
    runner.save(
        out_path,
        compact=compact,
        name="invertmeeg Phantom All-Solver Evaluation",
        description=(
            "All registered non-internal available solvers on the 4D BTi phantom "
            "dataset, evaluated with and without passing noise covariance "
            "(whitening mode)."
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=None,
        help="Optional list of solver names/aliases. Default: all available non-internal solvers.",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=DEFAULT_EXCLUDE,
        help=(
            "Solvers to exclude. Default: %(default)s "
            "(EM-based Champagne variants that are extremely slow on phantom data)."
        ),
    )
    parser.add_argument(
        "--include-internal",
        action="store_true",
        help="Include internal solvers from the registry.",
    )
    parser.add_argument(
        "--include-neural-networks",
        action="store_true",
        help=(
            "Run neural-network solvers that require torch/simulation training. "
            "Disabled by default to keep runtime tractable."
        ),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["no_whitening", "whitening"],
        default=["no_whitening", "whitening"],
        help="Evaluation modes to run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phantom/all_solvers_phantom.json"),
        help="Output JSON path (runner-compatible full results).",
    )
    parser.add_argument(
        "--compact-output",
        type=Path,
        default=Path("results/phantom/all_solvers_phantom_compact.json"),
        help="Output JSON path for compact runner-style results.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/phantom/all_solvers_phantom_summary.json"),
        help="Output path for per-solver summaries and failure details.",
    )
    parser.add_argument(
        "--per-run-timeout",
        type=float,
        default=60.0,
        help=(
            "Maximum seconds per solver/run pair before marking the run as failed "
            "and continuing."
        ),
    )
    parser.add_argument(
        "--peak-time",
        type=float,
        default=PEAK_TIME,
        help="Center time (seconds) of the phantom peak window.",
    )
    parser.add_argument(
        "--peak-window",
        type=float,
        default=PEAK_WINDOW,
        help="Width (seconds) of the time window around --peak-time.",
    )
    parser.add_argument(
        "--orientation",
        choices=["auto", "fixed", "pca", "free"],
        default=None,
        help="Optional orientation override applied to all selected solvers.",
    )
    args = parser.parse_args()

    solver_ids = _resolve_solver_ids(args.solvers, args.exclude, args.include_internal)
    if not solver_ids:
        raise SystemExit("No solvers selected after applying filters.")

    print(f"Selected {len(solver_ids)} solvers.", flush=True)
    print(
        f"Peak evaluation window: center={args.peak_time:.3f}s width={args.peak_window:.3f}s.",
        flush=True,
    )
    print("Loading 4D BTi phantom data...", flush=True)
    data_path = phantom_4dbti.data_path(verbose=False)

    runs: dict[int, PhantomRun] = {}
    for rid in RUN_IDS:
        epochs, evoked, cov = load_run(rid, data_path)
        forward = make_fwd(evoked.info)
        pos = pos_from_forward(forward)
        adjacency = _build_adjacency(forward["src"])
        y_true = _make_ground_truth(pos, ACTUAL_POS[rid - 1])
        runs[rid] = PhantomRun(
            epochs=epochs,
            evoked=evoked,
            cov=cov,
            forward=forward,
            pos=pos,
            adjacency=adjacency,
            y_true=y_true,
        )

    all_results = []
    summaries = []
    print(flush=True)
    for mode in args.modes:
        print(f"Mode: {mode}", flush=True)
        for i, solver_id in enumerate(solver_ids, start=1):
            skip_reason = None
            if (not args.include_neural_networks) and _is_neural_network_solver(solver_id):
                skip_reason = (
                    "Skipped by default: neural-network solver requires simulation "
                    "training. Re-run with --include-neural-networks to evaluate it."
                )
            result, summary = _evaluate_solver_mode(
                solver_id,
                mode,
                runs,
                skip_reason,
                per_run_timeout=args.per_run_timeout,
                peak_time=args.peak_time,
                peak_window=args.peak_window,
                orientation=args.orientation,
            )
            all_results.append(result)
            summaries.append(summary)
            mean_mm = summary["mean_peak_error_mm"]
            failed = summary["n_failed_runs"]
            print(
                f"[{i:>3}/{len(solver_ids):>3}] {solver_id:<35} "
                f"mean_peak_mm={mean_mm:8.3f} failed_runs={failed}"
                f"{' (skipped)' if skip_reason else ''}",
                flush=True,
            )
        print(flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.compact_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    _save_runner_like(all_results, solver_ids, runs[RUN_IDS[0]].forward, args.output, compact=False)
    _save_runner_like(
        all_results,
        solver_ids,
        runs[RUN_IDS[0]].forward,
        args.compact_output,
        compact=True,
    )

    args.summary_output.write_text(json.dumps(summaries, indent=2))

    print(f"Saved full results:    {args.output}", flush=True)
    print(f"Saved compact results: {args.compact_output}", flush=True)
    print(f"Saved summary:         {args.summary_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
