"""Phase 6 validation: whitening impact across three noise-covariance conditions.

Evaluates every fast solver on the *noisy* dataset (SNR -5..0 dB) under:
  (a) noise_cov = None   — no whitening
  (b) noise_cov = estimated — from independent baseline noise (what the runner uses)
  (c) noise_cov = true   — from the actual noise realization (oracle)

All three conditions use the exact same simulated data (fixed seed) so
differences are purely due to the covariance passed to each solver.

Configuration:  biosemi32 / ico2 / 50 samples / seed 42
Excludes:       ReciPSIICOS variants, neural networks, SESAME

Usage:
    cd invert-package
    .venv/bin/python scripts/eval_phase6_whitening.py
"""

from __future__ import annotations

import importlib.util as _ilu
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path

import mne
import numpy as np
from tqdm import tqdm

from invert.benchmark.runner import (
    _make_inverse_operator_with_covariance,
    _make_mne_covariance,
    get_solver_class,
    resolve_solvers,
)
from invert.forward import create_forward_model, get_info
from invert.simulate import SimulationConfig, SimulationGenerator
from invert.util.util import pos_from_forward

# Import evaluate_all via spec loader (avoids circular import)
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_SAMPLES = 50
RANDOM_SEED = 42
SAMPLING = "ico2"
SENSOR_KIND = "biosemi32"

NOISY_DATASET = dict(
    n_sources=(1, 3),
    n_orders=(0, 2),
    snr_range=(-5.0, 0.0),
    n_timepoints=50,
)

EXCLUDE_SOLVERS = [
    "SESAME",
    "ReciPSIICOS-Plain",
    "ReciPSIICOS-Whitened",
]

CONDITIONS = ["none", "estimated", "true"]

METRICS = ["mle", "emd", "sd", "ap", "correlation"]
METRIC_KEYS = {
    "mle": "Mean_Localization_Error",
    "emd": "EMD",
    "sd": "sd",
    "ap": "average_precision",
    "correlation": "correlation",
}
# Lower is better for these; higher is better for the rest
LOWER_IS_BETTER = {"mle", "emd", "sd"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evaluate_solver_on_condition(
    solver_name: str,
    forward: mne.Forward,
    info: mne.Info,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    adjacency,
    pos: np.ndarray,
    noise_cov: mne.Covariance | None,
    n_samples: int,
) -> list[dict[str, float]]:
    """Run one solver on all samples with a given noise_cov and return per-sample metrics."""
    solver_cls = get_solver_class(solver_name)
    per_sample: list[dict[str, float]] = []
    require_data = solver_cls().require_data

    for i in range(n_samples):
        solver = solver_cls()
        evoked = mne.EvokedArray(x_batch[i], info, tmin=0.0, verbose=0)
        try:
            _make_inverse_operator_with_covariance(
                solver=solver,
                forward=forward,
                require_data=require_data,
                evoked=evoked,
                alpha="auto",
                noise_cov=noise_cov,
            )
            stc = solver.apply_inverse_operator(evoked)
            y_pred = stc.data
            raw = evaluate_all(y_batch[i], y_pred, adjacency, adjacency, pos, pos)
            per_sample.append({m: float(raw[METRIC_KEYS[m]]) for m in METRICS})
        except Exception:
            traceback.print_exc()
            per_sample.append({m: float("nan") for m in METRICS})

    return per_sample


def aggregate(per_sample: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Aggregate per-sample metrics into mean/std/median."""
    agg: dict[str, dict[str, float]] = {}
    for m in METRICS:
        vals = np.array([s[m] for s in per_sample])
        agg[m] = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals)),
            "median": float(np.nanmedian(vals)),
        }
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("Phase 6: Whitening validation  (noisy dataset, ico2, 50 samples)")
    print("=" * 72)

    # Forward model
    info = get_info(kind=SENSOR_KIND)
    fwd = create_forward_model(sampling=SAMPLING, info=info)
    pos = pos_from_forward(fwd)
    adjacency = mne.spatial_src_adjacency(fwd["src"], verbose=0)
    n_sources = fwd["sol"]["data"].shape[1]
    print(f"Forward: {SENSOR_KIND}, {SAMPLING}, {n_sources} sources")

    # Resolve solvers
    solver_names = resolve_solvers(
        categories=[
            "beamformer", "bayesian", "minimum_norm", "loreta",
            "music", "matching_pursuit", "other",
        ],
        exclude=EXCLUDE_SOLVERS,
    )
    print(f"Solvers: {len(solver_names)}")

    # Generate simulation data once (deterministic)
    sim_config = SimulationConfig(
        batch_size=N_SAMPLES,
        n_sources=NOISY_DATASET["n_sources"],
        n_orders=NOISY_DATASET["n_orders"],
        snr_range=NOISY_DATASET["snr_range"],
        n_timepoints=NOISY_DATASET["n_timepoints"],
        random_seed=RANDOM_SEED,
        estimate_noise_cov=True,
        return_noise_cov=True,
    )
    gen = SimulationGenerator(fwd, config=sim_config)
    x_batch, y_batch, sim_info = next(gen.generate())
    print(f"Simulated {N_SAMPLES} samples  (SNR {NOISY_DATASET['snr_range']})")

    # Build averaged noise covariances
    noise_cov_est_array = np.mean(
        [sim_info.iloc[j]["noise_cov_est"] for j in range(N_SAMPLES)], axis=0,
    )
    noise_cov_true_array = np.mean(
        [sim_info.iloc[j]["noise_cov_true"] for j in range(N_SAMPLES)], axis=0,
    )

    noise_covs = {
        "none": None,
        "estimated": _make_mne_covariance(noise_cov_est_array, info, nfree=N_SAMPLES),
        "true": _make_mne_covariance(noise_cov_true_array, info, nfree=N_SAMPLES),
    }

    # Evaluate all (solver x condition)
    # results[solver][condition] = {"mle": {"mean":..., "std":..., "median":...}, ...}
    results: dict[str, dict[str, dict]] = {}

    for solver_name in tqdm(solver_names, desc="Solvers", position=0):
        results[solver_name] = {}
        for cond in CONDITIONS:
            per_sample = evaluate_solver_on_condition(
                solver_name, fwd, info, x_batch, y_batch,
                adjacency, pos, noise_covs[cond], N_SAMPLES,
            )
            results[solver_name][cond] = {
                "aggregate": aggregate(per_sample),
                "samples": per_sample,
            }

    # Save full results
    out_dir = Path("results/release")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase6_whitening_validation.json"

    payload = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_samples": N_SAMPLES,
            "random_seed": RANDOM_SEED,
            "sampling": SAMPLING,
            "sensor_kind": SENSOR_KIND,
            "n_sources": n_sources,
            "dataset": NOISY_DATASET,
            "conditions": CONDITIONS,
            "solvers": solver_names,
        },
        "results": {
            solver: {
                cond: data["aggregate"]
                for cond, data in cond_data.items()
            }
            for solver, cond_data in results.items()
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nFull results saved to {out_path}")

    # Print summary table and save markdown report
    report = build_report(results, solver_names)
    print(report)

    journal_dir = Path(__file__).resolve().parents[2] / ".lab_journal"
    journal_dir.mkdir(parents=True, exist_ok=True)
    report_path = journal_dir / "phase6_whitening_validation.md"
    report_path.write_text(report)
    print(f"Report saved to {report_path}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_report(
    results: dict[str, dict[str, dict]],
    solver_names: list[str],
) -> str:
    lines: list[str] = []
    lines.append("# Phase 6: Whitening Validation Report")
    lines.append("")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Config:** {SENSOR_KIND}/{SAMPLING}, {N_SAMPLES} samples, "
                 f"seed={RANDOM_SEED}")
    lines.append(f"**Dataset:** noisy (SNR {NOISY_DATASET['snr_range']})")
    lines.append(f"**Solvers:** {len(solver_names)}")
    lines.append(f"**Conditions:** none | estimated | true (oracle)")
    lines.append("")

    # Per-metric tables
    for metric in METRICS:
        lib = metric in LOWER_IS_BETTER
        arrow = "lower=better" if lib else "higher=better"
        lines.append(f"## {metric.upper()} ({arrow})")
        lines.append("")

        # Header
        lines.append(
            f"| {'Solver':<28s} | {'none':>8s} | {'est':>8s} | {'true':>8s} "
            f"| {'est vs none':>11s} | {'true vs none':>12s} |"
        )
        lines.append(
            f"|{'-'*30}|{'-'*10}|{'-'*10}|{'-'*10}"
            f"|{'-'*13}|{'-'*14}|"
        )

        for solver in solver_names:
            vals = {}
            for cond in CONDITIONS:
                vals[cond] = results[solver][cond]["aggregate"][metric]["mean"]

            v_none = vals["none"]
            v_est = vals["estimated"]
            v_true = vals["true"]

            # Compute deltas (positive = improvement)
            if lib:  # lower is better → improvement = decrease
                delta_est = v_none - v_est
                delta_true = v_none - v_true
            else:  # higher is better → improvement = increase
                delta_est = v_est - v_none
                delta_true = v_true - v_none

            def fmt_delta(d: float) -> str:
                if np.isnan(d):
                    return "nan"
                sign = "+" if d > 0 else ""
                return f"{sign}{d:.2f}"

            def fmt_val(v: float) -> str:
                return f"{v:.2f}" if not np.isnan(v) else "nan"

            lines.append(
                f"| {solver:<28s} | {fmt_val(v_none):>8s} | {fmt_val(v_est):>8s} "
                f"| {fmt_val(v_true):>8s} | {fmt_delta(delta_est):>11s} "
                f"| {fmt_delta(delta_true):>12s} |"
            )

        lines.append("")

    # Summary statistics
    lines.append("## Summary: Average improvement across all solvers")
    lines.append("")
    lines.append(
        f"| {'Metric':<12s} | {'Mean(none)':>10s} | {'Mean(est)':>10s} "
        f"| {'Mean(true)':>10s} | {'est vs none':>11s} | {'true vs none':>12s} |"
    )
    lines.append(
        f"|{'-'*14}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*13}|{'-'*14}|"
    )

    for metric in METRICS:
        lib = metric in LOWER_IS_BETTER
        vals_none = []
        vals_est = []
        vals_true = []
        for solver in solver_names:
            vn = results[solver]["none"]["aggregate"][metric]["mean"]
            ve = results[solver]["estimated"]["aggregate"][metric]["mean"]
            vt = results[solver]["true"]["aggregate"][metric]["mean"]
            if not (np.isnan(vn) or np.isnan(ve) or np.isnan(vt)):
                vals_none.append(vn)
                vals_est.append(ve)
                vals_true.append(vt)

        if vals_none:
            mn = np.mean(vals_none)
            me = np.mean(vals_est)
            mt = np.mean(vals_true)
            if lib:
                de = mn - me
                dt = mn - mt
            else:
                de = me - mn
                dt = mt - mn
            sign_e = "+" if de > 0 else ""
            sign_t = "+" if dt > 0 else ""
            lines.append(
                f"| {metric:<12s} | {mn:>10.2f} | {me:>10.2f} "
                f"| {mt:>10.2f} | {sign_e}{de:>10.2f} | {sign_t}{dt:>11.2f} |"
            )
        else:
            lines.append(f"| {metric:<12s} | {'nan':>10s} | {'nan':>10s} "
                         f"| {'nan':>10s} | {'nan':>11s} | {'nan':>12s} |")

    lines.append("")

    # Solvers that regressed
    lines.append("## Solvers where whitening HURT (estimated worse than none)")
    lines.append("")
    regressed = []
    for solver in solver_names:
        regressions = []
        for metric in METRICS:
            lib = metric in LOWER_IS_BETTER
            vn = results[solver]["none"]["aggregate"][metric]["mean"]
            ve = results[solver]["estimated"]["aggregate"][metric]["mean"]
            if np.isnan(vn) or np.isnan(ve):
                continue
            if lib and ve > vn * 1.02:  # >2% worse
                regressions.append(f"{metric}: {vn:.2f} -> {ve:.2f}")
            elif not lib and ve < vn * 0.98:
                regressions.append(f"{metric}: {vn:.2f} -> {ve:.2f}")
        if regressions:
            regressed.append((solver, regressions))

    if regressed:
        for solver, regs in regressed:
            lines.append(f"- **{solver}**: {'; '.join(regs)}")
    else:
        lines.append("None detected (all solvers improved or stayed the same).")

    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
