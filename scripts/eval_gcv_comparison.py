"""Compare GCV variants (GCV, MGCV, RGCV, R1GCV) across selected solvers.

For each solver × GCV method combination, runs a quick benchmark and reports
metrics.  The script does NOT use BenchmarkRunner because we need to pass
``regularisation_method`` to the solver constructor.

Usage:
    python scripts/eval_gcv_comparison.py
"""

from __future__ import annotations

import importlib
import importlib.util as _ilu
import json
import logging
import sys
import time
from pathlib import Path

import mne
import numpy as np
from tqdm import tqdm

# Load evaluate_all via spec loader to avoid triggering invert.models import
_spec = _ilu.spec_from_file_location(
    "invert.evaluate.evaluate",
    str(Path(__file__).resolve().parents[1] / "invert" / "evaluate" / "evaluate.py"),
)
assert _spec is not None and _spec.loader is not None
_eval_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_eval_mod)
evaluate_all = _eval_mod.evaluate_all

from invert.forward import create_forward_model, get_info
from invert.simulate import SimulationConfig, SimulationGenerator
from invert.util.util import pos_from_forward

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────
SOLVERS = {
    # name -> (module_path, class_name)
    "MNE": ("invert.solvers.minimum_norm.mne", "SolverMNE"),
    "wMNE": ("invert.solvers.minimum_norm.wmne", "SolverWMNE"),
    "dSPM": ("invert.solvers.minimum_norm.dspm", "SolverDSPM"),
    "sLORETA": ("invert.solvers.minimum_norm.sloreta", "SolverSLORETA"),
    "eLORETA": ("invert.solvers.minimum_norm.eloreta", "SolverELORETA"),
    "LAURA": ("invert.solvers.minimum_norm.laura", "SolverLAURA"),
    "LCMV": ("invert.solvers.beamformers.lcmv", "SolverLCMV"),
    "ESMV": ("invert.solvers.beamformers.esmv", "SolverESMV"),
}

GCV_METHODS = ["GCV", "MGCV", "RGCV", "R1GCV"]

N_SAMPLES = 10
RANDOM_SEED = 42


# ── Datasets (same as BenchmarkRunner defaults) ───────────────────────────
DATASETS = {
    "focal_source": dict(n_sources=(1, 1), n_orders=(0, 0), snr_range=(0.9, 0.99)),
    "multi_source": dict(n_sources=(2, 5), n_orders=(0, 0), snr_range=(0.9, 0.99)),
    "extended_source": dict(n_sources=(1, 3), n_orders=(1, 5), snr_range=(0.9, 0.99)),
    "noisy": dict(n_sources=(1, 3), n_orders=(0, 3), snr_range=(0.1, 0.4)),
}


def get_solver(name: str, reg_method: str):
    """Instantiate a solver with the given regularisation method."""
    module_path, class_name = SOLVERS[name]
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(regularisation_method=reg_method)


def run_comparison():
    info = get_info(kind="biosemi32")
    fwd = create_forward_model(sampling="ico2", info=info)
    pos = pos_from_forward(fwd)
    adjacency = mne.spatial_src_adjacency(fwd["src"], verbose=0)

    # results[dataset][solver][method] = {metric: value}
    results: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    # Track which alpha index each method picks
    alpha_picks: dict[str, dict[str, dict[str, list[int]]]] = {}

    total = len(DATASETS) * len(SOLVERS) * len(GCV_METHODS)
    pbar = tqdm(total=total, desc="Overall")

    for ds_name, ds_kwargs in DATASETS.items():
        results[ds_name] = {}
        alpha_picks[ds_name] = {}

        # Generate data once per dataset
        sim_config = SimulationConfig(
            batch_size=N_SAMPLES,
            n_timepoints=10,
            random_seed=RANDOM_SEED,
            **ds_kwargs,
        )
        gen = SimulationGenerator(fwd, config=sim_config)
        x_batch, y_batch, _ = next(gen.generate())

        for solver_name in SOLVERS:
            results[ds_name][solver_name] = {}
            alpha_picks[ds_name][solver_name] = {}

            for method in GCV_METHODS:
                pbar.set_postfix_str(f"{ds_name}/{solver_name}/{method}")

                np.random.seed(RANDOM_SEED)
                solver = get_solver(solver_name, method)

                # Build inverse operators once with first sample
                evoked0 = mne.EvokedArray(x_batch[0], info, tmin=0.0, verbose=0)
                if solver.require_data:
                    solver.make_inverse_operator(fwd, evoked0, alpha="auto")
                else:
                    solver.make_inverse_operator(fwd, alpha="auto")

                sample_metrics: list[dict] = []
                selected_alphas: list[int] = []

                for i in range(N_SAMPLES):
                    evoked = mne.EvokedArray(x_batch[i], info, tmin=0.0, verbose=0)
                    stc = solver.apply_inverse_operator(evoked)
                    selected_alphas.append(
                        solver.last_reg_idx if solver.last_reg_idx is not None else -1
                    )
                    y_pred = stc.data
                    metrics = evaluate_all(
                        y_batch[i], y_pred, adjacency, adjacency, pos, pos
                    )
                    sample_metrics.append(metrics)

                # Aggregate: median over samples
                agg = {}
                for key in sample_metrics[0]:
                    vals = [m[key] for m in sample_metrics]
                    agg[key] = float(np.nanmedian(vals))

                results[ds_name][solver_name][method] = agg
                alpha_picks[ds_name][solver_name][method] = selected_alphas
                pbar.update(1)

    pbar.close()
    return results, alpha_picks


def print_results(results, alpha_picks):
    """Print a comparison table for each metric."""
    all_metrics = list(next(iter(next(iter(next(iter(results.values())).values())).values())).keys())

    # ── Per-metric, per-dataset tables ──
    for metric in all_metrics:
        print(f"\n{'=' * 90}")
        print(f"  METRIC: {metric}")
        print(f"{'=' * 90}")

        for ds_name in results:
            print(f"\n  Dataset: {ds_name}")
            header = f"  {'Solver':<12}" + "".join(f"{m:>12}" for m in GCV_METHODS) + "    best"
            print(header)
            print(f"  {'-' * (len(header) - 2)}")

            for solver_name in SOLVERS:
                vals = []
                for method in GCV_METHODS:
                    v = results[ds_name][solver_name][method].get(metric, float("nan"))
                    vals.append(v)

                # Determine best: lower is better for MLE/EMD/sd, higher for correlation/AP
                lower_better = metric in {"Mean_Localization_Error", "EMD", "sd"}
                if lower_better:
                    best_idx = int(np.nanargmin(vals))
                else:
                    best_idx = int(np.nanargmax(vals))

                row = f"  {solver_name:<12}"
                for j, v in enumerate(vals):
                    marker = " *" if j == best_idx else "  "
                    row += f"{v:>10.4f}{marker}"
                row += f"    {GCV_METHODS[best_idx]}"
                print(row)

    # ── Summary: wins per method across all datasets and solvers ──
    print(f"\n{'=' * 90}")
    print("  SUMMARY: Method wins per metric (across all datasets × solvers)")
    print(f"{'=' * 90}")

    for metric in all_metrics:
        lower_better = metric in {"Mean_Localization_Error", "EMD", "sd"}
        wins = {m: 0 for m in GCV_METHODS}
        for ds_name in results:
            for solver_name in SOLVERS:
                vals = [
                    results[ds_name][solver_name][m].get(metric, float("nan"))
                    for m in GCV_METHODS
                ]
                if lower_better:
                    best_idx = int(np.nanargmin(vals))
                else:
                    best_idx = int(np.nanargmax(vals))
                wins[GCV_METHODS[best_idx]] += 1

        total_combos = len(results) * len(SOLVERS)
        print(f"\n  {metric}:")
        for m in GCV_METHODS:
            bar = "#" * wins[m]
            print(f"    {m:<8} {wins[m]:>3}/{total_combos}  {bar}")

    # ── Alpha index distribution ──
    print(f"\n{'=' * 90}")
    print("  SELECTED ALPHA INDICES (median over samples, per dataset)")
    print(f"{'=' * 90}")
    for ds_name in alpha_picks:
        print(f"\n  Dataset: {ds_name}")
        header = f"  {'Solver':<12}" + "".join(f"{m:>8}" for m in GCV_METHODS)
        print(header)
        print(f"  {'-' * (len(header) - 2)}")
        for solver_name in SOLVERS:
            row = f"  {solver_name:<12}"
            for method in GCV_METHODS:
                idxs = alpha_picks[ds_name][solver_name][method]
                median_idx = float(np.median(idxs))
                row += f"{median_idx:>8.1f}"
            print(row)


if __name__ == "__main__":
    t0 = time.time()
    results, alpha_picks = run_comparison()
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s\n")
    print_results(results, alpha_picks)

    # Save raw results
    out_path = Path("results/gcv_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nRaw results saved to {out_path}")
