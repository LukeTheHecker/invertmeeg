"""Experiment A+B+C: Fine alpha grid, γ/c sweep, and composite GCV.

A — Fine grid:   n_reg_params=50 (vs default 10)
B — γ/c sweep:   MGCV c∈{0.85–1.10}, RGCV γ∈{0.3–2.0}, R1GCV γ∈{0.5,1.5}
C — Composite:   geometric mean of GCV × RGCV scores

For each solver × GCV method × dataset, runs 10 samples and reports median
metrics plus a global mean-rank summary.

Usage:
    python scripts/eval_gcv_sweep.py
"""

from __future__ import annotations

import importlib
import importlib.util as _ilu
import json
import time
from copy import deepcopy
from pathlib import Path

import mne
import numpy as np
from tqdm import tqdm

# Load evaluate_all via spec loader (avoids invert.models import issue)
_spec = _ilu.spec_from_file_location(
    "invert.evaluate.evaluate",
    str(Path(__file__).resolve().parents[1] / "invert" / "evaluate" / "evaluate.py"),
)
assert _spec is not None and _spec.loader is not None
_eval_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_eval_mod)
evaluate_all = _eval_mod.evaluate_all

from invert.forward import create_forward_model, get_info  # noqa: E402
from invert.simulate import SimulationConfig, SimulationGenerator  # noqa: E402
from invert.util.util import pos_from_forward  # noqa: E402

# ── Configuration ──────────────────────────────────────────────────────────

SOLVERS = {
    "MNE": ("invert.solvers.minimum_norm.mne", "SolverMNE"),
    "wMNE": ("invert.solvers.minimum_norm.wmne", "SolverWMNE"),
    "dSPM": ("invert.solvers.minimum_norm.dspm", "SolverDSPM"),
    "sLORETA": ("invert.solvers.minimum_norm.sloreta", "SolverSLORETA"),
    "eLORETA": ("invert.solvers.minimum_norm.eloreta", "SolverELORETA"),
    "LAURA": ("invert.solvers.minimum_norm.laura", "SolverLAURA"),
    "LCMV": ("invert.solvers.beamformers.lcmv", "SolverLCMV"),
    "ESMV": ("invert.solvers.beamformers.esmv", "SolverESMV"),
}

# (label, method, gamma)
# method is passed to regularise_gcv; gamma overrides the solver attribute
METHODS = [
    # Baseline
    ("GCV", "gcv", 1.0),
    # MGCV sweep  (c < 1 = bias toward LESS reg, c > 1 = MORE)
    ("MGCV-0.85", "mgcv", 0.85),
    ("MGCV-0.90", "mgcv", 0.90),
    ("MGCV-0.95", "mgcv", 0.95),
    ("MGCV-1.05", "mgcv", 1.05),
    ("MGCV-1.10", "mgcv", 1.10),
    # RGCV sweep  (γ < 1 = bias toward MORE reg, γ > 1 = LESS)
    ("RGCV-0.3", "rgcv", 0.3),
    ("RGCV-0.5", "rgcv", 0.5),
    ("RGCV-0.7", "rgcv", 0.7),
    ("RGCV-0.9", "rgcv", 0.9),
    ("RGCV-1.1", "rgcv", 1.1),
    ("RGCV-1.3", "rgcv", 1.3),
    ("RGCV-1.5", "rgcv", 1.5),
    ("RGCV-2.0", "rgcv", 2.0),
    # R1GCV at two γ values
    ("R1GCV-0.5", "r1gcv", 0.5),
    ("R1GCV-1.5", "r1gcv", 1.5),
    # Composite (geomean of GCV × RGCV)
    ("Comp-0.5", "composite", 0.5),
    ("Comp-1.5", "composite", 1.5),
]

N_REG_PARAMS = 50
N_SAMPLES = 10
RANDOM_SEED = 42

DATASETS = {
    "focal_source": dict(n_sources=(1, 1), n_orders=(0, 0), snr_range=(0.9, 0.99)),
    "multi_source": dict(n_sources=(2, 5), n_orders=(0, 0), snr_range=(0.9, 0.99)),
    "extended_source": dict(n_sources=(1, 3), n_orders=(1, 5), snr_range=(0.9, 0.99)),
    "noisy": dict(n_sources=(1, 3), n_orders=(0, 3), snr_range=(0.1, 0.4)),
}

# Metrics where lower is better
LOWER_BETTER = {"Mean_Localization_Error", "EMD", "sd"}


# ── Helpers ────────────────────────────────────────────────────────────────


def get_solver_cls(name: str):
    module_path, class_name = SOLVERS[name]
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def build_solver(solver_name: str, fwd, info, x_sample):
    """Create solver with fine grid and build inverse operators."""
    cls = get_solver_cls(solver_name)
    solver = cls(regularisation_method="GCV", n_reg_params=N_REG_PARAMS)
    evoked = mne.EvokedArray(x_sample, info, tmin=0.0, verbose=0)
    if solver.require_data:
        solver.make_inverse_operator(deepcopy(fwd), evoked, alpha="auto")
    else:
        solver.make_inverse_operator(deepcopy(fwd), alpha="auto")
    return solver


def apply_with_method(solver, solver_name, evoked, method, gamma):
    """Apply inverse operator using a specific GCV variant.

    Mutates solver settings, calls apply_inverse_operator, and returns
    (source_data, selected_alpha_idx).
    """
    solver.regularisation_method = method.upper()
    # Set the appropriate gamma attribute
    if method in ("gcv",):
        solver.gcv_gamma = gamma
    elif method == "mgcv":
        solver.mgcv_gamma = gamma
    elif method in ("rgcv", "composite"):
        solver.rgcv_gamma = gamma
    elif method == "r1gcv":
        solver.r1gcv_gamma = gamma
    solver.use_last_alpha = False
    solver.last_reg_idx = None
    stc = solver.apply_inverse_operator(evoked)
    return stc.data, solver.last_reg_idx


# ── Main ───────────────────────────────────────────────────────────────────


def run():
    info = get_info(kind="biosemi32")
    fwd = create_forward_model(sampling="ico2", info=info)
    pos = pos_from_forward(fwd)
    adjacency = mne.spatial_src_adjacency(fwd["src"], verbose=0)

    # results[ds][solver][method] = {metric: median_value}
    results: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    # alpha_idx[ds][solver][method] = median selected alpha index
    alpha_idx: dict[str, dict[str, dict[str, float]]] = {}

    method_labels = [m[0] for m in METHODS]
    total = len(DATASETS) * len(SOLVERS) * len(METHODS)
    pbar = tqdm(total=total, desc="Overall")

    for ds_name, ds_kwargs in DATASETS.items():
        results[ds_name] = {}
        alpha_idx[ds_name] = {}

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
            alpha_idx[ds_name][solver_name] = {}

            # Build inverse operators once per (solver, dataset)
            np.random.seed(RANDOM_SEED)
            solver = build_solver(solver_name, fwd, info, x_batch[0])

            for label, method, gamma in METHODS:
                pbar.set_postfix_str(f"{ds_name}/{solver_name}/{label}")

                sample_metrics: list[dict] = []
                sample_idxs: list[int] = []

                for i in range(N_SAMPLES):
                    evoked = mne.EvokedArray(x_batch[i], info, tmin=0.0, verbose=0)
                    y_pred, idx = apply_with_method(
                        solver, solver_name, evoked, method, gamma
                    )
                    sample_idxs.append(idx if idx is not None else -1)
                    metrics = evaluate_all(
                        y_batch[i], y_pred, adjacency, adjacency, pos, pos
                    )
                    sample_metrics.append(metrics)

                # Aggregate: median over samples
                agg = {}
                for key in sample_metrics[0]:
                    vals = [m[key] for m in sample_metrics]
                    agg[key] = float(np.nanmedian(vals))

                results[ds_name][solver_name][label] = agg
                alpha_idx[ds_name][solver_name][label] = float(np.median(sample_idxs))
                pbar.update(1)

    pbar.close()
    return results, alpha_idx, method_labels


# ── Reporting ──────────────────────────────────────────────────────────────


def print_results(results, alpha_idx, method_labels):
    all_metrics = list(
        next(iter(next(iter(next(iter(results.values())).values())).values())).keys()
    )

    # ── Per-metric summary: mean rank across datasets × solvers ──────
    print("\n" + "=" * 100)
    print("  MEAN RANK across all (dataset × solver) combos  [lower = better]")
    print("  (For each combo, methods are ranked 1..N; ties share the mean rank)")
    print("=" * 100)

    for metric in all_metrics:
        lower = metric in LOWER_BETTER
        # Collect values: rows = (ds, solver), cols = methods
        all_vals = []
        for ds_name in results:
            for solver_name in SOLVERS:
                row = []
                for label in method_labels:
                    v = results[ds_name][solver_name][label].get(metric, float("nan"))
                    row.append(v)
                all_vals.append(row)

        all_vals = np.array(all_vals)  # (n_combos, n_methods)
        n_combos, n_methods = all_vals.shape

        # Compute ranks per row
        ranks = np.zeros_like(all_vals)
        for i in range(n_combos):
            row = all_vals[i]
            if lower:
                order = np.argsort(row)
            else:
                order = np.argsort(-row)
            # Assign ranks (1-based), handle NaN
            r = np.empty(n_methods)
            r[:] = np.nan
            for rank_pos, idx in enumerate(order):
                if np.isfinite(row[idx]):
                    r[idx] = rank_pos + 1
            ranks[i] = r

        mean_ranks = np.nanmean(ranks, axis=0)
        sorted_order = np.argsort(mean_ranks)

        direction = "↓" if lower else "↑"
        print(f"\n  {metric} ({direction})")
        print(f"  {'Rank':<6} {'Method':<14} {'Mean Rank':>10}  {'Wins':>5}")
        print(f"  {'-' * 40}")

        # Count wins (rank == 1)
        wins = np.sum(ranks == 1, axis=0).astype(int)

        for pos, idx in enumerate(sorted_order[:10]):  # top 10
            print(
                f"  {pos + 1:<6} {method_labels[idx]:<14} {mean_ranks[idx]:>10.2f}"
                f"  {wins[idx]:>5}"
            )

    # ── Per-dataset × solver breakdown for top methods ───────────────
    # Find top 5 methods by average mean-rank across metrics
    overall_mean_rank = np.zeros(len(method_labels))
    for metric in all_metrics:
        lower = metric in LOWER_BETTER
        for ds_name in results:
            for solver_name in SOLVERS:
                vals = [
                    results[ds_name][solver_name][label].get(metric, float("nan"))
                    for label in method_labels
                ]
                vals = np.array(vals)
                if lower:
                    order = np.argsort(vals)
                else:
                    order = np.argsort(-vals)
                for rank_pos, idx in enumerate(order):
                    if np.isfinite(vals[idx]):
                        overall_mean_rank[idx] += rank_pos + 1
    n_total = len(results) * len(SOLVERS) * len(all_metrics)
    overall_mean_rank /= n_total

    top5_idx = np.argsort(overall_mean_rank)[:6]
    top5_labels = [method_labels[i] for i in top5_idx]

    print("\n" + "=" * 100)
    print(f"  TOP METHODS (by overall mean rank): {top5_labels}")
    print("=" * 100)

    for metric in all_metrics:
        lower = metric in LOWER_BETTER
        direction = "↓" if lower else "↑"
        print(f"\n  {metric} ({direction})")
        for ds_name in results:
            header = f"    {ds_name + ':':<20}" + "".join(
                f"{l:>14}" for l in top5_labels
            )
            print(header)
            for solver_name in SOLVERS:
                vals = [
                    results[ds_name][solver_name][l].get(metric, float("nan"))
                    for l in top5_labels
                ]
                # Mark best
                if lower:
                    best_i = int(np.nanargmin(vals))
                else:
                    best_i = int(np.nanargmax(vals))
                row = f"      {solver_name:<14}"
                for j, v in enumerate(vals):
                    marker = " *" if j == best_i else "  "
                    row += f"{v:>12.4f}{marker}"
                print(row)
            print()

    # ── Selected alpha indices ──
    print("\n" + "=" * 100)
    print(f"  MEDIAN ALPHA INDEX (out of {N_REG_PARAMS}) — top methods")
    print("=" * 100)
    for ds_name in alpha_idx:
        print(f"\n  {ds_name}:")
        header = f"    {'Solver':<14}" + "".join(f"{l:>12}" for l in top5_labels)
        print(header)
        for solver_name in SOLVERS:
            row = f"    {solver_name:<14}"
            for label in top5_labels:
                idx = alpha_idx[ds_name][solver_name][label]
                row += f"{idx:>12.1f}"
            print(row)


if __name__ == "__main__":
    t0 = time.time()
    results, alpha_idx, method_labels = run()
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")
    print_results(results, alpha_idx, method_labels)

    # Save raw results
    out_path = Path("results/gcv_sweep.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nRaw results saved to {out_path}")
