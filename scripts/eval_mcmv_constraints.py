"""Evaluate MCMV constraint selection strategies.

Compares similarity-based, distance-based, and LCMV-preloc constraint
selection against an LCMV baseline across focal, multi-source, and noisy
datasets.

Usage:
    python scripts/eval_mcmv_constraints.py
    python scripts/eval_mcmv_constraints.py --n-samples 10 --seed 42
"""

import argparse
import importlib.util as _ilu
import json
import sys
import time
from pathlib import Path

import mne
import numpy as np
from tqdm import tqdm

# Direct-load evaluate_all to avoid invert.evaluate.__init__ side-effects
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
from invert.solvers.beamformers.lcmv import SolverLCMV
from invert.solvers.beamformers.mcmv import SolverMCMV
from invert.util.util import pos_from_forward

# ---------------------------------------------------------------------------
# Dataset definitions (subset of benchmark datasets, no extended_source)
# ---------------------------------------------------------------------------
DATASETS = {
    "focal_source": dict(
        n_sources=1, n_orders=0, snr_range=(3.0, 5.0), n_timepoints=50,
    ),
    "multi_source": dict(
        n_sources=(2, 4), n_orders=0, snr_range=(1.0, 3.0), n_timepoints=50,
    ),
    "noisy": dict(
        n_sources=(1, 3), n_orders=0, snr_range=(-5.0, 0.0), n_timepoints=50,
    ),
}

# ---------------------------------------------------------------------------
# Solver variant definitions
# ---------------------------------------------------------------------------
VARIANTS = [
    # (label, solver_class, make_kwargs)
    ("LCMV (baseline)", SolverLCMV, {}),
    ("MCMV-sim-k3", SolverMCMV, dict(k_constraints=3, constraint_mode="similarity")),
    ("MCMV-sim-k5", SolverMCMV, dict(k_constraints=5, constraint_mode="similarity")),
    ("MCMV-dist-k3", SolverMCMV, dict(k_constraints=3, constraint_mode="distance")),
    ("MCMV-dist-k5", SolverMCMV, dict(k_constraints=5, constraint_mode="distance")),
    ("MCMV-preloc", SolverMCMV, dict(constraint_mode="lcmv_preloc", n_preloc_peaks=5)),
]

METRIC_KEYS = ["Mean_Localization_Error", "EMD", "sd", "average_precision", "correlation"]
SHORT_NAMES = {"Mean_Localization_Error": "MLE", "EMD": "EMD", "sd": "SD",
               "average_precision": "AP", "correlation": "Corr"}


def make_noise_cov(sim_info, info):
    """Average per-sample noise covs into an MNE Covariance object."""
    noise_cov_array = np.mean(
        [sim_info.iloc[j]["noise_cov_est"] for j in range(len(sim_info))],
        axis=0,
    )
    ch_names = list(info["ch_names"])
    return mne.Covariance(
        data=noise_cov_array,
        names=ch_names,
        bads=list(info.get("bads", [])),
        projs=list(info.get("projs", [])),
        nfree=len(sim_info),
    )


def run_variant(label, solver_cls, make_kwargs, forward, info, x_batch,
                y_batch, adjacency, pos, noise_cov):
    """Run a single solver variant on all samples, return per-sample metrics."""
    n_samples = len(x_batch)
    per_sample = []

    for i in range(n_samples):
        solver = solver_cls()
        evoked = mne.EvokedArray(x_batch[i], info, tmin=0.0, verbose=0)
        solver.make_inverse_operator(
            forward, evoked, alpha="auto", noise_cov=noise_cov, **make_kwargs
        )
        stc = solver.apply_inverse_operator(evoked)
        y_pred = stc.data
        metrics = evaluate_all(y_batch[i], y_pred, adjacency, adjacency, pos, pos)
        per_sample.append(metrics)

    return per_sample


def median_metrics(per_sample):
    """Compute median of each metric across samples."""
    out = {}
    for k in METRIC_KEYS:
        vals = [s[k] for s in per_sample]
        out[k] = float(np.nanmedian(vals))
    return out


def print_table(title, rows):
    """Print a formatted comparison table.

    rows: list of (label, median_dict)
    """
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    header = f"{'Variant':<20}"
    for k in METRIC_KEYS:
        header += f" {SHORT_NAMES[k]:>8}"
    print(header)
    print("-" * 80)
    for label, med in rows:
        line = f"{label:<20}"
        for k in METRIC_KEYS:
            line += f" {med[k]:>8.2f}"
        print(line)


def main():
    parser = argparse.ArgumentParser(description="MCMV constraint selection eval")
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling", type=str, default="ico2")
    parser.add_argument("--kind", type=str, default="biosemi32")
    parser.add_argument("--save-json", type=str, default=None,
                        help="Path to save JSON results")
    args = parser.parse_args()

    print(f"Config: n_samples={args.n_samples}, seed={args.seed}, "
          f"sampling={args.sampling}, kind={args.kind}")

    info = get_info(kind=args.kind)
    fwd = create_forward_model(sampling=args.sampling, info=info)
    pos = pos_from_forward(fwd, verbose=0)
    adjacency = mne.spatial_src_adjacency(fwd["src"], verbose=0)

    all_results = {}  # {dataset: {variant_label: per_sample_metrics}}

    for ds_name, ds_params in DATASETS.items():
        print(f"\n--- Generating {ds_name} ({args.n_samples} samples) ---")
        sim_config = SimulationConfig(
            batch_size=args.n_samples,
            n_sources=ds_params["n_sources"],
            n_orders=ds_params["n_orders"],
            snr_range=ds_params["snr_range"],
            n_timepoints=ds_params["n_timepoints"],
            random_seed=args.seed,
            estimate_noise_cov=True,
            return_noise_cov=True,
        )
        gen = SimulationGenerator(fwd, config=sim_config)
        x_batch, y_batch, sim_info = next(gen.generate())
        noise_cov = make_noise_cov(sim_info, info)

        ds_results = {}
        for label, solver_cls, make_kwargs in tqdm(VARIANTS, desc=ds_name):
            t0 = time.time()
            per_sample = run_variant(
                label, solver_cls, make_kwargs,
                fwd, info, x_batch, y_batch, adjacency, pos, noise_cov,
            )
            elapsed = time.time() - t0
            ds_results[label] = per_sample
            med = median_metrics(per_sample)
            tqdm.write(f"  {label:<20} MLE={med['Mean_Localization_Error']:.1f}  "
                       f"EMD={med['EMD']:.1f}  Corr={med['correlation']:.3f}  "
                       f"({elapsed:.1f}s)")

        all_results[ds_name] = ds_results

    # Print per-dataset tables
    for ds_name, ds_results in all_results.items():
        rows = [(label, median_metrics(ds_results[label])) for label, _, _ in VARIANTS]
        print_table(f"Dataset: {ds_name}", rows)

    # Aggregated table (median across all datasets)
    print_table_aggregated(all_results)

    # Save JSON if requested
    if args.save_json:
        save_results_json(all_results, args, args.save_json)


def print_table_aggregated(all_results):
    """Print aggregated table: for each variant, pool all samples across datasets."""
    rows = []
    for label, _, _ in VARIANTS:
        pooled = []
        for ds_results in all_results.values():
            pooled.extend(ds_results[label])
        rows.append((label, median_metrics(pooled)))
    print_table("AGGREGATED (all datasets)", rows)


def save_results_json(all_results, args, path):
    """Save full results to JSON."""
    out = {
        "config": {
            "n_samples": args.n_samples,
            "seed": args.seed,
            "sampling": args.sampling,
            "kind": args.kind,
        },
        "datasets": {},
    }
    for ds_name, ds_results in all_results.items():
        ds_out = {}
        for label, per_sample in ds_results.items():
            ds_out[label] = {
                "median": median_metrics(per_sample),
                "per_sample": per_sample,
            }
        out["datasets"][ds_name] = ds_out

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nResults saved to {p}")


if __name__ == "__main__":
    main()
