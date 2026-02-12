"""Compare noise modes (None vs structured) with and without whitening.

Conditions:
  1. none_raw        — correlation_mode=None, no whitening
  2. none_whitened    — correlation_mode=None, whitened (should ≈ none_raw)
  3. struct_raw       — banded/diagonal/low_rank mix, no whitening
  4. struct_whitened   — banded/diagonal/low_rank mix, whitened

Saves results/release/leaderboard-noise-comparison.json
"""

from __future__ import annotations

import copy
import importlib.util as _ilu
import json
import time
from pathlib import Path

import mne
import numpy as np
from tqdm import tqdm

# --- bootstrap evaluate_all the same way runner.py does ---
_spec = _ilu.spec_from_file_location(
    "invert.evaluate.evaluate",
    str(Path(__file__).resolve().parents[1] / "invert" / "evaluate" / "evaluate.py"),
)
assert _spec is not None and _spec.loader is not None
_eval_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_eval_mod)
evaluate_all = _eval_mod.evaluate_all

from invert.benchmark.datasets import BENCHMARK_DATASETS  # noqa: E402
from invert.benchmark.runner import get_solver_class  # noqa: E402
from invert.forward import create_forward_model, get_info  # noqa: E402
from invert.simulate import SimulationGenerator  # noqa: E402
from invert.simulate.config import SimulationConfig  # noqa: E402
from invert.util.util import pos_from_forward  # noqa: E402

# ── configuration ──────────────────────────────────────────────────────────
SOLVERS = [
    "Champagne",
    "ESMV",
    "LCMV",
    "dSPM",
    "eLORETA",
    "LORETA",
    "sLORETA",
    "LAURA",
    "MNE",
    "SMAP",
    "WMNE",
]
N_SAMPLES = 50
SEED = 42
STRUCTURED_MODES = ["banded", "diagonal", "low_rank"]
METRICS = [
    "Mean_Localization_Error",
    "EMD",
    "sd",
    "average_precision",
    "correlation",
]


# ── helpers ────────────────────────────────────────────────────────────────
def generate_data(forward, ds_config, noise_mode, n_samples=N_SAMPLES, seed=SEED):
    """Generate simulation data.  Returns (x, y, avg_noise_cov)."""
    if noise_mode == "structured":
        chunk_sizes = _split_evenly(n_samples, len(STRUCTURED_MODES))
        x_all, y_all, covs = [], [], []
        for mode, size in zip(STRUCTURED_MODES, chunk_sizes):
            cfg = SimulationConfig(
                batch_size=size,
                n_sources=ds_config.n_sources,
                n_orders=ds_config.n_orders,
                snr_range=ds_config.snr_range,
                n_timepoints=ds_config.n_timepoints,
                correlation_mode=mode,
                random_seed=seed,
                return_noise_cov=True,
                estimate_noise_cov=True,
                noise_cov_n_baseline=1000,
            )
            gen = SimulationGenerator(forward, config=cfg)
            x, y, info = next(gen.generate())
            x_all.append(x)
            y_all.append(y)
            for j in range(len(info)):
                covs.append(info.iloc[j]["noise_cov_est"])
            seed += 1  # different seed per chunk for diversity
        return np.concatenate(x_all), np.concatenate(y_all), np.mean(covs, axis=0)
    else:
        cfg = SimulationConfig(
            batch_size=n_samples,
            n_sources=ds_config.n_sources,
            n_orders=ds_config.n_orders,
            snr_range=ds_config.snr_range,
            n_timepoints=ds_config.n_timepoints,
            correlation_mode=None,
            random_seed=seed,
            return_noise_cov=True,
            estimate_noise_cov=True,
            noise_cov_n_baseline=1000,
        )
        gen = SimulationGenerator(forward, config=cfg)
        x, y, info = next(gen.generate())
        covs = [info.iloc[j]["noise_cov_est"] for j in range(len(info))]
        return x, y, np.mean(covs, axis=0)


def _split_evenly(total, n_chunks):
    base, extra = divmod(total, n_chunks)
    return [base + (1 if i < extra else 0) for i in range(n_chunks)]


def compute_whitener(noise_cov, eps=1e-10):
    """Full-rank symmetric whitener: W s.t. W @ cov @ W.T ≈ I."""
    C = 0.5 * (noise_cov + noise_cov.T)
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, eps)
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return W


def whiten_forward(forward, W):
    fwd = copy.deepcopy(forward)
    fwd["sol"]["data"] = W @ fwd["sol"]["data"]
    return fwd


def run_solver_on_data(solver_name, forward, mne_info, x, y, adjacency, pos):
    """Run one solver on all samples, return list of metric dicts."""
    solver_cls = get_solver_class(solver_name)
    n = len(x)

    solver = solver_cls()

    # Seed for reproducibility
    np.random.seed(SEED)

    sample_metrics = []

    if solver.require_recompute:
        for i in range(n):
            s = solver_cls()
            evoked = mne.EvokedArray(x[i], mne_info, tmin=0.0, verbose=0)
            if s.require_data:
                s.make_inverse_operator(forward, evoked, alpha="auto")
            else:
                s.make_inverse_operator(forward, alpha="auto")
            stc = s.apply_inverse_operator(evoked)
            m = evaluate_all(y[i], stc.data, adjacency, adjacency, pos, pos)
            sample_metrics.append(m)
    else:
        if solver.require_data:
            evoked0 = mne.EvokedArray(x[0], mne_info, tmin=0.0, verbose=0)
            solver.make_inverse_operator(forward, evoked0, alpha="auto")
        else:
            solver.make_inverse_operator(forward, alpha="auto")

        if hasattr(solver, "inverse_operators") and solver.inverse_operators:
            if len(solver.inverse_operators) > 1:
                _, opt_idx = solver.regularise_gcv(x[0])
            else:
                opt_idx = 0
            inv_mat = solver.inverse_operators[opt_idx].data[0]
            for i in range(n):
                y_pred = inv_mat @ x[i]
                m = evaluate_all(y[i], y_pred, adjacency, adjacency, pos, pos)
                sample_metrics.append(m)
        else:
            # fallback: apply_inverse_operator per sample
            for i in range(n):
                evoked = mne.EvokedArray(x[i], mne_info, tmin=0.0, verbose=0)
                stc = solver.apply_inverse_operator(evoked)
                m = evaluate_all(y[i], stc.data, adjacency, adjacency, pos, pos)
                sample_metrics.append(m)

    return sample_metrics


def aggregate(sample_metrics):
    """Aggregate list of metric dicts → {metric: mean}."""
    out = {}
    for metric in METRICS:
        vals = [m[metric] for m in sample_metrics]
        out[metric] = float(np.mean(vals))
    return out


# ── main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()

    info = get_info(kind="biosemi32")
    fwd = create_forward_model(sampling="ico3", info=info)
    pos = pos_from_forward(fwd)
    adjacency = mne.spatial_src_adjacency(fwd["src"], verbose=0)

    CONDITIONS = [
        ("none", False),
        ("none", True),
        ("structured", False),
        ("structured", True),
    ]
    cond_labels = {
        ("none", False): "none_raw",
        ("none", True): "none_whitened",
        ("structured", False): "struct_raw",
        ("structured", True): "struct_whitened",
    }

    # results[cond_label][ds_name][solver_name] = {metric: mean}
    results: dict[str, dict[str, dict[str, dict[str, float]]]] = {
        v: {} for v in cond_labels.values()
    }

    for ds_name, ds_config in BENCHMARK_DATASETS.items():
        print(f"\n{'=' * 70}")
        print(f"DATASET: {ds_name}")
        print(f"{'=' * 70}")

        # Generate data for both noise modes (reuse across whitening conditions)
        gen_data: dict[str, tuple] = {}
        for noise_mode in ["none", "structured"]:
            print(f"  Generating {noise_mode} noise data...")
            x, y, noise_cov = generate_data(fwd, ds_config, noise_mode)
            W = compute_whitener(noise_cov)
            x_white = np.array([W @ x[i] for i in range(len(x))])
            fwd_white = whiten_forward(fwd, W)
            gen_data[noise_mode] = (x, y, x_white, fwd_white)

        for noise_mode, whiten in CONDITIONS:
            cond = cond_labels[(noise_mode, whiten)]
            x_raw, y, x_white, fwd_white = gen_data[noise_mode]
            x_use = x_white if whiten else x_raw
            fwd_use = fwd_white if whiten else fwd
            results[cond][ds_name] = {}

            print(f"\n  Condition: {cond}")
            for solver_name in tqdm(SOLVERS, desc=f"    {cond}/{ds_name}", leave=False):
                sm = run_solver_on_data(
                    solver_name, fwd_use, info, x_use, y, adjacency, pos
                )
                results[cond][ds_name][solver_name] = aggregate(sm)

    # ── save ───────────────────────────────────────────────────────────────
    out_path = Path("results/release/leaderboard-noise-comparison-1000bl.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # ── print summary ──────────────────────────────────────────────────────
    ms = {
        "Mean_Localization_Error": "MLE",
        "EMD": "EMD",
        "sd": "SD",
        "average_precision": "AP",
        "correlation": "Corr",
    }
    lower_better = {"Mean_Localization_Error", "EMD", "sd"}

    print(f"\n{'=' * 100}")
    print("SUMMARY: Effect of structured noise and whitening")
    print(f"{'=' * 100}")

    for ds_name in BENCHMARK_DATASETS:
        print(f"\n--- {ds_name} ---")
        header = f"{'Solver':<15}"
        for cond in cond_labels.values():
            header += f" | {cond:>16}"
        print(header)
        print("-" * len(header))

        for solver_name in SOLVERS:
            for metric in METRICS:
                row = f"  {ms[metric]:<13}"
                vals = []
                for cond in cond_labels.values():
                    v = results[cond][ds_name][solver_name][metric]
                    vals.append(v)
                    row += f" | {v:>16.3f}"
                # Mark if struct_raw→struct_whitened improved
                raw_v = results["struct_raw"][ds_name][solver_name][metric]
                wht_v = results["struct_whitened"][ds_name][solver_name][metric]
                if metric in lower_better:
                    better = wht_v < raw_v * 0.97
                    worse = wht_v > raw_v * 1.03
                else:
                    better = wht_v > raw_v * 1.03
                    worse = wht_v < raw_v * 0.97
                tag = (
                    " << whitening helps"
                    if better
                    else (" >> whitening hurts" if worse else "")
                )
                row += tag
                print(row)
            print(f"  {'---':<13}" + "   ".rjust(19) * 4)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed / 60:.1f} min")
