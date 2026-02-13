"""Phase 3 evaluation: solvers that accept but ignore noise_cov.

Runs a benchmark on the 5 Phase 3 solvers + reference solvers.

Usage:
    python scripts/phase3_eval.py baseline
    python scripts/phase3_eval.py after
    python scripts/phase3_eval.py compare
"""

import json
import sys
from pathlib import Path

from invert.benchmark import BenchmarkRunner
from invert.forward import create_forward_model, get_info

TARGET_SOLVERS = [
    # --- Reference solvers (known-good, for rank context) ---
    "MNE",
    "Champagne",
    "LCMV",
    "dSPM",
    # --- Phase 1/2 solvers (regression check for prepare_forward removal) ---
    "sLORETA",
    "eLORETA",
    "LAURA",
    "WMNE",
    "VBSBL",
    # --- Phase 3 solvers under modification ---
    "MCE",
    "GFTMCE",
    "UNIG",
    "MCMV",
    # "EBB",  # commented out in solver registry
]

N_SAMPLES = 10
RANDOM_SEED = 42

DATASETS = ["focal_source", "multi_source", "extended_source", "noisy"]


def run_eval(tag: str = "baseline") -> Path:
    info = get_info(kind="biosemi32")
    fwd = create_forward_model(sampling="ico2", info=info)

    runner = BenchmarkRunner(
        fwd,
        info,
        solvers=TARGET_SOLVERS,
        n_samples=N_SAMPLES,
        n_jobs=-1,
        random_seed=RANDOM_SEED,
    )
    runner.run()

    out_path = Path(f"results/phase3_eval_{tag}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    runner.save(
        out_path,
        compact=False,
        name=f"Phase 3 Eval ({tag})",
        description=f"Phase 3 evaluation ({N_SAMPLES} samples, seed={RANDOM_SEED}) — {tag}",
    )
    print(f"\nResults saved to {out_path}")
    return out_path


def _avg_metric(summary: dict, solver: str, metric: str) -> float:
    vals = []
    for ds in DATASETS:
        key = f"{solver} | {ds}"
        entry = summary.get(key, {})
        if metric in entry:
            vals.append(entry[metric])
    return sum(vals) / len(vals) if vals else float("nan")


def print_comparison(before_path: Path, after_path: Path) -> None:
    before = json.loads(before_path.read_text())
    after = json.loads(after_path.read_text())

    ref_solvers = {"MNE", "Champagne", "LCMV", "dSPM"}

    print("\n" + "=" * 100)
    print("Phase 3: BEFORE vs AFTER comparison")
    print("=" * 100)

    # Global ranks
    b_ranks = before.get("global_ranks", {})
    a_ranks = after.get("global_ranks", {})

    print(f"\n{'Solver':<22} {'Before':>8} {'After':>8} {'Delta':>8}  Notes")
    print("-" * 70)
    for solver in TARGET_SOLVERS:
        b = b_ranks.get(solver, float("nan"))
        a = a_ranks.get(solver, float("nan"))
        delta = a - b
        marker = ""
        if solver in ref_solvers:
            marker = "(ref)"
        elif delta < -0.5:
            marker = "IMPROVED"
        elif delta > 0.5:
            marker = "REGRESSED"
        print(f"{solver:<22} {b:>8.2f} {a:>8.2f} {delta:>+8.2f}  {marker}")

    # Per-metric averages
    b_summary = before.get("summary", {})
    a_summary = after.get("summary", {})

    for metric, label, lower_better in [
        ("mean_localization_error", "MLE (lower=better)", True),
        ("emd", "EMD (lower=better)", True),
        ("correlation", "Correlation (higher=better)", False),
    ]:
        print(f"\n--- {label} ---")
        print(f"  {'Solver':<22} {'Before':>10} {'After':>10} {'Delta':>10} {'%':>8}")
        print(f"  {'-' * 62}")
        for solver in TARGET_SOLVERS:
            b = _avg_metric(b_summary, solver, metric)
            a = _avg_metric(a_summary, solver, metric)
            delta = a - b
            if b != 0 and b == b:
                pct = 100.0 * delta / abs(b)
            else:
                pct = float("nan")
            note = ""
            if solver not in ref_solvers:
                if lower_better and delta < -0.5:
                    note = " <--"
                elif not lower_better and delta > 0.01:
                    note = " <--"
            print(
                f"  {solver:<22} {b:>10.3f} {a:>10.3f} {delta:>+10.3f} {pct:>+7.1f}%{note}"
            )

    # Per-dataset ranks
    for dataset in DATASETS:
        b_ds = before.get("ranks", {}).get(dataset, {})
        a_ds = after.get("ranks", {}).get(dataset, {})
        if not b_ds:
            continue
        print(f"\n--- Ranks: {dataset} ---")
        print(f"  {'Solver':<22} {'Before':>8} {'After':>8} {'Delta':>8}")
        print(f"  {'-' * 54}")
        for solver in TARGET_SOLVERS:
            b = b_ds.get(solver, float("nan"))
            a = a_ds.get(solver, float("nan"))
            delta = a - b
            print(f"  {solver:<22} {b:>8.2f} {a:>8.2f} {delta:>+8.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        tag = sys.argv[1]
    else:
        tag = "baseline"

    if tag == "compare":
        before_path = Path("results/phase3_eval_baseline.json")
        after_path = Path("results/phase3_eval_after.json")
        if not before_path.exists():
            print(f"Error: {before_path} not found. Run baseline first.")
            sys.exit(1)
        if not after_path.exists():
            print(f"Error: {after_path} not found. Run 'after' eval first.")
            sys.exit(1)
        print_comparison(before_path, after_path)
    else:
        run_eval(tag=tag)
