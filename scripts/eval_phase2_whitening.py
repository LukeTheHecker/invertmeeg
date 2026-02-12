"""Phase 2 whitening rollout benchmark.

Tests the 5 solvers modified in Phase 2 (standardize ad-hoc whitening)
plus reference solvers for comparison.

Usage:
    python scripts/eval_phase2_whitening.py              # run baseline (before)
    python scripts/eval_phase2_whitening.py after         # run after changes
    python scripts/eval_phase2_whitening.py compare       # compare before/after
"""

import json
import sys
from pathlib import Path

from invert.benchmark import BenchmarkRunner
from invert.forward import create_forward_model, get_info

PHASE2_SOLVERS = [
    # --- Reference solvers (unchanged, for rank stability check) ---
    "MNE",
    "dSPM",
    "LCMV",
    "FLEX-MUSIC",
    # --- Phase 2 targets ---
    "VBSBL",
    "ReciPSIICOS-Plain",
    "ReciPSIICOS-Whitened",
    "Champagne",
    "NLChampagne",
]

N_SAMPLES = 10
RANDOM_SEED = 42


def run_eval(tag: str = "before") -> Path:
    info = get_info(kind="biosemi32")
    fwd = create_forward_model(sampling="ico2", info=info)

    runner = BenchmarkRunner(
        fwd,
        info,
        solvers=PHASE2_SOLVERS,
        n_samples=N_SAMPLES,
        n_jobs=1,
        random_seed=RANDOM_SEED,
    )
    runner.run()

    out_path = Path(f"results/phase2_whitening_{tag}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    runner.save(
        out_path,
        compact=False,
        name=f"Phase 2 Whitening ({tag})",
        description=(
            f"Phase 2 whitening rollout eval ({N_SAMPLES} samples, "
            f"seed={RANDOM_SEED}, n_jobs=1) — {tag}"
        ),
    )
    print(f"\nResults saved to {out_path}")
    return out_path


def print_comparison(before_path: Path, after_path: Path) -> None:
    before = json.loads(before_path.read_text())
    after = json.loads(after_path.read_text())

    ref_solvers = {"MNE", "dSPM", "LCMV", "FLEX-MUSIC"}

    print("\n" + "=" * 100)
    print("Phase 2 Whitening: BEFORE vs AFTER")
    print("=" * 100)

    b_ranks = before.get("global_ranks", {})
    a_ranks = after.get("global_ranks", {})

    print(f"\n{'Solver':<25} {'Before':>8} {'After':>8} {'Delta':>8}  Notes")
    print("-" * 75)
    for solver in PHASE2_SOLVERS:
        b = b_ranks.get(solver, float("nan"))
        a = a_ranks.get(solver, float("nan"))
        delta = a - b
        marker = ""
        if solver in ref_solvers:
            marker = "(ref)"
        elif delta < -1:
            marker = "IMPROVED"
        elif delta > 1:
            marker = "REGRESSED"
        print(f"{solver:<25} {b:>8.1f} {a:>8.1f} {delta:>+8.1f}  {marker}")

    # Per-metric comparison for Phase 2 targets only
    for metric in ["mle", "emd", "correlation"]:
        b_met = before.get("metrics", {}).get(metric, {})
        a_met = after.get("metrics", {}).get(metric, {})
        if not b_met or not a_met:
            continue
        print(f"\n--- {metric.upper()} ---")
        print(f"  {'Solver':<25} {'Before':>10} {'After':>10} {'Delta':>10}")
        print(f"  {'-' * 60}")
        for solver in PHASE2_SOLVERS:
            b = b_met.get(solver, float("nan"))
            a = a_met.get(solver, float("nan"))
            delta = a - b
            tag = "(ref)" if solver in ref_solvers else ""
            print(f"  {solver:<25} {b:>10.2f} {a:>10.2f} {delta:>+10.2f}  {tag}")


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "before"

    if tag == "compare":
        bp = Path("results/phase2_whitening_before.json")
        ap = Path("results/phase2_whitening_after.json")
        if not bp.exists():
            print(f"Error: {bp} not found. Run 'before' first.")
            sys.exit(1)
        if not ap.exists():
            print(f"Error: {ap} not found. Run 'after' first.")
            sys.exit(1)
        print_comparison(bp, ap)
    else:
        run_eval(tag=tag)
