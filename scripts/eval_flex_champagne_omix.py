"""Fast research evaluation for Flex-Champagne variants on ongoing-mixture data.

This script is intended for short inner-loop experiments while developing
ongoing-mixture improvements. It keeps runtime low by default and compares
baseline and tuned solver variants in one run.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from invert.benchmark import BenchmarkRunner
from invert.benchmark.datasets import BENCHMARK_DATASETS
from invert.forward import create_8channel_montage, create_forward_model, get_info


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick eval for Flex-Champagne ongoing-mixture research."
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=8,
        help="Number of simulated samples per dataset (default: 8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data generation and solver determinism.",
    )
    parser.add_argument(
        "--sampling",
        type=str,
        default="ico2",
        help="Forward model sampling (default: ico2).",
    )
    parser.add_argument(
        "--info-kind",
        type=str,
        default="biosemi32",
        help="MNE montage name for get_info (ignored with --use-custom-8ch).",
    )
    parser.add_argument(
        "--use-custom-8ch",
        action="store_true",
        help="Use the package's custom evenly-spaced 8-channel montage.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ongoing_mixture"],
        choices=sorted(BENCHMARK_DATASETS.keys()),
        help="Datasets to include. Default targets ongoing_mixture only.",
    )
    parser.add_argument(
        "--with-reference-solvers",
        action="store_true",
        help="Also include ESMV and IR-ESMV for context.",
    )
    return parser.parse_args()


def _mean_total_seconds(result_payload: dict, solver_name: str) -> float:
    rows = [
        r["timing"]["total_seconds"]
        for r in result_payload.get("results", [])
        if r.get("solver_name") == solver_name
    ]
    if not rows:
        return float("nan")
    return float(sum(rows) / len(rows))


def main() -> None:
    args = _parse_args()

    if args.use_custom_8ch:
        info, _ = create_8channel_montage()
        channel_tag = "custom8"
    else:
        info = get_info(kind=args.info_kind)
        channel_tag = args.info_kind
    fwd = create_forward_model(sampling=args.sampling, info=info)

    datasets = {name: BENCHMARK_DATASETS[name] for name in args.datasets}
    solvers = [
        "flex-champagne",
        "flex-champagne-omix",
        "flex-nl-champagne",
        "flex-nl-champagne-omix",
    ]
    if args.with_reference_solvers:
        solvers.extend(["esmv", "ir-esmv"])

    runner = BenchmarkRunner(
        fwd,
        info,
        solvers=solvers,
        datasets=datasets,
        n_samples=args.n_samples,
        random_seed=args.seed,
        n_jobs=-1,
    )
    runner.run()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_root = Path(__file__).resolve().parents[1]
    out_dir = package_root / "results" / "research" / "flex_champagne_omix"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"quick_eval_{channel_tag}_{args.sampling}_{ts}.json"
    runner.save(
        out_path,
        compact=True,
        name="Flex-Champagne Ongoing-Mixture Research Eval",
        description=(
            f"Quick inner-loop run (channels={channel_tag}, "
            f"sampling={args.sampling}, n_samples={args.n_samples}, seed={args.seed})."
        ),
    )

    payload = json.loads(out_path.read_text())
    ranks = payload.get("global_ranks", {})
    print(f"\nSaved: {out_path}")
    print("\nSolver                    Rank   MeanTime(s)")
    print("---------------------------------------------")
    for solver in solvers:
        rank = ranks.get(solver, float("nan"))
        mean_time = _mean_total_seconds(payload, solver)
        print(f"{solver:24s} {rank:6.2f} {mean_time:12.3f}")


if __name__ == "__main__":
    main()
