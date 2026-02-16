"""Sweep beamformer covariance-regularization strategies on a few solvers.

Runs the BenchmarkRunner on a small set of beamformers under different
covariance regularization strategies and saves per-strategy JSON results plus
an aggregated summary ranking.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations, permutations
from dataclasses import dataclass
from pathlib import Path

from invert.benchmark import BenchmarkRunner
from invert.forward import create_forward_model, get_info


LOWER_IS_BETTER = {"mean_localization_error", "emd", "spatial_dispersion"}
HIGHER_IS_BETTER = {"average_precision", "correlation"}


@dataclass(frozen=True)
class RegConfig:
    name: str
    approach_set: tuple[str, ...]
    solver_params: dict


def _aggregate_overall(results: list) -> dict[str, float]:
    """Average dataset means over all (solver,dataset) results."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for r in results:
        for metric, stats in r.metrics.items():
            totals[metric] = totals.get(metric, 0.0) + float(stats.mean)
            counts[metric] = counts.get(metric, 0) + 1
    return {k: totals[k] / float(counts[k]) for k in totals}


def _rank_configs(overall: dict[str, dict[str, float]]) -> list[tuple[str, float]]:
    """Return [(config_name, mean_rank)] (lower is better)."""
    config_names = list(overall)
    metrics = sorted(set().union(*(overall[c].keys() for c in config_names)))
    ranks_per_config = {c: [] for c in config_names}
    for metric in metrics:
        values = []
        for c in config_names:
            if metric in overall[c]:
                values.append((c, float(overall[c][metric])))
        if not values:
            continue
        reverse = metric in HIGHER_IS_BETTER
        values.sort(key=lambda x: x[1], reverse=reverse)
        for rank, (c, _v) in enumerate(values, start=1):
            ranks_per_config[c].append(float(rank))
    mean_rank = {
        c: (sum(ranks) / float(len(ranks))) if ranks else float("inf")
        for c, ranks in ranks_per_config.items()
    }
    return sorted(mean_rank.items(), key=lambda x: x[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--diag-beta", type=float, default=0.05)
    parser.add_argument("--cond-target", type=float, default=1e4)
    parser.add_argument(
        "--pair-mode",
        choices=("combinations", "permutations"),
        default="permutations",
        help="How to build pairwise approach sets.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/beamformer_reg_sweep"),
        help="Directory to write JSON results.",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    solvers = [
        "MVAB",
        "LCMV",
        "LCMV-MVPURE",
        "SMV",
        "WNMV",
        "HOCMV",
        "MCMV",
        "UNIG",
        "HOCMCMV",
        "SAM",
        "ESMV",
        "ESMV-MVPURE",
        "ESMV2",
        "ESMV3",
        "AdaptFlexESMV",
    ]

    approaches = ["oas", "diag", "rank", "cond"]
    single_sets = [[a] for a in approaches]
    if args.pair_mode == "permutations":
        pair_sets = [list(pair) for pair in permutations(approaches, 2)]
    else:
        pair_sets = [list(pair) for pair in combinations(approaches, 2)]
    approach_sets = single_sets + pair_sets

    def _build_solver_params(approach_set: list[str]) -> dict:
        use_rank = "rank" in approach_set
        cov_steps: list[str] = []
        seen_cov_steps: set[str] = set()
        for approach in approach_set:
            if approach == "oas" and "oas" not in seen_cov_steps:
                cov_steps.append("oas")
                seen_cov_steps.add("oas")
            elif approach == "diag" and "trace" not in seen_cov_steps:
                cov_steps.append("trace")
                seen_cov_steps.add("trace")
            elif approach == "cond" and "cond" not in seen_cov_steps:
                cov_steps.append("cond")
                seen_cov_steps.add("cond")
        if not cov_steps:
            cov_steps = ["grid"]

        reg = "+".join(cov_steps)
        params = {
            "init__reduce_rank": bool(use_rank),
            "init__rank": "auto",
            "cov_reg": reg,
            "cov_reg_beta": float(args.diag_beta),
            "cov_reg_cond_target": float(args.cond_target),
        }
        return params

    configs = []
    seen_param_keys: dict[str, str] = {}
    skipped_duplicate_configs: list[dict[str, str]] = []
    for approach_set in approach_sets:
        name = "+".join(approach_set)
        cfg_params = _build_solver_params(approach_set)
        param_key = json.dumps(cfg_params, sort_keys=True)
        if param_key in seen_param_keys:
            skipped_duplicate_configs.append(
                {"name": name, "duplicate_of": seen_param_keys[param_key]}
            )
            continue
        seen_param_keys[param_key] = name
        configs.append(
            RegConfig(name=name, approach_set=tuple(approach_set), solver_params=cfg_params)
        )

    info = get_info(kind="biosemi32")
    fwd = create_forward_model(sampling="ico2", info=info)

    overall_by_config: dict[str, dict[str, float]] = {}
    per_config_paths: dict[str, str] = {}
    failed_configs: dict[str, str] = {}

    for cfg in configs:
        solver_params = {s: dict(cfg.solver_params) for s in solvers}
        runner = BenchmarkRunner(
            fwd,
            info,
            solvers=list(solvers),
            n_samples=int(args.n_samples),
            n_jobs=int(args.n_jobs),
            random_seed=int(args.seed),
            solver_params=solver_params,
        )
        try:
            results = runner.run()
        except Exception as exc:
            failed_configs[cfg.name] = str(exc)
            continue

        safe_name = cfg.name.replace("+", "_plus_")
        out_path = out_dir / f"beamformer_reg_{safe_name}.json"
        runner.save(out_path, compact=True, name="Beamformer reg sweep", description=cfg.name)
        per_config_paths[cfg.name] = str(out_path)
        overall_by_config[cfg.name] = _aggregate_overall(results)

    ranking = _rank_configs(overall_by_config)
    summary = {
        "solvers": solvers,
        "approaches": approaches,
        "pair_mode": args.pair_mode,
        "n_samples": int(args.n_samples),
        "seed": int(args.seed),
        "diag_beta": float(args.diag_beta),
        "cond_target": float(args.cond_target),
        "configs": [
            {
                "name": c.name,
                "approach_set": list(c.approach_set),
                "uses_rank_reduction": bool(c.solver_params["init__reduce_rank"]),
                "cov_reg_pipeline": str(c.solver_params["cov_reg"]),
                "solver_params": c.solver_params,
            }
            for c in configs
        ],
        "evaluated_configs": sorted(overall_by_config),
        "skipped_duplicate_configs": skipped_duplicate_configs,
        "overall_metrics": overall_by_config,
        "ranking_mean_rank": ranking,
        "result_files": per_config_paths,
        "failed_configs": failed_configs,
        "metric_conventions": {
            "lower_is_better": sorted(LOWER_IS_BETTER),
            "higher_is_better": sorted(HIGHER_IS_BETTER),
        },
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
