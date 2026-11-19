"""Run or resume the BioSemi32/ico3 development leaderboard."""

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from invert.benchmark import BENCHMARK_DATASETS, BenchmarkResult, BenchmarkRunner
from invert.benchmark.runner import _evaluation_protocol_metadata
from invert.forward import create_forward_model, get_info
from invert.solver_registry import list_solver_ids

CHECKPOINTS = Path("results/release/protocol-v2-cases")
OUTPUT = Path("results/release/leaderboard-protocol-v2.json.gz")
CATEGORIES = [
    "bayesian",
    "beamformer",
    "dipoles",
    "hybrids",
    "matching_pursuit",
    "minimum_norm",
    "loreta",
    "music",
    "other",
]
INELIGIBLE = {
    name: "Evaluated separately on grouped recordings with an explicit transport cost."
    for name in ("mwe", "remwe")
}


def build_runner():
    info = get_info(kind="biosemi32")
    fwd = create_forward_model(sampling="ico3", info=info)
    return BenchmarkRunner(
        fwd,
        info,
        n_samples=50,
        random_seed=42,
        categories=CATEGORIES,
        exclude_solvers=[
            "ReciPSIICOS-Whitened",
            "champagne-ar-em",
            "champagne-tem",
            *(
                name
                for name in INELIGIBLE
                if name in list_solver_ids(include_internal=True)
            ),
        ],
        n_jobs=int(os.environ.get("INVERT_BENCHMARK_JOBS", "2")),
    )


def manifest(runner):
    source = hashlib.sha256()
    root = Path(__file__).resolve().parents[1] / "invert"
    for path in sorted(root.rglob("*.py")):
        source.update(path.relative_to(root).as_posix().encode() + b"\0")
        source.update(path.read_bytes())
    return {
        "protocol": _evaluation_protocol_metadata(),
        "seed": runner.random_seed,
        "n_samples": runner.n_samples,
        "datasets": {k: v.model_dump(mode="json") for k, v in runner.datasets.items()},
        "solvers": runner.solvers,
        "solver_params": runner.solver_params,
        "leadfield_sha256": hashlib.sha256(
            runner.forward["sol"]["data"].tobytes()
        ).hexdigest(),
        "source_sha256": source.hexdigest(),
    }


def read_measurement(dataset, solver, n_samples):
    path = CHECKPOINTS / f"{dataset}--{solver}.json"
    if not path.exists():
        return None
    result = BenchmarkResult.model_validate_json(path.read_bytes())
    if (result.dataset_name, result.solver_name, result.case_count) != (
        dataset,
        solver,
        n_samples,
    ):
        raise ValueError(f"Inconsistent measurement in {path}")
    if [s.case_id for s in result.samples] != list(range(n_samples)):
        raise ValueError(f"Missing or unordered case IDs in {path}")
    return result


def save_measurement(result):
    path = CHECKPOINTS / f"{result.dataset_name}--{result.solver_name}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(result.model_dump_json())
    temporary.replace(path)


def run_scenario(dataset):
    runner = build_runner()
    runner.datasets = {dataset: BENCHMARK_DATASETS[dataset]}
    runner.solvers = [
        solver
        for solver in runner.solvers
        if read_measurement(dataset, solver, runner.n_samples) is None
    ]
    if runner.solvers:
        runner.run(on_result=save_measurement)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume", action="store_true", help="Reuse this run's completed measurements."
    )
    parser.add_argument("--scenario-workers", type=int, default=1)
    args = parser.parse_args()
    runner = build_runner()
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    manifest_path = CHECKPOINTS / "manifest.json"
    configuration = manifest(runner)
    if args.resume:
        if json.loads(manifest_path.read_text()) != configuration:
            raise ValueError("Checkpoint configuration differs from the current run.")
    elif any(CHECKPOINTS.iterdir()):
        raise FileExistsError(
            f"Existing measurements in {CHECKPOINTS}. Use --resume to continue."
        )
    else:
        manifest_path.write_text(json.dumps(configuration, indent=2))

    if args.scenario_workers == 1:
        for dataset in runner.datasets:
            run_scenario(dataset)
    else:
        with ProcessPoolExecutor(max_workers=args.scenario_workers) as pool:
            list(pool.map(run_scenario, runner.datasets))

    results = []
    for dataset in runner.datasets:
        for solver in runner.solvers:
            result = read_measurement(dataset, solver, runner.n_samples)
            if result is None:
                raise RuntimeError(f"Missing result for {dataset}/{solver}")
            results.append(result)
    runner._results = results
    runner.save(
        OUTPUT,
        compact=True,
        name="invertmeeg development leaderboard",
        description="BioSemi32/ico3, 50 independent recordings per scenario.",
        ineligible_solvers=INELIGIBLE,
        provenance=configuration,
    )


if __name__ == "__main__":
    main()
