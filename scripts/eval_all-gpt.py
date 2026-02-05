import os
from pathlib import Path


def _set_default_env() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(repo_root / ".mne_config"))
    os.environ.setdefault("MPLCONFIGDIR", str(repo_root / ".mplconfig"))
    os.environ.setdefault("MNE_DATA", "/Users/lukas/mne_data")
    os.environ.setdefault("SUBJECTS_DIR", "/Users/lukas/mne_data/MNE-fsaverage-data")


_set_default_env()

from invert.forward import create_forward_model, get_info  # noqa: E402
from invert.benchmark import BenchmarkRunner  # noqa: E402
from invert.benchmark.datasets import BENCHMARK_DATASETS  # noqa: E402

if __name__ == '__main__':
    info = get_info(kind="biosemi32")
    fwd = create_forward_model(sampling="ico2", info=info)

    runner = BenchmarkRunner(
        fwd,
        info,
        n_samples=10,
        datasets={k: BENCHMARK_DATASETS[k] for k in ["multi_dipole", "multi_patch"]},
        solvers=[
            "SubspaceSBLPlus",
            "Chimera",
            "NLChampagne",
            "CovCNN",
            "CovCNN-KL",
            "CovCNN-KL-FLEXOMP",
            "CovCNN-KL-Diff",
            "CovCNN-KL-Adapt",
        ],
        solver_params={
            # Faster ANN iteration for gpt runs; use scripts/bench.py tier C for full leaderboard.
            "CovCNN": {"epochs": 120, "patience": 40},
            "CovCNN-KL": {"epochs": 120, "patience": 40},
            "CovCNN-KL-FLEXOMP": {"epochs": 120, "patience": 40},
            "CovCNN-KL-Diff": {"epochs": 120, "patience": 40},
            "CovCNN-KL-Adapt": {"epochs": 120, "patience": 40},
        },
        n_jobs=-1,
        random_seed=0,
    )
    runner.run()
    out_path = Path("results/benchmark_results-gpt.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    runner.save(out_path, compact=False)
