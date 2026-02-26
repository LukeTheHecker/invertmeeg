"""Run the official invertmeeg leaderboard evaluation.

Uses biosemi32/ico2, 50 samples per dataset, all solver categories plus CovCNN
variants.  Saves compact results to results/release/leaderboard.json.
"""

from pathlib import Path

from invert.benchmark import (
    BenchmarkRunner,
    RECOMMENDED_DEPTH_BY_CATEGORY,
    RECOMMENDED_DEPTH_BY_SOLVER,
)
from invert.forward import create_forward_model, get_info

from invert.benchmark.datasets import DatasetConfig  # noqa: E402

BENCHMARK_DATASETS: dict[str, DatasetConfig] = {
    "extended_contiguous_gaussian": DatasetConfig(
        name="Extended Contiguous Gaussian",
        description="Multiple extended contiguous Gaussian patch sources with spatial smoothing",
        n_sources=(2, 4),
        n_orders=(1, 3),
        snr_range=(1.0, 3.0),
        source_spatial_model="contiguous_gaussian",
        patch_rank=(1, 3),
        n_timepoints=50,
        random_seed=42,
    ),
}

if __name__ == "__main__":
    info = get_info(kind="biosemi32")
    fwd = create_forward_model(sampling="ico2", info=info)

    runner = BenchmarkRunner(
        fwd,
        info,
        datasets=BENCHMARK_DATASETS,
        n_samples=50,  # should be 50 for the full leaderboard
        categories=[
            "beamformer",
            "bayesian",
            "minimum_norm",
            "loreta",
            "music",
            "matching_pursuit",
            "other",
        ],
        exclude_solvers = [
            "SESAME",
            "EBB",
            "ReciPSIICOS-Plain",
            "ReciPSIICOS-Whitened",
            "champagne-ar-em",
            "champagne-tem",
        ],
        # solvers=[
        #     "CovCNN",
        #     "CovCNN-KL",
        #     "CovCNN-KL-FLEXOMP",
        # ],
        solver_params={
            # Script-level ANN speedup: keep global solver defaults unchanged.
            "CovCNN": {"epochs": 120, "patience": 40},
            "CovCNN-KL": {"epochs": 250, "patience": 80},
            "CovCNN-KL-FLEXOMP": {"epochs": 250, "patience": 80},
        },
        depth_by_category=RECOMMENDED_DEPTH_BY_CATEGORY,
        depth_by_solver=RECOMMENDED_DEPTH_BY_SOLVER,
        random_seed=42,
        n_jobs=-1,
    )
    runner.run()

    out_path = Path("results/release/leaderboard-contiguous.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    runner.save(
        out_path,
        compact=True,
        name="invertmeeg Leaderboard",
        description=(
            "Official leaderboard run using biosemi32/ico2 with 50 samples "
            "per dataset across all solver categories."
        ),
    )
