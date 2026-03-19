"""Run the official invertmeeg leaderboard evaluation.

Uses biosemi32/ico2, 50 samples per dataset, all solver categories plus CovCNN
variants.  Saves compact results to results/release/leaderboard.json.
"""

from pathlib import Path

import invert.benchmark.runner as runner_mod
from invert.benchmark import (
    RECOMMENDED_DEPTH_BY_CATEGORY,
    RECOMMENDED_DEPTH_BY_SOLVER,
    BenchmarkRunner,
)
from invert.forward import create_forward_model, get_info

# ANN training batch size (default would be 12*n_dipoles which is very large)
runner_mod._default_nn_batch_size = lambda fwd: 2048

if __name__ == "__main__":
    info = get_info(kind="biosemi32")
    fwd = create_forward_model(sampling="ico2", info=info)

    runner = BenchmarkRunner(
        fwd,
        info,
        n_samples=20,  # should be 100 for the full leaderboard
        categories=[
            "bayesian",
            "beamformer",
            "dipoles",
            "hybrids",
            "matching_pursuit",
            "minimum_norm",
            "loreta",
            "music",
            "other",
        ],
        # solvers=[
        #     "CovCNN",
        #     "CovCNN-KL",
        #     "FC",
        # ],
        exclude_solvers=[
            "ReciPSIICOS-Whitened",
            "champagne-ar-em",
            "champagne-tem",
        ],
        # solver_params={
        #     "CovCNN": {"epochs": 200, "patience": 60, "n_dense_units": 24},
        #     "CovCNN-KL": {"epochs": 200, "patience": 60, "n_dense_units": 24},
        #     "FC": {"epochs": 200, "patience": 60, "n_dense_units": 600, "train_n_timepoints": 20},
        # },
        depth_by_category=RECOMMENDED_DEPTH_BY_CATEGORY,
        depth_by_solver=RECOMMENDED_DEPTH_BY_SOLVER,
        random_seed=42,
        n_jobs=-1,
    )
    runner.run()

    out_path = Path("results/release/leaderboard-new.json")
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
