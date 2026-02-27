"""Run the official invertmeeg leaderboard evaluation.

Uses biosemi32/ico2, 50 samples per dataset, all solver categories plus CovCNN
variants.  Saves compact results to results/release/leaderboard.json.
"""

from pathlib import Path

from invert.benchmark import (
    RECOMMENDED_DEPTH_BY_CATEGORY,
    RECOMMENDED_DEPTH_BY_SOLVER,
    BenchmarkRunner,
)
from invert.forward import create_forward_model, get_info

if __name__ == "__main__":
    info = get_info(kind="biosemi32")
    fwd = create_forward_model(sampling="ico3", info=info)

    runner = BenchmarkRunner(
        fwd,
        info,
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
        exclude_solvers=[
            "SESAME",
            "EBB",
            "ReciPSIICOS-Plain",
            "ReciPSIICOS-Whitened",
            "champagne-ar-em",
            "champagne-tem",
            "exhaustive-ml",
            "cmem",  # scipy.optimize.minimize L-BFGS-B crashes with ThreadPoolExecutor
        ],
        # solvers=[
        #     "CovCNN",
        #     "CovCNN-KL",
        #     "CovCNN-KL-FLEXOMP",
        # ],
        # solver_params={
        #     # Script-level ANN speedup: keep global solver defaults unchanged.
        #     "CovCNN": {"epochs": 120, "patience": 40},
        #     "CovCNN-KL": {"epochs": 250, "patience": 80},
        #     "CovCNN-KL-FLEXOMP": {"epochs": 250, "patience": 80},
        # },
        depth_by_category=RECOMMENDED_DEPTH_BY_CATEGORY,
        depth_by_solver=RECOMMENDED_DEPTH_BY_SOLVER,
        random_seed=42,
        n_jobs=-1,
    )
    runner.run()

    out_path = Path("results/release/leaderboard-test.json")
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
