"""Quick leaderboard check for solvers modified today.

Runs the same setup as eval_all_release.py (biosemi32/ico3, 50 samples)
but only for the subset of solvers we want to validate.
"""

from pathlib import Path

from invert.benchmark import BenchmarkRunner
from invert.forward import create_forward_model, get_info

if __name__ == "__main__":
    info = get_info(kind="biosemi32")
    fwd = create_forward_model(sampling="ico3", info=info)

    runner = BenchmarkRunner(
        fwd,
        info,
        n_samples=50,
        solvers=[
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
        ],
        n_jobs=-1,
    )
    runner.run()

    out_path = Path("results/release/leaderboard-check.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    runner.save(
        out_path,
        compact=True,
        name="Leaderboard Check",
        description="Targeted check for solvers modified on 2026-02-11.",
    )
    print(f"\nSaved to {out_path}")
