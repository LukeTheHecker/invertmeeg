"""Compare legacy dSPM vs. MNE-style dSPM on the benchmark datasets.

Usage:
    python scripts/compare_dspm_mne.py
"""

from pathlib import Path

from invert.benchmark import BenchmarkRunner
from invert.forward import create_forward_model, get_info

SOLVERS = ["dSPM", "dSPM-MNE"]
N_SAMPLES = 50
RANDOM_SEED = 42


def main() -> Path:
    info = get_info(kind="biosemi32")
    fwd = create_forward_model(sampling="ico2", info=info)

    runner = BenchmarkRunner(
        fwd,
        info,
        solvers=SOLVERS,
        n_samples=N_SAMPLES,
        n_jobs=-1,
        random_seed=RANDOM_SEED,
    )
    runner.run()

    out_path = Path("results/compare_dspm_mne.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    runner.save(
        out_path,
        compact=False,
        name="dSPM vs dSPM-MNE",
        description=(
            f"Benchmark comparison for {SOLVERS} "
            f"({N_SAMPLES} samples, seed={RANDOM_SEED})."
        ),
    )
    print(f"\nSaved results to {out_path}")
    return out_path


if __name__ == "__main__":
    main()
