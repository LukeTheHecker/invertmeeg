"""Leaderboard check with SNR-compensated datasets.

The old noise pipeline applied SNR_linear to amplitude (not power), effectively
doubling the dB value.  To match old effective noise levels with the new
power-domain formula we double every SNR boundary.
"""

from pathlib import Path

from invert.benchmark import BenchmarkRunner
from invert.benchmark.datasets import DatasetConfig
from invert.forward import create_forward_model, get_info

# Double the dB values to compensate for the formula change:
#   old effective SNR_dB ≈ 2 * nominal SNR_dB
COMPENSATED_DATASETS = {
    "focal_source": DatasetConfig(
        name="Focal Source",
        description="Single focal dipole source with moderate noise",
        n_sources=1,
        n_orders=0,
        snr_range=(6.0, 10.0),  # was (3, 5)
        n_timepoints=50,
    ),
    "multi_source": DatasetConfig(
        name="Multi Source",
        description="Multiple focal dipole sources with low-to-moderate noise",
        n_sources=(2, 4),
        n_orders=0,
        snr_range=(2.0, 6.0),  # was (1, 3)
        n_timepoints=50,
    ),
    "extended_source": DatasetConfig(
        name="Extended Source",
        description="Multiple extended patch sources with spatial smoothing",
        n_sources=(2, 4),
        n_orders=(1, 3),
        snr_range=(2.0, 6.0),  # was (1, 3)
        n_timepoints=50,
    ),
    "noisy": DatasetConfig(
        name="Noisy",
        description="Variable sources under challenging low-SNR conditions",
        n_sources=(1, 3),
        n_orders=(0, 2),
        snr_range=(-10.0, 0.0),  # was (-5, 0)
        n_timepoints=50,
    ),
}

if __name__ == "__main__":
    info = get_info(kind="biosemi32")
    fwd = create_forward_model(sampling="ico3", info=info)

    runner = BenchmarkRunner(
        fwd,
        info,
        n_samples=50,
        datasets=COMPENSATED_DATASETS,
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

    out_path = Path("results/release/leaderboard-check-snr.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    runner.save(
        out_path,
        compact=True,
        name="Leaderboard Check (SNR-compensated)",
        description="SNR ranges doubled to compensate for amplitude->power formula change.",
    )
    print(f"\nSaved to {out_path}")
