"""Run private benchmark profiles for marketing demos.

These benchmark outputs are meant for internal use (sales decks, PDF summaries)
and therefore default to writing into /tmp, not into the repository.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("MNE_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "mne-config"))

from invert.benchmark import BenchmarkRunner, DatasetConfig
from invert.forward import create_8channel_montage, create_forward_model, get_info
from invert.forward.forward import _get_fsaverage_dir

EVAL_RELEASE_CATEGORIES = [
    "beamformer",
    "bayesian",
    "minimum_norm",
    "loreta",
    "music",
    "matching_pursuit",
    "other",
]

EVAL_RELEASE_EXCLUDE_SOLVERS = [
    "SESAME",
    "EBB",
    "ReciPSIICOS-Plain",
    "ReciPSIICOS-Whitened",
    "champagne-ar-em",
    "champagne-tem",
]


@dataclass(frozen=True, slots=True)
class Profile:
    profile_id: str
    name: str
    description: str
    sampling: str
    info_kind: str | None = None
    use_8ch_montage: bool = False
    datasets: dict[str, DatasetConfig] | None = None
    categories: list[str] | None = None
    solvers: list[str] | None = None
    exclude_solvers: list[str] | None = None


def _profiles() -> dict[str, Profile]:
    return {
        "low-channel-neurofeedback": Profile(
            profile_id="low-channel-neurofeedback",
            name="Low-channel neurofeedback (8ch) — private benchmark",
            description=(
                "Low-channel EEG benchmark (8 channels) intended for neurofeedback/BCI "
                "product scenarios. Uses eval_all_release-style registry/category solver "
                "selection with low-channel datasets."
            ),
            sampling="ico2",
            use_8ch_montage=True,
            datasets={
                "nf_focal": DatasetConfig(
                    name="Neurofeedback (Focal)",
                    description="Single focal source, moderate-to-high SNR",
                    n_sources=1,
                    n_orders=0,
                    snr_range=(3.0, 10.0),
                    n_timepoints=50,
                ),
                "nf_multi": DatasetConfig(
                    name="Neurofeedback (Multi)",
                    description="2–3 focal sources, moderate SNR",
                    n_sources=(2, 3),
                    n_orders=0,
                    snr_range=(1.0, 6.0),
                    n_timepoints=50,
                ),
                "nf_noisy": DatasetConfig(
                    name="Neurofeedback (Noisy)",
                    description="Challenging low-SNR case",
                    n_sources=(1, 3),
                    n_orders=(0, 1),
                    snr_range=(-3.0, 1.0),
                    n_timepoints=50,
                ),
            },
            categories=EVAL_RELEASE_CATEGORIES,
            exclude_solvers=EVAL_RELEASE_EXCLUDE_SOLVERS,
        ),
        "epilepsy-software": Profile(
            profile_id="epilepsy-software",
            name="Epilepsy-oriented software (64ch) — private benchmark",
            description=(
                "Benchmark tailored to teams building epilepsy/clinical-research software: "
                "higher channel count, focal and noisy scenarios, and eval_all_release-style "
                "registry/category solver selection."
            ),
            sampling="ico2",
            info_kind="biosemi64",
            datasets={
                "epi_focal": DatasetConfig(
                    name="Epilepsy (Focal)",
                    description="Single focal source, realistic SNR",
                    n_sources=1,
                    n_orders=0,
                    snr_range=(0.0, 4.0),
                    n_timepoints=50,
                ),
                "epi_multi": DatasetConfig(
                    name="Epilepsy (Multi)",
                    description="2–4 focal sources, lower SNR",
                    n_sources=(2, 4),
                    n_orders=0,
                    snr_range=(-2.0, 2.0),
                    n_timepoints=50,
                ),
                "epi_extended": DatasetConfig(
                    name="Epilepsy (Extended)",
                    description="Extended/patch sources",
                    n_sources=(2, 4),
                    n_orders=(1, 3),
                    snr_range=(-2.0, 2.0),
                    source_spatial_model="contiguous_gaussian",
                    patch_rank=(1, 2),
                    n_timepoints=50,
                ),
                "epi_noisy": DatasetConfig(
                    name="Epilepsy (Very Noisy)",
                    description="Hard stress-test scenario",
                    n_sources=(1, 3),
                    n_orders=(0, 2),
                    snr_range=(-6.0, -1.0),
                    n_timepoints=50,
                ),
            },
            categories=EVAL_RELEASE_CATEGORIES,
            exclude_solvers=EVAL_RELEASE_EXCLUDE_SOLVERS,
        ),
        "platform-leaderboard": Profile(
            profile_id="platform-leaderboard",
            name="Method selection platform (32ch) — private leaderboard run",
            description=(
                "Broad run across solver families for showcasing method-selection and "
                "benchmarking capabilities. Mirrors the public leaderboard setup but "
                "writes results outside the repo."
            ),
            sampling="ico2",
            info_kind="biosemi32",
            categories=EVAL_RELEASE_CATEGORIES,
            exclude_solvers=EVAL_RELEASE_EXCLUDE_SOLVERS,
        ),
    }


def _build_info(profile: Profile):
    if profile.use_8ch_montage:
        info, _ = create_8channel_montage(sfreq=256)
        return info
    if profile.info_kind is None:
        msg = f"Profile {profile.profile_id!r} must define info_kind or use_8ch_montage"
        raise ValueError(msg)
    return get_info(kind=profile.info_kind)


def main() -> None:
    profiles = _profiles()

    parser = argparse.ArgumentParser(
        description="Run invertmeeg marketing benchmark profiles (private outputs)."
    )
    parser.add_argument(
        "--profile",
        required=True,
        choices=sorted(profiles.keys()),
        help="Which marketing benchmark profile to run.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.environ.get("INVERT_MARKETING_BENCH_DIR", "/tmp/invertmeeg-marketing"),
        help=(
            "Output directory for JSON artifacts. Defaults to "
            "/tmp/invertmeeg-marketing (override via INVERT_MARKETING_BENCH_DIR)."
        ),
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of samples per dataset (default: 50).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel workers (-1 = all cores, default: -1).",
    )
    args = parser.parse_args()

    profile = profiles[args.profile]
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Benchmark metrics call mne.vertex_to_mni(), which requires SUBJECTS_DIR.
    fs_dir = Path(_get_fsaverage_dir())
    os.environ.setdefault("SUBJECTS_DIR", str(fs_dir.parent))

    info = _build_info(profile)
    fwd = create_forward_model(sampling=profile.sampling, info=info)

    runner = BenchmarkRunner(
        fwd,
        info,
        solvers=profile.solvers,
        categories=profile.categories,
        exclude_solvers=profile.exclude_solvers,
        datasets=profile.datasets,
        n_samples=args.n_samples,
        random_seed=args.random_seed,
        n_jobs=args.n_jobs,
    )
    runner.run()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"{profile.profile_id}__{timestamp}.json"
    runner.save(
        out_path,
        compact=True,
        name=profile.name,
        description=profile.description,
    )
    print(out_path)


if __name__ == "__main__":
    main()
