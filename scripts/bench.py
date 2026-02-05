from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def _set_default_env() -> None:
    """Set offline-safe defaults for MNE/MPL config locations."""
    repo_root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(repo_root / ".mne_config"))
    os.environ.setdefault("MPLCONFIGDIR", str(repo_root / ".mplconfig"))

    # Prefer an existing MNE data directory; do not trigger downloads.
    candidates = [
        Path(os.environ.get("MNE_DATA", "")) if os.environ.get("MNE_DATA") else None,
        Path("/Users/lukas/mne_data"),
        Path.home() / "mne_data",
    ]
    for cand in candidates:
        if cand and cand.exists():
            os.environ.setdefault("MNE_DATA", str(cand))
            subjects_dir = cand / "MNE-fsaverage-data"
            if subjects_dir.exists():
                os.environ.setdefault("SUBJECTS_DIR", str(subjects_dir))
            break


_set_default_env()

from invert.benchmark import BenchmarkRunner  # noqa: E402
from invert.benchmark.datasets import BENCHMARK_DATASETS  # noqa: E402
from invert.forward import create_forward_model, get_info  # noqa: E402


def _csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    return items or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run benchmark tiers quickly (offline-safe)."
    )
    parser.add_argument("--tier", choices=["A", "B", "C"], default="A")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument(
        "--datasets", type=str, default=None, help="Comma-separated dataset names."
    )
    parser.add_argument(
        "--solvers", type=str, default=None, help="Comma-separated solver names."
    )
    parser.add_argument(
        "--categories", type=str, default=None, help="Comma-separated category names."
    )
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Save compact output (no per-sample metrics).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Save full output (includes per-sample metrics).",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--sampling", type=str, default="ico2")
    parser.add_argument("--info-kind", type=str, default="biosemi32")

    # Optional ANN training overrides (applied to ANN solvers in the run)
    parser.add_argument("--ann-epochs", type=int, default=None)
    parser.add_argument("--ann-patience", type=int, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tier_defaults = {
        "A": {
            "n_samples": 10,
            "datasets": ["multi_dipole", "multi_patch"],
            "solvers": [
                "SubspaceSBLPlus",
                "Chimera",
                "NLChampagne",
                "CovCNN",
                "CovCNN-KL",
                "CovCNN-KL-FLEXOMP",
                "CovCNN-KL-Diff",
                "CovCNN-KL-Adapt",
            ],
            "compact": False,
        },
        "B": {
            "n_samples": 25,
            "datasets": ["single_dipole", "multi_dipole", "multi_patch"],
            "solvers": [
                "SubspaceSBLPlus",
                "Chimera",
                "NLChampagne",
                "CovCNN",
                "CovCNN-KL",
                "CovCNN-KL-FLEXOMP",
                "CovCNN-KL-Diff",
                "CovCNN-KL-Adapt",
            ],
            "compact": False,
        },
        "C": {
            "n_samples": 50,
            "datasets": ["single_dipole", "multi_dipole", "multi_patch"],
            "categories": [
                "beamformer",
                "bayesian",
                "minimum_norm",
                "loreta",
                "music",
                "matching_pursuit",
                "other",
            ],
            "solvers": [
                "CovCNN",
                "CovCNN-KL",
                "CovCNN-KL-FLEXOMP",
                "CovCNN-KL-Diff",
                "CovCNN-KL-Adapt",
            ],
            "compact": True,
        },
    }

    defaults: dict[str, Any] = dict(tier_defaults[str(args.tier)])

    n_samples = (
        int(args.n_samples)
        if args.n_samples is not None
        else int(defaults["n_samples"])
    )
    datasets_arg = _csv_list(args.datasets)
    solvers_arg = _csv_list(args.solvers)
    categories_arg = _csv_list(args.categories)

    datasets: list[str] | None = datasets_arg or defaults.get("datasets")
    solvers: list[str] | None = solvers_arg or defaults.get("solvers")
    categories: list[str] | None = categories_arg or defaults.get("categories")

    if args.compact and args.full:
        raise SystemExit("Choose only one of --compact or --full.")
    if args.compact:
        compact = True
    elif args.full:
        compact = False
    else:
        compact = bool(defaults.get("compact", False))

    out = args.out
    if out is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = str(Path("results") / "dev" / f"{stamp}_tier{args.tier}.json")

    # Subset dataset configs
    ds_dict = dict(BENCHMARK_DATASETS)
    if datasets is not None:
        missing = sorted(set(datasets) - set(ds_dict))
        if missing:
            raise SystemExit(
                f"Unknown datasets: {missing}. Available: {sorted(ds_dict)}"
            )
        ds_dict = {k: ds_dict[k] for k in datasets}

    # Optional ANN training overrides
    solver_params: dict[str, dict[str, int]] = {}
    if args.ann_epochs is not None or args.ann_patience is not None:
        for name in solvers or []:
            if not isinstance(name, str):
                continue
            if name.startswith("CovCNN"):
                solver_params[name] = {}
                if args.ann_epochs is not None:
                    solver_params[name]["epochs"] = int(args.ann_epochs)
                if args.ann_patience is not None:
                    solver_params[name]["patience"] = int(args.ann_patience)

    info = get_info(kind=str(args.info_kind))
    fwd = create_forward_model(sampling=str(args.sampling), info=info)

    runner = BenchmarkRunner(
        fwd,
        info,
        solvers=solvers,
        categories=categories,
        datasets=ds_dict,
        n_samples=n_samples,
        n_jobs=int(args.n_jobs),
        random_seed=args.random_seed,
        solver_params=solver_params or None,
    )
    runner.run()

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    runner.save(out_path, compact=compact)


if __name__ == "__main__":
    main()
