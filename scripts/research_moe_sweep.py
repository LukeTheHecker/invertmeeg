"""Sweep mixture-of-experts (MoE) solvers across channel counts and SNR.

This is a lightweight research harness to answer:
  - Does the MoE improvement hold across different montages (channel counts)?
  - How sensitive is the gate to noise level shifts (SNR ranges)?

By default it runs a small subset of solvers (including ``hydra``) on the
official benchmark datasets, across several montage presets, and stores compact
result JSON files under ``invert-package/results/research/moe_sweep/``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from invert.benchmark import BenchmarkRunner
from invert.benchmark.datasets import BENCHMARK_DATASETS, DatasetConfig
from invert.forward import (
    create_8channel_montage,
    create_forward_model,
    create_muse_montage,
    get_info,
)
from invert.simulate import SimulationConfig, SimulationGenerator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research sweep for MoE reliability across channels/SNR."
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=8,
        help="Simulated samples per dataset (default: 8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for simulations and solver determinism.",
    )
    parser.add_argument(
        "--sampling",
        type=str,
        default="ico3",
        help="Forward model sampling (default: ico3).",
    )
    parser.add_argument(
        "--info-kinds",
        nargs="+",
        default=["custom8", "biosemi32", "biosemi64", "biosemi256"],
        help=(
            "Montage identifiers. Use any MNE standard montage name (e.g. "
            "biosemi32) plus special tags: custom8, muse."
        ),
    )
    parser.add_argument(
        "--snr-shifts-db",
        nargs="+",
        type=float,
        default=[0.0],
        help=(
            "Additive shift (dB) applied to every dataset snr_range. "
            "Example: --snr-shifts-db -3 0 3"
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=sorted(BENCHMARK_DATASETS.keys()),
        choices=sorted(BENCHMARK_DATASETS.keys()),
        help="Datasets to include (default: all official benchmark datasets).",
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=[
            "hydra",
            "flex-greedy-ml",
            "subspace-sbl-plus",
            "subspace-sbl",
            "flex-ssm",
            "adapt-flex-esmv",
        ],
        help="Solver ids/aliases to evaluate (default includes the top globals + hydra).",
    )
    parser.add_argument(
        "--probe-gate",
        action="store_true",
        help="Also probe Hydra's sensor-variance gate statistics (fast).",
    )
    parser.add_argument(
        "--hydra-threshold",
        type=float,
        default=1e5,
        help="Variance threshold used for gate probing (default matches Hydra).",
    )
    return parser.parse_args()


def _resolve_info(kind: str):
    if kind == "custom8":
        info, _ = create_8channel_montage()
        return info, "custom8"
    if kind == "muse":
        info, _ = create_muse_montage()
        return info, "muse"
    info = get_info(kind=kind)
    return info, kind


def _shift_snr(cfg: DatasetConfig, shift_db: float) -> DatasetConfig:
    if float(shift_db) == 0.0:
        return cfg
    lo, hi = cfg.snr_range
    return cfg.model_copy(update={"snr_range": (float(lo + shift_db), float(hi + shift_db))})


def _probe_gate(
    *,
    forward,
    datasets: dict[str, DatasetConfig],
    n_samples: int,
    seed: int,
    threshold: float,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for ds_name, ds_cfg in datasets.items():
        sim_kwargs = dict(
            batch_size=n_samples,
            n_sources=ds_cfg.n_sources,
            n_orders=ds_cfg.n_orders,
            snr_range=ds_cfg.snr_range,
            n_timepoints=ds_cfg.n_timepoints,
            source_spatial_model=ds_cfg.source_spatial_model,
            patch_rank=ds_cfg.patch_rank,
            simulation_mode=getattr(ds_cfg, "simulation_mode", "patches"),
            random_seed=seed,
            estimate_noise_cov=True,
            return_noise_cov=True,
        )
        for key in (
            "background_beta",
            "background_mixture_alpha",
            "amplitude_range",
            "amplitude_distribution",
            "correlation_mode",
            "correlation_modes",
            "correlation_mode_weights",
            "noise_temporal_beta",
            "noise_low_rank_dim",
            "source_extent",
        ):
            val = getattr(ds_cfg, key, None)
            if val is not None:
                sim_kwargs[key] = val
        sim_config = SimulationConfig(**sim_kwargs)
        gen = SimulationGenerator(forward, config=sim_config)
        x_batch, _y, _info = next(gen.generate())
        mean_vars = np.mean(np.var(x_batch, axis=2, ddof=1), axis=1)
        out[ds_name] = {
            "p10": float(np.quantile(mean_vars, 0.1)),
            "median": float(np.median(mean_vars)),
            "p90": float(np.quantile(mean_vars, 0.9)),
            "frac_ge_threshold": float(np.mean(mean_vars >= threshold)),
        }
    return out


def main() -> None:
    args = _parse_args()

    package_root = Path(__file__).resolve().parents[1]
    out_dir = package_root / "results" / "research" / "moe_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_datasets = {name: BENCHMARK_DATASETS[name] for name in args.datasets}

    print("\ninfo        snr_shift  hydra  best_non_hydra  saved")
    print("----------------------------------------------------")
    for info_kind in args.info_kinds:
        info, channel_tag = _resolve_info(info_kind)
        fwd = create_forward_model(sampling=args.sampling, info=info)

        for shift_db in args.snr_shifts_db:
            datasets = {k: _shift_snr(v, shift_db) for k, v in base_datasets.items()}

            runner = BenchmarkRunner(
                fwd,
                info,
                solvers=list(args.solvers),
                datasets=datasets,
                n_samples=int(args.n_samples),
                random_seed=int(args.seed),
                n_jobs=-1,
            )
            runner.run()

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            shift_tag = f"{shift_db:+.1f}db".replace("+", "p").replace("-", "m").replace(".", "d")
            out_path = out_dir / f"moe_{channel_tag}_{args.sampling}_snrshift_{shift_tag}_{ts}.json"
            runner.save(
                out_path,
                compact=True,
                name="MoE Sweep",
                description=(
                    f"MoE solver sweep (channels={channel_tag}, sampling={args.sampling}, "
                    f"n_samples={args.n_samples}, seed={args.seed}, snr_shift_db={shift_db})."
                ),
            )

            payload = json.loads(out_path.read_text())
            global_ranks = dict(payload.get("global_ranks", {}))
            hydra_rank = global_ranks.get("hydra", float("nan"))
            non_hydra = [(k, v) for k, v in global_ranks.items() if k != "hydra"]
            best_non_hydra = min(non_hydra, key=lambda kv: kv[1]) if non_hydra else ("-", float("nan"))

            print(
                f"{channel_tag:10s} {shift_db:8.1f} {hydra_rank:6.2f} "
                f"{best_non_hydra[0]:14s} {best_non_hydra[1]:6.2f}  {out_path.name}"
            )

            if args.probe_gate:
                gate = _probe_gate(
                    forward=fwd,
                    datasets=datasets,
                    n_samples=int(args.n_samples),
                    seed=int(args.seed),
                    threshold=float(args.hydra_threshold),
                )
                for ds_name, stats in gate.items():
                    frac = stats["frac_ge_threshold"]
                    print(
                        f"  gate {ds_name:14s} median={stats['median']:.3g} "
                        f"p10={stats['p10']:.3g} p90={stats['p90']:.3g} "
                        f"frac>={args.hydra_threshold:.0e}={frac:.2f}"
                    )


if __name__ == "__main__":
    main()
