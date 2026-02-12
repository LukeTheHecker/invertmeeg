#!/usr/bin/env python
"""ANN testbench for CovCNN-KL hyperparameter exploration.

Trains CovCNN-KL with varying hyperparameters and evaluates on a fixed
held-out test set so results are directly comparable.

Usage:
    cd invert-package
    .venv/bin/python scripts/ann_testbench.py              # full sweep (~24 experiments)
    .venv/bin/python scripts/ann_testbench.py --quick       # reduced sweep (~8 key experiments)
    .venv/bin/python scripts/ann_testbench.py --only baseline units=100 batch_size=1024
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

# Import evaluate_all directly to avoid pulling in seaborn via __init__
_spec = _ilu.spec_from_file_location(
    "invert.evaluate.evaluate",
    str(Path(__file__).resolve().parents[1] / "invert" / "evaluate" / "evaluate.py"),
)
assert _spec is not None and _spec.loader is not None
_eval_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_eval_mod)
evaluate_all = _eval_mod.evaluate_all
from invert.forward import create_forward_model, get_info
from invert.simulate import SimulationConfig, SimulationGenerator
from invert.simulate.spatial import build_adjacency
from invert.solvers.neural_networks.covcnn_kl import SolverCovCNNKL
from invert.util.util import pos_from_forward

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ann_testbench")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """One training + eval run."""

    label: str
    # Architecture
    n_dense_units: int = 500
    n_dense_layers: int = 2
    activation_function: str = "tanh"
    # Training
    epochs: int = 500
    learning_rate: float = 1e-3
    patience: int = 300
    temperature: float = 1.0
    # Data
    batch_size: int | None = None  # None = use default (max(2048, 6*n_dipoles))
    batch_repetitions: int = 1
    # Target processing
    target_power: float = 0.5
    gamma_power: float = 1.5


def get_experiments(
    n_dipoles: int, quick: bool = False, round2: bool = False, round3: bool = False
) -> list[ExperimentConfig]:
    """Define the experiments to run.  Edit this list to explore dimensions."""
    experiments = []

    # --- Baseline ---
    experiments.append(ExperimentConfig(label="baseline"))

    if quick:
        # Focused subset: one from each dimension
        experiments.append(ExperimentConfig(label="units=100", n_dense_units=100))
        experiments.append(ExperimentConfig(label="units=500", n_dense_units=500))
        experiments.append(ExperimentConfig(label="layers=1", n_dense_layers=1))
        experiments.append(ExperimentConfig(label="batch_size=1024", batch_size=1024))
        experiments.append(
            ExperimentConfig(label="batch_reps=10", batch_repetitions=10)
        )
        experiments.append(
            ExperimentConfig(label="epochs=500", epochs=500, patience=500)
        )
        experiments.append(ExperimentConfig(label="lr=3e-4", learning_rate=3e-4))
        return experiments

    if round2:
        # Round 2: combine the best strategies from round 1
        # Winners: batch_size=1024, epochs=500; losers: batch_reps>1, units<300
        experiments.append(
            ExperimentConfig(
                label="bs1024+ep500",
                batch_size=1024,
                epochs=500,
                patience=500,
            )
        )
        experiments.append(
            ExperimentConfig(
                label="bs1024+ep500+pat150",
                batch_size=1024,
                epochs=500,
                patience=150,
            )
        )
        experiments.append(
            ExperimentConfig(
                label="bs2048+ep500",
                batch_size=2048,
                epochs=500,
                patience=500,
            )
        )
        experiments.append(
            ExperimentConfig(
                label="bs2048+ep300",
                batch_size=2048,
                epochs=300,
                patience=150,
            )
        )
        experiments.append(
            ExperimentConfig(
                label="bs1024+ep1000",
                batch_size=1024,
                epochs=1000,
                patience=300,
            )
        )
        experiments.append(
            ExperimentConfig(
                label="bs2048+ep500+u500",
                batch_size=2048,
                epochs=500,
                patience=300,
                n_dense_units=500,
            )
        )
        experiments.append(
            ExperimentConfig(
                label="bs4096+ep300",
                batch_size=4096,
                epochs=300,
                patience=150,
            )
        )
        return experiments

    if round3:
        # Round 3: Push data limits hard. R2 winner was bs2048+ep500+u500.
        # Now test: even larger batches, more epochs, and a few arch tweaks.

        # -- Scaling batch size with u500 (the R2 winner arch) --
        experiments.append(
            ExperimentConfig(
                label="bs2048+ep500+u500",
                batch_size=2048,
                epochs=500,
                patience=300,
                n_dense_units=500,
            )
        )
        experiments.append(
            ExperimentConfig(
                label="bs4096+ep500+u500",
                batch_size=4096,
                epochs=500,
                patience=300,
                n_dense_units=500,
            )
        )
        experiments.append(
            ExperimentConfig(
                label="bs8192+ep500+u500",
                batch_size=8192,
                epochs=500,
                patience=300,
                n_dense_units=500,
            )
        )
        experiments.append(
            ExperimentConfig(
                label="bs16384+ep500+u500",
                batch_size=16384,
                epochs=500,
                patience=300,
                n_dense_units=500,
            )
        )

        # -- More epochs with large data --
        experiments.append(
            ExperimentConfig(
                label="bs4096+ep1000+u500",
                batch_size=4096,
                epochs=1000,
                patience=400,
                n_dense_units=500,
            )
        )
        experiments.append(
            ExperimentConfig(
                label="bs8192+ep1000+u500",
                batch_size=8192,
                epochs=1000,
                patience=400,
                n_dense_units=500,
            )
        )

        # -- Wider network with massive data --
        experiments.append(
            ExperimentConfig(
                label="bs8192+ep500+u800",
                batch_size=8192,
                epochs=500,
                patience=300,
                n_dense_units=800,
            )
        )
        experiments.append(
            ExperimentConfig(
                label="bs8192+ep500+u1000",
                batch_size=8192,
                epochs=500,
                patience=300,
                n_dense_units=1000,
            )
        )

        # -- Deeper network --
        experiments.append(
            ExperimentConfig(
                label="bs4096+ep500+u500+L3",
                batch_size=4096,
                epochs=500,
                patience=300,
                n_dense_units=500,
                n_dense_layers=3,
            )
        )

        # -- ReLU (one test — bounded tanh may be better for KL targets) --
        experiments.append(
            ExperimentConfig(
                label="bs4096+ep500+u500+relu",
                batch_size=4096,
                epochs=500,
                patience=300,
                n_dense_units=500,
                activation_function="relu",
            )
        )

        # -- Learning rate with large batch --
        experiments.append(
            ExperimentConfig(
                label="bs8192+ep500+u500+lr3e-3",
                batch_size=8192,
                epochs=500,
                patience=300,
                n_dense_units=500,
                learning_rate=3e-3,
            )
        )

        return experiments

    # --- Vary network width ---
    for units in [100, 200, 500, 800]:
        experiments.append(
            ExperimentConfig(
                label=f"units={units}",
                n_dense_units=units,
            )
        )

    # --- Vary network depth ---
    for layers in [1, 3, 4]:
        experiments.append(
            ExperimentConfig(
                label=f"layers={layers}",
                n_dense_layers=layers,
            )
        )

    # --- Vary batch size (training data volume per epoch) ---
    for bs in [128, 512, 1024, 2048]:
        experiments.append(
            ExperimentConfig(
                label=f"batch_size={bs}",
                batch_size=bs,
            )
        )

    # --- Vary batch repetitions (reuse each batch N times before regenerating) ---
    for reps in [5, 10, 30]:
        experiments.append(
            ExperimentConfig(
                label=f"batch_reps={reps}",
                batch_repetitions=reps,
            )
        )

    # --- Vary epochs ---
    for ep in [100, 500, 1000]:
        experiments.append(
            ExperimentConfig(
                label=f"epochs={ep}",
                epochs=ep,
                patience=ep,  # don't early-stop for epoch sweeps
            )
        )

    # --- Vary learning rate ---
    for lr in [3e-4, 3e-3, 1e-2]:
        experiments.append(
            ExperimentConfig(
                label=f"lr={lr}",
                learning_rate=lr,
            )
        )

    # --- Vary patience ---
    for pat in [30, 150, 300]:
        experiments.append(
            ExperimentConfig(
                label=f"patience={pat}",
                patience=pat,
            )
        )

    return experiments


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def generate_test_set(
    fwd, sim_config: SimulationConfig, n_samples: int = 20, seed: int = 99
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a fixed test set of (x, y) pairs."""
    rng_cfg = sim_config.model_copy(
        update={"random_seed": seed, "batch_size": n_samples}
    )
    sim_gen = SimulationGenerator(fwd, config=rng_cfg)
    gen = sim_gen.generate()
    x_batch, y_batch, _info = next(gen)
    return x_batch, y_batch


def evaluate_solver(
    solver: SolverCovCNNKL,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    info: mne.Info,
    adjacency,
    pos: np.ndarray,
) -> dict[str, float]:
    """Evaluate a trained solver on the test set and return mean metrics."""
    all_metrics: dict[str, list[float]] = {}
    for i in range(x_batch.shape[0]):
        evoked = mne.EvokedArray(x_batch[i], info, tmin=0.0, verbose=0)
        stc = solver.apply_inverse_operator(evoked)
        y_pred = stc.data
        metrics = evaluate_all(y_batch[i], y_pred, adjacency, adjacency, pos, pos)
        for k, v in metrics.items():
            all_metrics.setdefault(k, []).append(float(v) if np.isfinite(v) else np.nan)

    return {k: float(np.nanmean(v)) for k, v in all_metrics.items()}


def train_and_evaluate(
    exp: ExperimentConfig,
    fwd,
    info: mne.Info,
    sim_config: SimulationConfig,
    x_test: np.ndarray,
    y_test: np.ndarray,
    adjacency,
    pos: np.ndarray,
    n_dipoles: int,
) -> dict:
    """Train CovCNN-KL with given config, evaluate, return results dict."""
    bs = exp.batch_size if exp.batch_size is not None else 2 * n_dipoles
    train_sim_config = sim_config.model_copy(
        update={
            "batch_size": bs,
            "batch_repetitions": exp.batch_repetitions,
        }
    )

    solver = SolverCovCNNKL()

    logger.info("=" * 60)
    logger.info("EXPERIMENT: %s", exp.label)
    logger.info(
        "  units=%d, layers=%d, epochs=%d, lr=%g, patience=%d",
        exp.n_dense_units,
        exp.n_dense_layers,
        exp.epochs,
        exp.learning_rate,
        exp.patience,
    )
    logger.info(
        "  batch_size=%d, batch_reps=%d, temp=%g",
        bs,
        exp.batch_repetitions,
        exp.temperature,
    )
    logger.info("=" * 60)

    t0 = time.perf_counter()
    solver.make_inverse_operator(
        fwd,
        train_sim_config,
        n_dense_units=exp.n_dense_units,
        n_dense_layers=exp.n_dense_layers,
        activation_function=exp.activation_function,
        epochs=exp.epochs,
        learning_rate=exp.learning_rate,
        patience=exp.patience,
        temperature=exp.temperature,
        target_power=exp.target_power,
        gamma_power=exp.gamma_power,
        alpha="auto",
    )
    train_time = time.perf_counter() - t0

    # Count parameters
    from invert.solvers.neural_networks.torch_utils import count_trainable_parameters

    n_params = count_trainable_parameters(solver.model)

    # Evaluate
    t1 = time.perf_counter()
    metrics = evaluate_solver(solver, x_test, y_test, info, adjacency, pos)
    eval_time = time.perf_counter() - t1

    result = {
        "label": exp.label,
        "n_params": n_params,
        "train_time_s": round(train_time, 1),
        "eval_time_s": round(eval_time, 1),
        **{k: round(v, 4) for k, v in metrics.items()},
    }
    logger.info("RESULT [%s]: %s", exp.label, json.dumps(result, indent=2))
    return result


def print_summary(results: list[dict]) -> None:
    """Print a formatted summary table."""
    print("\n" + "=" * 110)
    print("SUMMARY  (MLE/EMD/SD: lower=better; AvgPrec/Corr: higher=better)")
    print("=" * 110)
    header = (
        f"{'Label':<25} {'Params':>10} {'Train(s)':>9} "
        f"{'MLE':>8} {'EMD':>8} {'SD':>8} {'AvgPrec':>8} {'Corr':>8}"
    )
    print(header)
    print("-" * 110)
    for r in results:
        if "error" in r:
            print(f"{r['label']:<25} ERROR: {r['error']}")
            continue
        print(
            f"{r['label']:<25} "
            f"{r.get('n_params', '?'):>10} "
            f"{r.get('train_time_s', '?'):>9} "
            f"{r.get('Mean_Localization_Error', float('nan')):>8.4f} "
            f"{r.get('EMD', float('nan')):>8.4f} "
            f"{r.get('sd', float('nan')):>8.4f} "
            f"{r.get('average_precision', float('nan')):>8.4f} "
            f"{r.get('correlation', float('nan')):>8.4f}"
        )
    print("=" * 110)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="CovCNN-KL hyperparameter testbench")
    parser.add_argument(
        "--quick", action="store_true", help="Run reduced experiment set"
    )
    parser.add_argument(
        "--round2",
        action="store_true",
        help="Run round-2 experiments (combine winners)",
    )
    parser.add_argument(
        "--round3",
        action="store_true",
        help="Run round-3 experiments (push data limits)",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        help="Run only experiments whose labels match these prefixes",
    )
    parser.add_argument(
        "--n-test", type=int, default=20, help="Number of test samples (default: 20)"
    )
    args = parser.parse_args()

    mne.set_log_level("WARNING")

    # Forward model
    info = get_info(kind="biosemi32")
    fwd = create_forward_model(sampling="ico2", info=info)
    n_dipoles = fwd["sol"]["data"].shape[1]
    n_channels = fwd["sol"]["data"].shape[0]
    logger.info(
        "Forward model: biosemi32/ico2, n_channels=%d, n_dipoles=%d",
        n_channels,
        n_dipoles,
    )

    # Adjacency & positions for evaluation
    adjacency = build_adjacency(fwd, verbose=0)
    pos = pos_from_forward(fwd)

    # Simulation config (same as benchmark default)
    sim_config = SimulationConfig()

    # Generate fixed test set
    logger.info("Generating test set (n=%d)...", args.n_test)
    x_test, y_test = generate_test_set(fwd, sim_config, n_samples=args.n_test, seed=99)
    logger.info("Test set shapes: x=%s, y=%s", x_test.shape, y_test.shape)

    # Get experiments
    experiments = get_experiments(
        n_dipoles, quick=args.quick, round2=args.round2, round3=args.round3
    )

    # Filter if --only specified
    if args.only:
        experiments = [
            e for e in experiments if any(e.label.startswith(p) for p in args.only)
        ]
        if not experiments:
            logger.error("No experiments matched --only %s", args.only)
            sys.exit(1)

    logger.info("Running %d experiments", len(experiments))

    # Run all experiments, saving incrementally
    out_path = Path("results/ann_testbench_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []

    for i, exp in enumerate(experiments):
        logger.info("[%d/%d] Starting: %s", i + 1, len(experiments), exp.label)
        try:
            result = train_and_evaluate(
                exp, fwd, info, sim_config, x_test, y_test, adjacency, pos, n_dipoles
            )
            results.append(result)
        except Exception as e:
            logger.error("FAILED [%s]: %s", exp.label, e, exc_info=True)
            results.append({"label": exp.label, "error": str(e)})

        # Save after each experiment so we don't lose progress
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    logger.info("All results saved to %s", out_path)
    print_summary(results)


if __name__ == "__main__":
    main()
