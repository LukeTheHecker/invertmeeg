#!/usr/bin/env python
"""Architecture testbench for new CovCNN-KL variants.

Tests 4 architectural experiments against the CovCNN-KL baseline:
1. RawCNN-KL     — raw EEG data input (no covariance)
2. RawCNN-KL-LCMV — raw data + LCMV beamformer power maps
3. RawCNN-KL-Eigen — raw data + SVD eigenspace reduction
4. IterCNN-KL    — covariance + learned iterative refinement

Uses reduced settings for faster iteration:
  batch_size=2048, epochs=500, n_dense_units=300, patience=300

Usage:
    cd invert-package
    .venv/bin/python scripts/arch_testbench.py
    .venv/bin/python scripts/arch_testbench.py --only baseline rawcnn
    .venv/bin/python scripts/arch_testbench.py --n-test 10
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mne
import numpy as np

import importlib.util as _ilu

# Import evaluate_all directly to avoid seaborn dependency
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
from invert.util.util import pos_from_forward

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("arch_testbench")


# ---------------------------------------------------------------------------
# Solver imports (lazy to catch import errors gracefully)
# ---------------------------------------------------------------------------

def _import_solver(name: str):
    """Import a solver class by name."""
    if name == "CovCNN-KL":
        from invert.solvers.neural_networks.covcnn_kl import SolverCovCNNKL
        return SolverCovCNNKL
    elif name == "RawCNN-KL":
        from invert.solvers.neural_networks.rawcnn_kl import SolverRawCNNKL
        return SolverRawCNNKL
    elif name == "RawCNN-KL-LCMV":
        from invert.solvers.neural_networks.rawcnn_kl_lcmv import SolverRawCNNKLLCMV
        return SolverRawCNNKLLCMV
    elif name == "RawCNN-KL-Eigen":
        from invert.solvers.neural_networks.rawcnn_kl_eigen import SolverRawCNNKLEigen
        return SolverRawCNNKLEigen
    elif name == "IterCNN-KL":
        from invert.solvers.neural_networks.itercnn_kl import SolverIterCNNKL
        return SolverIterCNNKL
    else:
        raise ValueError(f"Unknown solver: {name}")


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

@dataclass
class ArchExperiment:
    """One architecture experiment."""
    label: str
    solver_name: str
    # Architecture
    n_dense_units: int = 300
    n_dense_layers: int = 2
    activation_function: str = "tanh"
    # Training
    epochs: int = 500
    learning_rate: float = 1e-3
    patience: int = 300
    temperature: float = 1.0
    # Data
    batch_size: int = 2048
    # Target
    target_power: float = 0.5
    gamma_power: float = 1.5
    # Solver-specific params
    extra_params: dict = field(default_factory=dict)


def get_experiments() -> list[ArchExperiment]:
    """Define the architecture experiments."""
    experiments = []

    # Baseline: CovCNN-KL with reduced settings (for fair comparison)
    experiments.append(ArchExperiment(
        label="baseline-covcnn-kl",
        solver_name="CovCNN-KL",
    ))

    # Experiment 1: Raw Data Model
    experiments.append(ArchExperiment(
        label="rawcnn-kl",
        solver_name="RawCNN-KL",
    ))

    # Experiment 2: Raw Data + LCMV
    experiments.append(ArchExperiment(
        label="rawcnn-kl-lcmv",
        solver_name="RawCNN-KL-LCMV",
    ))

    # Experiment 3: Raw Data + Eigenspace (K=8)
    experiments.append(ArchExperiment(
        label="rawcnn-kl-eigen-k8",
        solver_name="RawCNN-KL-Eigen",
        extra_params={"n_components": 8},
    ))

    # Experiment 3b: Raw Data + Eigenspace (K=5)
    experiments.append(ArchExperiment(
        label="rawcnn-kl-eigen-k5",
        solver_name="RawCNN-KL-Eigen",
        extra_params={"n_components": 5},
    ))

    # Experiment 4: Iterative Refinement (2 steps)
    experiments.append(ArchExperiment(
        label="itercnn-kl-2step",
        solver_name="IterCNN-KL",
        extra_params={"n_refinement_steps": 2},
    ))

    # Experiment 4b: Iterative Refinement (3 steps)
    experiments.append(ArchExperiment(
        label="itercnn-kl-3step",
        solver_name="IterCNN-KL",
        extra_params={"n_refinement_steps": 3},
    ))

    return experiments


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_test_set(
    fwd, sim_config: SimulationConfig, n_samples: int = 20, seed: int = 99
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a fixed test set."""
    rng_cfg = sim_config.model_copy(update={"random_seed": seed, "batch_size": n_samples})
    sim_gen = SimulationGenerator(fwd, config=rng_cfg)
    gen = sim_gen.generate()
    x_batch, y_batch, _info = next(gen)
    return x_batch, y_batch


def evaluate_solver(
    solver,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    info: mne.Info,
    adjacency,
    pos: np.ndarray,
) -> dict[str, float]:
    """Evaluate a trained solver on the test set."""
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
    exp: ArchExperiment,
    fwd,
    info: mne.Info,
    sim_config: SimulationConfig,
    x_test: np.ndarray,
    y_test: np.ndarray,
    adjacency,
    pos: np.ndarray,
) -> dict:
    """Train a solver with given config, evaluate, return results."""
    solver_cls = _import_solver(exp.solver_name)
    solver = solver_cls()

    train_sim_config = sim_config.model_copy(update={
        "batch_size": exp.batch_size,
    })

    logger.info("=" * 60)
    logger.info("EXPERIMENT: %s (solver=%s)", exp.label, exp.solver_name)
    logger.info("  units=%d, layers=%d, epochs=%d, lr=%g, patience=%d, batch_size=%d",
                exp.n_dense_units, exp.n_dense_layers, exp.epochs,
                exp.learning_rate, exp.patience, exp.batch_size)
    if exp.extra_params:
        logger.info("  extra: %s", exp.extra_params)
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
        **exp.extra_params,
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
        "solver": exp.solver_name,
        "n_params": n_params,
        "train_time_s": round(train_time, 1),
        "eval_time_s": round(eval_time, 1),
        **{k: round(v, 4) for k, v in metrics.items()},
    }
    logger.info("RESULT [%s]: %s", exp.label, json.dumps(result, indent=2))
    return result


def print_summary(results: list[dict]) -> None:
    """Print a formatted summary table."""
    print("\n" + "=" * 120)
    print("ARCHITECTURE COMPARISON  (MLE/EMD/SD: lower=better; AvgPrec/Corr: higher=better)")
    print("=" * 120)
    header = (
        f"{'Label':<25} {'Solver':<18} {'Params':>10} {'Train(s)':>9} "
        f"{'MLE':>8} {'EMD':>8} {'SD':>8} {'AvgPrec':>8} {'Corr':>8}"
    )
    print(header)
    print("-" * 120)
    for r in results:
        if "error" in r:
            print(f"{r['label']:<25} ERROR: {r['error']}")
            continue
        print(
            f"{r['label']:<25} "
            f"{r.get('solver', '?'):<18} "
            f"{r.get('n_params', '?'):>10} "
            f"{r.get('train_time_s', '?'):>9} "
            f"{r.get('Mean_Localization_Error', float('nan')):>8.4f} "
            f"{r.get('EMD', float('nan')):>8.4f} "
            f"{r.get('sd', float('nan')):>8.4f} "
            f"{r.get('average_precision', float('nan')):>8.4f} "
            f"{r.get('correlation', float('nan')):>8.4f}"
        )
    print("=" * 120)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Architecture testbench for CovCNN-KL variants")
    parser.add_argument("--only", nargs="*", help="Run only experiments whose labels contain these substrings")
    parser.add_argument("--n-test", type=int, default=20, help="Number of test samples (default: 20)")
    args = parser.parse_args()

    mne.set_log_level("WARNING")

    # Forward model
    info = get_info(kind="biosemi32")
    fwd = create_forward_model(sampling="ico2", info=info)
    n_dipoles = fwd["sol"]["data"].shape[1]
    n_channels = fwd["sol"]["data"].shape[0]
    logger.info("Forward model: biosemi32/ico2, n_channels=%d, n_dipoles=%d", n_channels, n_dipoles)

    # Adjacency & positions
    adjacency = build_adjacency(fwd, verbose=0)
    pos = pos_from_forward(fwd)

    # Simulation config
    sim_config = SimulationConfig()

    # Fixed test set
    logger.info("Generating test set (n=%d)...", args.n_test)
    x_test, y_test = generate_test_set(fwd, sim_config, n_samples=args.n_test, seed=99)
    logger.info("Test set shapes: x=%s, y=%s", x_test.shape, y_test.shape)

    # Get experiments
    experiments = get_experiments()

    # Filter if --only
    if args.only:
        experiments = [e for e in experiments if any(s in e.label for s in args.only)]
        if not experiments:
            logger.error("No experiments matched --only %s", args.only)
            sys.exit(1)

    logger.info("Running %d experiments", len(experiments))

    # Run
    out_path = Path("results/arch_testbench_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []

    for i, exp in enumerate(experiments):
        logger.info("[%d/%d] Starting: %s", i + 1, len(experiments), exp.label)
        try:
            result = train_and_evaluate(
                exp, fwd, info, sim_config, x_test, y_test, adjacency, pos
            )
            results.append(result)
        except Exception as e:
            logger.error("FAILED [%s]: %s", exp.label, e, exc_info=True)
            results.append({"label": exp.label, "solver": exp.solver_name, "error": str(e)})

        # Save incrementally
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    logger.info("All results saved to %s", out_path)
    print_summary(results)


if __name__ == "__main__":
    main()
