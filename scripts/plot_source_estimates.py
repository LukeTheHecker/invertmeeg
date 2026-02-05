#!/usr/bin/env python
"""Simulate one sample per benchmark dataset, invert with OmniChampagne,
and save glass-brain and surface plots to figures/."""

import sys
from pathlib import Path

import mne

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from invert.benchmark.datasets import BENCHMARK_DATASETS
from invert.forward import create_forward_model, get_info
from invert.simulate import SimulationConfig, SimulationGenerator
from invert.solvers.bayesian.omni_champagne import SolverOmniChampagne
from invert.viz import plot_glass_brain, plot_surface

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def main():
    mne.set_log_level("WARNING")

    # Create custom info with desired channel layout
    info = get_info(kind="biosemi32")

    # Create forward model once (ico3 for speed) using our custom info
    print("Creating forward model …")
    fwd = create_forward_model(sampling="ico3", info=info)
    src = fwd["src"]

    for ds_name, ds_cfg in BENCHMARK_DATASETS.items():
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds_name}")
        print(f"{'=' * 60}")

        # Simulate a single sample
        sim_config = SimulationConfig(
            batch_size=1,
            n_sources=ds_cfg.n_sources,
            n_orders=ds_cfg.n_orders,
            snr_range=ds_cfg.snr_range,
            n_timepoints=ds_cfg.n_timepoints,
            random_seed=42,
        )
        gen = SimulationGenerator(fwd, config=sim_config)
        x, y, sim_info = next(gen.generate())

        # x shape: (1, n_channels, n_timepoints)
        x_sample = x[0]  # (n_channels, n_timepoints)

        # Create Evoked object from simulated sensor data
        evoked = mne.EvokedArray(x_sample, info, tmin=0.0, verbose=0)

        # Invert with OmniChampagne
        print("Running OmniChampagne …")
        solver = SolverOmniChampagne()
        solver.make_inverse_operator(fwd, evoked, alpha="auto")
        stc = solver.apply_inverse_operator(evoked)

        # --- Glass brain plot (matplotlib) ---
        fig = plot_glass_brain(
            stc,
            src,
            threshold=0.25,
            cmap="hot",
            title=f"OmniChampagne — {ds_name}",
        )
        path = FIGURES_DIR / f"glass_brain_{ds_name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")
        import matplotlib.pyplot as plt

        plt.close(fig)

        # --- Surface plot (PyVista) ---
        path = FIGURES_DIR / f"surface_{ds_name}.png"
        plot_surface(
            stc,
            src,
            threshold=0.25,
            cmap="hot",
            views=["lateral", "medial", "dorsal"],
            title=f"OmniChampagne — {ds_name}",
            screenshot_path=str(path),
            show=False,
        )
        print(f"Saved {path}")

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
