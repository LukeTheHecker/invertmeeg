"""Evaluate MSP solver convergence and performance.

Runs MSP on a few simulated samples and prints detailed convergence diagnostics.
"""

import time
import logging
import numpy as np

from invert.forward import create_forward_model, get_info
from invert.simulate import SimulationConfig, SimulationGenerator
from invert.solvers.bayesian.msp import SolverMSP

import mne

# Enable verbose logging for MSP
logging.basicConfig(level=logging.DEBUG, format="%(message)s")
logger = logging.getLogger(__name__)


def patch_msp_for_diagnostics(solver):
    """Monkey-patch MSP's make_inverse_operator to print convergence info."""
    original_make = solver.make_inverse_operator.__func__

    def make_with_diagnostics(self, forward, mne_obj=None, *args, **kwargs):
        # We'll intercept the ReML loop by temporarily enabling verbose
        old_verbose = self.verbose
        self.verbose = True
        # Set logging to DEBUG
        msp_logger = logging.getLogger("invert.solvers.bayesian.msp")
        old_level = msp_logger.level
        msp_logger.setLevel(logging.DEBUG)

        t0 = time.time()
        result = original_make(self, forward, mne_obj, *args, **kwargs)
        elapsed = time.time() - t0

        self.verbose = old_verbose
        msp_logger.setLevel(old_level)

        print(f"\n  Total make_inverse_operator time: {elapsed:.2f}s")
        print(f"  Final lambda range: [{self.lambdas.min():.2f}, {self.lambdas.max():.2f}]")
        scales = np.exp(self.lambdas)
        n_active = np.sum(scales > 1e-6)
        print(f"  Active components (scale > 1e-6): {n_active}/{len(self.lambdas)}")
        print(f"  Top 10 scales: {np.sort(scales)[::-1][:10]}")
        return result

    import types
    solver.make_inverse_operator = types.MethodType(make_with_diagnostics, solver)
    return solver


def make_detailed_msp(forward, evoked, noise_cov=None, max_iter=512, **kwargs):
    """Run MSP with detailed per-iteration diagnostics."""
    from invert.solvers.bayesian.msp import SolverMSP

    solver = SolverMSP(max_iter=max_iter, **kwargs)
    solver.verbose = True

    # Manually step through to add timing info
    t_start = time.time()

    # First time the patterns generation
    t0 = time.time()
    solver.make_inverse_operator(forward, mne_obj=evoked, noise_cov=noise_cov)
    t_total = time.time() - t_start

    return solver, t_total


def run_convergence_study():
    """Run MSP on simulated data and show convergence."""
    print("=" * 70)
    print("MSP Convergence Study")
    print("=" * 70)

    # Setup
    info = get_info(kind="biosemi32")
    fwd = create_forward_model(sampling="ico2", info=info)
    leadfield = fwd["sol"]["data"]
    n_channels, n_dipoles = leadfield.shape
    print(f"\nForward model: {n_channels} channels, {n_dipoles} dipoles")

    # Create simulations
    configs = {
        "focal": SimulationConfig(
            batch_size=3, n_sources=(1, 1), n_orders=(0, 0),
            snr_range=(5.0, 5.0), n_timepoints=50, random_seed=42,
            estimate_noise_cov=True, return_noise_cov=True,
        ),
        "multi": SimulationConfig(
            batch_size=3, n_sources=(2, 4), n_orders=(0, 2),
            snr_range=(1.0, 3.0), n_timepoints=50, random_seed=42,
            estimate_noise_cov=True, return_noise_cov=True,
        ),
    }

    for dataset_name, config in configs.items():
        print(f"\n{'=' * 70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 70}")

        gen = SimulationGenerator(fwd, config=config)
        x_batch, y_batch, sim_info = next(gen.generate())

        for sample_idx in range(min(2, len(x_batch))):
            print(f"\n--- Sample {sample_idx + 1} ---")
            x = x_batch[sample_idx]  # (n_channels, n_timepoints)
            y_true = y_batch[sample_idx]  # (n_dipoles, n_timepoints)

            # Get noise cov
            noise_cov_arr = sim_info.iloc[sample_idx].get("noise_cov_est", None)
            if noise_cov_arr is not None:
                from invert.benchmark.runner import _make_mne_covariance
                noise_cov = _make_mne_covariance(noise_cov_arr, info, nfree=200)
            else:
                noise_cov = None

            evoked = mne.EvokedArray(x, info, tmin=0.0, verbose=0)

            # Run MSP with detailed logging
            print(f"\nRunning MSP (max_iter=128)...")
            solver, t_total = make_detailed_msp(
                fwd, evoked, noise_cov=noise_cov, max_iter=128
            )
            print(f"Total time: {t_total:.2f}s")

            # Evaluate reconstruction quality
            stc = solver.apply_inverse_operator(evoked)
            y_pred = stc.data

            # Simple metrics
            corr = np.corrcoef(
                np.abs(y_true).max(axis=1),
                np.abs(y_pred).max(axis=1)
            )[0, 1]
            print(f"Correlation (peak amplitudes): {corr:.3f}")

            # Only do first sample in detail to keep output manageable
            if sample_idx >= 1:
                break


def run_iteration_profiling():
    """Profile where time is spent per iteration."""
    print("\n" + "=" * 70)
    print("MSP Per-Iteration Profiling")
    print("=" * 70)

    info = get_info(kind="biosemi32")
    fwd = create_forward_model(sampling="ico2", info=info)

    config = SimulationConfig(
        batch_size=1, n_sources=(1, 1), n_orders=(0, 0),
        snr_range=(5.0, 5.0), n_timepoints=50, random_seed=42,
        estimate_noise_cov=True, return_noise_cov=True,
    )
    gen = SimulationGenerator(fwd, config=config)
    x_batch, y_batch, sim_info = next(gen.generate())

    x = x_batch[0]
    evoked = mne.EvokedArray(x, info, tmin=0.0, verbose=0)

    noise_cov_arr = sim_info.iloc[0].get("noise_cov_est", None)
    if noise_cov_arr is not None:
        from invert.benchmark.runner import _make_mne_covariance
        noise_cov = _make_mne_covariance(noise_cov_arr, info, nfree=200)
    else:
        noise_cov = None

    # Create solver and manually run the ReML loop with timing
    solver = SolverMSP(max_iter=100, verbose=False)

    # Call make_inverse_operator but we'll instrument the inner loop
    # For now, just time the whole thing at different max_iter values
    for max_it in [10, 50, 100, 200, 512]:
        solver_test = SolverMSP(max_iter=max_it, verbose=False)
        t0 = time.time()
        solver_test.make_inverse_operator(fwd, mne_obj=evoked, noise_cov=noise_cov)
        elapsed = time.time() - t0

        # Evaluate quality
        stc = solver_test.apply_inverse_operator(evoked)
        y_pred = stc.data
        y_true = y_batch[0]
        corr = np.corrcoef(
            np.abs(y_true).max(axis=1),
            np.abs(y_pred).max(axis=1)
        )[0, 1]

        n_active = np.sum(np.exp(solver_test.lambdas) > 1e-6)
        print(f"max_iter={max_it:>4d}  time={elapsed:6.2f}s  corr={corr:.3f}  "
              f"active={n_active}/{len(solver_test.lambdas)}  "
              f"lambda_range=[{solver_test.lambdas.min():.1f}, {solver_test.lambdas.max():.1f}]")


if __name__ == "__main__":
    # Suppress MNE info messages
    mne.set_log_level("WARNING")

    run_iteration_profiling()
    print("\n\n")
    run_convergence_study()
