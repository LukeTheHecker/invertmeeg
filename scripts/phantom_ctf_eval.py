"""Evaluate solvers on the Brainstorm CTF phantom dataset.

Based on the MNE tutorial:
https://mne.tools/stable/auto_tutorials/inverse/85_brainstorm_phantom_ctf.html

Single dipole at [18, 0, 49] mm, 23 Hz sinusoidal stimulus.
Two recordings: 20 µA (low) and 200 µA (high current).
Empty room recording available for noise covariance.
"""

import argparse
import json
import traceback
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.datasets.brainstorm import bst_phantom_ctf

from invert import Solver

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ground truth dipole position (mm, head coordinates)
ACTUAL_POS_MM = np.array([18.0, 0.0, 49.0])
ACTUAL_POS = ACTUAL_POS_MM / 1000.0  # meters

DIP_FREQ = 23.0  # Hz
HEAD_RADIUS = 0.080  # meters
SOURCE_RADIUS = 0.060  # meters — tighter than head to remove shallow sources
SOURCE_GRID_MM = 7.0
MINDIST_MM = 5.0

ALL_SOLVERS = [
    "eLORETA", "sLORETA", "SR-eLORETA",
    "ESMV", "ESMV-MVPURE", "Flex-ESMV", "LCMV",
    "MNE", "wMNE", "dSPM",
    "Champagne", "Subspace-SBL",
    "RAP-MUSIC", "FLEX-MUSIC", "OMP",
    "ECD",
]
FAST_SOLVERS = ["eLORETA", "ESMV", "LCMV", "MNE", "dSPM"]


def load_phantom_data(data_path, recording="20uA"):
    """Load CTF phantom recording and empty room."""
    if recording == "20uA":
        raw_path = data_path / "phantom_20uA_20150603_03.ds"
    elif recording == "200uA":
        raw_path = data_path / "phantom_200uA_20150709_01.ds"
    else:
        raise ValueError(f"Unknown recording: {recording}")

    raw = mne.io.read_raw_ctf(str(raw_path), preload=True, verbose=False)

    # Detect stimulus events from the DAC sinusoid
    sinusoid, _ = raw[raw.ch_names.index("HDAC006-4408")]
    crossings = np.where(np.diff(sinusoid > 0.5) > 0)[1] + raw.first_samp
    # Waveform crosses 0.5 twice per cycle — take every other for true period
    crossings = crossings[::2]

    events = np.zeros((len(crossings), 3), dtype=int)
    events[:, 0] = crossings
    events[:, 2] = 1

    # Epoch around one full cycle
    tmin = -0.5 / DIP_FREQ
    tmax = 0.5 / DIP_FREQ
    epochs = mne.Epochs(
        raw, events, tmin=tmin, tmax=tmax,
        baseline=(None, None), preload=True, verbose=False,
    )
    evoked = epochs.average()

    # Noise covariance from empty room
    raw_er = mne.io.read_raw_ctf(
        str(data_path / "emptyroom_20150709_01.ds"), preload=True, verbose=False,
    )
    noise_cov = mne.compute_raw_covariance(raw_er, verbose=False)

    return epochs, evoked, noise_cov


def make_forward(info):
    """Build spherical forward model for the CTF phantom."""
    sphere = mne.make_sphere_model(
        r0=(0.0, 0.0, 0.0), head_radius=HEAD_RADIUS, verbose=False,
    )
    src = mne.setup_volume_source_space(
        subject=None,
        pos=SOURCE_GRID_MM,
        sphere=(0.0, 0.0, 0.0, SOURCE_RADIUS),
        mindist=MINDIST_MM,
        exclude=0.0,
        verbose=False,
    )
    # Inject the true dipole position
    rr_used = src[0]["rr"][src[0]["vertno"]]
    rr = np.vstack([rr_used, ACTUAL_POS])
    rr = np.unique(np.round(rr, 6), axis=0)

    norms = np.linalg.norm(rr, axis=1, keepdims=True)
    nn = np.zeros_like(rr)
    mask = norms.squeeze() > 0
    nn[mask] = rr[mask] / norms[mask]
    nn[~mask] = np.array([1.0, 0.0, 0.0])

    src = mne.setup_volume_source_space(
        subject=None,
        pos={"rr": rr, "nn": nn},
        sphere=(0.0, 0.0, 0.0, SOURCE_RADIUS),
        mindist=0.0,
        exclude=0.0,
        verbose=False,
    )
    fwd = mne.make_forward_solution(
        info, trans=None, src=src, bem=sphere,
        meg=True, eeg=False, verbose=False,
    )
    return fwd, sphere


def source_positions(fwd):
    return np.concatenate([s["rr"][s["vertno"]] for s in fwd["src"]], axis=0)


def peak_localization_error(stc, fwd, actual_pos):
    """Return peak position and error in mm."""
    power = np.max(np.abs(stc.data), axis=1)
    pos = source_positions(fwd)
    # Handle free orientation: collapse if needed
    n_src = pos.shape[0]
    if power.shape[0] != n_src and power.shape[0] % n_src == 0:
        n_orient = power.shape[0] // n_src
        power = np.max(power.reshape(n_src, n_orient), axis=1)
    peak_idx = int(np.argmax(power))
    peak_pos = pos[peak_idx]
    err_mm = 1e3 * np.linalg.norm(peak_pos - actual_pos)
    return err_mm, peak_pos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solvers", nargs="+", default=None,
        help="Solver names (default: ALL_SOLVERS).",
    )
    parser.add_argument("--fast", action="store_true", help="Run fast subset only.")
    parser.add_argument(
        "--recording", choices=["20uA", "200uA", "both"], default="both",
        help="Which recording to use.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Save results to JSON.",
    )
    args = parser.parse_args()

    solvers = FAST_SOLVERS if args.fast else (args.solvers or ALL_SOLVERS)
    recordings = ["20uA", "200uA"] if args.recording == "both" else [args.recording]

    print("Loading Brainstorm CTF phantom data...")
    data_path = bst_phantom_ctf.data_path(verbose=False)

    all_results = []
    for rec in recordings:
        print(f"\n{'=' * 70}")
        print(f"Recording: {rec}")
        print(f"{'=' * 70}")

        epochs, evoked, noise_cov = load_phantom_data(data_path, rec)
        fwd, sphere = make_forward(evoked.info)
        n_sources = source_positions(fwd).shape[0]
        print(f"  {len(epochs)} epochs, {n_sources} sources")

        # MNE dipole fit baseline
        dip = mne.fit_dipole(
            evoked.copy().crop(0, 0), noise_cov, sphere, verbose=False,
        )[0]
        dip_err = 1e3 * np.linalg.norm(dip.pos[0] - ACTUAL_POS)
        print(f"  MNE dipole fit: {dip_err:.1f} mm (GOF {dip.gof[0]:.1f}%)")

        print(f"\n  {'Solver':<25} {'Error (mm)':>12}")
        print(f"  {'-' * 40}")

        for solver_name in solvers:
            try:
                solver = Solver(solver_name)
                is_bf = getattr(solver, "_is_beamformer", False)
                mne_obj = epochs if is_bf else evoked
                solver.make_inverse_operator(fwd, mne_obj, noise_cov=noise_cov)
                stc = solver.apply_inverse_operator(evoked.copy().crop(0, 0))
                err_mm, peak_pos = peak_localization_error(stc, fwd, ACTUAL_POS)
                print(f"  {solver_name:<25} {err_mm:>10.1f}")
                all_results.append({
                    "solver": solver_name,
                    "recording": rec,
                    "error_mm": err_mm,
                    "peak_pos": peak_pos.tolist(),
                })
            except Exception as e:
                print(f"  {solver_name:<25} {'FAILED':>10}  ({e})")
                traceback.print_exc()
                all_results.append({
                    "solver": solver_name,
                    "recording": rec,
                    "error_mm": float("nan"),
                    "error_msg": str(e),
                })

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(all_results, indent=2))
        print(f"\nSaved results to {args.output}")

    # Summary
    if all_results:
        df = pd.DataFrame(all_results)
        print(f"\n{'=' * 70}")
        print("Summary (sorted by mean error across recordings):")
        print(f"{'=' * 70}")
        summary = df.groupby("solver")["error_mm"].mean().sort_values()
        for solver, err in summary.items():
            print(f"  {solver:<25} {err:>8.1f} mm")


if __name__ == "__main__":
    main()
