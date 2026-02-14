"""Evaluate focal-source solvers on 4D BTi phantom dataset."""
import os.path as op
import warnings
import traceback

import mne
import numpy as np
import pandas as pd

from invert import Solver
from mne.datasets import phantom_4dbti

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Config
RUN_IDS = [1, 2, 3, 4]
BAD_CHANNELS = ["A173", "A213", "A232"]
PEAK_TIME = 0.07

actual_pos = 0.01 * np.array([
    [0.16, 1.61, 5.13],
    [0.17, 1.35, 4.15],
    [0.16, 1.05, 3.19],
    [0.13, 0.80, 2.26],
]) @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

# Solvers to test - focal source methods
SOLVERS = [
    "mne",
    "wmne",
    "loreta",
    "sloreta",
    "esmv",
    "lcmv",
    "champagne",
    "gamma_map",
    "rap_music",
    "ecd",
]

def load_run(run_id, data_path):
    raw = mne.io.read_raw_bti(
        op.join(data_path, f"{run_id}/e,rfhp1.0Hz"),
        rename_channels=False, preload=True, verbose=False
    )
    raw.info["bads"] = BAD_CHANNELS
    events = mne.find_events(raw, stim_channel="TRIGGER", mask=4350, mask_type="not_and", verbose=False)
    epochs = mne.Epochs(raw, events=events, event_id=8192, tmin=-0.2, tmax=0.4,
                        preload=True, baseline=(None, 0.0), verbose=False)
    evoked = epochs.average()
    cov = mne.compute_covariance(epochs, tmax=0.0, verbose=False)
    return evoked, cov

def make_fwd(info):
    sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.0), head_radius=0.080, verbose=False)
    src = mne.setup_volume_source_space(
        subject=None, pos=8.0, sphere=(0.0, 0.0, 0.0, 0.072),
        mindist=5.0, exclude=0.0, verbose=False
    )
    fwd = mne.make_forward_solution(info, trans=None, src=src, bem=sphere,
                                     meg=True, eeg=False, verbose=False)
    return fwd, sphere

def source_positions(fwd):
    return np.concatenate([s["rr"][s["vertno"]] for s in fwd["src"]], axis=0)

def peak_position(source_mat, fwd):
    power = np.max(np.abs(source_mat), axis=1)
    idx = int(np.argmax(power))
    return source_positions(fwd)[idx]

# Load data
print("Loading phantom data...")
data_path = phantom_4dbti.data_path(verbose=False)
runs = {}
for rid in RUN_IDS:
    evoked, cov = load_run(rid, data_path)
    fwd, sphere = make_fwd(evoked.info)
    runs[rid] = {"evoked": evoked, "cov": cov, "fwd": fwd, "sphere": sphere}

# MNE dipole fit baseline
print("\nMNE Dipole Fit baseline:")
for rid in RUN_IDS:
    r = runs[rid]
    dip = mne.fit_dipole(r["evoked"].copy().crop(0.07, 0.07), r["cov"], r["sphere"], verbose=False)[0]
    err = 1e3 * np.linalg.norm(dip.pos[0] - actual_pos[rid-1])
    print(f"  Run {rid}: {err:.1f} mm (GOF {dip.gof[0]:.1f}%)")

# Test each solver
print(f"\n{'='*70}")
print(f"{'Solver':<20} {'Run1':>8} {'Run2':>8} {'Run3':>8} {'Run4':>8} {'Mean':>8}")
print(f"{'='*70}")

results = []
for solver_name in SOLVERS:
    errors = []
    failed = False
    for rid in RUN_IDS:
        r = runs[rid]
        try:
            solver = Solver(solver_name)
            solver.make_inverse_operator(r["fwd"], r["evoked"], noise_cov=r["cov"])
            stc = solver.apply_inverse_operator(r["evoked"].copy().crop(PEAK_TIME, PEAK_TIME))
            pos_hat = peak_position(stc.data, r["fwd"])
            err = 1e3 * np.linalg.norm(pos_hat - actual_pos[rid-1])
            errors.append(err)
        except Exception as e:
            print(f"  {solver_name} run {rid} FAILED: {e}")
            traceback.print_exc()
            failed = True
            break
    
    if not failed and len(errors) == 4:
        mean_err = np.mean(errors)
        print(f"{solver_name:<20} {errors[0]:>7.1f}  {errors[1]:>7.1f}  {errors[2]:>7.1f}  {errors[3]:>7.1f}  {mean_err:>7.1f}")
        results.append({"solver": solver_name, "mean_mm": mean_err, **{f"run{i+1}": e for i, e in enumerate(errors)}})

print(f"{'='*70}")
if results:
    df = pd.DataFrame(results).sort_values("mean_mm")
    print(f"\nRanked by mean error:")
    for _, row in df.iterrows():
        print(f"  {row['solver']:<20} {row['mean_mm']:.1f} mm")
