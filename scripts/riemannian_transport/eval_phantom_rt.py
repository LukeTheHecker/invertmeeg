"""Phantom datasets: RT with controlled geometry mismatch.

Uses exact ground truth dipole positions from two phantom datasets
(CTF single-dipole and 4D BTi four-dipole) to quantify RT's ability
to correct a deliberately wrong sphere model.

Conditions:
  1. Correct sphere  — true r0, true radius (upper bound)
  2. Wrong sphere    — shifted center + wrong radius
  3. Wrong sphere+RT — same wrong sphere + Riemannian transport

Reports localization error in mm for each condition.

Usage:
    cd invert-package
    .venv/bin/python scripts/riemannian_transport/eval_phantom_rt.py
"""

import os
import time
import warnings

import mne
import numpy as np
from mne.datasets.brainstorm import bst_phantom_ctf
from mne.datasets import phantom_4dbti

from invert import Solver
from invert.cortical_folding.riemannian_transport import (
    riemannian_leadfield_transport,
)

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# ── Parameters ──────────────────────────────────────────────────────────
SOLVERS = ['eloreta', 'esmv', 'champagne', 'subspace-sbl']
ALPHA = 0.80
REG = 0.01
TRANSPORT = 'bures'
ALPHA_MODE = 'geodesic'

# Sphere mismatch parameters
WRONG_CENTER = np.array([0.010, 0.005, 0.0])  # 10mm x, 5mm y shift
WRONG_RADIUS = 0.070  # true = 0.080

SOURCE_GRID_MM = 7.0
SOURCE_RADIUS = 0.060
MINDIST_MM = 5.0

FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures', 'phantom')
os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Shared utilities ────────────────────────────────────────────────────

def make_source_space(extra_pos=None):
    """Build volume source space, optionally injecting known positions."""
    src_grid = mne.setup_volume_source_space(
        subject=None, pos=SOURCE_GRID_MM,
        sphere=(0., 0., 0., SOURCE_RADIUS),
        mindist=MINDIST_MM, exclude=0.0, verbose=0)
    rr = src_grid[0]['rr'][src_grid[0]['vertno']]
    if extra_pos is not None:
        rr = np.vstack([rr, np.atleast_2d(extra_pos)])
    rr = np.unique(np.round(rr, 6), axis=0)
    norms = np.linalg.norm(rr, axis=1, keepdims=True)
    nn = np.zeros_like(rr)
    mask = norms.squeeze() > 0
    nn[mask] = rr[mask] / norms[mask]
    nn[~mask] = [1., 0., 0.]
    src = mne.setup_volume_source_space(
        subject=None, pos={'rr': rr, 'nn': nn},
        sphere=(0., 0., 0., SOURCE_RADIUS),
        mindist=0.0, exclude=0.0, verbose=0)
    return src, rr


def make_forward(info, r0, radius, src):
    """Build forward with given sphere parameters."""
    sphere = mne.make_sphere_model(r0=tuple(r0), head_radius=radius, verbose=0)
    fwd = mne.make_forward_solution(
        info, trans=None, src=src, bem=sphere,
        meg=True, eeg=False, verbose=0)
    return fwd


def get_peak_error(stc, pos_grid, true_pos):
    """Get localization error in mm from peak source to true position."""
    power = np.max(np.abs(stc.data), axis=1)
    n_src = pos_grid.shape[0]
    if power.shape[0] != n_src and power.shape[0] % n_src == 0:
        power = np.max(power.reshape(n_src, power.shape[0] // n_src), axis=1)
    peak_idx = int(np.argmax(power))
    return 1e3 * np.linalg.norm(pos_grid[peak_idx] - true_pos)


def run_solver(solver_name, fwd, evoked, noise_cov, peak_time):
    """Run solver and return STC cropped to peak time."""
    solver = Solver(solver_name)
    solver.make_inverse_operator(
        fwd, evoked, noise_cov=noise_cov, alpha='auto', verbose=0)
    return solver.apply_inverse_operator(evoked.copy().crop(peak_time, peak_time))


# ── CTF Phantom ─────────────────────────────────────────────────────────

def run_ctf_phantom():
    """CTF phantom: single dipole at [18, 0, 49] mm."""
    print('\n' + '=' * 70)
    print('CTF PHANTOM — Single dipole at [18, 0, 49] mm')
    print(f'Wrong sphere: center={WRONG_CENTER*1000}mm, radius={WRONG_RADIUS*1000}mm')
    print('=' * 70)

    ACTUAL_POS = np.array([18.0, 0.0, 49.0]) / 1000.0
    DIP_FREQ = 23.0

    data_path = bst_phantom_ctf.data_path(verbose=False)

    def load_ctf(name):
        raw = mne.io.read_raw_ctf(
            str(data_path / name), preload=True, verbose=0)
        sinusoid, _ = raw[raw.ch_names.index('HDAC006-4408')]
        crossings = np.where(np.diff(sinusoid > 0.5) > 0)[1] + raw.first_samp
        crossings = crossings[::2]
        events = np.zeros((len(crossings), 3), dtype=int)
        events[:, 0] = crossings
        events[:, 2] = 1
        epochs = mne.Epochs(
            raw, events, tmin=-0.5/DIP_FREQ, tmax=0.5/DIP_FREQ,
            baseline=(None, None), preload=True, verbose=0)
        return raw, epochs, epochs.average()

    # Load recordings
    raw_200, epochs_200, evoked_200 = load_ctf('phantom_200uA_20150709_01.ds')
    raw_20, epochs_20, evoked_20 = load_ctf('phantom_20uA_20150603_03.ds')
    raw_er = mne.io.read_raw_ctf(
        str(data_path / 'emptyroom_20150709_01.ds'), preload=True, verbose=0)
    noise_cov = mne.compute_raw_covariance(raw_er, verbose=0)

    # Source space (shared across all conditions)
    src, pos_grid = make_source_space(ACTUAL_POS)
    print(f'  Source space: {pos_grid.shape[0]} points')

    # Data covariance for RT (from 200µA recording)
    raw_meg = raw_200.copy().pick('meg')
    raw_meg.apply_proj()
    X = raw_meg.get_data().T
    C_data = (X.T @ X) / X.shape[0]
    cov_chs = list(raw_meg.ch_names)

    results = []

    for rec_label, evoked, epochs in [
        ('200µA', evoked_200, epochs_200),
        ('20µA', evoked_20, epochs_20),
    ]:
        print(f'\n  --- {rec_label} ---')

        # Build forwards
        fwd_correct = make_forward(evoked.info, [0,0,0], 0.080, src)
        fwd_wrong = make_forward(evoked.info, WRONG_CENTER, WRONG_RADIUS, src)

        # Align C_data to fwd_wrong channels
        fwd_ch = [ch['ch_name'] for ch in fwd_wrong['info']['chs']]
        common = [ch for ch in fwd_ch if ch in cov_chs]
        fwd_wrong_picked = mne.pick_channels_forward(fwd_wrong, common, verbose=0)
        fwd_ch_picked = [ch['ch_name'] for ch in fwd_wrong_picked['info']['chs']]
        idx = [cov_chs.index(ch) for ch in fwd_ch_picked]
        C_aligned = C_data[np.ix_(idx, idx)]

        # RT
        fwd_wrong_rt = riemannian_leadfield_transport(
            C_aligned, fwd_wrong_picked,
            alpha=ALPHA, reg=REG, transport=TRANSPORT, alpha_mode=ALPHA_MODE)

        # Also pick correct and wrong fwd to same channels
        fwd_correct_picked = mne.pick_channels_forward(
            fwd_correct, fwd_ch_picked, verbose=0)

        for solver_name in SOLVERS:
            for cond, fwd_use in [
                ('correct', fwd_correct_picked),
                ('wrong', fwd_wrong_picked),
                ('wrong_rt', fwd_wrong_rt),
            ]:
                try:
                    stc = run_solver(solver_name, fwd_use, evoked, noise_cov, 0.0)
                    err = get_peak_error(stc, pos_grid, ACTUAL_POS)
                    results.append({
                        'dataset': 'CTF',
                        'recording': rec_label,
                        'solver': solver_name,
                        'condition': cond,
                        'error_mm': err,
                    })
                    print(f'    {solver_name:15} {cond:12} {err:6.1f} mm')
                except Exception as e:
                    print(f'    {solver_name:15} {cond:12} FAILED: {e}')
                    results.append({
                        'dataset': 'CTF',
                        'recording': rec_label,
                        'solver': solver_name,
                        'condition': cond,
                        'error_mm': np.nan,
                    })

    del raw_200, raw_20, raw_er, raw_meg
    return results


# ── 4D BTi Phantom ──────────────────────────────────────────────────────

def run_4dbti_phantom():
    """4D BTi phantom: 4 dipoles at known positions."""
    print('\n' + '=' * 70)
    print('4D BTi PHANTOM — 4 dipoles at known positions')
    print(f'Wrong sphere: center={WRONG_CENTER*1000}mm, radius={WRONG_RADIUS*1000}mm')
    print('=' * 70)

    ACTUAL_POS = 0.01 * np.array([
        [0.16, 1.61, 5.13],
        [0.17, 1.35, 4.15],
        [0.16, 1.05, 3.19],
        [0.13, 0.80, 2.26],
    ]) @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

    BAD_CHANNELS = ["A173", "A213", "A232"]
    PEAK_TIME = 0.07

    data_path = phantom_4dbti.data_path()
    src, pos_grid = make_source_space(ACTUAL_POS)
    print(f'  Source space: {pos_grid.shape[0]} points')

    results = []

    for run_id in [1, 2, 3, 4]:
        true_pos = ACTUAL_POS[run_id - 1]
        print(f'\n  --- Run {run_id} (dipole at '
              f'[{true_pos[0]*1e3:.1f}, {true_pos[1]*1e3:.1f}, '
              f'{true_pos[2]*1e3:.1f}] mm) ---')

        raw_fname = os.path.join(data_path, f'{run_id}', 'e,rfhp1.0Hz')
        raw = mne.io.read_raw_bti(
            raw_fname, rename_channels=False, preload=True, verbose=0)
        raw.info['bads'] = BAD_CHANNELS

        events = mne.find_events(
            raw, stim_channel='TRIGGER',
            mask=4350, mask_type='not_and', verbose=0)
        epochs = mne.Epochs(
            raw, events, event_id=8192, tmin=-0.2, tmax=0.4,
            preload=True, baseline=(None, 0.0), verbose=0)
        evoked = epochs.average()
        evoked.drop_channels(evoked.info['bads'])
        noise_cov = mne.compute_covariance(epochs, tmax=0.0, verbose=0)

        # Data covariance
        raw_meg = raw.copy().pick('meg')
        raw_meg.drop_channels(BAD_CHANNELS)
        X = raw_meg.get_data().T
        C_data = (X.T @ X) / X.shape[0]
        cov_chs = list(raw_meg.ch_names)

        # Build forwards
        fwd_correct = make_forward(evoked.info, [0,0,0], 0.080, src)
        fwd_wrong = make_forward(evoked.info, WRONG_CENTER, WRONG_RADIUS, src)

        # Align and RT
        fwd_ch = [ch['ch_name'] for ch in fwd_wrong['info']['chs']]
        common = [ch for ch in fwd_ch if ch in cov_chs]
        fwd_wrong_p = mne.pick_channels_forward(fwd_wrong, common, verbose=0)
        fwd_ch_p = [ch['ch_name'] for ch in fwd_wrong_p['info']['chs']]
        idx = [cov_chs.index(ch) for ch in fwd_ch_p]
        C_aligned = C_data[np.ix_(idx, idx)]

        fwd_wrong_rt = riemannian_leadfield_transport(
            C_aligned, fwd_wrong_p,
            alpha=ALPHA, reg=REG, transport=TRANSPORT, alpha_mode=ALPHA_MODE)

        fwd_correct_p = mne.pick_channels_forward(
            fwd_correct, fwd_ch_p, verbose=0)

        for solver_name in SOLVERS:
            for cond, fwd_use in [
                ('correct', fwd_correct_p),
                ('wrong', fwd_wrong_p),
                ('wrong_rt', fwd_wrong_rt),
            ]:
                try:
                    stc = run_solver(
                        solver_name, fwd_use, evoked, noise_cov, PEAK_TIME)
                    err = get_peak_error(stc, pos_grid, true_pos)
                    results.append({
                        'dataset': '4D-BTi',
                        'recording': f'run{run_id}',
                        'solver': solver_name,
                        'condition': cond,
                        'error_mm': err,
                    })
                    print(f'    {solver_name:15} {cond:12} {err:6.1f} mm')
                except Exception as e:
                    print(f'    {solver_name:15} {cond:12} FAILED: {e}')
                    results.append({
                        'dataset': '4D-BTi',
                        'recording': f'run{run_id}',
                        'solver': solver_name,
                        'condition': cond,
                        'error_mm': np.nan,
                    })

        del raw, raw_meg

    return results


# ── Main ────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    print('=' * 70)
    print('PHANTOM DATASETS — RIEMANNIAN TRANSPORT WITH WRONG SPHERE')
    print(f'Solvers: {", ".join(SOLVERS)}')
    print(f'Transport: {TRANSPORT}/{ALPHA_MODE}, α={ALPHA}')
    print('=' * 70)

    all_results = []
    all_results.extend(run_ctf_phantom())
    all_results.extend(run_4dbti_phantom())

    # Summary tables
    print(f'\n{"=" * 80}')
    print('SUMMARY — Mean localization error (mm)')
    print('=' * 80)

    import pandas as pd
    df = pd.DataFrame(all_results)

    for dataset in ['CTF', '4D-BTi']:
        sub = df[df['dataset'] == dataset]
        if sub.empty:
            continue
        print(f'\n  {dataset}:')
        print(f'  {"Solver":<15} {"Correct":>10} {"Wrong":>10} '
              f'{"Wrong+RT":>10} {"RT effect":>12}')
        print(f'  {"-"*60}')
        for solver in SOLVERS:
            ss = sub[sub['solver'] == solver]
            for cond in ['correct', 'wrong', 'wrong_rt']:
                vals = ss[ss['condition'] == cond]['error_mm']
                if cond == 'correct':
                    c_mean = vals.mean()
                elif cond == 'wrong':
                    w_mean = vals.mean()
                else:
                    rt_mean = vals.mean()
            gap = w_mean - c_mean
            rescue = w_mean - rt_mean
            pct = rescue / gap * 100 if abs(gap) > 0.1 else 0
            print(f'  {solver:<15} {c_mean:>9.1f}  {w_mean:>9.1f}  '
                  f'{rt_mean:>9.1f}  {pct:>+10.0f}% gap')

    # Overall
    print(f'\n  Overall across all datasets:')
    print(f'  {"Solver":<15} {"Correct":>10} {"Wrong":>10} '
          f'{"Wrong+RT":>10}')
    print(f'  {"-"*50}')
    for solver in SOLVERS:
        ss = df[df['solver'] == solver]
        for cond in ['correct', 'wrong', 'wrong_rt']:
            vals = ss[ss['condition'] == cond]['error_mm']
            if cond == 'correct':
                print(f'  {solver:<15} {vals.mean():>9.1f}', end='')
            elif cond == 'wrong':
                print(f'  {vals.mean():>9.1f}', end='')
            else:
                print(f'  {vals.mean():>9.1f}')

    elapsed = time.time() - t_start
    print(f'\nTotal time: {elapsed / 60:.1f} min')


if __name__ == '__main__':
    main()
