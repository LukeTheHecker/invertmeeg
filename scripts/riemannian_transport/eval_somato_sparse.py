"""Somato dataset: sparse solvers with RT, tracking peak positions.

Uses sparse/dipolar solvers (ECD, SESAME, Subspace-SBL, Champagne) on
the somatosensory dataset. For each solver × condition, reports:
  - Peak vertex MNI coordinate
  - Distance to central sulcus (postcentral gyrus label)
  - Peak activation map saved as brain plot with marker

Three conditions: correct MRI, wrong MRI (fsaverage), wrong MRI + RT.

Usage:
    cd invert-package
    .venv/bin/python scripts/riemannian_transport/eval_somato_sparse.py
"""

import os
import time
import warnings
from pathlib import Path

import mne
import numpy as np

from invert import Solver
from invert.cortical_folding.riemannian_transport import (
    riemannian_leadfield_transport,
)
from invert.forward.forward import _get_fsaverage_dir

warnings.filterwarnings('ignore', category=RuntimeWarning)

import pyvista
pyvista.OFF_SCREEN = True

# ── Parameters ──────────────────────────────────────────────────────────
SOLVERS = ['ecd', 'sesame', 'subspace-sbl', 'champagne']
ALPHA = 0.80
REG = 0.01
TRANSPORT = 'bures'
ALPHA_MODE = 'geodesic'
PEAK_TIME = 0.04  # ~40ms for somatosensory N20m

FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures', 'somato_sparse')
os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Helpers ─────────────────────────────────────────────────────────────

def get_peak_info(stc, src, subjects_dir, subject, peak_time=PEAK_TIME):
    """Get peak vertex position (MNI), hemisphere, and vertex index."""
    t_idx = np.argmin(np.abs(stc.times - peak_time))
    act = np.abs(stc.data[:, t_idx])
    peak_idx = int(np.argmax(act))
    n_lh = len(stc.vertices[0])

    if peak_idx < n_lh:
        hemi = 'lh'
        vtx = stc.vertices[0][peak_idx]
        # Use vertno-indexed position from source space
        lh_vertno = src[0]['vertno']
        src_idx = np.searchsorted(lh_vertno, vtx)
        pos = src[0]['rr'][vtx] if vtx < src[0]['rr'].shape[0] else src[0]['rr'][lh_vertno[min(src_idx, len(lh_vertno)-1)]]
    else:
        hemi = 'rh'
        vtx = stc.vertices[1][peak_idx - n_lh]
        rh_vertno = src[1]['vertno']
        src_idx = np.searchsorted(rh_vertno, vtx)
        pos = src[1]['rr'][vtx] if vtx < src[1]['rr'].shape[0] else src[1]['rr'][rh_vertno[min(src_idx, len(rh_vertno)-1)]]

    return {
        'peak_idx': peak_idx,
        'hemi': hemi,
        'vtx': vtx,
        'pos_m': pos,
        'pos_mm': pos * 1000,
    }


def dist_to_label(peak_info, label, src):
    """Euclidean distance from peak to nearest vertex in label (mm)."""
    src_idx = 0 if label.hemi == 'lh' else 1
    rr = src[src_idx]['rr']
    # Only use label vertices that are within the rr array
    valid = label.vertices[label.vertices < rr.shape[0]]
    if len(valid) == 0:
        return np.nan
    label_pos = rr[valid]
    dists = np.linalg.norm(label_pos - peak_info['pos_m'], axis=1)
    return np.min(dists) * 1000


def save_peak_plot(stc, subject, subjects_dir, peak_info, name, output_dir):
    """Save brain plot with peak marker, viewed from the peak hemisphere."""
    stc_plot = abs(stc)
    stc_plot.subject = subject
    hemi = peak_info['hemi']

    brain = stc_plot.plot(
        subjects_dir=subjects_dir, subject=subject,
        surface='inflated', hemi=hemi, cortex='low_contrast',
        initial_time=PEAK_TIME, time_unit='s',
        time_viewer=False, size=(800, 600), background='white')
    brain.add_foci(
        [peak_info['vtx']], coords_as_verts=True,
        map_surface='inflated',
        scale_factor=0.5, color='springgreen',
        alpha=0.9, hemi=hemi)
    fname = os.path.join(output_dir, f'{name}_{hemi}.png')
    brain.save_image(fname)
    brain.close()

    # Also save the other hemisphere
    other = 'rh' if hemi == 'lh' else 'lh'
    brain2 = stc_plot.plot(
        subjects_dir=subjects_dir, subject=subject,
        surface='inflated', hemi=other, cortex='low_contrast',
        initial_time=PEAK_TIME, time_unit='s',
        time_viewer=False, size=(800, 600), background='white')
    fname2 = os.path.join(output_dir, f'{name}_{other}.png')
    brain2.save_image(fname2)
    brain2.close()

    print(f'      Saved {name}_{{{hemi},{other}}}.png')


# ── Main ────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    print('=' * 75)
    print('SOMATO — SPARSE SOLVERS + RIEMANNIAN TRANSPORT')
    print(f'Solvers: {", ".join(SOLVERS)}')
    print(f'Transport: {TRANSPORT}/{ALPHA_MODE}, α={ALPHA}')
    print(f'Expected: contralateral (RH) postcentral gyrus at ~40ms')
    print('=' * 75)

    # ── Load data ───────────────────────────────────────────────────────
    data_path = Path(mne.datasets.somato.data_path(download=False))
    subject = "01"
    subjects_dir = str(data_path / "derivatives" / "freesurfer" / "subjects")
    raw_fname = data_path / "sub-01" / "meg" / "sub-01_task-somato_meg.fif"
    fwd_fname = data_path / "derivatives" / "sub-01" / "sub-01_task-somato-fwd.fif"

    print('\nLoading and preprocessing...')
    raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=0)
    raw.filter(1.0, 40.0, verbose=0)
    events = mne.find_events(raw, verbose=0)

    picks = mne.pick_types(raw.info, meg="grad", eeg=False, exclude="bads")
    epochs = mne.Epochs(raw, events, event_id=1, tmin=-0.2, tmax=0.4,
                        picks=picks, baseline=(None, 0.0), preload=True,
                        verbose=0)
    evoked = epochs.average()
    # Crop to fit window for sparse solvers
    evoked_fit = evoked.copy().crop(0.02, 0.06)

    noise_cov = mne.compute_covariance(
        epochs, tmin=-0.2, tmax=0.0, method="empirical", verbose=0)

    # Data covariance for RT
    print('Computing data covariance...')
    raw_meg = raw.copy().pick('grad')
    raw_meg.load_data()
    X = raw_meg.get_data().T
    C_data = (X.T @ X) / X.shape[0]
    cov_chs = list(raw_meg.ch_names)
    del raw_meg

    # ── Forward models ──────────────────────────────────────────────────
    print('Loading correct forward (subject MRI)...')
    fwd_correct = mne.read_forward_solution(fwd_fname, verbose=0)
    fwd_correct = mne.pick_types_forward(fwd_correct, meg="grad", eeg=False)

    print('Building wrong forward (fsaverage)...')
    fs_dir = _get_fsaverage_dir(verbose=0)
    fs_subjects_dir = os.path.dirname(fs_dir)
    info_grad = mne.pick_info(evoked.info,
                              mne.pick_types(evoked.info, meg='grad'))
    src_fs = mne.setup_source_space(
        'fsaverage', spacing='oct6', subjects_dir=fs_subjects_dir,
        add_dist=False, verbose=0)
    model_fs = mne.make_bem_model(
        'fsaverage', subjects_dir=fs_subjects_dir,
        conductivity=(0.3,), verbose=0)
    bem_fs = mne.make_bem_solution(model_fs)
    fwd_wrong = mne.make_forward_solution(
        info_grad, trans='fsaverage', src=src_fs, bem=bem_fs,
        eeg=False, mindist=5.0, verbose=0)

    # Align C_data to fwd_wrong channels
    fwd_ch = [ch['ch_name'] for ch in fwd_wrong['info']['chs']]
    common = [ch for ch in fwd_ch if ch in cov_chs]
    fwd_wrong = mne.pick_channels_forward(fwd_wrong, common, verbose=0)
    fwd_ch = [ch['ch_name'] for ch in fwd_wrong['info']['chs']]
    idx = [cov_chs.index(ch) for ch in fwd_ch]
    C_aligned = C_data[np.ix_(idx, idx)]

    # Also pick correct fwd to same channels
    fwd_correct = mne.pick_channels_forward(fwd_correct, fwd_ch, verbose=0)

    print('Applying Riemannian transport...')
    fwd_wrong_rt = riemannian_leadfield_transport(
        C_aligned, fwd_wrong,
        alpha=ALPHA, reg=REG, transport=TRANSPORT, alpha_mode=ALPHA_MODE)

    # ── Labels for S1 ──────────────────────────────────────────────────
    # Postcentral gyrus = primary somatosensory cortex
    labels_correct = mne.read_labels_from_annot(
        subject, parc='aparc', subjects_dir=subjects_dir, verbose=0)
    postcentral_correct = {
        l.hemi: l for l in labels_correct if 'postcentral' in l.name}

    labels_fs = mne.read_labels_from_annot(
        'fsaverage', parc='aparc', subjects_dir=fs_subjects_dir, verbose=0)
    postcentral_fs = {
        l.hemi: l for l in labels_fs if 'postcentral' in l.name}

    del raw

    # ── Run solvers ─────────────────────────────────────────────────────
    conditions = [
        ('correct', fwd_correct, subject, subjects_dir, postcentral_correct),
        ('wrong', fwd_wrong, 'fsaverage', fs_subjects_dir, postcentral_fs),
        ('wrong_rt', fwd_wrong_rt, 'fsaverage', fs_subjects_dir, postcentral_fs),
    ]

    results = []
    for solver_name in SOLVERS:
        print(f'\n  === {solver_name} ===')
        for cond_name, fwd, subj, subj_dir, postcen in conditions:
            label = f'{solver_name}_{cond_name}'
            fwd_ch_use = [ch['ch_name'] for ch in fwd['info']['chs']]
            ev = evoked.copy().pick(fwd_ch_use)
            ev_fit = ev.copy().crop(0.02, 0.06)

            cov_ch = [ch for ch in fwd_ch_use if ch in noise_cov.ch_names]
            ncov = noise_cov.copy().pick_channels(cov_ch)

            print(f'    {cond_name}...', end=' ', flush=True)
            try:
                solver = Solver(solver_name)
                solver.make_inverse_operator(
                    fwd, ev_fit, alpha='auto', noise_cov=ncov, verbose=0)
                stc = solver.apply_inverse_operator(ev)

                peak = get_peak_info(stc, fwd['src'], subj_dir, subj)

                # Distance to postcentral gyrus
                if peak['hemi'] in postcen:
                    d_s1 = dist_to_label(peak, postcen[peak['hemi']], fwd['src'])
                else:
                    # Peak on wrong hemisphere — compute distance to RH S1
                    d_s1 = np.nan

                results.append({
                    'solver': solver_name,
                    'condition': cond_name,
                    'hemi': peak['hemi'],
                    'pos_mm': peak['pos_mm'],
                    'dist_to_S1_mm': d_s1,
                })

                print(f'{peak["hemi"]} '
                      f'[{peak["pos_mm"][0]:+.1f}, {peak["pos_mm"][1]:+.1f}, '
                      f'{peak["pos_mm"][2]:+.1f}]mm  '
                      f'→S1: {d_s1:.1f}mm')

                save_peak_plot(stc, subj, subj_dir, peak, label, FIGURES_DIR)

            except Exception as e:
                print(f'FAILED: {e}')
                import traceback; traceback.print_exc()
                results.append({
                    'solver': solver_name,
                    'condition': cond_name,
                    'hemi': '?',
                    'pos_mm': np.array([np.nan]*3),
                    'dist_to_S1_mm': np.nan,
                })

    # ── Summary ─────────────────────────────────────────────────────────
    print(f'\n{"=" * 75}')
    print('SUMMARY — Peak positions and distance to postcentral gyrus (S1)')
    print('Expected: RH postcentral gyrus (contralateral to left median nerve)')
    print('=' * 75)
    print(f'\n{"Solver":<15} {"Cond":<12} {"Hemi":<5} '
          f'{"Peak position (mm)":<28} {"→S1 (mm)":<10}')
    print('-' * 75)
    for r in results:
        p = r['pos_mm']
        pos_str = f'[{p[0]:+6.1f}, {p[1]:+6.1f}, {p[2]:+6.1f}]'
        d_str = f'{r["dist_to_S1_mm"]:.1f}' if not np.isnan(r['dist_to_S1_mm']) else 'n/a'
        correct_hemi = '*' if r['hemi'] == 'rh' else ' '
        print(f'{r["solver"]:<15} {r["condition"]:<12} '
              f'{r["hemi"]}{correct_hemi}   {pos_str:<28} {d_str:<10}')

    # RT effect table
    print(f'\n{"Solver":<15} {"→S1 correct":<14} {"→S1 wrong":<14} '
          f'{"→S1 wrong+RT":<14} {"RT helped?":<12}')
    print('-' * 70)
    for solver_name in SOLVERS:
        sr = [r for r in results if r['solver'] == solver_name]
        c = next((r for r in sr if r['condition'] == 'correct'), None)
        w = next((r for r in sr if r['condition'] == 'wrong'), None)
        rt = next((r for r in sr if r['condition'] == 'wrong_rt'), None)
        if c and w and rt:
            c_d = c['dist_to_S1_mm']
            w_d = w['dist_to_S1_mm']
            rt_d = rt['dist_to_S1_mm']
            helped = 'YES' if rt_d < w_d else 'no'
            # Also check hemisphere
            hemi_note = ''
            if c['hemi'] != w['hemi']:
                hemi_note += f' (wrong→{w["hemi"]})'
            if rt['hemi'] != w['hemi'] and rt['hemi'] == c['hemi']:
                hemi_note += f' RT fixed hemi!'
            print(f'{solver_name:<15} {c_d:>10.1f}mm  {w_d:>10.1f}mm  '
                  f'{rt_d:>10.1f}mm   {helped}{hemi_note}')

    elapsed = time.time() - t_start
    print(f'\nTotal time: {elapsed / 60:.1f} min')
    print(f'Figures: {FIGURES_DIR}/')


if __name__ == '__main__':
    main()
