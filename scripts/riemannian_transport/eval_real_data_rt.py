"""Riemannian transport on real MEG datasets with wrong MRI.

Tests RT across three MNE datasets where we know the expected source
location. For each dataset:
  1. Correct MRI — subject's own anatomy (upper bound)
  2. Wrong MRI   — fsaverage anatomy (template)
  3. Wrong MRI+RT — fsaverage + Riemannian transport

Datasets:
  - Somatosensory (somato): median nerve stim → contralateral S1 at ~40ms
  - MNE Sample (auditory): left-ear oddball → right auditory cortex at ~93ms
  - SPM Faces: face perception → fusiform/occipital at ~170ms

Brain plots saved per-dataset under figures/{dataset_name}/.

Usage:
    cd invert-package
    .venv/bin/python scripts/riemannian_transport/eval_real_data_rt.py
"""

import os
import time
from pathlib import Path

import mne
import numpy as np
from mne.preprocessing import ICA, create_eog_epochs

from invert import Solver
from invert.cortical_folding.riemannian_transport import (
    riemannian_leadfield_transport,
)
from invert.forward.forward import _get_fsaverage_dir

import pyvista
pyvista.OFF_SCREEN = True

# ── Parameters ──────────────────────────────────────────────────────────
SOLVERS = ['eloreta', 'esmv']
ALPHA = 1.0
REG = 0.01
TRANSPORT = 'bures'
ALPHA_MODE = 'geodesic'

FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')


# ── Shared utilities ────────────────────────────────────────────────────

def build_fsaverage_fwd_meg(info, spacing='ico4'):
    """Build fsaverage forward model for MEG (single-layer BEM)."""
    fs_dir = _get_fsaverage_dir(verbose=0)
    subjects_dir = os.path.dirname(fs_dir)
    src = mne.setup_source_space(
        'fsaverage', spacing=spacing, subjects_dir=subjects_dir,
        add_dist=False, verbose=0)
    model = mne.make_bem_model(
        'fsaverage', subjects_dir=subjects_dir,
        conductivity=(0.3,), verbose=0)
    bem = mne.make_bem_solution(model)
    fwd = mne.make_forward_solution(
        info, trans='fsaverage', src=src, bem=bem,
        eeg=False, mindist=5.0, verbose=0)
    return fwd, subjects_dir


def compute_data_cov(raw, picks_type='meg'):
    """Compute sensor covariance from raw data."""
    raw_copy = raw.copy().pick(picks_type)
    if hasattr(raw_copy, 'apply_proj'):
        raw_copy.apply_proj()
    raw_copy.load_data()
    X = raw_copy.get_data().T
    return (X.T @ X) / X.shape[0], list(raw_copy.ch_names)


def align_fwd_cov(fwd, C_data_full, cov_ch_names):
    """Pick forward to common channels, reorder C_data to match."""
    fwd_ch = [ch['ch_name'] for ch in fwd['info']['chs']]
    common = [ch for ch in fwd_ch if ch in cov_ch_names]
    fwd = mne.pick_channels_forward(fwd, common, verbose=0)
    fwd_ch = [ch['ch_name'] for ch in fwd['info']['chs']]
    idx = [cov_ch_names.index(ch) for ch in fwd_ch]
    return fwd, C_data_full[np.ix_(idx, idx)]


def save_brain_plots(stc, subject, subjects_dir, name, output_dir,
                     peak_time=0.1, clim=None):
    """Save lh and rh brain plots as PNG."""
    stc_plot = abs(stc)
    stc_plot.subject = subject
    plot_kw = dict(
        subjects_dir=subjects_dir, subject=subject,
        surface='inflated', cortex='low_contrast',
        initial_time=peak_time, time_unit='s',
        time_viewer=False, size=(800, 600), background='white')
    if clim is not None:
        plot_kw['clim'] = clim
    for hemi in ['lh', 'rh']:
        brain = stc_plot.plot(hemi=hemi, **plot_kw)
        fname = os.path.join(output_dir, f'{name}_{hemi}.png')
        brain.save_image(fname)
        brain.close()
    print(f'      Saved {name}_{{lh,rh}}.png')


def run_dataset(name, evoked, fwd_correct, fwd_wrong, fwd_wrong_rt,
                noise_cov, subject_correct, subjects_dir_correct,
                fs_subjects_dir, peak_time, fit_window=None):
    """Run solvers on correct/wrong/wrong+RT and save plots."""
    output_dir = os.path.join(FIGURES_DIR, name)
    os.makedirs(output_dir, exist_ok=True)

    conditions = [
        ('correct', fwd_correct, subject_correct, subjects_dir_correct),
        ('wrong', fwd_wrong, 'fsaverage', fs_subjects_dir),
        ('wrong_rt', fwd_wrong_rt, 'fsaverage', fs_subjects_dir),
    ]

    results = []
    for solver_name in SOLVERS:
        for cond_name, fwd, subj, subj_dir in conditions:
            label = f'{solver_name}_{cond_name}'
            fwd_ch = [ch['ch_name'] for ch in fwd['info']['chs']]
            ev = evoked.copy().pick(fwd_ch)

            # Fit on restricted window if specified
            ev_fit = ev.copy().crop(*fit_window) if fit_window else ev

            # Noise cov: pick channels present in both
            cov_ch = [ch for ch in fwd_ch if ch in noise_cov.ch_names]
            ncov = noise_cov.copy().pick_channels(cov_ch)

            print(f'    {label}...', end=' ', flush=True)
            t0 = time.time()
            try:
                solver = Solver(solver_name)
                solver.make_inverse_operator(
                    fwd, ev_fit, alpha='auto', noise_cov=ncov, verbose=0)
                stc = solver.apply_inverse_operator(ev)
                elapsed = time.time() - t0
                print(f'{elapsed:.1f}s')

                save_brain_plots(
                    stc, subj, subj_dir, label, output_dir,
                    peak_time=peak_time)
                results.append((solver_name, cond_name, 'ok'))

            except Exception as e:
                print(f'FAILED: {e}')
                results.append((solver_name, cond_name, str(e)))

    return results


# ── Dataset: Somatosensory ──────────────────────────────────────────────

def run_somato():
    """Somatosensory median nerve stimulation → S1 at ~40ms."""
    print('\n' + '=' * 70)
    print('DATASET: Somatosensory (somato)')
    print('Expected: contralateral sensorimotor cortex at ~40ms')
    print('=' * 70)

    data_path = Path(mne.datasets.somato.data_path(download=False))
    subject = "01"
    raw_fname = data_path / f"sub-{subject}" / "meg" / f"sub-{subject}_task-somato_meg.fif"
    fwd_fname = data_path / "derivatives" / f"sub-{subject}" / f"sub-{subject}_task-somato-fwd.fif"
    subjects_dir = str(data_path / "derivatives" / "freesurfer" / "subjects")

    # Load and preprocess
    raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=0)
    raw.filter(1.0, 40.0, verbose=0)
    events = mne.find_events(raw, verbose=0)
    picks = mne.pick_types(raw.info, meg="grad", eeg=False, exclude="bads")
    epochs = mne.Epochs(raw, events, event_id=1, tmin=-0.2, tmax=0.4,
                        picks=picks, baseline=(None, 0.0), preload=True,
                        verbose=0)
    evoked = epochs.average()

    # Noise cov
    rank = mne.compute_rank(epochs, tol="auto", tol_kind="relative", verbose=0)
    noise_cov = mne.compute_covariance(
        epochs, tmin=-0.2, tmax=0.0, method="empirical", rank=rank, verbose=0)

    # Data covariance for RT
    C_data, cov_chs = compute_data_cov(raw, 'grad')

    # Correct forward
    fwd_correct = mne.read_forward_solution(fwd_fname, verbose=0)
    fwd_correct = mne.pick_types_forward(fwd_correct, meg="grad", eeg=False)

    # Wrong forward (fsaverage)
    # Need MEG info with grad only
    info_grad = mne.pick_info(evoked.info,
                              mne.pick_types(evoked.info, meg='grad'))
    fwd_wrong, fs_subj_dir = build_fsaverage_fwd_meg(info_grad)
    fwd_wrong = mne.pick_types_forward(fwd_wrong, meg="grad", eeg=False)

    # Align channels
    fwd_correct, _ = align_fwd_cov(fwd_correct, C_data, cov_chs)
    fwd_wrong, C_data_aligned = align_fwd_cov(fwd_wrong, C_data, cov_chs)

    # RT
    print('  Applying Riemannian transport...')
    fwd_wrong_rt = riemannian_leadfield_transport(
        C_data_aligned, fwd_wrong,
        alpha=ALPHA, reg=REG, transport=TRANSPORT, alpha_mode=ALPHA_MODE)

    del raw

    return run_dataset(
        'somato', evoked, fwd_correct, fwd_wrong, fwd_wrong_rt,
        noise_cov, subject, subjects_dir, fs_subj_dir,
        peak_time=0.04, fit_window=(0.02, 0.06))


# ── Dataset: MNE Sample (auditory) ─────────────────────────────────────

def run_sample_auditory():
    """MNE Sample left auditory → right auditory cortex at ~93ms."""
    print('\n' + '=' * 70)
    print('DATASET: MNE Sample (Left Auditory)')
    print('Expected: right auditory cortex at ~93ms')
    print('=' * 70)

    data_root = Path(mne.datasets.sample.data_path())
    sample_dir = data_root / "MEG" / "sample"
    subjects_dir = str(data_root / "subjects")

    evoked = mne.read_evokeds(
        sample_dir / "sample_audvis-ave.fif",
        condition="Left Auditory", baseline=(None, 0.0), verbose=0)
    noise_cov = mne.read_cov(
        sample_dir / "sample_audvis-cov.fif", verbose=0)

    # Use MEG only for clean RT
    evoked.pick('meg')

    fwd_correct = mne.read_forward_solution(
        sample_dir / "sample_audvis-meg-eeg-oct-6-fwd.fif", verbose=0)
    fwd_correct = mne.pick_types_forward(
        fwd_correct, meg=True, eeg=False)

    # Data cov from raw
    raw = mne.io.read_raw_fif(
        sample_dir / "sample_audvis_raw.fif", preload=True, verbose=0)
    raw.filter(1.0, 40.0, verbose=0)
    C_data, cov_chs = compute_data_cov(raw, 'meg')

    # Wrong forward
    info_meg = mne.pick_info(evoked.info,
                             mne.pick_types(evoked.info, meg=True))
    fwd_wrong, fs_subj_dir = build_fsaverage_fwd_meg(info_meg)

    # Align
    fwd_correct, _ = align_fwd_cov(fwd_correct, C_data, cov_chs)
    fwd_wrong, C_data_aligned = align_fwd_cov(fwd_wrong, C_data, cov_chs)

    # RT
    print('  Applying Riemannian transport...')
    fwd_wrong_rt = riemannian_leadfield_transport(
        C_data_aligned, fwd_wrong,
        alpha=ALPHA, reg=REG, transport=TRANSPORT, alpha_mode=ALPHA_MODE)

    del raw

    return run_dataset(
        'sample_auditory', evoked, fwd_correct, fwd_wrong, fwd_wrong_rt,
        noise_cov, 'sample', subjects_dir, fs_subj_dir,
        peak_time=0.093, fit_window=(0.05, 0.15))


# ── Dataset: SPM Faces ──────────────────────────────────────────────────

def run_spm_faces():
    """SPM Faces (faces - scrambled) → fusiform at ~170ms."""
    print('\n' + '=' * 70)
    print('DATASET: SPM Faces (faces - scrambled)')
    print('Expected: fusiform/occipitotemporal at ~170ms')
    print('=' * 70)

    data_path = Path(mne.datasets.spm_face.data_path())
    subjects_dir = str(data_path / "subjects")
    spm_path = data_path / "MEG" / "spm"

    raw_fname = spm_path / "SPM_CTF_MEG_example_faces1_3D.ds"
    trans_fname = spm_path / "SPM_CTF_MEG_example_faces1_3D_raw-trans.fif"
    src_fname = data_path / "subjects" / "spm" / "bem" / "spm-oct-6-src.fif"
    bem_fname = (data_path / "subjects" / "spm" / "bem"
                 / "spm-5120-5120-5120-bem-sol.fif")

    # Load and preprocess
    raw = mne.io.read_raw_ctf(raw_fname, preload=True, verbose=0)
    raw.resample(100, verbose=0)
    raw.filter(1.0, None, verbose=0)

    # ICA for EOG
    ica = ICA(n_components=0.95, max_iter="auto", random_state=0)
    ica.fit(raw, reject=dict(mag=5e-12), verbose=0)
    eog_epochs = create_eog_epochs(
        raw, ch_name="MRT31-2908", reject=dict(mag=5e-12), verbose=0)
    eog_inds, _ = ica.find_bads_eog(eog_epochs, ch_name="MRT31-2908",
                                     verbose=0)
    ica.exclude += eog_inds[:1]

    events = mne.find_events(raw, stim_channel="UPPT001", verbose=0)
    epochs = mne.Epochs(
        raw, events, {"faces": 1, "scrambled": 2}, -0.2, 0.6,
        picks="meg", baseline=None, preload=True,
        reject=dict(mag=5e-12), verbose=0)
    ica.apply(epochs, verbose=0)

    evoked_faces = epochs["faces"].average()
    evoked_scrambled = epochs["scrambled"].average()
    contrast = mne.combine_evoked(
        [evoked_scrambled, evoked_faces], weights=[-1, 1])
    contrast.comment = "faces - scrambled"

    noise_cov = mne.compute_covariance(
        epochs, tmax=0.0, method="shrunk", rank=None, verbose=0)

    # Data cov
    C_data, cov_chs = compute_data_cov(raw, 'meg')

    # Correct forward
    fwd_correct = mne.make_forward_solution(
        contrast.info, trans=trans_fname, src=src_fname, bem=bem_fname,
        meg=True, eeg=False, verbose=0)

    # Wrong forward
    info_meg = mne.pick_info(contrast.info,
                             mne.pick_types(contrast.info, meg=True))
    fwd_wrong, fs_subj_dir = build_fsaverage_fwd_meg(info_meg)

    # Align
    fwd_correct, _ = align_fwd_cov(fwd_correct, C_data, cov_chs)
    fwd_wrong, C_data_aligned = align_fwd_cov(fwd_wrong, C_data, cov_chs)

    # RT
    print('  Applying Riemannian transport...')
    fwd_wrong_rt = riemannian_leadfield_transport(
        C_data_aligned, fwd_wrong,
        alpha=ALPHA, reg=REG, transport=TRANSPORT, alpha_mode=ALPHA_MODE)

    del raw

    return run_dataset(
        'spm_faces', contrast, fwd_correct, fwd_wrong, fwd_wrong_rt,
        noise_cov, 'spm', subjects_dir, fs_subj_dir,
        peak_time=0.170, fit_window=(0.12, 0.22))


# ── Main ────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    print('=' * 70)
    print('RIEMANNIAN TRANSPORT — MULTI-DATASET REAL DATA EVALUATION')
    print(f'Solvers: {", ".join(SOLVERS)}')
    print(f'Transport: {TRANSPORT}/{ALPHA_MODE}, α={ALPHA}, reg={REG}')
    print(f'Figures: {FIGURES_DIR}/')
    print('=' * 70)

    all_results = {}

    for name, runner in [
        ('somato', run_somato),
        ('sample_auditory', run_sample_auditory),
        ('spm_faces', run_spm_faces),
    ]:
        try:
            all_results[name] = runner()
        except Exception as e:
            print(f'\n  DATASET {name} FAILED: {e}')
            import traceback; traceback.print_exc()
            all_results[name] = [(s, c, 'dataset_error')
                                  for s in SOLVERS
                                  for c in ['correct', 'wrong', 'wrong_rt']]

    # Summary
    print(f'\n{"=" * 70}')
    print('OVERALL SUMMARY')
    print('=' * 70)
    for dataset, results in all_results.items():
        print(f'\n  {dataset}:')
        for solver, cond, status in results:
            print(f'    {solver:15} {cond:12} {status}')

    elapsed = time.time() - t_start
    print(f'\nTotal time: {elapsed / 60:.1f} min')
    print(f'Figures saved to: {FIGURES_DIR}/')


if __name__ == '__main__':
    main()
