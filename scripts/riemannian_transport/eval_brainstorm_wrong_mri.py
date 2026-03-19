"""Brainstorm auditory with WRONG MRI: Riemannian transport rescue test.

Uses real MEG data from the brainstorm auditory dataset but builds the
forward model from fsaverage anatomy instead of the subject's own MRI.
This simulates the common clinical scenario where no individual MRI is
available and a template brain must be used.

Compares three conditions:
  1. Correct MRI  — subject's own anatomy (upper bound)
  2. Wrong MRI    — fsaverage anatomy (baseline)
  3. Wrong MRI+RT — fsaverage anatomy + Riemannian transport (test)

Metrics (no ground truth):
  - Auditory cortex concentration (% power in Heschl + STG at N100)
  - Peak-to-Heschl distance (mm)
  - Focality (Gini coefficient)
  - Peak latency (should be ~100ms)

Usage:
    cd invert-package
    .venv/bin/python scripts/riemannian_transport/eval_brainstorm_wrong_mri.py
"""

import os
import time

import mne
import numpy as np
import pandas as pd
from mne.datasets.brainstorm import bst_auditory
from mne.io import read_raw_ctf

from invert import Solver
from invert.cortical_folding.riemannian_transport import (
    riemannian_leadfield_transport,
)
from invert.forward.forward import _get_fsaverage_dir

# ── Parameters ──────────────────────────────────────────────────────────
SOLVERS = ['subspace-sbl', 'champagne', 'eloreta', 'esmv']
ALPHA = 1.0
REG = 0.01
TRANSPORT = 'bures'
ALPHA_MODE = 'geodesic'
N100_WINDOW = (0.08, 0.12)
FIT_WINDOW = (0.05, 0.15)

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), 'results', 'brainstorm_wrong_mri')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Metrics ─────────────────────────────────────────────────────────────

def gini_coefficient(x):
    x = np.abs(x).ravel()
    if x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    return (2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum())) - (n + 1) / n


def get_auditory_labels(subject, subjects_dir):
    """Load Heschl's gyrus and STG labels for a subject."""
    labels = mne.read_labels_from_annot(
        subject, parc='aparc', subjects_dir=subjects_dir, verbose=0)
    aud = {}
    for label in labels:
        if 'transversetemporal' in label.name:
            aud['heschl_' + label.hemi] = label
        elif 'superiortemporal' in label.name:
            aud['stg_' + label.hemi] = label
    return aud


def compute_metrics(stc, labels_aud, src_positions):
    """Compute plausibility metrics for a source estimate."""
    t_mask = (stc.times >= N100_WINDOW[0]) & (stc.times <= N100_WINDOW[1])
    act_n100 = np.abs(stc.data[:, t_mask]).mean(axis=1)

    # Peak latency (search 50–200ms)
    gfp = np.sum(stc.data ** 2, axis=0)
    search_mask = (stc.times >= 0.05) & (stc.times <= 0.20)
    search_gfp = gfp.copy()
    search_gfp[~search_mask] = 0
    peak_latency = stc.times[np.argmax(search_gfp)]

    verts_lh, verts_rh = stc.vertices[0], stc.vertices[1]
    n_lh = len(verts_lh)

    def label_idx(label, verts, offset=0):
        return np.where(np.isin(verts, label.vertices))[0] + offset

    idx_heschl = np.concatenate([
        label_idx(labels_aud['heschl_lh'], verts_lh),
        label_idx(labels_aud['heschl_rh'], verts_rh, n_lh),
    ])
    idx_stg = np.concatenate([
        label_idx(labels_aud['stg_lh'], verts_lh),
        label_idx(labels_aud['stg_rh'], verts_rh, n_lh),
    ])
    idx_aud = np.union1d(idx_heschl, idx_stg)

    power_total = np.sum(act_n100 ** 2)
    power_aud = np.sum(act_n100[idx_aud] ** 2) if len(idx_aud) > 0 else 0
    concentration = power_aud / power_total if power_total > 0 else 0

    peak_idx = np.argmax(act_n100)
    peak_pos = src_positions[peak_idx]
    if len(idx_heschl) > 0:
        dists = np.linalg.norm(src_positions[idx_heschl] - peak_pos, axis=1)
        peak_to_heschl = np.min(dists) * 1000
    else:
        peak_to_heschl = np.nan

    return {
        'peak_latency_ms': peak_latency * 1000,
        'aud_concentration': concentration,
        'peak_to_heschl_mm': peak_to_heschl,
        'focality_gini': gini_coefficient(act_n100),
        'peak_hemi': 'lh' if peak_idx < n_lh else 'rh',
    }


def save_brain_plots(stc, subject, subjects_dir, name, output_dir):
    """Save left and right hemisphere brain plots."""
    import pyvista
    pyvista.OFF_SCREEN = True

    stc_plot = abs(stc)
    stc_plot.subject = subject
    for hemi in ['lh', 'rh']:
        brain = stc_plot.plot(
            subjects_dir=subjects_dir, subject=subject,
            surface='inflated', hemi=hemi, cortex='low_contrast',
            initial_time=0.1, time_unit='s', time_viewer=False,
            size=(800, 600), background='white')
        fname = os.path.join(output_dir, f'{name}_{hemi}.png')
        brain.save_image(fname)
        brain.close()
        print(f'      Saved {fname}')


# ── Preprocessing ───────────────────────────────────────────────────────

def load_and_preprocess():
    """Load brainstorm auditory data, return evoked + noise cov + data cov."""
    data_path = bst_auditory.data_path()
    subject = "bst_auditory"
    subjects_dir = data_path / "subjects"

    raw = read_raw_ctf(data_path / "MEG" / subject / "S01_AEF_20131218_01.ds")
    n_times_run1 = raw.n_times
    mne.io.concatenate_raws(
        [raw, read_raw_ctf(
            data_path / "MEG" / subject / "S01_AEF_20131218_02.ds")],
        on_mismatch="ignore")
    raw_erm = read_raw_ctf(
        data_path / "MEG" / subject / "S01_Noise_20131218_01.ds")
    raw.set_channel_types({"HEOG": "eog", "VEOG": "eog", "ECG": "ecg"})

    # Annotations + projectors
    annotations_df = pd.DataFrame()
    for idx in [1, 2]:
        csv_fname = (data_path / "MEG" / "bst_auditory"
                     / f"events_bad_0{idx}.csv")
        df = pd.read_csv(csv_fname, header=None,
                         names=["onset", "duration", "id", "label"])
        df["onset"] += n_times_run1 * (idx - 1)
        annotations_df = pd.concat([annotations_df, df], axis=0)

    saccades_events = df[df["label"] == "saccade"].values[:, :3].astype(int)
    raw.set_annotations(mne.Annotations(
        annotations_df["onset"].values / raw.info["sfreq"],
        annotations_df["duration"].values / raw.info["sfreq"],
        annotations_df["label"].values))

    saccade_epochs = mne.Epochs(
        raw, saccades_events, 1, 0.0, 0.5, preload=True,
        baseline=(None, None), reject_by_annotation=False)
    projs_saccade = mne.compute_proj_epochs(
        saccade_epochs, n_mag=1, n_eeg=0, desc_prefix="saccade")
    projs_eog = mne.read_proj(
        data_path / "MEG" / "bst_auditory" / "bst_auditory-eog-proj.fif")[0]
    raw.add_proj(projs_saccade)
    raw.add_proj(projs_eog)
    del saccade_epochs, saccades_events, projs_eog, projs_saccade

    raw.info["bads"] = [
        "MLO52-4408", "MRT51-4408", "MLO42-4408", "MLO43-4408"]

    # Epoching
    events = mne.find_events(raw, stim_channel="UPPT001")
    sound_data = raw[raw.ch_names.index("UADC001-4408")][0][0]
    onsets = np.where(np.abs(sound_data) > 2.0 * np.std(sound_data))[0]
    min_diff = int(0.5 * raw.info["sfreq"])
    diffs = np.concatenate([[min_diff + 1], np.diff(onsets)])
    onsets = onsets[diffs > min_diff]
    assert len(onsets) == len(events)
    events[:, 0] = onsets

    epochs = mne.Epochs(
        raw, events, dict(standard=1, deviant=2), -0.1, 0.5,
        picks=["meg", "eog"], baseline=(None, 0),
        reject=dict(mag=4e-12, eog=250e-6), preload=False, proj=True)
    epochs.drop_bad()
    epochs.set_annotations(None)

    epochs_standard = mne.concatenate_epochs(
        [epochs["standard"][range(40)], epochs["standard"][182:222]])
    epochs_standard.load_data().resample(600, npad="auto")

    evoked_std = epochs_standard.average()
    evoked_std.filter(l_freq=None, h_freq=40.0, fir_design="firwin")

    # Noise covariance
    noise_cov = mne.compute_raw_covariance(raw_erm, reject=dict(mag=4e-12))

    # Data covariance from task recording
    print('Computing data covariance from task recording (720s)...')
    raw_task = raw.copy().pick('meg').apply_proj()
    raw_task.load_data()
    raw_task_ch_names = list(raw_task.info['ch_names'])
    X = raw_task.get_data().T
    C_data_full = (X.T @ X) / X.shape[0]

    del raw, raw_erm, raw_task, epochs, epochs_standard, X

    return (subject, subjects_dir, data_path, evoked_std, noise_cov,
            C_data_full, raw_task_ch_names)


def build_subject_fwd(evoked_info, subject, subjects_dir, data_path):
    """Build forward model with the subject's own MRI."""
    src = mne.setup_source_space(
        subject, spacing="ico4", subjects_dir=subjects_dir,
        add_dist="patch", verbose=0)
    model = mne.make_bem_model(
        subject=subject, subjects_dir=subjects_dir,
        conductivity=(0.3,), verbose=0)
    bem = mne.make_bem_solution(model)
    fwd = mne.make_forward_solution(
        evoked_info, trans="fsaverage", src=src, bem=bem,
        eeg=False, mindist=5.0, verbose=0)
    return fwd


def build_fsaverage_fwd(evoked_info):
    """Build forward model with fsaverage anatomy (wrong MRI)."""
    fs_dir = _get_fsaverage_dir(verbose=0)
    subjects_dir = os.path.dirname(fs_dir)
    src = mne.setup_source_space(
        'fsaverage', spacing="ico4", subjects_dir=subjects_dir,
        add_dist="patch", verbose=0)
    model = mne.make_bem_model(
        subject='fsaverage', subjects_dir=subjects_dir,
        conductivity=(0.3,), verbose=0)
    bem = mne.make_bem_solution(model)
    fwd = mne.make_forward_solution(
        evoked_info, trans="fsaverage", src=src, bem=bem,
        eeg=False, mindist=5.0, verbose=0)
    return fwd, subjects_dir


def align_channels(fwd, C_data_full, raw_task_ch_names):
    """Pick fwd to common channels and reorder C_data to match."""
    fwd_ch = [ch['ch_name'] for ch in fwd['info']['chs']]
    common = [ch for ch in fwd_ch if ch in raw_task_ch_names]
    fwd = mne.pick_channels_forward(fwd, common)
    fwd_ch = [ch['ch_name'] for ch in fwd['info']['chs']]
    idx = [raw_task_ch_names.index(ch) for ch in fwd_ch]
    C_data = C_data_full[np.ix_(idx, idx)]
    return fwd, C_data


def src_positions(fwd):
    """Extract source positions from forward model."""
    return np.vstack([
        fwd['src'][0]['rr'][fwd['src'][0]['vertno']],
        fwd['src'][1]['rr'][fwd['src'][1]['vertno']],
    ])


# ── Main ────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    print('=' * 80)
    print('BRAINSTORM AUDITORY — WRONG MRI + RIEMANNIAN TRANSPORT')
    print(f'Solvers: {", ".join(SOLVERS)}')
    print(f'Transport: {TRANSPORT}/{ALPHA_MODE}, α={ALPHA}, reg={REG}')
    print('=' * 80)

    # ── Load data ───────────────────────────────────────────────────────
    (subject, subjects_dir, data_path, evoked_std, noise_cov,
     C_data_full, raw_task_ch_names) = load_and_preprocess()

    # ── Build forward models ────────────────────────────────────────────
    print('\nBuilding correct forward (subject MRI)...')
    fwd_correct = build_subject_fwd(
        evoked_std.info, subject, subjects_dir, data_path)
    fwd_correct, C_data_correct = align_channels(
        fwd_correct, C_data_full, raw_task_ch_names)
    print(f'  {fwd_correct["sol"]["data"].shape[0]} ch, '
          f'{fwd_correct["sol"]["data"].shape[1]} src')

    print('Building wrong forward (fsaverage MRI)...')
    fwd_wrong, fs_subjects_dir = build_fsaverage_fwd(evoked_std.info)
    fwd_wrong, C_data_wrong = align_channels(
        fwd_wrong, C_data_full, raw_task_ch_names)
    print(f'  {fwd_wrong["sol"]["data"].shape[0]} ch, '
          f'{fwd_wrong["sol"]["data"].shape[1]} src')

    # ── Riemannian transport on wrong forward ───────────────────────────
    print('\nApplying Riemannian transport to wrong forward...')
    fwd_wrong_rt = riemannian_leadfield_transport(
        C_data_wrong, fwd_wrong,
        alpha=ALPHA, reg=REG,
        transport=TRANSPORT, alpha_mode=ALPHA_MODE)
    print('  Done.')

    # ── Labels ──────────────────────────────────────────────────────────
    labels_correct = get_auditory_labels(subject, subjects_dir)
    labels_wrong = get_auditory_labels('fsaverage', fs_subjects_dir)

    # ── Conditions ──────────────────────────────────────────────────────
    conditions = [
        ('correct_mri', fwd_correct, subject, subjects_dir,
         labels_correct, src_positions(fwd_correct)),
        ('wrong_mri', fwd_wrong, 'fsaverage', fs_subjects_dir,
         labels_wrong, src_positions(fwd_wrong)),
        ('wrong_mri_rt', fwd_wrong_rt, 'fsaverage', fs_subjects_dir,
         labels_wrong, src_positions(fwd_wrong)),
    ]

    # ── Run solvers ─────────────────────────────────────────────────────
    results = []

    for solver_name in SOLVERS:
        for cond_name, fwd_use, subj, subj_dir, labels, pos in conditions:
            label = f'{solver_name}_{cond_name}'
            fwd_ch = [ch['ch_name'] for ch in fwd_use['info']['chs']]
            evoked_pick = evoked_std.copy().pick(fwd_ch)
            evoked_fit = evoked_pick.copy().crop(*FIT_WINDOW)
            cov_ch = [ch for ch in fwd_ch if ch in noise_cov.ch_names]
            ncov = noise_cov.copy().pick_channels(cov_ch)

            print(f'\n  {label}...')
            t0 = time.time()

            try:
                solver = Solver(solver_name)
                solver.make_inverse_operator(
                    fwd_use, evoked_fit,
                    alpha='auto', noise_cov=ncov, verbose=0)
                stc = solver.apply_inverse_operator(evoked_pick)
                elapsed = time.time() - t0

                metrics = compute_metrics(stc, labels, pos)
                metrics['solver'] = solver_name
                metrics['condition'] = cond_name
                metrics['time_s'] = elapsed
                results.append(metrics)

                print(f'    {metrics["peak_latency_ms"]:.0f}ms '
                      f'| Aud {metrics["aud_concentration"]:.1%} '
                      f'| →Heschl {metrics["peak_to_heschl_mm"]:.1f}mm '
                      f'| Gini {metrics["focality_gini"]:.3f} '
                      f'| {metrics["peak_hemi"]}')

                save_brain_plots(stc, subj, subj_dir, label, OUTPUT_DIR)

            except Exception as e:
                print(f'    FAILED: {e}')
                import traceback; traceback.print_exc()
                results.append({
                    'solver': solver_name, 'condition': cond_name,
                    'peak_latency_ms': np.nan, 'aud_concentration': np.nan,
                    'peak_to_heschl_mm': np.nan, 'focality_gini': np.nan,
                    'peak_hemi': '?', 'time_s': np.nan,
                })

    # ── Summary ─────────────────────────────────────────────────────────
    print(f'\n{"=" * 90}')
    print('SUMMARY')
    print('=' * 90)
    print(f'\n{"Solver":<15} {"Condition":<15} {"Peak ms":<9} {"Aud %":<8} '
          f'{"→Heschl":<9} {"Gini":<7} {"Hemi":<5}')
    print('-' * 75)
    for r in results:
        print(f'{r["solver"]:<15} {r["condition"]:<15} '
              f'{r["peak_latency_ms"]:>6.0f}   '
              f'{r["aud_concentration"]:>5.1%}   '
              f'{r["peak_to_heschl_mm"]:>6.1f}   '
              f'{r["focality_gini"]:>5.3f}  '
              f'{r["peak_hemi"]}')

    # Comparison table: wrong vs wrong+RT vs correct
    print(f'\n{"=" * 90}')
    print('RT RESCUE EFFECT (wrong MRI → wrong MRI + RT, vs correct MRI)')
    print('=' * 90)
    print(f'\n{"Solver":<15} {"Correct":<12} {"Wrong":<12} '
          f'{"Wrong+RT":<12} {"RT rescued?":<20}')
    print(f'{"":15} {"→Heschl":>10} {"→Heschl":>10} {"→Heschl":>10}')
    print('-' * 75)
    for solver_name in SOLVERS:
        def get(cond):
            return next((r for r in results
                         if r['solver'] == solver_name
                         and r['condition'] == cond), None)
        c, w, rt = get('correct_mri'), get('wrong_mri'), get('wrong_mri_rt')
        if c and w and rt:
            # How much of the wrong→correct gap did RT close?
            gap = w['peak_to_heschl_mm'] - c['peak_to_heschl_mm']
            rescue = w['peak_to_heschl_mm'] - rt['peak_to_heschl_mm']
            pct = rescue / gap * 100 if abs(gap) > 0.1 else 0
            note = f'{pct:+.0f}% gap closed' if abs(gap) > 0.1 else 'no gap'
            print(f'{solver_name:<15} '
                  f'{c["peak_to_heschl_mm"]:>9.1f}mm '
                  f'{w["peak_to_heschl_mm"]:>9.1f}mm '
                  f'{rt["peak_to_heschl_mm"]:>9.1f}mm '
                  f'  {note}')

    # Same for auditory concentration
    print(f'\n{"Solver":<15} {"Correct":<12} {"Wrong":<12} '
          f'{"Wrong+RT":<12} {"RT rescued?":<20}')
    print(f'{"":15} {"Aud %":>10} {"Aud %":>10} {"Aud %":>10}')
    print('-' * 75)
    for solver_name in SOLVERS:
        def get(cond):
            return next((r for r in results
                         if r['solver'] == solver_name
                         and r['condition'] == cond), None)
        c, w, rt = get('correct_mri'), get('wrong_mri'), get('wrong_mri_rt')
        if c and w and rt:
            print(f'{solver_name:<15} '
                  f'{c["aud_concentration"]:>9.1%}  '
                  f'{w["aud_concentration"]:>9.1%}  '
                  f'{rt["aud_concentration"]:>9.1%}')

    elapsed = time.time() - t_start
    print(f'\nTotal time: {elapsed / 60:.1f} min')
    print(f'Brain plots saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
