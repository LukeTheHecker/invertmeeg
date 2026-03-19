"""Brainstorm auditory: Riemannian transport evaluation on real MEG data.

Compares source reconstruction with and without Riemannian transport (RT)
across four solvers on the Brainstorm auditory oddball dataset.

Since there is no ground truth, we evaluate using anatomical priors:
  1. Auditory cortex concentration — fraction of N100 power in Heschl's
     gyrus + superior temporal gyrus (higher = more plausible)
  2. Peak-to-Heschl distance — distance from peak source to nearest
     Heschl's gyrus vertex in mm (lower = more plausible)
  3. Focality (Gini) — sparsity of source distribution (higher = more focal)
  4. Peak latency — time of max source power (should be ~100ms for N100)

Saves brain plots (lh/rh) for each solver × condition as PNG files.

Usage:
    cd invert-package
    .venv/bin/python scripts/riemannian_transport/eval_brainstorm_auditory.py
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

# ── Parameters ──────────────────────────────────────────────────────────
SOLVERS = ['subspace-sbl', 'champagne', 'eloreta', 'esmv']
ALPHA = 1.0
REG = 0.01
TRANSPORT = 'bures'
ALPHA_MODE = 'geodesic'
N100_WINDOW = (0.08, 0.12)  # seconds — for metric computation
FIT_WINDOW = (0.05, 0.15)  # seconds — restrict inverse fitting to N100 region

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), 'results', 'brainstorm_auditory')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Metrics ─────────────────────────────────────────────────────────────

def gini_coefficient(x):
    """Gini coefficient of array x (0 = uniform, 1 = maximally sparse)."""
    x = np.abs(x).ravel()
    if x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    return (2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum())) - (n + 1) / n


def compute_metrics(stc, labels_aud, src_positions, subject, subjects_dir):
    """Compute plausibility metrics for a source estimate.

    Parameters
    ----------
    stc : mne.SourceEstimate
    labels_aud : dict with 'heschl_lh', 'heschl_rh', 'stg_lh', 'stg_rh'
        Label objects for auditory cortex regions.
    src_positions : ndarray (n_src, 3)
        Source positions in meters.

    Returns
    -------
    dict of metric values
    """
    # Time index for N100 window
    t_mask = (stc.times >= N100_WINDOW[0]) & (stc.times <= N100_WINDOW[1])
    act_n100 = np.abs(stc.data[:, t_mask]).mean(axis=1)  # avg over N100 window

    # Peak latency (time of max global source power)
    gfp = np.sum(stc.data ** 2, axis=0)
    # Search within 0.05-0.20s to avoid edge effects
    search_mask = (stc.times >= 0.05) & (stc.times <= 0.20)
    search_gfp = gfp.copy()
    search_gfp[~search_mask] = 0
    peak_latency = stc.times[np.argmax(search_gfp)]

    # Source vertex indices in label
    verts_lh = stc.vertices[0]
    verts_rh = stc.vertices[1]
    n_lh = len(verts_lh)

    def label_indices(label, verts, offset=0):
        mask = np.isin(verts, label.vertices)
        return np.where(mask)[0] + offset

    idx_heschl = np.concatenate([
        label_indices(labels_aud['heschl_lh'], verts_lh),
        label_indices(labels_aud['heschl_rh'], verts_rh, n_lh),
    ])
    idx_stg = np.concatenate([
        label_indices(labels_aud['stg_lh'], verts_lh),
        label_indices(labels_aud['stg_rh'], verts_rh, n_lh),
    ])
    idx_auditory = np.union1d(idx_heschl, idx_stg)

    # Metric 1: Auditory cortex power concentration
    power_total = np.sum(act_n100 ** 2)
    power_auditory = np.sum(act_n100[idx_auditory] ** 2) if len(idx_auditory) > 0 else 0
    concentration = power_auditory / power_total if power_total > 0 else 0

    # Metric 2: Peak-to-Heschl distance
    peak_idx = np.argmax(act_n100)
    peak_pos = src_positions[peak_idx]
    if len(idx_heschl) > 0:
        heschl_positions = src_positions[idx_heschl]
        dists = np.linalg.norm(heschl_positions - peak_pos, axis=1)
        peak_to_heschl = np.min(dists) * 1000  # mm
    else:
        peak_to_heschl = np.nan

    # Metric 3: Focality (Gini)
    focality = gini_coefficient(act_n100)

    # Metric 4: Peak hemisphere
    peak_hemi = 'lh' if peak_idx < n_lh else 'rh'

    return {
        'peak_latency_ms': peak_latency * 1000,
        'aud_concentration': concentration,
        'peak_to_heschl_mm': peak_to_heschl,
        'focality_gini': focality,
        'peak_hemi': peak_hemi,
    }


def save_brain_plots(stc, subject, subjects_dir, name, output_dir):
    """Save left and right hemisphere brain plots as PNG."""
    import pyvista
    pyvista.OFF_SCREEN = True

    stc_plot = abs(stc)
    stc_plot.subject = subject

    for hemi in ['lh', 'rh']:
        brain = stc_plot.plot(
            subjects_dir=subjects_dir,
            subject=subject,
            surface='inflated',
            hemi=hemi,
            cortex='low_contrast',
            initial_time=0.1,
            time_unit='s',
            time_viewer=False,
            size=(800, 600),
            background='white',
        )
        fname = os.path.join(output_dir, f'{name}_{hemi}.png')
        brain.save_image(fname)
        brain.close()
        print(f'    Saved {fname}')


# ── Preprocessing (follows tutorials/brainstorm_auditory.ipynb) ─────────

def preprocess():
    """Load, preprocess, and return evoked + forward + covariances."""
    data_path = bst_auditory.data_path()
    subject = "bst_auditory"
    subjects_dir = data_path / "subjects"

    raw_fname1 = data_path / "MEG" / subject / "S01_AEF_20131218_01.ds"
    raw_fname2 = data_path / "MEG" / subject / "S01_AEF_20131218_02.ds"
    erm_fname = data_path / "MEG" / subject / "S01_Noise_20131218_01.ds"

    print('Loading raw data...')
    raw = read_raw_ctf(raw_fname1)
    n_times_run1 = raw.n_times
    mne.io.concatenate_raws(
        [raw, read_raw_ctf(raw_fname2)], on_mismatch="ignore")
    raw_erm = read_raw_ctf(erm_fname)
    raw.set_channel_types({"HEOG": "eog", "VEOG": "eog", "ECG": "ecg"})

    # Annotations and artifact projectors
    print('Setting up annotations and projectors...')
    annotations_df = pd.DataFrame()
    for idx in [1, 2]:
        csv_fname = (data_path / "MEG" / "bst_auditory"
                     / f"events_bad_0{idx}.csv")
        df = pd.read_csv(csv_fname, header=None,
                         names=["onset", "duration", "id", "label"])
        df["onset"] += n_times_run1 * (idx - 1)
        annotations_df = pd.concat([annotations_df, df], axis=0)

    saccades_events = df[df["label"] == "saccade"].values[:, :3].astype(int)
    annotations = mne.Annotations(
        annotations_df["onset"].values / raw.info["sfreq"],
        annotations_df["duration"].values / raw.info["sfreq"],
        annotations_df["label"].values,
    )
    raw.set_annotations(annotations)

    saccade_epochs = mne.Epochs(
        raw, saccades_events, 1, 0.0, 0.5, preload=True,
        baseline=(None, None), reject_by_annotation=False)
    projs_saccade = mne.compute_proj_epochs(
        saccade_epochs, n_mag=1, n_eeg=0, desc_prefix="saccade")
    proj_fname = (data_path / "MEG" / "bst_auditory"
                  / "bst_auditory-eog-proj.fif")
    projs_eog = mne.read_proj(proj_fname)[0]
    raw.add_proj(projs_saccade)
    raw.add_proj(projs_eog)
    del saccade_epochs, saccades_events, projs_eog, projs_saccade

    # Mark bad channels
    raw.info["bads"] = [
        "MLO52-4408", "MRT51-4408", "MLO42-4408", "MLO43-4408"]

    # Epoching
    print('Epoching...')
    tmin, tmax = -0.1, 0.5
    event_id = dict(standard=1, deviant=2)
    reject = dict(mag=4e-12, eog=250e-6)

    events = mne.find_events(raw, stim_channel="UPPT001")
    sound_data = raw[raw.ch_names.index("UADC001-4408")][0][0]
    onsets = np.where(np.abs(sound_data) > 2.0 * np.std(sound_data))[0]
    min_diff = int(0.5 * raw.info["sfreq"])
    diffs = np.concatenate([[min_diff + 1], np.diff(onsets)])
    onsets = onsets[diffs > min_diff]
    assert len(onsets) == len(events)
    events[:, 0] = onsets

    epochs = mne.Epochs(
        raw, events, event_id, tmin, tmax,
        picks=["meg", "eog"], baseline=(None, 0),
        reject=reject, preload=False, proj=True)
    epochs.drop_bad()
    epochs.set_annotations(None)

    epochs_standard = mne.concatenate_epochs(
        [epochs["standard"][range(40)], epochs["standard"][182:222]])
    epochs_standard.load_data().resample(600, npad="auto")
    epochs_deviant = epochs["deviant"].load_data()
    epochs_deviant.resample(600, npad="auto")

    evoked_std = epochs_standard.average()
    evoked_dev = epochs_deviant.average()
    for evoked in (evoked_std, evoked_dev):
        evoked.filter(l_freq=None, h_freq=40.0, fir_design="firwin")

    # Noise covariance from empty-room
    print('Computing noise covariance...')
    noise_cov = mne.compute_raw_covariance(raw_erm, reject=dict(mag=4e-12))

    # Data covariance from task recording (for Riemannian transport)
    print('Computing data covariance from task recording...')
    raw_task = raw.copy().pick('meg').apply_proj()
    raw_task.load_data()
    raw_task_info = raw_task.info.copy()
    X = raw_task.get_data().T  # (n_times, n_ch)
    C_data_full = (X.T @ X) / X.shape[0]

    del raw, raw_erm, raw_task, epochs, epochs_standard, epochs_deviant, X

    # Forward model
    print('Building forward model (ico4)...')
    src = mne.setup_source_space(
        subject, spacing="ico4", subjects_dir=subjects_dir, add_dist="patch")
    conductivity = (0.3,)
    model = mne.make_bem_model(
        subject=subject, subjects_dir=subjects_dir,
        conductivity=conductivity)
    bem = mne.make_bem_solution(model)
    fwd = mne.make_forward_solution(
        evoked_dev.info, trans="fsaverage", src=src, bem=bem,
        eeg=False, mindist=5.0)

    # Align channels: intersect fwd channels with raw_task channels
    raw_task_chs = list(raw_task_info['ch_names'])
    fwd_all_ch = [ch['ch_name'] for ch in fwd['info']['chs']]
    common_ch = [ch for ch in fwd_all_ch if ch in raw_task_chs]
    fwd = mne.pick_channels_forward(fwd, common_ch)
    print(f'  Forward picked to {fwd["sol"]["data"].shape[0]} channels')

    # Reorder C_data to match fwd channel order
    fwd_ch = [ch['ch_name'] for ch in fwd['info']['chs']]
    idx_map = [raw_task_chs.index(ch) for ch in fwd_ch]
    C_data = C_data_full[np.ix_(idx_map, idx_map)]
    print(f'  Data covariance: {C_data.shape}')

    return (subject, subjects_dir, evoked_std, evoked_dev, noise_cov, fwd,
            C_data, src)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    print('=' * 80)
    print('BRAINSTORM AUDITORY — RIEMANNIAN TRANSPORT EVALUATION')
    print(f'Solvers: {", ".join(SOLVERS)}')
    print(f'Transport: {TRANSPORT}/{ALPHA_MODE}, α={ALPHA}, reg={REG}')
    print(f'Output: {OUTPUT_DIR}')
    print('=' * 80)

    (subject, subjects_dir, evoked_std, evoked_dev, noise_cov, fwd,
     C_data, src) = preprocess()

    fwd_ch = [ch['ch_name'] for ch in fwd['info']['chs']]
    n_ch = len(fwd_ch)
    n_src = fwd['sol']['data'].shape[1]
    print(f'\nForward: {n_ch} channels, {n_src} sources (free orientation)')

    # Apply Riemannian transport
    print('\nApplying Riemannian transport...')
    fwd_rt = riemannian_leadfield_transport(
        C_data, fwd, alpha=ALPHA, reg=REG,
        transport=TRANSPORT, alpha_mode=ALPHA_MODE)
    print('  Done.')

    # Load auditory cortex labels
    print('Loading anatomical labels...')
    labels = mne.read_labels_from_annot(
        subject, parc='aparc', subjects_dir=subjects_dir, verbose=0)
    labels_aud = {}
    for label in labels:
        if 'transversetemporal' in label.name:
            key = 'heschl_lh' if label.hemi == 'lh' else 'heschl_rh'
            labels_aud[key] = label
        elif 'superiortemporal' in label.name:
            key = 'stg_lh' if label.hemi == 'lh' else 'stg_rh'
            labels_aud[key] = label
    print(f'  Heschl LH: {len(labels_aud.get("heschl_lh", []))} vertices, '
          f'RH: {len(labels_aud.get("heschl_rh", []))} vertices')

    # Source positions for distance computation
    src_positions = np.vstack([
        fwd['src'][0]['rr'][fwd['src'][0]['vertno']],
        fwd['src'][1]['rr'][fwd['src'][1]['vertno']],
    ])

    # Pick evoked and noise_cov to match fwd channels
    evoked_fwd = evoked_std.copy().pick(fwd_ch)
    noise_cov_fwd = noise_cov.copy().pick_channels(
        [ch for ch in fwd_ch if ch in noise_cov.ch_names])

    # Crop evoked to N100 window for fitting — forces sparse solvers to
    # focus on the auditory N100 rather than later components (P200/MMN).
    evoked_fit = evoked_fwd.copy().crop(*FIT_WINDOW)
    print(f'  Fit window: {FIT_WINDOW[0]*1000:.0f}–{FIT_WINDOW[1]*1000:.0f}ms '
          f'({evoked_fit.data.shape[1]} samples)')

    # Run solvers
    results = []
    for solver_name in SOLVERS:
        for condition, fwd_use in [('vanilla', fwd), ('riemannian', fwd_rt)]:
            label = f'{solver_name}_{condition}'
            print(f'\n  Running {label}...')
            t0 = time.time()

            try:
                solver = Solver(solver_name)
                solver.make_inverse_operator(
                    fwd_use, evoked_fit,
                    alpha='auto', noise_cov=noise_cov_fwd, verbose=0)
                stc = solver.apply_inverse_operator(evoked_fwd)
                elapsed = time.time() - t0
                print(f'    Solved in {elapsed:.1f}s')

                # Compute metrics
                metrics = compute_metrics(
                    stc, labels_aud, src_positions, subject, subjects_dir)
                metrics['solver'] = solver_name
                metrics['condition'] = condition
                metrics['time_s'] = elapsed
                results.append(metrics)

                print(f'    Peak latency: {metrics["peak_latency_ms"]:.0f}ms '
                      f'({metrics["peak_hemi"]})')
                print(f'    Aud concentration: '
                      f'{metrics["aud_concentration"]:.1%}')
                print(f'    Peak-to-Heschl: '
                      f'{metrics["peak_to_heschl_mm"]:.1f}mm')
                print(f'    Focality (Gini): '
                      f'{metrics["focality_gini"]:.3f}')

                # Save brain plots
                save_brain_plots(
                    stc, subject, subjects_dir, label, OUTPUT_DIR)

            except Exception as e:
                print(f'    FAILED: {e}')
                results.append({
                    'solver': solver_name,
                    'condition': condition,
                    'peak_latency_ms': np.nan,
                    'aud_concentration': np.nan,
                    'peak_to_heschl_mm': np.nan,
                    'focality_gini': np.nan,
                    'peak_hemi': '?',
                    'time_s': np.nan,
                })

    # Summary table
    print(f'\n{"=" * 80}')
    print('SUMMARY')
    print('=' * 80)
    print(f'\n{"Solver":<15} {"Cond":<12} {"Peak ms":<9} {"Aud %":<8} '
          f'{"→Heschl":<9} {"Gini":<7} {"Hemi":<5}')
    print('-' * 70)
    for r in results:
        print(f'{r["solver"]:<15} {r["condition"]:<12} '
              f'{r["peak_latency_ms"]:>6.0f}   '
              f'{r["aud_concentration"]:>5.1%}   '
              f'{r["peak_to_heschl_mm"]:>6.1f}   '
              f'{r["focality_gini"]:>5.3f}  '
              f'{r["peak_hemi"]}')

    # Comparison: vanilla vs riemannian
    print(f'\n{"=" * 80}')
    print('EFFECT OF RIEMANNIAN TRANSPORT')
    print('=' * 80)
    print(f'\n{"Solver":<15} {"Δ Aud %":<10} {"Δ →Heschl":<11} '
          f'{"Δ Gini":<9} {"Note":<20}')
    print('-' * 70)
    for solver_name in SOLVERS:
        van = next((r for r in results
                    if r['solver'] == solver_name
                    and r['condition'] == 'vanilla'), None)
        riem = next((r for r in results
                     if r['solver'] == solver_name
                     and r['condition'] == 'riemannian'), None)
        if van and riem:
            d_aud = riem['aud_concentration'] - van['aud_concentration']
            d_heschl = riem['peak_to_heschl_mm'] - van['peak_to_heschl_mm']
            d_gini = riem['focality_gini'] - van['focality_gini']
            note = ''
            if d_aud > 0.01:
                note += 'more auditory '
            elif d_aud < -0.01:
                note += 'less auditory '
            if d_heschl < -1:
                note += 'closer to Heschl '
            elif d_heschl > 1:
                note += 'farther from Heschl '
            print(f'{solver_name:<15} {d_aud:>+8.1%}   '
                  f'{d_heschl:>+8.1f}mm  '
                  f'{d_gini:>+7.3f}   '
                  f'{note.strip()}')

    elapsed = time.time() - t_start
    print(f'\nTotal time: {elapsed / 60:.1f} min')
    print(f'Brain plots saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
