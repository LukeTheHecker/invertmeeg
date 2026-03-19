"""Evaluate Riemannian leadfield transport as a function of available EEG
data amount and SNR.

Simulates continuous raw EEG using the mixed_cortical_activity scenario
(diffuse 1/f background + sparse patches, realistic noise) at 1 kHz for
different durations (30s, 1min, 10min) and SNR levels (-5, 0, 5, 100 dB).
For each condition, estimates the sensor covariance from the raw data,
applies Riemannian transport, and evaluates inverse accuracy (MLE, EMD)
against a vanilla template baseline.

Uses MNI-aligned distance metrics (Talairach HEAD→MRI→MNI) for
anatomically meaningful localization accuracy.

Usage:
    cd invert-package
    .venv/bin/python scripts/eval_riemannian_data_amount.py
"""

import os
import sys
import time
import importlib.util

import mne
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.covariance import ledoit_wolf

from invert import Solver
from invert.simulate import SimulationConfig, SimulationGenerator
from invert.forward.forward import _get_fsaverage_dir, get_info
from invert.cortical_folding.riemannian_transport import (
    riemannian_leadfield_transport,
)

sys.stdout.reconfigure(line_buffering=True)

# Evaluation imports
import invert as _invert_pkg

_eval_path = os.path.join(
    os.path.dirname(_invert_pkg.__file__), 'evaluate', 'evaluate.py')
_spec = importlib.util.spec_from_file_location(
    'invert.evaluate.evaluate', _eval_path, submodule_search_locations=[])
_eval_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _eval_mod
_spec.loader.exec_module(_eval_mod)
eval_emd = _eval_mod.eval_emd
eval_mean_localization_error = _eval_mod.eval_mean_localization_error


# ── Parameters ──────────────────────────────────────────────────────────
SFREQ = 1000  # Hz
DURATIONS_SEC = [30, 60, 600]  # 30s, 1min, 10min
SNRS_DB = [-5, 0, 5, 100]
SOLVER_NAME = 'subspace-sbl'
N_EVAL_TRIALS = 100
ALPHA = 0.80  # Riemannian blending (updated default)
REG = 0.01
BATCH_TIMEPOINTS = 500  # timepoints per simulation batch (0.5s at 1kHz)
RANDOM_SEED = 42

# ── Paths (set up once for MNI transforms) ──
SAMPLE_PATH = mne.datasets.sample.data_path()
SUBJECTS_DIR_SAMPLE = os.path.join(SAMPLE_PATH, 'subjects')
FS_DIR = _get_fsaverage_dir(verbose=0)
SUBJECTS_DIR_FS = os.path.dirname(FS_DIR)


def build_D_mni(fwd_true, fwd_est):
    """Build MNI-aligned cross-distance matrix via Talairach transforms."""
    trans_s = mne.read_trans(os.path.join(
        SAMPLE_PATH, 'MEG', 'sample', 'sample_audvis_raw-trans.fif'))
    trans_f = mne.read_trans(os.path.join(FS_DIR, 'bem', 'fsaverage-trans.fif'))
    tal_s = mne.read_talxfm('sample', SUBJECTS_DIR_SAMPLE)
    tal_f = mne.read_talxfm('fsaverage', SUBJECTS_DIR_FS)

    def _to_mni(pos, trans, tal):
        T = tal['trans'] @ trans['trans']
        ph = np.column_stack([pos, np.ones(len(pos))])
        return (T @ ph.T).T[:, :3]

    p_true = _to_mni(fwd_true['source_rr'], trans_s, tal_s)
    if fwd_est is fwd_true:
        p_est = p_true
    else:
        p_est = _to_mni(fwd_est['source_rr'], trans_f, tal_f)
    return cdist(p_true, p_est)


def build_forward_models():
    """Build ground truth (sample) and template (fsaverage) forward models."""
    trans_fs = os.path.join(FS_DIR, 'bem', 'fsaverage-trans.fif')
    info = get_info()

    print('Building ground truth (sample subject)...')
    fwd_true = mne.make_forward_solution(
        info,
        trans=os.path.join(SAMPLE_PATH, 'MEG', 'sample',
                           'sample_audvis_raw-trans.fif'),
        src=mne.setup_source_space('sample', spacing='ico3', surface='white',
                                   subjects_dir=SUBJECTS_DIR_SAMPLE,
                                   add_dist=False, verbose=0),
        bem=os.path.join(SUBJECTS_DIR_SAMPLE, 'sample', 'bem',
                         'sample-5120-5120-5120-bem-sol.fif'),
        eeg=True, mindist=0.0, verbose=0,
    )
    fwd_true = mne.convert_forward_solution(
        fwd_true, surf_ori=True, force_fixed=True, verbose=0)

    print('Building template (fsaverage)...')
    fwd_tpl = mne.make_forward_solution(
        info, trans=trans_fs,
        src=mne.setup_source_space('fsaverage', spacing='ico3',
                                   surface='white',
                                   subjects_dir=SUBJECTS_DIR_FS,
                                   add_dist=False, verbose=0),
        bem=os.path.join(FS_DIR, 'bem',
                         'fsaverage-5120-5120-5120-bem-sol.fif'),
        eeg=True, mindist=0.0, verbose=0,
    )
    fwd_tpl = mne.convert_forward_solution(
        fwd_tpl, surf_ori=True, force_fixed=True, verbose=0)

    return fwd_true, fwd_tpl


def simulate_sensor_covariance(fwd_true, duration_sec, snr_db, rng_seed):
    """Estimate sensor covariance from simulated continuous raw EEG.

    Uses the mixed_cortical_activity scenario: diffuse 1/f background +
    sparse cortical patches + realistic sensor noise. This produces
    full-rank covariances even at high SNR because all dipoles contribute
    through the background activity.
    """
    n_ch = fwd_true['sol']['data'].shape[0]
    total_samples = int(duration_sec * SFREQ)
    n_batches = max(1, total_samples // BATCH_TIMEPOINTS)

    # mixed_cortical_activity config from datasets.py, with fixed SNR
    config = SimulationConfig(
        batch_size=1,
        n_sources=(2, 4),
        n_timepoints=BATCH_TIMEPOINTS,
        snr_range=(snr_db, snr_db),
        simulation_mode='mixture',
        background_beta=(1.0, 2.5),
        background_mixture_alpha=(0.10, 0.30),
        amplitude_range=(0.2, 2.0),
        amplitude_distribution='log_uniform',
        correlation_mode='realistic',
        noise_temporal_beta=(0.3, 1.2),
        noise_low_rank_dim=(3, 10),
        source_extent=(3, 20),
        verbose=0,
        random_seed=rng_seed,
    )
    gen = SimulationGenerator(fwd_true, config)

    # Collect all samples for Ledoit-Wolf shrinkage estimation
    samples = []
    for _ in range(n_batches):
        batch = next(gen.generate())
        samples.append(batch.x[0].T)  # (n_tp, n_ch)
    X = np.concatenate(samples, axis=0)  # (n_total, n_ch)

    C, shrinkage = ledoit_wolf(X)
    print(f'    LW shrinkage={shrinkage:.4f} (n={X.shape[0]})')
    return C


def evaluate_fwd(fwd_est, fwd_true, eval_batch, info_mne, solver_name, D):
    """Evaluate a forward model on pre-generated test trials using D_mni."""
    pos_true = fwd_true['source_rr']
    pos_est = fwd_est['source_rr']
    adj_true = mne.spatial_src_adjacency(fwd_true['src'], verbose=0)
    adj_est = mne.spatial_src_adjacency(fwd_est['src'], verbose=0)

    solver = Solver(solver_name)
    mles, emds = [], []
    n_trials = eval_batch.x.shape[0]
    for trial in range(n_trials):
        ev = mne.EvokedArray(eval_batch.x[trial], info_mne, tmin=0)
        try:
            solver.make_inverse_operator(fwd_est, ev, alpha='auto', verbose=0)
            stc = solver.apply_inverse_operator(ev)
            y_true = np.abs(eval_batch.y[trial]).mean(axis=1)
            y_pred = np.abs(stc.data).mean(axis=1)
            mle = eval_mean_localization_error(
                y_true, y_pred, adj_true, adj_est,
                pos_true, pos_est, D)
            emd = eval_emd(D, y_true, y_pred)
            mles.append(mle * 1000)
            emds.append(emd * 1000)
        except Exception:
            pass
    return np.median(mles), np.median(emds)


def main():
    fwd_true, fwd_tpl = build_forward_models()

    info_mne = mne.create_info(
        ch_names=fwd_true['info']['ch_names'], sfreq=250., ch_types='eeg')

    # MNI-aligned distance matrix (anatomically correct cross-subject metric)
    print('\nBuilding MNI-aligned distance matrix...')
    D_mni = build_D_mni(fwd_true, fwd_tpl)

    # Generate evaluation trials (fixed across all conditions)
    print(f'Generating {N_EVAL_TRIALS} evaluation trials...')
    eval_config = SimulationConfig(
        batch_size=N_EVAL_TRIALS, n_sources=2, n_timepoints=200,
        snr_range=(5, 15), verbose=0, random_seed=99,
    )
    eval_batch = next(SimulationGenerator(fwd_true, eval_config).generate())

    # Vanilla baseline (no transport)
    print(f'\nEvaluating vanilla template baseline...')
    mle_vanilla, emd_vanilla = evaluate_fwd(
        fwd_tpl, fwd_true, eval_batch, info_mne, SOLVER_NAME, D_mni)
    print(f'  Vanilla:  MNI MLE={mle_vanilla:.1f} mm  MNI EMD={emd_vanilla:.1f} mm\n')

    # Sweep duration × SNR
    results = {}
    duration_labels = {30: '30s', 60: '1min', 600: '10min'}

    print('=' * 76)
    print(f'{"Duration":<10} {"SNR (dB)":<10} {"MNI MLE":<12} '
          f'{"MNI EMD":<12} {"ΔMLE":<12} {"ΔEMD":<12} {"Time (s)":<10}')
    print('-' * 76)

    for dur in DURATIONS_SEC:
        for snr in SNRS_DB:
            t0 = time.time()

            # Simulate raw EEG and estimate covariance
            C_data = simulate_sensor_covariance(
                fwd_true, dur, snr, rng_seed=RANDOM_SEED + dur + snr)

            # Riemannian transport (LW-shrunk cov → reg=0, use default alpha)
            fwd_riem = riemannian_leadfield_transport(
                C_data, fwd_tpl, alpha=ALPHA, reg=0)

            # Evaluate with MNI-aligned distances
            mle, emd = evaluate_fwd(
                fwd_riem, fwd_true, eval_batch, info_mne, SOLVER_NAME, D_mni)

            elapsed = time.time() - t0
            delta_mle = (1 - mle / mle_vanilla) * 100
            delta_emd = (1 - emd / emd_vanilla) * 100

            results[(dur, snr)] = (mle, emd)
            print(f'{duration_labels[dur]:<10} {snr:<10} {mle:<12.1f} '
                  f'{emd:<12.1f} {delta_mle:+11.1f}% {delta_emd:+11.1f}% '
                  f'{elapsed:<10.0f}')

    # Summary table
    print('\n' + '=' * 72)
    print('SUMMARY')
    print('=' * 72)
    print(f'\nVanilla template:  MNI MLE={mle_vanilla:.1f} mm  '
          f'MNI EMD={emd_vanilla:.1f} mm\n')

    # MLE table
    print('MNI MLE (mm) — rows: duration, cols: SNR (dB)')
    print(f'{"":>10}', end='')
    for snr in SNRS_DB:
        print(f'{snr:>10} dB', end='')
    print()
    for dur in DURATIONS_SEC:
        print(f'{duration_labels[dur]:>10}', end='')
        for snr in SNRS_DB:
            mle, _ = results[(dur, snr)]
            print(f'{mle:>10.1f}  ', end='')
        print()

    # EMD table
    print(f'\nMNI EMD (mm) — rows: duration, cols: SNR (dB)')
    print(f'{"":>10}', end='')
    for snr in SNRS_DB:
        print(f'{snr:>10} dB', end='')
    print()
    for dur in DURATIONS_SEC:
        print(f'{duration_labels[dur]:>10}', end='')
        for snr in SNRS_DB:
            _, emd = results[(dur, snr)]
            print(f'{emd:>10.1f}  ', end='')
        print()

    # Improvement table
    print(f'\nΔ MNI MLE vs vanilla (%) — positive = better')
    print(f'{"":>10}', end='')
    for snr in SNRS_DB:
        print(f'{snr:>10} dB', end='')
    print()
    for dur in DURATIONS_SEC:
        print(f'{duration_labels[dur]:>10}', end='')
        for snr in SNRS_DB:
            mle, _ = results[(dur, snr)]
            delta = (1 - mle / mle_vanilla) * 100
            print(f'{delta:>+10.1f}% ', end='')
        print()


if __name__ == '__main__':
    main()
