"""Evaluate Riemannian transport across three geometry-mismatch scenarios.

Scenario 1 — Optimal (GM == IM):
    Perfect geometry.  Transport should be near-identity.
    Prediction: slight degradation (overfits sensor noise).

Scenario 2 — Sub-optimal (GM ≈ IM):
    Same cortical anatomy, but wrong skull conductivity and/or co-registration.
    Prediction: large improvement ("auto-calibrator" for global errors).

Scenario 3 — Mismatch (GM ≠ IM):
    Template brain (fsaverage) for inversion.
    Prediction: solid improvement, limited by high-freq cortical mismatch.

Usage:
    cd invert-package
    .venv/bin/python scripts/eval_riemannian_scenarios.py
"""

import os
import sys
import time
import importlib.util

import mne
import numpy as np
from scipy.spatial.distance import cdist

from invert import Solver
from invert.simulate import SimulationConfig, SimulationGenerator
from invert.forward.forward import _get_fsaverage_dir, get_info
from invert.cortical_folding.riemannian_transport import (
    riemannian_leadfield_transport,
)
from invert.cortical_folding.fast_bem import (
    make_bem_solution_factored,
    rebuild_bem_from_coeff,
)

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
SOLVER_NAME = 'subspace-sbl'
N_EVAL_TRIALS = 30
COV_DURATION_SEC = 60       # 1 min of simulated EEG for covariance
COV_SNR_DB = 5              # moderate clinical SNR
SFREQ = 1000
BATCH_TIMEPOINTS = 500
RANDOM_SEED = 42

# Transport parameters (updated after MNI-aligned evaluation 2026-03-19)
ALPHA = 1.0
REG = 0.01
TRANSPORT = 'bures'
ALPHA_MODE = 'geodesic'

# Scenario 2 parameters
TRUE_SIGMA = 0.006          # true skull conductivity (S/m)
WRONG_SIGMAS = [0.012, 0.020, 0.040]  # typical mis-specifications
COREG_SHIFTS_MM = [5, 10, 15]  # co-registration errors (mm)


# ── Infrastructure ──────────────────────────────────────────────────────

def build_sample_fwd(info, subjects_dir, trans_path, bem_sol_path):
    """Build ground truth forward from pre-computed sample BEM."""
    src = mne.setup_source_space(
        'sample', spacing='ico3', surface='white',
        subjects_dir=subjects_dir, add_dist=False, verbose=0)
    fwd = mne.make_forward_solution(
        info, trans=trans_path, src=src, bem=bem_sol_path,
        eeg=True, mindist=0.0, verbose=0)
    return mne.convert_forward_solution(
        fwd, surf_ori=True, force_fixed=True, verbose=0)


def build_sample_fwd_with_bem(info, subjects_dir, trans_path, bem, src):
    """Build forward from an in-memory BEM solution + source space."""
    fwd = mne.make_forward_solution(
        info, trans=trans_path, src=src, bem=bem,
        eeg=True, mindist=0.0, verbose=0)
    return mne.convert_forward_solution(
        fwd, surf_ori=True, force_fixed=True, verbose=0)


def build_fsaverage_fwd(info):
    """Build template (fsaverage) forward model."""
    fs_dir = _get_fsaverage_dir(verbose=0)
    subjects_dir = os.path.dirname(fs_dir)
    trans = os.path.join(fs_dir, 'bem', 'fsaverage-trans.fif')
    src = mne.setup_source_space(
        'fsaverage', spacing='ico3', surface='white',
        subjects_dir=subjects_dir, add_dist=False, verbose=0)
    fwd = mne.make_forward_solution(
        info, trans=trans, src=src,
        bem=os.path.join(fs_dir, 'bem',
                         'fsaverage-5120-5120-5120-bem-sol.fif'),
        eeg=True, mindist=0.0, verbose=0)
    return mne.convert_forward_solution(
        fwd, surf_ori=True, force_fixed=True, verbose=0)


def simulate_sensor_covariance(fwd, duration_sec, snr_db, rng_seed):
    """Estimate sensor covariance from simulated raw EEG."""
    total_samples = int(duration_sec * SFREQ)
    n_batches = max(1, total_samples // BATCH_TIMEPOINTS)
    config = SimulationConfig(
        batch_size=1, n_sources=(2, 4), n_timepoints=BATCH_TIMEPOINTS,
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
        verbose=0, random_seed=rng_seed,
    )
    gen = SimulationGenerator(fwd, config)
    samples = []
    for _ in range(n_batches):
        batch = next(gen.generate())
        samples.append(batch.x[0].T)
    X = np.concatenate(samples, axis=0)
    return (X.T @ X) / X.shape[0]


def evaluate_fwd(fwd_inv, fwd_true, eval_batch, info_mne, eval_ctx):
    """Evaluate inverse accuracy. Returns (median_mle_mm, median_emd_mm)."""
    solver = Solver(SOLVER_NAME)
    mles, emds = [], []
    for trial in range(eval_batch.x.shape[0]):
        ev = mne.EvokedArray(eval_batch.x[trial], info_mne, tmin=0)
        try:
            solver.make_inverse_operator(fwd_inv, ev, alpha='auto', verbose=0)
            stc = solver.apply_inverse_operator(ev)
            y_true = np.abs(eval_batch.y[trial]).mean(axis=1)
            y_pred = np.abs(stc.data).mean(axis=1)
            mle = eval_mean_localization_error(
                y_true, y_pred,
                eval_ctx['adj_true'], eval_ctx['adj_est'],
                eval_ctx['pos_true'], eval_ctx['pos_est'],
                eval_ctx['D_cross'])
            emd = eval_emd(eval_ctx['D_cross'], y_true, y_pred)
            mles.append(mle * 1000)
            emds.append(emd * 1000)
        except Exception:
            pass
    return np.median(mles), np.median(emds)


def apply_transport(C_data, fwd_inv):
    """Apply Riemannian transport to a forward model."""
    return riemannian_leadfield_transport(
        C_data, fwd_inv,
        alpha=ALPHA, reg=REG,
        transport=TRANSPORT, alpha_mode=ALPHA_MODE)


def make_eval_ctx(fwd_true, fwd_inv):
    """Pre-compute evaluation context (positions, adjacency, distances)."""
    pos_true = fwd_true['source_rr']
    pos_est = fwd_inv['source_rr']
    return {
        'pos_true': pos_true,
        'pos_est': pos_est,
        'adj_true': mne.spatial_src_adjacency(fwd_true['src'], verbose=0),
        'adj_est': mne.spatial_src_adjacency(fwd_inv['src'], verbose=0),
        'D_cross': cdist(pos_true, pos_est),
    }


def make_shifted_trans(trans_path, shift_mm):
    """Load a trans and add a random direction shift (reproducible)."""
    trans = mne.read_trans(trans_path)
    rng = np.random.RandomState(12345)
    direction = rng.randn(3)
    direction /= np.linalg.norm(direction)
    trans['trans'][:3, 3] += direction * (shift_mm / 1000.0)
    return trans


def run_condition(label, fwd_inv, fwd_true, eval_batch, info_mne,
                  C_data, eval_ctx):
    """Run vanilla + riemannian for one condition, print results."""
    # Vanilla
    mle_v, emd_v = evaluate_fwd(fwd_inv, fwd_true, eval_batch, info_mne,
                                eval_ctx)
    # Riemannian
    fwd_riem = apply_transport(C_data, fwd_inv)
    mle_r, emd_r = evaluate_fwd(fwd_riem, fwd_true, eval_batch, info_mne,
                                eval_ctx)
    delta_mle = (1 - mle_r / mle_v) * 100 if mle_v > 0 else 0
    delta_emd = (1 - emd_r / emd_v) * 100 if emd_v > 0 else 0
    print(f'  {label}')
    print(f'    Vanilla:    MLE={mle_v:6.1f} mm  EMD={emd_v:6.1f} mm')
    print(f'    Riemannian: MLE={mle_r:6.1f} mm  EMD={emd_r:6.1f} mm  '
          f'(ΔMLE={delta_mle:+.1f}%  ΔEMD={delta_emd:+.1f}%)')
    return label, mle_v, emd_v, mle_r, emd_r


# ── Main ────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    print('=' * 80)
    print('RIEMANNIAN TRANSPORT — THREE SCENARIO EVALUATION')
    print(f'Solver: {SOLVER_NAME}  |  Transport: {TRANSPORT}/{ALPHA_MODE}  '
          f'|  α={ALPHA}  |  reg={REG}')
    print(f'Covariance: {COV_DURATION_SEC}s at SNR={COV_SNR_DB}dB  '
          f'|  Eval trials: {N_EVAL_TRIALS}')
    print('=' * 80)

    # ── Setup ───────────────────────────────────────────────────────────
    sample_path = mne.datasets.sample.data_path()
    subjects_dir_sample = os.path.join(sample_path, 'subjects')
    trans_sample = os.path.join(
        sample_path, 'MEG', 'sample', 'sample_audvis_raw-trans.fif')
    bem_sol_sample = os.path.join(
        subjects_dir_sample, 'sample', 'bem',
        'sample-5120-5120-5120-bem-sol.fif')
    info = get_info()

    print('\nBuilding ground truth forward (sample subject)...')
    fwd_true = build_sample_fwd(info, subjects_dir_sample, trans_sample,
                                 bem_sol_sample)
    n_ch = fwd_true['sol']['data'].shape[0]
    n_dip = fwd_true['sol']['data'].shape[1]
    print(f'  {n_ch} channels, {n_dip} dipoles')

    info_mne = mne.create_info(
        ch_names=fwd_true['info']['ch_names'], sfreq=250., ch_types='eeg')

    print(f'Generating {N_EVAL_TRIALS} evaluation trials...')
    eval_config = SimulationConfig(
        batch_size=N_EVAL_TRIALS, n_sources=2, n_timepoints=200,
        snr_range=(5, 15), verbose=0, random_seed=99)
    eval_batch = next(SimulationGenerator(fwd_true, eval_config).generate())

    print(f'Simulating {COV_DURATION_SEC}s of EEG for covariance...')
    C_data = simulate_sensor_covariance(
        fwd_true, COV_DURATION_SEC, COV_SNR_DB, RANDOM_SEED)

    all_results = []

    # ── Scenario 1: Optimal (GM == IM) ──────────────────────────────────
    print(f'\n{"=" * 80}')
    print('SCENARIO 1: OPTIMAL (GM == IM)')
    print('Perfect geometry — expect slight degradation from noise overfitting')
    print('=' * 80)

    eval_ctx_same = make_eval_ctx(fwd_true, fwd_true)
    all_results.append(run_condition(
        'Perfect geometry', fwd_true, fwd_true,
        eval_batch, info_mne, C_data, eval_ctx_same))

    # ── Scenario 2: Sub-optimal (GM ≈ IM) ──────────────────────────────
    print(f'\n{"=" * 80}')
    print('SCENARIO 2: SUB-OPTIMAL (GM ≈ IM)')
    print(f'Same cortical anatomy, wrong calibration (true σ_skull={TRUE_SIGMA})')
    print('=' * 80)

    # Build factored BEM for sample subject (to sweep conductivity)
    print('\n  Building factored BEM for sample subject...')
    t0 = time.time()
    bem_model = mne.make_bem_model(
        'sample', ico=4, conductivity=(0.3, TRUE_SIGMA, 0.3),
        subjects_dir=subjects_dir_sample, verbose=0)
    bem, coeff, coeff_h = make_bem_solution_factored(bem_model)
    src_sample = mne.setup_source_space(
        'sample', spacing='ico3', surface='white',
        subjects_dir=subjects_dir_sample, add_dist=False, verbose=0)
    print(f'  Done ({time.time()-t0:.1f}s)')

    # 2a: Wrong conductivity only
    print(f'\n  --- 2a: Wrong skull conductivity ---')
    for sigma in WRONG_SIGMAS:
        rebuild_bem_from_coeff(bem, coeff, coeff_h, (0.3, sigma, 0.3))
        fwd_wrong_sigma = build_sample_fwd_with_bem(
            info, subjects_dir_sample, trans_sample, bem, src_sample)
        all_results.append(run_condition(
            f'σ_skull={sigma:.3f} (true={TRUE_SIGMA})',
            fwd_wrong_sigma, fwd_true, eval_batch, info_mne,
            C_data, eval_ctx_same))

    # 2b: Wrong co-registration only
    print(f'\n  --- 2b: Wrong co-registration ---')
    # Restore true conductivity for coreg tests
    rebuild_bem_from_coeff(bem, coeff, coeff_h, (0.3, TRUE_SIGMA, 0.3))
    for shift in COREG_SHIFTS_MM:
        trans_shifted = make_shifted_trans(trans_sample, shift)
        fwd_shifted = build_sample_fwd_with_bem(
            info, subjects_dir_sample, trans_shifted, bem, src_sample)
        all_results.append(run_condition(
            f'Coreg shift={shift}mm',
            fwd_shifted, fwd_true, eval_batch, info_mne,
            C_data, eval_ctx_same))

    # 2c: Combined (wrong sigma + wrong coreg)
    print(f'\n  --- 2c: Combined errors ---')
    for sigma in [0.012, 0.020]:
        rebuild_bem_from_coeff(bem, coeff, coeff_h, (0.3, sigma, 0.3))
        trans_shifted = make_shifted_trans(trans_sample, 10)
        fwd_combined = build_sample_fwd_with_bem(
            info, subjects_dir_sample, trans_shifted, bem, src_sample)
        all_results.append(run_condition(
            f'σ={sigma:.3f} + 10mm shift',
            fwd_combined, fwd_true, eval_batch, info_mne,
            C_data, eval_ctx_same))

    # ── Scenario 3: Mismatch (GM ≠ IM) ─────────────────────────────────
    print(f'\n{"=" * 80}')
    print('SCENARIO 3: MISMATCH (GM ≠ IM)')
    print('Template brain (fsaverage) for inversion')
    print('=' * 80)

    print('\n  Building fsaverage forward...')
    fwd_tpl = build_fsaverage_fwd(info)
    eval_ctx_cross = make_eval_ctx(fwd_true, fwd_tpl)
    all_results.append(run_condition(
        'fsaverage template', fwd_tpl, fwd_true,
        eval_batch, info_mne, C_data, eval_ctx_cross))

    # ── Summary ─────────────────────────────────────────────────────────
    print(f'\n{"=" * 80}')
    print('SUMMARY')
    print('=' * 80)
    print(f'\n{"Condition":<35} {"Van MLE":>8} {"Riem MLE":>9} '
          f'{"ΔMLE":>7} {"Van EMD":>8} {"Riem EMD":>9} {"ΔEMD":>7}')
    print('-' * 90)
    for label, mle_v, emd_v, mle_r, emd_r in all_results:
        d_mle = (1 - mle_r / mle_v) * 100 if mle_v > 0 else 0
        d_emd = (1 - emd_r / emd_v) * 100 if emd_v > 0 else 0
        print(f'{label:<35} {mle_v:>7.1f}  {mle_r:>8.1f}  {d_mle:>+6.1f}% '
              f'{emd_v:>7.1f}  {emd_r:>8.1f}  {d_emd:>+6.1f}%')

    elapsed = time.time() - t_start
    print(f'\nTotal time: {elapsed/60:.1f} min')


if __name__ == '__main__':
    main()
