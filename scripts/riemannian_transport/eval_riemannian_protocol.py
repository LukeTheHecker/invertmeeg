"""Systematic optimization of Riemannian leadfield transport protocol.

Sweeps (alpha, reg_strategy, transport_method) across 12 data conditions
(duration × SNR) to find the protocol that is most robust across realistic
clinical scenarios.

Evaluates with both D_head (raw head coordinates) and D_mni (anatomically
aligned via Talairach) distance metrics.

What we optimize:
  - alpha ∈ {0.25, 0.50, 0.75, 1.0}
  - reg_strategy ∈ {fixed(r) for r in [0.001, 0.01, 0.05, 0.1, 0.5],
                     ledoit_wolf, oracle_adaptive}
  - transport ∈ {whiten_recolor+blend, bures+geodesic}

What we evaluate robustness over:
  - SNR ∈ {-5, 0, 5, 100} dB
  - duration ∈ {30s, 60s, 600s}

Usage:
    cd invert-package
    .venv/bin/python scripts/eval_riemannian_protocol.py
"""

import os
import sys
import time
import importlib.util

import mne
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from sklearn.covariance import ledoit_wolf

from invert import Solver
from invert.simulate import SimulationConfig, SimulationGenerator
from invert.forward.forward import _get_fsaverage_dir, get_info
from invert.cortical_folding.riemannian_transport import (
    riemannian_leadfield_transport,
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
SFREQ = 1000
DURATIONS_SEC = [30, 60, 600]
SNRS_DB = [-5, 0, 5, 100]
ALPHAS = [0.25, 0.50, 0.75, 1.0]
FIXED_REGS = [0.001, 0.01, 0.05, 0.1, 0.5]
SOLVER_NAME = 'subspace-sbl'
N_EVAL_TRIALS = 30
BATCH_TIMEPOINTS = 500  # 0.5s at 1kHz
RANDOM_SEED = 42
DURATION_LABELS = {30: '30s', 60: '1min', 600: '10min'}


# ── Phase 1: Infrastructure ────────────────────────────────────────────

def build_forward_models():
    """Build ground truth (sample) and template (fsaverage) forward models."""
    fs_dir = _get_fsaverage_dir(verbose=0)
    subjects_dir_fs = os.path.dirname(fs_dir)
    trans_fs = os.path.join(fs_dir, 'bem', 'fsaverage-trans.fif')
    sample_path = mne.datasets.sample.data_path()
    subjects_dir_sample = os.path.join(sample_path, 'subjects')
    info = get_info()

    print('Building ground truth (sample subject)...')
    fwd_true = mne.make_forward_solution(
        info,
        trans=os.path.join(sample_path, 'MEG', 'sample',
                           'sample_audvis_raw-trans.fif'),
        src=mne.setup_source_space('sample', spacing='ico3', surface='white',
                                   subjects_dir=subjects_dir_sample,
                                   add_dist=False, verbose=0),
        bem=os.path.join(subjects_dir_sample, 'sample', 'bem',
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
                                   subjects_dir=subjects_dir_fs,
                                   add_dist=False, verbose=0),
        bem=os.path.join(fs_dir, 'bem',
                         'fsaverage-5120-5120-5120-bem-sol.fif'),
        eeg=True, mindist=0.0, verbose=0,
    )
    fwd_tpl = mne.convert_forward_solution(
        fwd_tpl, surf_ori=True, force_fixed=True, verbose=0)

    return fwd_true, fwd_tpl


def build_mni_distance_matrix(fwd_true, fwd_tpl):
    """Build distance matrix with both position sets in MNI space.

    Transforms: HEAD → MRI (via trans) → MNI (via Talairach).
    """
    fs_dir = _get_fsaverage_dir(verbose=0)
    sample_path = mne.datasets.sample.data_path()

    trans_s = mne.read_trans(os.path.join(
        sample_path, 'MEG', 'sample', 'sample_audvis_raw-trans.fif'))
    trans_f = mne.read_trans(os.path.join(fs_dir, 'bem', 'fsaverage-trans.fif'))
    tal_s = mne.read_talxfm('sample', os.path.join(sample_path, 'subjects'))
    tal_f = mne.read_talxfm('fsaverage', os.path.dirname(fs_dir))

    def _to_mni(pos_head, trans, tal):
        T = tal['trans'] @ trans['trans']  # HEAD→MRI→MNI
        ph = np.column_stack([pos_head, np.ones(len(pos_head))])
        return (T @ ph.T).T[:, :3]

    pos_true_mni = _to_mni(fwd_true['source_rr'], trans_s, tal_s)
    pos_tpl_mni = _to_mni(fwd_tpl['source_rr'], trans_f, tal_f)
    return cdist(pos_true_mni, pos_tpl_mni), pos_true_mni, pos_tpl_mni


def generate_eval_trials(fwd_true):
    """Generate fixed evaluation trials."""
    eval_config = SimulationConfig(
        batch_size=N_EVAL_TRIALS, n_sources=2, n_timepoints=200,
        snr_range=(5, 15), verbose=0, random_seed=99,
    )
    return next(SimulationGenerator(fwd_true, eval_config).generate())


# ── Phase 2: Pre-compute covariances ───────────────────────────────────

def simulate_covariances(fwd_true, duration_sec, snr_db, rng_seed):
    """Simulate raw EEG and return sample cov + LW-shrunk cov.

    Returns (C_sample, C_lw, shrinkage).
    """
    total_samples = int(duration_sec * SFREQ)
    n_batches = max(1, total_samples // BATCH_TIMEPOINTS)

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

    samples = []
    for _ in range(n_batches):
        batch = next(gen.generate())
        samples.append(batch.x[0].T)  # (n_tp, n_ch)
    X = np.concatenate(samples, axis=0)  # (n_total, n_ch)

    # Sample covariance
    C_sample = (X.T @ X) / X.shape[0]

    # Ledoit-Wolf shrinkage
    C_lw, shrinkage = ledoit_wolf(X)

    return C_sample, C_lw, shrinkage


# ── Phase 3: Transport + evaluation ────────────────────────────────────

def regularize_cov(C, strategy, reg_param=None):
    """Apply regularization strategy to covariance matrix.

    Strategies:
      - 'fixed': C + reg_param * trace(C)/n * I
      - 'ledoit_wolf': C is already LW-shrunk, just return it
      - 'oracle_adaptive': eigenvalue-based adaptive reg — clip eigenvalues
        below noise floor estimated from smallest eigenvalue cluster
    """
    n = C.shape[0]
    if strategy == 'ledoit_wolf':
        return C.copy()
    elif strategy == 'fixed':
        C_reg = C.copy()
        C_reg += reg_param * np.trace(C) / n * np.eye(n)
        return C_reg
    elif strategy == 'oracle_adaptive':
        # Estimate noise floor from median of bottom half of eigenvalues
        eigvals = eigh(C, eigvals_only=True)
        noise_floor = np.median(eigvals[:n // 2])
        # Regularize: clip eigenvalues to at least noise_floor
        # Equivalent to C + max(0, noise_floor - lambda_min) * I
        # but we use a soft version: add noise_floor * I
        C_reg = C.copy()
        C_reg += noise_floor * np.eye(n)
        return C_reg
    else:
        raise ValueError(f'Unknown strategy: {strategy}')


def _transport_with_prereg(C_data_reg, fwd_tpl, alpha,
                           transport='whiten_recolor', alpha_mode='blend'):
    """Riemannian transport with pre-regularized covariance.

    C_data_reg is already regularized externally (fixed, LW, or adaptive).
    We pass reg=0 so the shared function skips its own diagonal loading on
    C_data, but the template covariance still gets its mandatory reg=0.01.
    """
    return riemannian_leadfield_transport(
        C_data_reg, fwd_tpl, alpha=alpha, reg=0,
        transport=transport, alpha_mode=alpha_mode)


def evaluate_fwd(fwd_est, fwd_true, eval_batch, info_mne, solver_name,
                 eval_ctx=None):
    """Evaluate a forward model on pre-generated test trials.

    eval_ctx: dict with pre-computed pos_true, adj_true, adj_est, D_cross
    to avoid recomputing on every call.
    """
    if eval_ctx is None:
        pos_true = fwd_true['source_rr']
        pos_est = fwd_est['source_rr']
        adj_true = mne.spatial_src_adjacency(fwd_true['src'], verbose=0)
        adj_est = mne.spatial_src_adjacency(fwd_est['src'], verbose=0)
        D_cross = cdist(pos_true, pos_est)
    else:
        pos_true = eval_ctx['pos_true']
        pos_est = eval_ctx['pos_est']
        adj_true = eval_ctx['adj_true']
        adj_est = eval_ctx['adj_est']
        D_cross = eval_ctx['D_cross']

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
                pos_true, pos_est, D_cross)
            emd = eval_emd(D_cross, y_true, y_pred)
            mles.append(mle * 1000)
            emds.append(emd * 1000)
        except Exception:
            pass
    return np.median(mles), np.median(emds)


# ── Phase 4: Analysis ──────────────────────────────────────────────────

def print_analysis(results, mle_vanilla, emd_vanilla):
    """Print robustness analysis of all (alpha, reg_strategy) combos."""
    # Collect all protocol keys
    protocols = set()
    for key in results:
        dur, snr, alpha, strat = key
        protocols.add((alpha, strat))
    protocols = sorted(protocols)

    conditions = [(d, s) for d in DURATIONS_SEC for s in SNRS_DB]

    # Compute robustness metrics for each protocol
    print('\n' + '=' * 80)
    print('ROBUSTNESS ANALYSIS')
    print('=' * 80)
    print(f'\nVanilla baseline:  MLE={mle_vanilla:.1f} mm  '
          f'EMD={emd_vanilla:.1f} mm\n')

    rows = []
    for alpha, strat in protocols:
        mles = []
        for dur, snr in conditions:
            key = (dur, snr, alpha, strat)
            if key in results:
                mle, emd = results[key]
                mles.append(mle)
        if not mles:
            continue
        mean_mle = np.mean(mles)
        worst_mle = np.max(mles)
        median_mle = np.median(mles)
        std_mle = np.std(mles)
        rows.append((alpha, strat, mean_mle, worst_mle, median_mle, std_mle))

    # Sort by mean MLE (lower is better)
    rows.sort(key=lambda r: r[2])

    print(f'{"Rank":<5} {"Alpha":<7} {"Reg Strategy":<20} '
          f'{"Mean MLE":<10} {"Worst MLE":<11} {"Median MLE":<12} '
          f'{"Std MLE":<9} {"vs Vanilla":<10}')
    print('-' * 90)
    for i, (alpha, strat, mean_mle, worst_mle, median_mle, std_mle) in enumerate(rows[:15]):
        delta = (1 - mean_mle / mle_vanilla) * 100
        print(f'{i+1:<5} {alpha:<7.2f} {strat:<20} '
              f'{mean_mle:<10.1f} {worst_mle:<11.1f} {median_mle:<12.1f} '
              f'{std_mle:<9.1f} {delta:>+8.1f}%')

    # Heatmap for 1min duration (most practical)
    print('\n' + '=' * 80)
    print('HEATMAP: MLE (mm) at duration=1min')
    print('Rows: alpha, Columns: reg_strategy')
    print('=' * 80)

    # Collect all strategies
    all_strats = sorted(set(s for _, s in protocols))

    # Header
    print(f'\n{"α \\ reg":<8}', end='')
    for strat in all_strats:
        print(f'{strat:>16}', end='')
    print()
    print('-' * (8 + 16 * len(all_strats)))

    for alpha in ALPHAS:
        print(f'{alpha:<8.2f}', end='')
        for strat in all_strats:
            # Average over SNRs at 1min duration
            mles = []
            for snr in SNRS_DB:
                key = (60, snr, alpha, strat)
                if key in results:
                    mle, _ = results[key]
                    mles.append(mle)
            if mles:
                avg = np.mean(mles)
                print(f'{avg:>16.1f}', end='')
            else:
                print(f'{"—":>16}', end='')
        print()

    # Per-SNR breakdown for best protocol
    best_alpha, best_strat = rows[0][0], rows[0][1]
    print(f'\n{"=" * 80}')
    print(f'BEST PROTOCOL: alpha={best_alpha}, reg={best_strat}')
    print(f'{"=" * 80}')
    print(f'\n{"Duration":<10}', end='')
    for snr in SNRS_DB:
        print(f'{snr:>10} dB', end='')
    print(f'{"Mean":>10}')
    print('-' * (10 + 12 * len(SNRS_DB) + 10))

    for dur in DURATIONS_SEC:
        print(f'{DURATION_LABELS[dur]:<10}', end='')
        row_mles = []
        for snr in SNRS_DB:
            key = (dur, snr, best_alpha, best_strat)
            if key in results:
                mle, _ = results[key]
                row_mles.append(mle)
                print(f'{mle:>10.1f}  ', end='')
            else:
                print(f'{"—":>10}  ', end='')
        if row_mles:
            print(f'{np.mean(row_mles):>8.1f}', end='')
        print()

    # Recommendation
    print(f'\n{"=" * 80}')
    print('RECOMMENDATION')
    print(f'{"=" * 80}')
    best = rows[0]
    runner_up = rows[1] if len(rows) > 1 else None
    print(f'\n  Best: alpha={best[0]}, reg={best[1]}')
    print(f'    Mean MLE: {best[2]:.1f} mm (vs vanilla {mle_vanilla:.1f} mm, '
          f'{(1-best[2]/mle_vanilla)*100:+.1f}%)')
    print(f'    Worst-case: {best[3]:.1f} mm, Std: {best[5]:.1f} mm')
    if runner_up:
        print(f'\n  Runner-up: alpha={runner_up[0]}, reg={runner_up[1]}')
        print(f'    Mean MLE: {runner_up[2]:.1f} mm, '
              f'Worst-case: {runner_up[3]:.1f} mm')


# ── Main ────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    # Phase 1: Infrastructure
    print('=' * 80)
    print('RIEMANNIAN PROTOCOL OPTIMIZATION')
    print('=' * 80)

    fwd_true, fwd_tpl = build_forward_models()

    info_mne = mne.create_info(
        ch_names=fwd_true['info']['ch_names'], sfreq=250., ch_types='eeg')

    print(f'\nGenerating {N_EVAL_TRIALS} evaluation trials...')
    eval_batch = generate_eval_trials(fwd_true)

    # Pre-compute evaluation contexts — both D_head and D_mni
    print('Pre-computing evaluation contexts...')
    adj_true = mne.spatial_src_adjacency(fwd_true['src'], verbose=0)
    adj_est = mne.spatial_src_adjacency(fwd_tpl['src'], verbose=0)

    D_head = cdist(fwd_true['source_rr'], fwd_tpl['source_rr'])
    D_mni, pos_true_mni, pos_tpl_mni = build_mni_distance_matrix(fwd_true, fwd_tpl)

    eval_ctx_head = {
        'pos_true': fwd_true['source_rr'],
        'pos_est': fwd_tpl['source_rr'],
        'adj_true': adj_true, 'adj_est': adj_est,
        'D_cross': D_head,
    }
    eval_ctx_mni = {
        'pos_true': fwd_true['source_rr'],
        'pos_est': fwd_tpl['source_rr'],
        'adj_true': adj_true, 'adj_est': adj_est,
        'D_cross': D_mni,
    }
    print(f'  D_head NN: {D_head.min(axis=1).mean()*1000:.1f} mm '
          f'(centroid offset: {np.linalg.norm(fwd_true["source_rr"].mean(0) - fwd_tpl["source_rr"].mean(0))*1000:.1f} mm)')
    print(f'  D_mni  NN: {D_mni.min(axis=1).mean()*1000:.1f} mm '
          f'(centroid offset: {np.linalg.norm(pos_true_mni.mean(0) - pos_tpl_mni.mean(0))*1000:.1f} mm)')

    print('Evaluating vanilla baseline...')
    mle_vanilla_h, emd_vanilla_h = evaluate_fwd(
        fwd_tpl, fwd_true, eval_batch, info_mne, SOLVER_NAME,
        eval_ctx=eval_ctx_head)
    mle_vanilla_m, emd_vanilla_m = evaluate_fwd(
        fwd_tpl, fwd_true, eval_batch, info_mne, SOLVER_NAME,
        eval_ctx=eval_ctx_mni)
    print(f'  Vanilla (head): MLE={mle_vanilla_h:.1f} mm  EMD={emd_vanilla_h:.1f} mm')
    print(f'  Vanilla (MNI):  MLE={mle_vanilla_m:.1f} mm  EMD={emd_vanilla_m:.1f} mm')

    # Phase 2: Pre-compute covariances
    print(f'\n{"=" * 80}')
    print('PHASE 2: Pre-computing covariances (12 conditions)')
    print(f'{"=" * 80}')

    covariances = {}  # (dur, snr) -> (C_sample, C_lw)
    for dur in DURATIONS_SEC:
        for snr in SNRS_DB:
            label = f'{DURATION_LABELS[dur]}, SNR={snr}dB'
            print(f'  {label}...', end=' ', flush=True)
            t0 = time.time()
            rng_seed = RANDOM_SEED + dur + snr
            C_sample, C_lw, shrinkage = simulate_covariances(
                fwd_true, dur, snr, rng_seed)
            covariances[(dur, snr)] = (C_sample, C_lw)
            print(f'done ({time.time()-t0:.0f}s, LW shrinkage={shrinkage:.4f})')

    # Phase 3: Sweep
    print(f'\n{"=" * 80}')
    print('PHASE 3: Sweeping alpha × reg_strategy over all conditions')
    print(f'{"=" * 80}')

    # Build list of (strategy_name, strategy_type, reg_param)
    strategies = []
    for r in FIXED_REGS:
        strategies.append((f'fixed({r})', 'fixed', r))
    strategies.append(('ledoit_wolf', 'ledoit_wolf', None))
    strategies.append(('oracle_adaptive', 'oracle_adaptive', None))

    # Transport methods: (label, transport, alpha_mode)
    transport_methods = [
        ('WR+blend', 'whiten_recolor', 'blend'),
        ('bures+geo', 'bures', 'geodesic'),
    ]

    total = (len(DURATIONS_SEC) * len(SNRS_DB) * len(ALPHAS)
             * len(strategies) * len(transport_methods))
    print(f'Total evaluations: {total}')

    # (dur, snr, alpha, strat_name, transport_label) -> (mle_h, emd_h, mle_m, emd_m)
    results = {}
    eval_count = 0

    for dur in DURATIONS_SEC:
        for snr in SNRS_DB:
            C_sample, C_lw = covariances[(dur, snr)]
            label = f'{DURATION_LABELS[dur]}, SNR={snr}dB'
            print(f'\n  {label}:')

            for alpha in ALPHAS:
                for strat_name, strat_type, reg_param in strategies:
                    # Select base covariance
                    if strat_type == 'ledoit_wolf':
                        C_reg = regularize_cov(C_lw, 'ledoit_wolf')
                    else:
                        C_reg = regularize_cov(C_sample, strat_type, reg_param)

                    for t_label, transport, alpha_mode in transport_methods:
                        eval_count += 1

                        # Transport
                        try:
                            fwd_riem = _transport_with_prereg(
                                C_reg, fwd_tpl, alpha,
                                transport=transport, alpha_mode=alpha_mode)
                        except Exception as e:
                            print(f'    [{eval_count:3d}/{total}] '
                                  f'α={alpha:.2f} {strat_name:<16} '
                                  f'{t_label:<10} FAILED: {e}')
                            continue

                        # Evaluate with both metrics
                        mle_h, emd_h = evaluate_fwd(
                            fwd_riem, fwd_true, eval_batch, info_mne,
                            SOLVER_NAME, eval_ctx=eval_ctx_head)
                        mle_m, emd_m = evaluate_fwd(
                            fwd_riem, fwd_true, eval_batch, info_mne,
                            SOLVER_NAME, eval_ctx=eval_ctx_mni)

                        key = (dur, snr, alpha, strat_name, t_label)
                        results[key] = (mle_h, emd_h, mle_m, emd_m)
                        delta_h = (1 - mle_h / mle_vanilla_h) * 100
                        delta_m = (1 - mle_m / mle_vanilla_m) * 100
                        print(f'    [{eval_count:3d}/{total}] '
                              f'α={alpha:.2f} {strat_name:<16} '
                              f'{t_label:<10} '
                              f'head={mle_h:.1f}({delta_h:+.0f}%) '
                              f'MNI={mle_m:.1f}({delta_m:+.0f}%)')

    # Phase 4: Analysis — report both metrics
    print('\n' + '=' * 80)
    print('RESULTS SUMMARY (sorted by MNI MLE, 1min duration, mean over SNR)')
    print('=' * 80)

    # Collect protocols
    protocols = set()
    for key in results:
        dur, snr, alpha, strat, tlabel = key
        protocols.add((alpha, strat, tlabel))
    protocols = sorted(protocols)

    rows = []
    for alpha, strat, tlabel in protocols:
        mles_h, mles_m = [], []
        for snr in SNRS_DB:
            key = (60, snr, alpha, strat, tlabel)
            if key in results:
                mle_h, emd_h, mle_m, emd_m = results[key]
                mles_h.append(mle_h)
                mles_m.append(mle_m)
        if mles_m:
            rows.append((alpha, strat, tlabel,
                          np.mean(mles_h), np.mean(mles_m)))

    rows.sort(key=lambda r: r[4])  # sort by MNI MLE
    print(f'\n{"Rank":<5} {"α":<5} {"Reg":<16} {"Transport":<10} '
          f'{"Head MLE":<10} {"MNI MLE":<10}')
    print('-' * 60)
    for i, (a, s, t, mh, mm) in enumerate(rows[:20]):
        print(f'{i+1:<5} {a:<5.2f} {s:<16} {t:<10} '
              f'{mh:<10.1f} {mm:<10.1f}')

    # Full robustness (all durations) for top 5
    print(f'\n{"=" * 80}')
    print('TOP 5: Full robustness (all durations × SNR)')
    print(f'{"=" * 80}')
    for rank, (alpha, strat, tlabel, _, _) in enumerate(rows[:5]):
        print(f'\n  #{rank+1}: α={alpha}, {strat}, {tlabel}')
        for dur in DURATIONS_SEC:
            mles_h, mles_m = [], []
            for snr in SNRS_DB:
                key = (dur, snr, alpha, strat, tlabel)
                if key in results:
                    mh, eh, mm, em = results[key]
                    mles_h.append(mh)
                    mles_m.append(mm)
            if mles_h:
                print(f'    {DURATION_LABELS[dur]:>5s}: '
                      f'head={np.mean(mles_h):.1f}±{np.std(mles_h):.1f}  '
                      f'MNI={np.mean(mles_m):.1f}±{np.std(mles_m):.1f}')

    print(f'\nVanilla: head={mle_vanilla_h:.1f}  MNI={mle_vanilla_m:.1f}')

    elapsed = time.time() - t_start
    print(f'\nTotal time: {elapsed/60:.1f} min')


if __name__ == '__main__':
    main()
