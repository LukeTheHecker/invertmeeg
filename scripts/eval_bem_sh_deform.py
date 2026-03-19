"""BEM SH deformation: hybrid co-deform + skull-only SH.

Architecture (from lab journal findings):
  1. Co-deform affine (9p): offset+stretch+rotation applied to BOTH skull and sources
     → captures coupled brain+skull translation/scaling
     → optimized at ico2 (fast, generalizes well)
  2. Skull-only SH l=2+3 (12p): radial deformation on inner skull AFTER co-deform
     → captures skull-specific shape residual beyond affine
     → optimized at ico3 to avoid overfitting (~1s/eval, manageable for 12p)
  3. Sigma: ratio-constrained grid sweep at each stage

Key improvements over v1:
  - No decoupled brain — co-deform handles brain+skull coupling
  - SH optimized at ico3 to avoid ico2 overfitting
  - Source count validation at every eval
  - Separate sigma seeds per approach
"""
import os, sys, time, importlib.util
import numpy as np
import mne
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize as sp_minimize
from scipy.spatial.distance import cdist

from mne.bem import _check_complete_surface

from invert.cortical_folding.fast_bem import (
    make_bem_solution_factored, rebuild_bem_from_coeff, partial_K_update,
)
from invert.cortical_folding.sh_analysis import cart_to_sph, real_sh_matrix
from invert.forward.forward import _get_fsaverage_dir, get_info
from invert.simulate import SimulationConfig, SimulationGenerator
from invert import Solver

sys.stdout.reconfigure(line_buffering=True)

# ── Eval functions ──
_eval_path = os.path.join(os.path.dirname(__file__), '..', 'invert', 'evaluate', 'evaluate.py')
_spec = importlib.util.spec_from_file_location('eval_mod', _eval_path, submodule_search_locations=[])
_eval_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_mod)
eval_emd = _eval_mod.eval_emd
eval_mle = _eval_mod.eval_mean_localization_error

# ── Setup ──
fs_dir = _get_fsaverage_dir(verbose=0)
subjects_dir = os.path.dirname(fs_dir)
trans_fs = os.path.join(fs_dir, 'bem', 'fsaverage-trans.fif')
info = get_info()
sample_path = mne.datasets.sample.data_path()
subjects_dir_sample = os.path.join(sample_path, 'subjects')

print('Building ground truth (sample subject)...')
fwd_true = mne.make_forward_solution(
    info,
    trans=os.path.join(sample_path, 'MEG', 'sample', 'sample_audvis_raw-trans.fif'),
    src=mne.setup_source_space('sample', spacing='ico3', surface='white',
        subjects_dir=subjects_dir_sample, add_dist=False, verbose=0),
    bem=os.path.join(subjects_dir_sample, 'sample', 'bem',
        'sample-5120-5120-5120-bem-sol.fif'),
    eeg=True, mindist=0.0, verbose=0)
fwd_true = mne.convert_forward_solution(fwd_true, surf_ori=True, force_fixed=True, verbose=0)
pos_true = fwd_true['source_rr']
n_src_true = fwd_true['sol']['data'].shape[1]

print('Simulating sensor covariance...')
config_cov = SimulationConfig(batch_size=500, n_sources=(1, 5), n_timepoints=50,
    snr_range=(0, 10), verbose=0, random_seed=42)
C_data = np.zeros((61, 61))
gen = SimulationGenerator(fwd_true, config_cov)
for _ in range(10):
    batch = next(gen.generate())
    for s in range(batch.x.shape[0]):
        x = batch.x[s]; C_data += x @ x.T / x.shape[1]
C_data /= 5000

print('Simulating evaluation trials...')
n_trials = 30
eval_config = SimulationConfig(batch_size=n_trials, n_sources=2, n_timepoints=200,
    snr_range=(5, 15), verbose=0, random_seed=99)
eval_batch = next(SimulationGenerator(fwd_true, eval_config).generate())
info_mne = mne.create_info(ch_names=fwd_true['info']['ch_names'], sfreq=250., ch_types='eeg')
adj_true = mne.spatial_src_adjacency(fwd_true['src'], verbose=0)
print('Data ready.')


# ── BEM setup ──
def setup_bem(ico):
    print(f'Building BEM (ico{ico})...')
    bem_model = mne.make_bem_model('fsaverage', ico=ico, conductivity=(0.3, 0.006, 0.3),
        subjects_dir=subjects_dir, verbose=0)
    bem, coeff, coeff_h = make_bem_solution_factored(bem_model)
    src = mne.setup_source_space('fsaverage', spacing='ico3', surface='white',
        subjects_dir=subjects_dir, add_dist=False, verbose=0)
    pos = np.vstack([src[0]['rr'][src[0]['vertno']].copy(),
                      src[1]['rr'][src[1]['vertno']].copy()])
    nn = np.vstack([src[0]['nn'][src[0]['vertno']].copy(),
                     src[1]['nn'][src[1]['vertno']].copy()])
    skull_rr_orig = bem_model[2]['rr'].copy()
    skull_center = skull_rr_orig.mean(axis=0)
    src_center = pos.mean(axis=0)
    n_lh = len(src[0]['vertno'])

    radii_base = np.linalg.norm(skull_rr_orig - skull_center, axis=1)
    dirs_skull = (skull_rr_orig - skull_center) / radii_base[:, None]
    theta_skull, phi_skull = cart_to_sph(skull_rr_orig, skull_center)
    Y_skull = real_sh_matrix(theta_skull, phi_skull, l_max=3)

    outer_rr = bem_model[1]['rr']
    outer_radii = np.linalg.norm(outer_rr - outer_rr.mean(axis=0), axis=1)
    dirs_outer = (outer_rr - outer_rr.mean(axis=0)) / outer_radii[:, None]

    return dict(bem_model=bem_model, bem=bem, coeff=coeff, coeff_h=coeff_h,
                src=src, pos=pos, nn=nn, n_lh=n_lh,
                skull_rr_orig=skull_rr_orig,
                skull_center=skull_center, src_center=src_center,
                radii_base=radii_base, dirs_skull=dirs_skull,
                Y_skull=Y_skull,
                outer_rr=outer_rr, outer_radii=outer_radii,
                dirs_outer=dirs_outer)


# ── Helpers ──

def _apply_transform(pts, center, offset, stretch, rotation):
    p = pts - center
    if np.any(rotation != 0):
        R = Rotation.from_euler('XYZ', rotation, degrees=True).as_matrix()
        p = p @ R.T
    return p * stretch + center + offset


def subspace_cost(G, k=10):
    _, U_d = np.linalg.eigh(C_data); U_d = U_d[:, -k:]
    _, U_m = np.linalg.eigh(G @ G.T); U_m = U_m[:, -k:]
    S = np.linalg.svd(U_d.T @ U_m, compute_uv=False)
    return 1 - np.mean(S**2)


def ratio_to_sigma(log_r_skull, r_scalp, sigma_brain=0.3):
    return sigma_brain / (10 ** log_r_skull), sigma_brain * r_scalp


_LR_GRID = np.linspace(np.log10(5), np.log10(200), 10)
_RS_GRID = np.linspace(0.5, 2.0, 8)

MAX_DEFORM_MM = 15.0


def deform_skull_sh(sh_coeffs, env):
    full_coeffs = np.zeros(16)
    full_coeffs[4:4 + len(sh_coeffs)] = sh_coeffs
    delta_r = env['Y_skull'] @ full_coeffs
    return env['skull_center'] + (env['radii_base'] + delta_r)[:, None] * env['dirs_skull']


def check_sh_bounds(sh_coeffs, env):
    full_coeffs = np.zeros(16)
    full_coeffs[4:4 + len(sh_coeffs)] = sh_coeffs
    delta_r = env['Y_skull'] @ full_coeffs
    if np.max(np.abs(delta_r)) > MAX_DEFORM_MM / 1000:
        return False
    r_new = env['radii_base'] + delta_r
    if np.min(r_new) < 0.030:
        return False
    from scipy.spatial import cKDTree
    _, idx = cKDTree(env['dirs_outer']).query(env['dirs_skull'])
    if np.any(r_new > env['outer_radii'][idx] - 0.002):
        return False
    return True


def _max_deform_mm(sh, env):
    full_c = np.zeros(16); full_c[4:4+len(sh)] = sh
    return np.max(np.abs(env['Y_skull'] @ full_c)) * 1000


def _make_src_with_pos(env, offset, stretch, rotation):
    pos = _apply_transform(env['pos'], env['src_center'], offset, stretch, rotation)
    nn = env['nn'].copy()
    if np.any(rotation != 0):
        R = Rotation.from_euler('XYZ', rotation, degrees=True).as_matrix()
        nn = nn @ R.T
    src = env['src'].copy()
    n_lh = env['n_lh']
    for i, h in enumerate(src):
        sl = slice(0, n_lh) if i == 0 else slice(n_lh, None)
        h['rr'] = h['rr'].copy(); h['rr'][h['vertno']] = pos[sl]
        h['nn'] = h['nn'].copy(); h['nn'][h['vertno']] = nn[sl]
    return src


def build_fwd_codeform(env, sigma_skull, sigma_scalp, offset, stretch, rotation,
                        sh_coeffs=None):
    """Build forward model: co-deform skull+sources (affine) + optional SH skull-only."""
    bem, coeff, coeff_h = env['bem'], env['coeff'], env['coeff_h']

    # Step 1: SH deformation on skull (skull-specific shape, BEFORE affine)
    if sh_coeffs is not None and np.any(sh_coeffs != 0):
        skull_rr = deform_skull_sh(sh_coeffs, env)
    else:
        skull_rr = env['skull_rr_orig'].copy()

    # Step 2: Affine co-deform on skull (translation+stretch+rotation)
    skull_rr = _apply_transform(skull_rr, env['skull_center'], offset, stretch, rotation)
    env['bem']['surfs'][2]['rr'] = skull_rr
    _check_complete_surface(env['bem']['surfs'][2])
    coeff_updated = partial_K_update(env['bem']['surfs'], coeff, changed_idx=2)
    env['bem']['surfs'][2]['rr'] = env['skull_rr_orig']

    rebuild_bem_from_coeff(bem, coeff_updated, coeff_h, (sigma_scalp, sigma_skull, 0.3))

    # Step 3: Affine co-deform on sources (same transform as skull)
    src = _make_src_with_pos(env, offset, stretch, rotation)

    fwd = mne.make_forward_solution(info, trans=trans_fs, src=src,
        bem=bem, eeg=True, mindist=0.0, verbose=0)
    return mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, verbose=0)


def inner_sigma_sweep(env, offset, stretch, rotation, sh_coeffs=None):
    """Sweep sigma grid at fixed geometry. Returns (cost, lr, rs)."""
    if sh_coeffs is not None and np.any(sh_coeffs != 0):
        skull_rr = deform_skull_sh(sh_coeffs, env)
    else:
        skull_rr = env['skull_rr_orig'].copy()
    skull_rr = _apply_transform(skull_rr, env['skull_center'], offset, stretch, rotation)
    env['bem']['surfs'][2]['rr'] = skull_rr
    _check_complete_surface(env['bem']['surfs'][2])
    coeff_updated = partial_K_update(env['bem']['surfs'], env['coeff'], changed_idx=2)
    env['bem']['surfs'][2]['rr'] = env['skull_rr_orig']

    src = _make_src_with_pos(env, offset, stretch, rotation)

    n_src_base = len(env['pos'])
    best_cost, best_lr, best_rs = np.inf, None, None
    for lr in _LR_GRID:
        for rs in _RS_GRID:
            ss, sc = ratio_to_sigma(lr, rs)
            try:
                rebuild_bem_from_coeff(env['bem'], coeff_updated, env['coeff_h'],
                                       (sc, ss, 0.3))
                fwd = mne.make_forward_solution(info, trans=trans_fs, src=src,
                    bem=env['bem'], eeg=True, mindist=0.0, verbose=0)
                fwd = mne.convert_forward_solution(fwd, surf_ori=True,
                    force_fixed=True, verbose=0)
                n = fwd['sol']['data'].shape[1]
                if n < 0.90 * n_src_base:
                    continue
                c = subspace_cost(fwd['sol']['data'])
                if c < best_cost:
                    best_cost, best_lr, best_rs = c, lr, rs
            except Exception:
                pass
    if best_lr is None:
        best_lr, best_rs = np.log10(50), 1.0
        ss, sc = ratio_to_sigma(best_lr, best_rs)
        try:
            fwd = build_fwd_codeform(env, ss, sc, offset, stretch, rotation, sh_coeffs)
            best_cost = subspace_cost(fwd['sol']['data'])
        except Exception:
            best_cost = 1.0
    return best_cost, best_lr, best_rs


def evaluate_fwd(fwd_est, solver_name='esmv'):
    pos_est = fwd_est['source_rr']
    adj_est = mne.spatial_src_adjacency(fwd_est['src'], verbose=0)
    D_cross = cdist(pos_true, pos_est)
    solver = Solver(solver_name)
    mles, emds = [], []
    for trial in range(n_trials):
        evoked = mne.EvokedArray(eval_batch.x[trial], info_mne, tmin=0)
        try:
            solver.make_inverse_operator(fwd_est, evoked, alpha='auto', verbose=0)
            stc = solver.apply_inverse_operator(evoked)
            y_true_avg = np.abs(eval_batch.y[trial]).mean(axis=1)
            y_pred_avg = np.abs(stc.data).mean(axis=1)
            mle = eval_mle(y_true_avg, y_pred_avg, adj_true, adj_est,
                           pos_true, pos_est, D_cross)
            emd_val = eval_emd(D_cross, y_true_avg, y_pred_avg)
            mles.append(mle * 1000)
            emds.append(emd_val * 1000)
        except Exception:
            pass
    return np.array(mles), np.array(emds)


def _fast_cost(env, offset, stretch, rotation, sigma_skull, sigma_scalp,
               sh_coeffs=None, min_src_frac=0.90):
    """Single eval with source count validation."""
    try:
        fwd = build_fwd_codeform(env, sigma_skull, sigma_scalp,
                                  offset, stretch, rotation, sh_coeffs)
        n = fwd['sol']['data'].shape[1]
        if n < min_src_frac * len(env['pos']):
            return 1.0
        return subspace_cost(fwd['sol']['data'])
    except Exception:
        return 1.0


# ══════════════════════════════════════════════════════════════════════
# STAGE 1: Co-deform affine optimization at ico2
# ══════════════════════════════════════════════════════════════════════
t0 = time.time()
env2 = setup_bem(ico=2)
print('ico2 BEM ready.')

SEED_LR = np.log10(8)
SEED_RS = 1.7
SEED_SS, SEED_SC = ratio_to_sigma(SEED_LR, SEED_RS)

# Stage 1a: offset+stretch (6p)
print('\n' + '=' * 70)
print('Stage 1a: offset + stretch (6p, ico2)')
print('=' * 70)
n_eval = [0]
t1 = time.time()

def cost_6p(params):
    n_eval[0] += 1
    if n_eval[0] % 50 == 0:
        print(f'  eval {n_eval[0]} ({time.time()-t1:.0f}s)...', flush=True)
    return _fast_cost(env2, params[0:3], params[3:6], np.zeros(3), SEED_SS, SEED_SC)

res6 = sp_minimize(cost_6p, [0, 0, 0, 1, 1, 1], method='Powell',
                    options={'maxiter': 200, 'ftol': 1e-6})
off6, str6 = res6.x[0:3], res6.x[3:6]
print(f'  off=[{off6[0]*1000:.1f},{off6[1]*1000:.1f},{off6[2]*1000:.1f}]mm, '
      f'str=[{str6[0]:.3f},{str6[1]:.3f},{str6[2]:.3f}], '
      f'cost={res6.fun:.5f}, {n_eval[0]} evals, {time.time()-t1:.0f}s')

# Stage 1b: +rotation (9p)
print('\n' + '=' * 70)
print('Stage 1b: + rotation (9p, ico2)')
print('=' * 70)
n_eval[0] = 0
t2 = time.time()

def cost_9p(params):
    n_eval[0] += 1
    if n_eval[0] % 50 == 0:
        print(f'  eval {n_eval[0]} ({time.time()-t2:.0f}s)...', flush=True)
    return _fast_cost(env2, params[0:3], params[3:6], params[6:9], SEED_SS, SEED_SC)

x0_9 = list(off6) + list(str6) + [0, 0, 0]
res9 = sp_minimize(cost_9p, x0_9, method='Powell',
                    options={'maxiter': 300, 'ftol': 1e-6})
off9, str9, rot9 = res9.x[0:3], res9.x[3:6], res9.x[6:9]
print(f'  off=[{off9[0]*1000:.1f},{off9[1]*1000:.1f},{off9[2]*1000:.1f}]mm, '
      f'str=[{str9[0]:.3f},{str9[1]:.3f},{str9[2]:.3f}], '
      f'rot=[{rot9[0]:.1f},{rot9[1]:.1f},{rot9[2]:.1f}]deg, '
      f'cost={res9.fun:.5f}, {n_eval[0]} evals, {time.time()-t2:.0f}s')

# Stage 1c: sigma sweep
print('\n' + '=' * 70)
print('Stage 1c: sigma sweep at optimal geometry (ico2)')
print('=' * 70)
t3 = time.time()
cost_cd, lr_cd, rs_cd = inner_sigma_sweep(env2, off9, str9, rot9)
ss_cd, sc_cd = ratio_to_sigma(lr_cd, rs_cd)
print(f'  sigma_skull={ss_cd*1000:.1f}, sigma_scalp={sc_cd:.3f}, '
      f'r={10**lr_cd:.0f}:1, cost={cost_cd:.5f}, {time.time()-t3:.0f}s')

# Sanity check
cost_zero = _fast_cost(env2, off9, str9, rot9, ss_cd, sc_cd, sh_coeffs=np.zeros(12))
print(f'  Zero-SH sanity: {cost_zero:.5f} (should match {cost_cd:.5f})')


# ══════════════════════════════════════════════════════════════════════
# STAGE 2: SH skull deformation at ico3 (avoids ico2 overfitting)
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('Stage 2: SH l=2+3 skull-only deformation (12p, ico3)')
print('=' * 70)

env3 = setup_bem(ico=3)

# Verify co-deform transfers to ico3
cost_cd_ico3 = _fast_cost(env3, off9, str9, rot9, ss_cd, sc_cd)
print(f'  Co-deform at ico3: cost={cost_cd_ico3:.5f} (ico2: {cost_cd:.5f}, '
      f'ratio={cost_cd_ico3/cost_cd:.3f})')

# Optimize SH at ico3
n_eval[0] = 0
t4 = time.time()

def cost_sh_ico3(sh_params):
    n_eval[0] += 1
    if n_eval[0] % 20 == 0:
        print(f'  eval {n_eval[0]} ({time.time()-t4:.0f}s)...', flush=True)
    if not check_sh_bounds(sh_params, env3):
        return 1.0
    return _fast_cost(env3, off9, str9, rot9, ss_cd, sc_cd, sh_coeffs=sh_params)

x0_sh = np.zeros(12)
res_sh = sp_minimize(cost_sh_ico3, x0_sh, method='Powell',
                      options={'maxiter': 300, 'ftol': 1e-6})
sh_best = res_sh.x
max_deform = _max_deform_mm(sh_best, env3)
print(f'  SH l=2: [{", ".join(f"{c:.4f}" for c in sh_best[:5])}]')
print(f'  SH l=3: [{", ".join(f"{c:.4f}" for c in sh_best[5:])}]')
print(f'  max deformation: {max_deform:.1f}mm')
print(f'  cost={res_sh.fun:.5f} (baseline={cost_cd_ico3:.5f}, '
      f'delta={res_sh.fun - cost_cd_ico3:+.5f})')
print(f'  {n_eval[0]} evals, {time.time()-t4:.0f}s')

# Stage 2b: sigma re-sweep with SH
print('\n  Sigma re-sweep at SH geometry (ico3)...')
t5 = time.time()
cost_sh_sigma, lr_sh, rs_sh = inner_sigma_sweep(env3, off9, str9, rot9, sh_coeffs=sh_best)
ss_sh, sc_sh = ratio_to_sigma(lr_sh, rs_sh)
print(f'  sigma_skull={ss_sh*1000:.1f}, sigma_scalp={sc_sh:.3f}, '
      f'r={10**lr_sh:.0f}:1, cost={cost_sh_sigma:.5f}, {time.time()-t5:.0f}s')


# ══════════════════════════════════════════════════════════════════════
# FINAL EVALUATION (ico3)
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('Final evaluation (ico3 BEM)')
print('=' * 70)

configs = [
    ('Vanilla (sigma=4.2)',
     dict(sigma_skull=0.0042, sigma_scalp=0.3,
          offset=np.zeros(3), stretch=np.ones(3), rotation=np.zeros(3),
          sh_coeffs=None),
     0.0),
    ('Co-deform (off+str+rot+sigma)',
     dict(sigma_skull=ss_cd, sigma_scalp=sc_cd,
          offset=off9, stretch=str9, rotation=rot9,
          sh_coeffs=None),
     0.0),
    ('+ SH l=2+3 (12p, ico3-optimized)',
     dict(sigma_skull=ss_cd, sigma_scalp=sc_cd,
          offset=off9, stretch=str9, rotation=rot9,
          sh_coeffs=sh_best),
     max_deform),
    ('+ SH l=2+3 + sigma resweep',
     dict(sigma_skull=ss_sh, sigma_scalp=sc_sh,
          offset=off9, stretch=str9, rotation=rot9,
          sh_coeffs=sh_best),
     max_deform),
]

print(f'\n{"Model":<45s} {"MLE":>6s} {"+-":>5s} {"EMD":>6s} {"+-":>5s} '
      f'{"cost":>7s} {"deform":>7s} {"n_src":>6s}')
print('-' * 90)

for name, kw, deform in configs:
    fwd = build_fwd_codeform(env3, **kw)
    c = subspace_cost(fwd['sol']['data'])
    n = fwd['sol']['data'].shape[1]
    mles, emds = evaluate_fwd(fwd)
    print(f'{name:<45s} {np.median(mles):5.1f} {np.std(mles):5.1f} '
          f'{np.median(emds):5.1f} {np.std(emds):5.1f} {c:7.5f} '
          f'{deform:6.1f}mm {n:6d}')

# Oracle
fwd_oracle = mne.make_forward_solution(
    info,
    trans=os.path.join(sample_path, 'MEG', 'sample', 'sample_audvis_raw-trans.fif'),
    src=mne.setup_source_space('sample', spacing='ico3', surface='white',
        subjects_dir=subjects_dir_sample, add_dist=False, verbose=0),
    bem=os.path.join(subjects_dir_sample, 'sample', 'bem',
        'sample-5120-5120-5120-bem-sol.fif'),
    eeg=True, mindist=0.0, verbose=0)
fwd_oracle = mne.convert_forward_solution(fwd_oracle, surf_ori=True,
    force_fixed=True, verbose=0)
c_oracle = subspace_cost(fwd_oracle['sol']['data'])
mles_o, emds_o = evaluate_fwd(fwd_oracle)
print(f'{"Oracle":<45s} {np.median(mles_o):5.1f} {np.std(mles_o):5.1f} '
      f'{np.median(emds_o):5.1f} {np.std(emds_o):5.1f} {c_oracle:7.5f} '
      f'{"---":>7s} {fwd_oracle["sol"]["data"].shape[1]:6d}')

print(f'\n{"="*70}')
print(f'Co-deform: off=[{off9[0]*1000:.1f},{off9[1]*1000:.1f},{off9[2]*1000:.1f}]mm, '
      f'str=[{str9[0]:.3f},{str9[1]:.3f},{str9[2]:.3f}], '
      f'rot=[{rot9[0]:.1f},{rot9[1]:.1f},{rot9[2]:.1f}]deg')
print(f'SH l=2: [{", ".join(f"{c:.4f}" for c in sh_best[:5])}]')
print(f'SH l=3: [{", ".join(f"{c:.4f}" for c in sh_best[5:])}]')
print(f'Max skull deformation: {max_deform:.1f}mm')
print(f'Sigma: skull={ss_sh*1000:.1f} mS/m, scalp={sc_sh:.3f} S/m')
print(f'\nTotal time: {time.time()-t0:.0f}s')
