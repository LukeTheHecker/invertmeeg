"""Re-evaluate SH 23p model with MNI-aligned distance metrics.

Rebuilds the best model from precomputed parameters and evaluates
with both D_head and D_mni. Also tests Riemannian transport on top.

Usage:
    cd invert-package
    .venv/bin/python scripts/eval_sh_mni.py
"""

import os, sys, importlib.util
import mne
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist
from scipy.linalg import sqrtm, inv

from mne.bem import _check_complete_surface
from invert.cortical_folding.fast_bem import (
    make_bem_solution_factored, rebuild_bem_from_coeff, partial_K_update,
)
from invert.cortical_folding.sh_analysis import cart_to_sph, real_sh_matrix
from invert.cortical_folding.riemannian_transport import riemannian_leadfield_transport
from invert.forward.forward import _get_fsaverage_dir, get_info
from invert.simulate import SimulationConfig, SimulationGenerator
from invert import Solver

sys.stdout.reconfigure(line_buffering=True)

import invert as _invert_pkg
_eval_path = os.path.join(os.path.dirname(_invert_pkg.__file__), "evaluate", "evaluate.py")
_spec = importlib.util.spec_from_file_location("eval_mod", _eval_path, submodule_search_locations=[])
_eval_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_mod)
eval_emd = _eval_mod.eval_emd
eval_mle = _eval_mod.eval_mean_localization_error

sys.path.insert(0, os.path.dirname(__file__))
from eval_hybrid_morphing import build_sample_forward

# ── Paths ──
SAMPLE_PATH = mne.datasets.sample.data_path()
SUBJECTS_DIR_SAMPLE = os.path.join(SAMPLE_PATH, "subjects")
FS_DIR = _get_fsaverage_dir(verbose=0)
SUBJECTS_DIR_FS = os.path.dirname(FS_DIR)

N_TRIALS = 30
SOLVER = "subspace-sbl"

# ── Best 23p params ──
OFF9 = np.array([-0.0003, 0.0075, 0.0112])
STR9 = np.array([1.007, 0.987, 0.946])
ROT9 = np.array([-1.9, -1.4, -2.8])
SH_EXP3 = np.array([-0.0046, 0.0092, 0.0040, -0.0005, -0.0044,
                      0.0022, -0.0018, -0.0002, -0.0013, -0.0025, 0.0053, -0.0005])

# Physiological sigma candidates (top from previous sweep)
SIGMA_CANDIDATES = [
    (0.016, 0.30),  # best EMD
    (0.016, 0.25),
    (0.020, 0.35),
    (0.013, 0.25),
    (0.010, 0.30),
    (0.020, 0.30),
]


def build_D_mni(fwd_true, fwd_est):
    trans_s = mne.read_trans(os.path.join(
        SAMPLE_PATH, "MEG", "sample", "sample_audvis_raw-trans.fif"))
    trans_f = mne.read_trans(os.path.join(FS_DIR, "bem", "fsaverage-trans.fif"))
    tal_s = mne.read_talxfm("sample", SUBJECTS_DIR_SAMPLE)
    tal_f = mne.read_talxfm("fsaverage", SUBJECTS_DIR_FS)

    def _to_mni(pos, trans, tal):
        T = tal["trans"] @ trans["trans"]
        ph = np.column_stack([pos, np.ones(len(pos))])
        return (T @ ph.T).T[:, :3]

    p_true = _to_mni(fwd_true["source_rr"], trans_s, tal_s)
    if fwd_est is fwd_true:
        p_est = p_true
    else:
        p_est = _to_mni(fwd_est["source_rr"], trans_f, tal_f)
    return cdist(p_true, p_est)


def evaluate(fwd_est, fwd_true, eval_batch, info_mne, D):
    adj_true = mne.spatial_src_adjacency(fwd_true["src"], verbose=0)
    adj_est = mne.spatial_src_adjacency(fwd_est["src"], verbose=0)
    solver = Solver(SOLVER)
    mles, emds = [], []
    for trial in range(N_TRIALS):
        ev = mne.EvokedArray(eval_batch.x[trial], info_mne, tmin=0)
        try:
            solver.make_inverse_operator(fwd_est, ev, alpha="auto", verbose=0)
            stc = solver.apply_inverse_operator(ev)
            y_true = np.abs(eval_batch.y[trial]).mean(axis=1)
            y_pred = np.abs(stc.data).mean(axis=1)
            mle = eval_mle(y_true, y_pred, adj_true, adj_est,
                           fwd_true["source_rr"], fwd_est["source_rr"], D)
            emd = eval_emd(D, y_true, y_pred)
            mles.append(mle * 1000)
            emds.append(emd * 1000)
        except Exception:
            pass
    return np.median(mles), np.median(emds)


def main():
    print("=" * 80)
    print("SH 23p MODEL: MNI-ALIGNED EVALUATION")
    print("=" * 80)

    # Setup
    print("\nSetup...")
    fwd_true = build_sample_forward(verbose=0)
    n_ch = fwd_true["sol"]["data"].shape[0]
    info_mne = mne.create_info(
        ch_names=fwd_true["info"]["ch_names"], sfreq=250., ch_types="eeg")
    info_fs = get_info()
    trans_fs = os.path.join(FS_DIR, "bem", "fsaverage-trans.fif")

    # Sensor covariance
    print("  Sensor covariance (seed=42)...")
    config_cov = SimulationConfig(
        batch_size=500, n_sources=(1, 5), n_timepoints=50,
        snr_range=(0, 10), verbose=0, random_seed=42)
    C_data = np.zeros((n_ch, n_ch))
    gen = SimulationGenerator(fwd_true, config_cov)
    for _ in range(10):
        batch = next(gen.generate())
        for s in range(batch.x.shape[0]):
            x = batch.x[s]
            C_data += x @ x.T / x.shape[1]
    C_data /= 5000

    # Eval data
    print("  Eval trials (seed=99, SNR 5-15)...")
    eval_config = SimulationConfig(
        batch_size=N_TRIALS, n_sources=2, n_timepoints=200,
        snr_range=(5, 15), verbose=0, random_seed=99)
    eval_batch = next(SimulationGenerator(fwd_true, eval_config).generate())

    # BEM
    print("  BEM (ico3)...")
    bem_model = mne.make_bem_model("fsaverage", ico=3,
        conductivity=(0.3, 0.006, 0.3), subjects_dir=SUBJECTS_DIR_FS, verbose=0)
    bem, coeff, coeff_h = make_bem_solution_factored(bem_model)
    skull_rr_orig = bem_model[2]["rr"].copy()
    skull_center = skull_rr_orig.mean(axis=0)
    radii_base = np.linalg.norm(skull_rr_orig - skull_center, axis=1)
    dirs_skull = (skull_rr_orig - skull_center) / radii_base[:, None]
    theta_skull, phi_skull = cart_to_sph(skull_rr_orig * 1000, skull_center * 1000)
    Y_skull = real_sh_matrix(theta_skull, phi_skull, l_max=3)

    src_fs = mne.setup_source_space("fsaverage", spacing="ico3", surface="white",
        subjects_dir=SUBJECTS_DIR_FS, add_dist=False, verbose=0)
    pos_fs = np.vstack([src_fs[0]["rr"][src_fs[0]["vertno"]].copy(),
                         src_fs[1]["rr"][src_fs[1]["vertno"]].copy()])
    nn_fs = np.vstack([src_fs[0]["nn"][src_fs[0]["vertno"]].copy(),
                        src_fs[1]["nn"][src_fs[1]["vertno"]].copy()])
    src_center = pos_fs.mean(axis=0)
    n_lh = len(src_fs[0]["vertno"])

    def _apply(pts, ctr, off, stch, rot):
        p = pts - ctr
        if np.any(rot != 0):
            R = Rotation.from_euler("XYZ", rot, degrees=True).as_matrix()
            p = p @ R.T
        return p * stch + ctr + off

    def build_fwd(sigma_skull, sigma_scalp, sh_coeffs=None):
        if sh_coeffs is not None:
            full_c = np.zeros(16)
            full_c[4:4+len(sh_coeffs)] = sh_coeffs
            delta_r = Y_skull @ full_c
            skull_rr = skull_center + (radii_base + delta_r)[:, None] * dirs_skull
        else:
            skull_rr = skull_rr_orig.copy()
        skull_rr = _apply(skull_rr, skull_center, OFF9, STR9, ROT9)
        bem["surfs"][2]["rr"] = skull_rr
        _check_complete_surface(bem["surfs"][2])
        coeff_up = partial_K_update(bem["surfs"], coeff, changed_idx=2)
        bem["surfs"][2]["rr"] = skull_rr_orig
        rebuild_bem_from_coeff(bem, coeff_up, coeff_h, (sigma_scalp, sigma_skull, 0.3))
        pos = _apply(pos_fs, src_center, OFF9, STR9, ROT9)
        nn = nn_fs @ Rotation.from_euler("XYZ", ROT9, degrees=True).as_matrix().T
        src = src_fs.copy()
        for i, h in enumerate(src):
            sl = slice(0, n_lh) if i == 0 else slice(n_lh, None)
            h["rr"] = h["rr"].copy(); h["rr"][h["vertno"]] = pos[sl]
            h["nn"] = h["nn"].copy(); h["nn"][h["vertno"]] = nn[sl]
        fwd = mne.make_forward_solution(info_fs, trans=trans_fs, src=src,
            bem=bem, eeg=True, mindist=0.0, verbose=0)
        return mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, verbose=0)

    # ── Build models ─────────────────────────────────────────────────────
    print("\nBuilding models...")

    # Vanilla
    fwd_vanilla = mne.make_forward_solution(
        info_fs, trans=trans_fs, src=src_fs,
        bem=os.path.join(FS_DIR, "bem", "fsaverage-5120-5120-5120-bem-sol.fif"),
        eeg=True, mindist=0.0, verbose=0)
    fwd_vanilla = mne.convert_forward_solution(
        fwd_vanilla, surf_ori=True, force_fixed=True, verbose=0)

    # Riemannian LF (α=0.80, bures+geodesic)
    fwd_riem = riemannian_leadfield_transport(C_data, fwd_vanilla)

    # Co-deform only (no SH)
    fwd_codeform = build_fwd(0.016, 0.30, sh_coeffs=None)

    # SH 23p at different sigmas
    sh_models = {}
    for ss, sc in SIGMA_CANDIDATES:
        label = f"SH σ={ss*1000:.0f}/{sc:.2f}"
        fwd_sh = build_fwd(ss, sc, SH_EXP3)
        sh_models[label] = fwd_sh

    # SH 23p + Riemannian on top (best σ)
    fwd_sh_best = sh_models["SH σ=16/0.30"]
    fwd_sh_riem = riemannian_leadfield_transport(C_data, fwd_sh_best)

    print("  All models built.")

    # ── Evaluate ─────────────────────────────────────────────────────────
    print(f"\nEvaluating with {SOLVER} ({N_TRIALS} trials)...\n")

    all_models = [
        ("Vanilla", fwd_vanilla),
        ("Riemannian LF (α=0.80)", fwd_riem),
        ("Co-deform (9p)", fwd_codeform),
    ]
    for label, fwd in sh_models.items():
        all_models.append((label, fwd))
    all_models.append(("SH + Riemannian LF", fwd_sh_riem))
    all_models.append(("Oracle", fwd_true))

    print(f"{'Model':<30s} {'head MLE':>8s} {'head EMD':>8s} "
          f"{'MNI MLE':>8s} {'MNI EMD':>8s} {'n':>5s}")
    print("-" * 75)

    for label, fwd_est in all_models:
        D_head = cdist(fwd_true["source_rr"], fwd_est["source_rr"])
        D_mni = build_D_mni(fwd_true, fwd_est)
        mle_h, emd_h = evaluate(fwd_est, fwd_true, eval_batch, info_mne, D_head)
        mle_m, emd_m = evaluate(fwd_est, fwd_true, eval_batch, info_mne, D_mni)
        n = fwd_est["sol"]["data"].shape[1]
        print(f"  {label:<28s} {mle_h:7.1f}  {emd_h:7.1f}  "
              f"{mle_m:7.1f}  {emd_m:7.1f}  {n:4d}")

    print("-" * 75)
    print("=" * 80)


if __name__ == "__main__":
    main()
