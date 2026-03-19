"""Riemannian Transport of the Leadfield (not the data).

Instead of x_aligned = W @ x, we define L_pseudo = W^{-1} @ L_template
and solve with the raw, untouched subject data. This preserves noise
structure and lets the solver adapt regularization to the warped geometry.

Math: if W @ x ≈ L_tpl @ s, then x ≈ W^{-1} @ L_tpl @ s = L_pseudo @ s.

Usage:
    cd invert-package
    .venv/bin/python scripts/iter_riemannian_leadfield.py
"""

import os, sys, importlib.util
import mne
import numpy as np
from scipy.linalg import sqrtm, inv
from scipy.spatial.distance import cdist

from invert import Solver
from invert.simulate import SimulationConfig, SimulationGenerator
from invert.forward.forward import _get_fsaverage_dir, get_info

# Eval functions
import invert as _invert_pkg
_eval_path = os.path.join(os.path.dirname(_invert_pkg.__file__), "evaluate", "evaluate.py")
_spec = importlib.util.spec_from_file_location("invert.evaluate.evaluate", _eval_path, submodule_search_locations=[])
_eval_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _eval_mod
_spec.loader.exec_module(_eval_mod)
eval_emd = _eval_mod.eval_emd
eval_mean_localization_error = _eval_mod.eval_mean_localization_error

sys.path.insert(0, os.path.dirname(__file__))
from eval_hybrid_morphing import build_sample_forward

SAMPLE_PATH = mne.datasets.sample.data_path()
FS_DIR = _get_fsaverage_dir(verbose=0)
SUBJECTS_DIR_FS = os.path.dirname(FS_DIR)

N_TRIALS = 30
EVAL_SOLVER = "ESMV"
RANDOM_SEED = 1


def evaluate(fwd_est, fwd_true, eval_batch, info_mne, data_x=None):
    """Evaluate with optional custom data."""
    pos_true = fwd_true["source_rr"]
    pos_est = fwd_est["source_rr"]
    adj_true = mne.spatial_src_adjacency(fwd_true["src"], verbose=0)
    adj_est = mne.spatial_src_adjacency(fwd_est["src"], verbose=0)
    D_cross = cdist(pos_true, pos_est)
    solver = Solver(EVAL_SOLVER)
    x = data_x if data_x is not None else eval_batch.x
    mles, emds = [], []
    for trial in range(N_TRIALS):
        evoked = mne.EvokedArray(x[trial], info_mne, tmin=0)
        try:
            solver.make_inverse_operator(fwd_est, evoked, alpha="auto", verbose=0)
            stc = solver.apply_inverse_operator(evoked)
            y_true = np.abs(eval_batch.y[trial]).mean(axis=1)
            y_pred = np.abs(stc.data).mean(axis=1)
            mle = eval_mean_localization_error(
                y_true, y_pred, adj_true, adj_est, pos_true, pos_est, D_cross)
            emd = eval_emd(D_cross, y_true, y_pred)
            mles.append(mle * 1000)
            emds.append(emd * 1000)
        except Exception:
            pass
    return np.median(mles), np.median(emds)


def compute_W(C_data, G_tpl, alpha, reg=0.01):
    """Compute partial Riemannian transport matrix."""
    n = G_tpl.shape[0]
    C_tpl = G_tpl @ G_tpl.T
    C_tpl += reg * np.trace(C_tpl) / n * np.eye(n)
    C_tpl /= np.trace(C_tpl) / n

    C_s = C_data.copy()
    C_s += reg * np.trace(C_s) / n * np.eye(n)
    C_s /= np.trace(C_s) / n

    W_full = sqrtm(C_tpl).real @ inv(sqrtm(C_s).real)
    return alpha * W_full + (1 - alpha) * np.eye(n)


def make_pseudo_fwd(fwd_tpl, W_inv):
    """Create a pseudo forward model with L_pseudo = W^{-1} @ L_template.

    Modifies the forward solution's leadfield matrix in-place-ish
    (creates a shallow copy with modified sol data).
    """
    fwd_pseudo = fwd_tpl.copy()
    G_tpl = fwd_tpl["sol"]["data"]
    G_pseudo = W_inv @ G_tpl
    # MNE Forward stores sol as a dict-like; we need to replace the data
    fwd_pseudo["sol"] = dict(fwd_pseudo["sol"])
    fwd_pseudo["sol"]["data"] = G_pseudo
    # Also update sol_grad if present
    if fwd_pseudo.get("_orig_sol") is not None:
        fwd_pseudo["_orig_sol"] = G_pseudo
    return fwd_pseudo


def main():
    print("=" * 60)
    print("RIEMANNIAN LEADFIELD TRANSPORT")
    print("=" * 60)

    # Setup
    print("\nSetup...")
    fwd_true = build_sample_forward(verbose=0)
    n_ch = fwd_true["sol"]["data"].shape[0]
    info_mne = mne.create_info(
        ch_names=fwd_true["info"]["ch_names"], sfreq=250., ch_types="eeg")

    # Sensor covariance
    print("  Sensor covariance...")
    config_cov = SimulationConfig(
        batch_size=500, n_sources=(1, 5), n_timepoints=50,
        snr_range=(0, 10), verbose=0, random_seed=RANDOM_SEED)
    C_data = np.zeros((n_ch, n_ch))
    gen = SimulationGenerator(fwd_true, config_cov)
    for _ in range(10):
        batch = next(gen.generate())
        for s in range(batch.x.shape[0]):
            x = batch.x[s]
            C_data += x @ x.T / x.shape[1]
    C_data /= 5000

    # Template forward model
    print("  Template forward model...")
    info_fs = get_info()
    trans_fs = os.path.join(FS_DIR, "bem", "fsaverage-trans.fif")
    src = mne.setup_source_space("fsaverage", spacing="ico3", surface="white",
        subjects_dir=SUBJECTS_DIR_FS, add_dist=False, verbose=0)
    fwd_tpl = mne.make_forward_solution(
        info_fs, trans=trans_fs, src=src,
        bem=os.path.join(FS_DIR, "bem", "fsaverage-5120-5120-5120-bem-sol.fif"),
        eeg=True, mindist=0.0, verbose=0)
    fwd_tpl = mne.convert_forward_solution(
        fwd_tpl, surf_ori=True, force_fixed=True, verbose=0)
    G_tpl = fwd_tpl["sol"]["data"]

    # Eval data
    print("  Eval trials...")
    eval_config = SimulationConfig(
        batch_size=N_TRIALS, n_sources=2, n_timepoints=200,
        snr_range=(50, 50), verbose=0, random_seed=RANDOM_SEED + 1)
    eval_batch = next(SimulationGenerator(fwd_true, eval_config).generate())

    # ── Baseline ─────────────────────────────────────────────────────────
    print("\nBaselines...")
    mle_v, emd_v = evaluate(fwd_tpl, fwd_true, eval_batch, info_mne)
    print(f"  Vanilla:           MLE={mle_v:.1f}, EMD={emd_v:.1f}")

    # ── Alpha sweep: data transport (reference) ──────────────────────────
    print("\nData transport (reference):")
    for alpha in [0.2, 0.4, 0.6, 0.8, 1.0]:
        W = compute_W(C_data, G_tpl, alpha)
        x_aligned = np.array([W @ eval_batch.x[i] for i in range(N_TRIALS)])
        mle, emd = evaluate(fwd_tpl, fwd_true, eval_batch, info_mne, x_aligned)
        print(f"  α={alpha:.1f} data:      MLE={mle:.1f}, EMD={emd:.1f}")

    # ── Alpha sweep: leadfield transport ─────────────────────────────────
    print("\nLeadfield transport (L_pseudo = W^{-1} @ L_tpl):")
    for alpha in [0.2, 0.4, 0.6, 0.8, 1.0]:
        W = compute_W(C_data, G_tpl, alpha)
        W_inv = inv(W)
        fwd_pseudo = make_pseudo_fwd(fwd_tpl, W_inv)
        mle, emd = evaluate(fwd_pseudo, fwd_true, eval_batch, info_mne)
        print(f"  α={alpha:.1f} leadfield: MLE={mle:.1f}, EMD={emd:.1f}")

    # ── Finer sweep around best region ───────────────────────────────────
    print("\nFine sweep (leadfield transport):")
    for alpha in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        W = compute_W(C_data, G_tpl, alpha)
        W_inv = inv(W)
        fwd_pseudo = make_pseudo_fwd(fwd_tpl, W_inv)
        mle, emd = evaluate(fwd_pseudo, fwd_true, eval_batch, info_mne)
        print(f"  α={alpha:.2f} leadfield: MLE={mle:.1f}, EMD={emd:.1f}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Reference: Vanilla=24.6, Riemannian data α=0.8: 20.8")
    print("Reference: Combined=18.9, Co-deform+SH+σ=14.4")
    print("=" * 60)


if __name__ == "__main__":
    main()
