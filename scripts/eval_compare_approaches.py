"""Head-to-head comparison: Depth Mapping vs Template Tuning.

Evaluates both approaches on the same cross-subject benchmark:
- **Depth mapping**: Estimate per-vertex cortical depth from EEG (unsupervised
  PCA on eLORETA features), build forward model with mesh normals.
- **Template tuning**: Optimize 11 global parameters (conductivity, offset,
  stretch, rotation) of the fsaverage template via subspace alignment.

Both are evaluated with the same 30-trial MLE + EMD benchmark using the
MNE sample subject as the unknown individual and fsaverage as the template.

Produces a multi-panel PyVista figure comparing source reconstructions.

Usage:
    cd invert-package
    .venv/bin/python scripts/eval_compare_approaches.py
"""

import os
import sys

import mne
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from invert import Solver
from invert.simulate import SimulationConfig, SimulationGenerator
from invert.forward.forward import _get_fsaverage_dir, get_info
from invert.cortical_folding.fast_bem import (
    make_bem_solution_factored,
    rebuild_bem_from_coeff,
    partial_K_update,
)
from invert.cortical_folding.cache import load_params
from invert.cortical_folding.sh_analysis import predict_depth_from_eeg
from mne.bem import _check_complete_surface

# Re-use helpers from eval_hybrid_morphing
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from eval_hybrid_morphing import (
    build_sample_forward,
    build_sphere_forward_for_subject,
    build_hybrid_forward,
    compute_mesh_normals,
    _compact_tris,
    _cap_depth_to_skull,
)

# Direct import of evaluation functions (bypass broken __init__.py chain)
import importlib.util
import invert as _invert_pkg
_eval_path = os.path.join(
    os.path.dirname(_invert_pkg.__file__), "evaluate", "evaluate.py"
)
_spec = importlib.util.spec_from_file_location(
    "invert.evaluate.evaluate", _eval_path, submodule_search_locations=[],
)
_eval_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _eval_mod
_spec.loader.exec_module(_eval_mod)
eval_emd = _eval_mod.eval_emd
eval_mean_localization_error = _eval_mod.eval_mean_localization_error

# ── Paths ────────────────────────────────────────────────────────────────
FIG_DIR = os.path.join(
    os.path.dirname(_invert_pkg.__file__), "cortical_folding"
)
SAMPLE_PATH = mne.datasets.sample.data_path()
SUBJECTS_DIR_SAMPLE = os.path.join(SAMPLE_PATH, "subjects")
FS_DIR = _get_fsaverage_dir(verbose=0)
SUBJECTS_DIR_FS = os.path.dirname(FS_DIR)

N_TRIALS = 30
EVAL_SOLVER = "ESMV"
RANDOM_SEED = 1


# ── Template Tuning forward model builder ────────────────────────────────

def build_template_tuning_fwd(info, C_data):
    """Build the winning template tuning forward model (co-deform + rotation).

    Loads cached parameters or uses hardcoded defaults.
    """
    cached = load_params(C_data)
    if cached is not None and cached["co_deform"]:
        sig_skull = cached["sigma_skull"]
        sig_scalp = cached["sigma_scalp"]
        offset = cached["offset"]
        stretch = cached["stretch"]
        rotation = cached["rotation"]
        cost = cached["cost"]
        print(f"  Template tuning: loaded from cache (cost={cost:.5f})")
    else:
        sig_skull = 0.0209
        sig_scalp = 0.210
        offset = np.array([-0.0003, 0.0069, 0.0099])
        stretch = np.array([1.005, 0.992, 0.985])
        rotation = np.array([-1.9, -0.8, -2.3])
        print("  Template tuning: using hardcoded params")

    print(f"    σ_skull={sig_skull*1000:.1f} mS/m, σ_scalp={sig_scalp*1000:.0f} mS/m")
    print(f"    offset=[{offset[0]*1000:.1f}, {offset[1]*1000:.1f}, "
          f"{offset[2]*1000:.1f}] mm")
    print(f"    stretch=[{stretch[0]:.3f}, {stretch[1]:.3f}, {stretch[2]:.3f}]")
    print(f"    rotation=[{rotation[0]:.1f}, {rotation[1]:.1f}, "
          f"{rotation[2]:.1f}] deg")

    # Build BEM
    trans_fs = os.path.join(FS_DIR, "bem", "fsaverage-trans.fif")
    bem_model = mne.make_bem_model(
        "fsaverage", ico=3, conductivity=(0.3, 0.006, 0.3),
        subjects_dir=SUBJECTS_DIR_FS, verbose=0,
    )
    bem, coeff, coeff_h = make_bem_solution_factored(bem_model)
    skull_rr_orig = bem_model[2]["rr"].copy()
    skull_center = skull_rr_orig.mean(axis=0)

    src = mne.setup_source_space(
        "fsaverage", spacing="ico3", surface="white",
        subjects_dir=SUBJECTS_DIR_FS, add_dist=False, verbose=0,
    )
    pos_base = np.vstack([
        src[0]["rr"][src[0]["vertno"]].copy(),
        src[1]["rr"][src[1]["vertno"]].copy(),
    ])
    nn_base = np.vstack([
        src[0]["nn"][src[0]["vertno"]].copy(),
        src[1]["nn"][src[1]["vertno"]].copy(),
    ])
    center = pos_base.mean(axis=0)
    n_lh = len(src[0]["vertno"])

    def _apply_transform(pts, ctr, off, stch, rot):
        p = pts - ctr
        if np.any(rot != 0):
            R = Rotation.from_euler("XYZ", rot, degrees=True).as_matrix()
            p = p @ R.T
        return p * stch + ctr + off

    # Co-deform inner skull
    new_skull_rr = _apply_transform(
        skull_rr_orig, skull_center, offset, stretch, rotation
    )
    bem["surfs"][2]["rr"] = new_skull_rr
    _check_complete_surface(bem["surfs"][2])
    coeff_updated = partial_K_update(bem["surfs"], coeff, changed_idx=2)
    bem["surfs"][2]["rr"] = skull_rr_orig
    rebuild_bem_from_coeff(
        bem, coeff_updated, coeff_h, (sig_scalp, sig_skull, 0.3)
    )

    # Deform sources
    pos = _apply_transform(pos_base, center, offset, stretch, rotation)
    nn = nn_base.copy()
    if np.any(rotation != 0):
        R = Rotation.from_euler("XYZ", rotation, degrees=True).as_matrix()
        nn = nn @ R.T

    for i, h in enumerate(src):
        sl = slice(0, n_lh) if i == 0 else slice(n_lh, None)
        h["rr"] = h["rr"].copy()
        h["rr"][h["vertno"]] = pos[sl]
        h["nn"] = h["nn"].copy()
        h["nn"][h["vertno"]] = nn[sl]

    fwd = mne.make_forward_solution(
        info, trans=trans_fs, src=src, bem=bem,
        eeg=True, mindist=0.0, verbose=0,
    )
    return mne.convert_forward_solution(
        fwd, surf_ori=True, force_fixed=True, verbose=0,
    )


# ── Combined approach: depth-informed template tuning ─────────────────────

def extract_shape_params(pos_morphed, pos_fs):
    """Extract stretch from two corresponding ico3 point clouds.

    Uses eigenvalue ratios of the spatial covariance matrix, which are
    fully coordinate-frame-independent (no rotation needed).  The eigenvalue
    ratios give per-principal-axis size ratios.  These are mapped to the
    fsaverage head frame via the template's PCA axes.

    Rotation is set to zero — the optimal template tuning rotation for the
    sample subject is <3 degrees, and extracting it requires resolving the
    coordinate frame ambiguity which dominates any real signal.

    Parameters
    ----------
    pos_morphed : (n, 3) — depth-mapped positions (sample head coords), meters
    pos_fs : (n, 3) — fsaverage source positions (fsaverage head coords), meters

    Returns
    -------
    stretch : (3,) per-axis stretch factors (in fsaverage head frame)
    rotation : (3,) Euler angles in degrees (all zeros)
    """
    # Center both
    P = pos_morphed - pos_morphed.mean(axis=0)
    Q = pos_fs - pos_fs.mean(axis=0)

    # Eigenvalues of spatial covariance = squared principal axis lengths
    # These are coordinate-frame-independent
    cov_m = P.T @ P / len(P)
    cov_f = Q.T @ Q / len(Q)
    vals_m = np.sort(np.linalg.eigvalsh(cov_m))[::-1]  # descending
    vals_f = np.sort(np.linalg.eigvalsh(cov_f))[::-1]

    # Per-principal-axis stretch
    stretch_pca = np.sqrt(vals_m / vals_f)

    # Map to fsaverage head frame: the template's PCA axes tell us
    # which head-frame axis corresponds to which principal axis
    _, vecs_f = np.linalg.eigh(cov_f)
    vecs_f = vecs_f[:, ::-1]  # descending order

    # The dominant PCA axis of fsaverage (AP direction) maps to y in head frame
    # For each head-frame axis, find which principal axis it aligns with
    stretch_head = np.ones(3)
    for head_ax in range(3):
        # Which PCA axis is most aligned with this head axis?
        alignment = np.abs(vecs_f[head_ax, :])
        best_pca = np.argmax(alignment)
        stretch_head[head_ax] = stretch_pca[best_pca]

    print(f"    PCA axis lengths (mm): morphed={np.sqrt(vals_m)*1000}, "
          f"fs={np.sqrt(vals_f)*1000}")
    print(f"    PCA stretch: {stretch_pca}")
    print(f"    Head-frame stretch: {stretch_head}")

    rotation_deg = np.zeros(3)
    return stretch_head, rotation_deg


def build_combined_fwd(info, C_data, stretch, rotation):
    """Build forward model with depth-derived stretch+rotation, optimized offset.

    Fixes stretch and rotation from depth mapping, fixes conductivity at
    r_skull=8 boundary (σ_skull=37.5 mS/m, σ_scalp=0.51 S/m), and
    optimizes only offset (3 params) via Powell.

    Uses ico2 BEM for fast optimization (~0.2s per K rebuild), then
    rebuilds the final model at ico3.
    """
    trans_fs = os.path.join(FS_DIR, "bem", "fsaverage-trans.fif")

    # Fixed conductivity: r_skull=8 (previous experiments' boundary optimum)
    sig_skull = 0.3 / 8     # = 0.0375 S/m = 37.5 mS/m
    sig_scalp = 0.3 * 1.7   # = 0.51 S/m

    # --- ico2 BEM for fast optimization ---
    bem_model_opt = mne.make_bem_model(
        "fsaverage", ico=2, conductivity=(0.3, 0.006, 0.3),
        subjects_dir=SUBJECTS_DIR_FS, verbose=0,
    )
    bem_opt, coeff_opt, coeff_h_opt = make_bem_solution_factored(bem_model_opt)
    skull_rr_orig_opt = bem_model_opt[2]["rr"].copy()
    skull_center_opt = skull_rr_orig_opt.mean(axis=0)

    # Source space (always ico3 for source positions)
    src_base = mne.setup_source_space(
        "fsaverage", spacing="ico3", surface="white",
        subjects_dir=SUBJECTS_DIR_FS, add_dist=False, verbose=0,
    )
    pos_base = np.vstack([
        src_base[0]["rr"][src_base[0]["vertno"]].copy(),
        src_base[1]["rr"][src_base[1]["vertno"]].copy(),
    ])
    nn_base = np.vstack([
        src_base[0]["nn"][src_base[0]["vertno"]].copy(),
        src_base[1]["nn"][src_base[1]["vertno"]].copy(),
    ])
    center = pos_base.mean(axis=0)
    n_lh = len(src_base[0]["vertno"])

    # Precompute rotation matrix (fixed)
    R_rot = Rotation.from_euler("XYZ", rotation, degrees=True).as_matrix()

    def _apply_transform(pts, ctr, off, stch):
        p = pts - ctr
        p = p @ R_rot.T
        return p * stch + ctr + off

    # Subspace alignment cost (precompute data eigenvectors once)
    _, U_d = np.linalg.eigh(C_data)
    U_d = U_d[:, -10:]

    def subspace_cost(G):
        _, U_m = np.linalg.eigh(G @ G.T)
        U_m = U_m[:, -10:]
        S = np.linalg.svd(U_d.T @ U_m, compute_uv=False)
        return 1 - np.mean(S**2)

    n_eval = [0]
    best_cost = [np.inf]

    def offset_cost(params):
        offset = np.array(params)
        n_eval[0] += 1
        try:
            # Co-deform inner skull (ico2 BEM)
            new_skull_rr = _apply_transform(
                skull_rr_orig_opt, skull_center_opt, offset, stretch,
            )
            bem_opt["surfs"][2]["rr"] = new_skull_rr
            _check_complete_surface(bem_opt["surfs"][2])
            coeff_up = partial_K_update(
                bem_opt["surfs"], coeff_opt, changed_idx=2,
            )
            bem_opt["surfs"][2]["rr"] = skull_rr_orig_opt
            # Rebuild γ only (K is cached in coeff_up)
            rebuild_bem_from_coeff(
                bem_opt, coeff_up, coeff_h_opt,
                (sig_scalp, sig_skull, 0.3),
            )

            # Deform sources
            pos = _apply_transform(pos_base, center, offset, stretch)
            nn = nn_base @ R_rot.T

            src = src_base.copy()
            for i, h in enumerate(src):
                sl = slice(0, n_lh) if i == 0 else slice(n_lh, None)
                h["rr"] = h["rr"].copy()
                h["rr"][h["vertno"]] = pos[sl]
                h["nn"] = h["nn"].copy()
                h["nn"][h["vertno"]] = nn[sl]

            fwd = mne.make_forward_solution(
                info, trans=trans_fs, src=src, bem=bem_opt,
                eeg=True, mindist=0.0, verbose=0,
            )
            fwd = mne.convert_forward_solution(
                fwd, surf_ori=True, force_fixed=True, verbose=0,
            )
            cost = subspace_cost(fwd["sol"]["data"])
            if cost < best_cost[0]:
                best_cost[0] = cost
            if n_eval[0] % 20 == 0:
                print(f"    eval {n_eval[0]}, cost={best_cost[0]:.5f}")
            return cost
        except Exception:
            return 1.0

    from scipy.optimize import minimize as sp_minimize

    print("  Optimizing offset (Powell, ico2 BEM, fixed stretch+rotation+σ)...")
    res = sp_minimize(offset_cost, [0, 0, 0], method="Powell",
                      options={"maxfev": 80, "ftol": 1e-4})
    offset_opt = res.x
    print(f"    offset=[{offset_opt[0]*1000:.1f}, {offset_opt[1]*1000:.1f}, "
          f"{offset_opt[2]*1000:.1f}] mm")
    print(f"    cost={best_cost[0]:.5f} ({n_eval[0]} evals)")

    # --- Final model at ico3 BEM ---
    print("  Rebuilding at ico3 for final model...")
    bem_model_final = mne.make_bem_model(
        "fsaverage", ico=3, conductivity=(0.3, 0.006, 0.3),
        subjects_dir=SUBJECTS_DIR_FS, verbose=0,
    )
    bem_f, coeff_f, coeff_h_f = make_bem_solution_factored(bem_model_final)
    skull_rr_f = bem_model_final[2]["rr"].copy()
    skull_center_f = skull_rr_f.mean(axis=0)

    new_skull_rr = _apply_transform(
        skull_rr_f, skull_center_f, offset_opt, stretch,
    )
    bem_f["surfs"][2]["rr"] = new_skull_rr
    _check_complete_surface(bem_f["surfs"][2])
    coeff_up = partial_K_update(bem_f["surfs"], coeff_f, changed_idx=2)
    bem_f["surfs"][2]["rr"] = skull_rr_f
    rebuild_bem_from_coeff(
        bem_f, coeff_up, coeff_h_f, (sig_scalp, sig_skull, 0.3),
    )

    pos = _apply_transform(pos_base, center, offset_opt, stretch)
    nn = nn_base @ R_rot.T
    src = src_base.copy()
    for i, h in enumerate(src):
        sl = slice(0, n_lh) if i == 0 else slice(n_lh, None)
        h["rr"] = h["rr"].copy()
        h["rr"][h["vertno"]] = pos[sl]
        h["nn"] = h["nn"].copy()
        h["nn"][h["vertno"]] = nn[sl]

    fwd = mne.make_forward_solution(
        info, trans=trans_fs, src=src, bem=bem_f,
        eeg=True, mindist=0.0, verbose=0,
    )
    return mne.convert_forward_solution(
        fwd, surf_ori=True, force_fixed=True, verbose=0,
    )


# ── Evaluation ───────────────────────────────────────────────────────────

def evaluate_model(fwd_est, fwd_true, eval_batch, info_mne, solver_name):
    """Evaluate a forward model with MLE + EMD over multiple trials."""
    pos_true = fwd_true["source_rr"]
    pos_est = fwd_est["source_rr"]
    adj_true = mne.spatial_src_adjacency(fwd_true["src"], verbose=0)
    adj_est = mne.spatial_src_adjacency(fwd_est["src"], verbose=0)
    D_cross = cdist(pos_true, pos_est)

    solver = Solver(solver_name)
    n_trials = eval_batch.x.shape[0]
    mles, emds = [], []
    for trial in range(n_trials):
        evoked = mne.EvokedArray(eval_batch.x[trial], info_mne, tmin=0)
        try:
            solver.make_inverse_operator(fwd_est, evoked, alpha="auto", verbose=0)
            stc = solver.apply_inverse_operator(evoked)
            y_true_avg = np.abs(eval_batch.y[trial]).mean(axis=1)
            y_pred_avg = np.abs(stc.data).mean(axis=1)
            mle = eval_mean_localization_error(
                y_true_avg, y_pred_avg,
                adj_true, adj_est, pos_true, pos_est, D_cross,
            )
            emd_val = eval_emd(D_cross, y_true_avg, y_pred_avg)
            mles.append(mle * 1000)
            emds.append(emd_val * 1000)
        except Exception as e:
            print(f"    Trial {trial} failed: {e}")
    return np.array(mles), np.array(emds)


# ── Riemannian covariance alignment ───────────────────────────────────────

def riemannian_align(C_data, fwd_tpl, eval_batch, reg=0.01):
    """Align subject's EEG to template geometry via Riemannian transport.

    Finds a linear transform W that maps the subject's sensor covariance
    structure to the template's expected covariance structure, then applies
    W to each trial.  The template inverse operator can then be applied
    to the aligned data as if it came from the template head.

    Parameters
    ----------
    C_data : (n_ch, n_ch) — subject's sensor covariance
    fwd_tpl : mne.Forward — template forward model
    eval_batch : batch object with .x (n_trials, n_ch, n_time)
    reg : float — regularization (fraction of trace added to diagonal)

    Returns
    -------
    aligned_x : (n_trials, n_ch, n_time) — aligned EEG data
    """
    from scipy.linalg import sqrtm, inv

    G_tpl = fwd_tpl["sol"]["data"]
    n_ch = G_tpl.shape[0]

    # Template "geometric" covariance: what sensor cov looks like through
    # the template forward model with unit-variance uncorrelated sources
    C_tpl = G_tpl @ G_tpl.T
    C_tpl += reg * np.trace(C_tpl) / n_ch * np.eye(n_ch)

    # Subject's data covariance (regularized)
    C_subj = C_data.copy()
    C_subj += reg * np.trace(C_subj) / n_ch * np.eye(n_ch)

    # Normalize both to unit trace so transport aligns spatial structure
    # only, not amplitude
    C_tpl = C_tpl / np.trace(C_tpl) * n_ch
    C_subj = C_subj / np.trace(C_subj) * n_ch

    # Riemannian transport: W = C_tpl^(1/2) @ C_subj^(-1/2)
    W = sqrtm(C_tpl).real @ inv(sqrtm(C_subj).real)

    # Apply to each trial
    n_trials = eval_batch.x.shape[0]
    aligned_x = np.array([W @ eval_batch.x[i] for i in range(n_trials)])

    print(f"    W condition number: {np.linalg.cond(W):.1f}")
    print(f"    Data scale change: {np.std(aligned_x) / np.std(eval_batch.x):.2f}x")

    return aligned_x


# ── Create FreeSurfer subject from forward model geometry ─────────────────

def create_depth_mapping_subject(fwd, name, subjects_dir):
    """Write a FreeSurfer subject from the depth mapping's compact ico3 mesh.

    The depth mapping forward model has ~642 active vertices per hemisphere
    on a deformed sphere.  We write a compact mesh (only active vertices)
    so the rendered brain shows the actual estimated geometry.

    Returns (subject_name, remapped_stc_func) where remapped_stc_func takes
    an STC from the depth mapping solver and returns one with vertex indices
    matching the compact mesh.
    """
    src = fwd["src"]
    subj_dir = os.path.join(subjects_dir, name, "surf")
    os.makedirs(subj_dir, exist_ok=True)
    new_vertno = []
    for i, hemi_name in enumerate(["lh", "rh"]):
        hemi = src[i]
        vertno = hemi["vertno"]
        pos = hemi["rr"][vertno] * 1000  # m -> mm
        # Remap triangulation to compact indices
        tris = hemi["use_tris"]
        if tris.max() >= len(vertno):
            vmap = np.full(hemi["rr"].shape[0], -1, dtype=int)
            vmap[vertno] = np.arange(len(vertno))
            tris = vmap[tris]
            valid = np.all(tris >= 0, axis=1)
            tris = tris[valid]
        out_path = os.path.join(subj_dir, f"{hemi_name}.white")
        mne.write_surface(out_path, pos.astype(np.float32), tris, overwrite=True)
        new_vertno.append(np.arange(len(vertno)))

    n_lh = len(src[0]["vertno"])

    def remap_stc(stc):
        """Remap STC vertex indices to match the compact mesh."""
        return mne.SourceEstimate(
            stc.data, new_vertno, stc.tmin, stc.tstep, subject=name,
        )

    return name, remap_stc


# ── Multi-panel figure using MNE's stc.plot() ───────────────────────────

def _render_brain(stc, subject, subjects_dir, peak_time, view="lateral"):
    """Render a brain plot and return the image as a numpy array."""
    import tempfile
    brain = stc.plot(
        subjects_dir=subjects_dir, subject=subject,
        surface="white", hemi="both", cortex="low_contrast",
        smoothing_steps=10, initial_time=peak_time,
        time_unit="s", size=(800, 600), verbose=0,
        background="white", show_traces=False, colorbar=False,
    )
    brain.show_view(view)
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    brain.save_image(f.name)
    brain.close()
    import matplotlib.image as mpimg
    img = mpimg.imread(f.name)
    # Crop: remove bottom 15% (time label) and top 5% (padding)
    h = img.shape[0]
    img = img[int(h * 0.02):int(h * 0.88), :]
    return img


def make_comparison_figure(
    stc_true, stc_vanilla, stc_depth, stc_tuning,
    fwd_true, fwd_vanilla, fwd_depth, fwd_tuning,
    results, peak_time, view="lateral",
    depth_subject="depth_mapped", depth_subjects_dir=None,
):
    """Create a 2x2 composite figure using MNE brain rendering + matplotlib."""
    import matplotlib.pyplot as plt

    if depth_subjects_dir is None:
        depth_subjects_dir = SUBJECTS_DIR_FS

    stc_true.subject = "sample"
    stc_vanilla.subject = "fsaverage"
    stc_depth.subject = depth_subject
    stc_tuning.subject = "fsaverage"

    panels = [
        ("Ground truth (sample MRI)", stc_true, "sample", SUBJECTS_DIR_SAMPLE),
        ("Vanilla template", stc_vanilla, "fsaverage", SUBJECTS_DIR_FS),
        ("Depth mapping", stc_depth, depth_subject, depth_subjects_dir),
        ("Template tuning", stc_tuning, "fsaverage", SUBJECTS_DIR_FS),
    ]

    imgs = []
    for title, stc, subj, sdir in panels:
        print(f"    Rendering {title} ({view})...")
        img = _render_brain(stc, subj, sdir, peak_time, view)
        imgs.append((title, img))

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.patch.set_facecolor("white")

    for idx, (title, img) in enumerate(imgs):
        ax = axes[idx // 2, idx % 2]
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

        if title == "Ground truth (sample MRI)":
            label = title
        else:
            mle_med = np.median(results[title][0])
            emd_med = np.median(results[title][1])
            n_src = results[title][2]
            label = (f"{title}\n"
                     f"MLE {mle_med:.1f} mm  |  EMD {emd_med:.1f} mm  |  "
                     f"{n_src} src")

        ax.set_title(label, fontsize=13, fontweight="bold", loc="left", pad=8)

    plt.tight_layout(pad=1.5)
    return fig


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("HEAD-TO-HEAD: DEPTH MAPPING vs TEMPLATE TUNING")
    print("=" * 70)

    # ── 1. Ground truth ──────────────────────────────────────────────────
    print("\n[1/6] Building ground truth (sample subject)...")
    fwd_true = build_sample_forward(verbose=0)
    n_ch, n_dip = fwd_true["sol"]["data"].shape
    print(f"  {n_ch} channels, {n_dip} dipoles")

    info_mne = mne.create_info(
        ch_names=fwd_true["info"]["ch_names"], sfreq=250.0, ch_types="eeg",
    )

    # ── 2. Simulate sensor covariance ────────────────────────────────────
    print("\n[2/6] Simulating sensor covariance (5000 samples)...")
    config_cov = SimulationConfig(
        batch_size=500, n_sources=(1, 5), n_timepoints=50,
        snr_range=(0, 10), verbose=0, random_seed=RANDOM_SEED,
    )
    C_data = np.zeros((n_ch, n_ch))
    gen = SimulationGenerator(fwd_true, config_cov)
    for _ in range(10):
        batch = next(gen.generate())
        for s in range(batch.x.shape[0]):
            x = batch.x[s]
            C_data += x @ x.T / x.shape[1]
    C_data /= 5000
    print("  Done.")

    # ── 3. Build vanilla template ────────────────────────────────────────
    print("\n[3/6] Building vanilla template forward model...")
    info_fs = get_info()
    trans_fs = os.path.join(FS_DIR, "bem", "fsaverage-trans.fif")
    src_vanilla = mne.setup_source_space(
        "fsaverage", spacing="ico3", surface="white",
        subjects_dir=SUBJECTS_DIR_FS, add_dist=False, verbose=0,
    )
    fwd_vanilla = mne.make_forward_solution(
        info_fs, trans=trans_fs, src=src_vanilla,
        bem=os.path.join(FS_DIR, "bem", "fsaverage-5120-5120-5120-bem-sol.fif"),
        eeg=True, mindist=0.0, verbose=0,
    )
    fwd_vanilla = mne.convert_forward_solution(
        fwd_vanilla, surf_ori=True, force_fixed=True, verbose=0,
    )
    n_vanilla = fwd_vanilla["sol"]["data"].shape[1]
    print(f"  Vanilla: {n_vanilla} dipoles")

    # ── 4. Build depth mapping forward model ─────────────────────────────
    print("\n[4/6] Building depth mapping forward model...")
    print("  Building sphere model in sample's coordinate system...")
    fwd_sphere, geo = build_sphere_forward_for_subject(fwd_true, verbose=0)
    n_sphere = fwd_sphere["sol"]["data"].shape[1]
    print(f"  Sphere: {n_sphere} dipoles")

    print("  Estimating depth (SH regression, l<=5, eLORETA)...")
    depth_result = predict_depth_from_eeg(
        fwd_true, fwd_sphere, geo,
        l_max=5, n_samples=5000, verbose=1,
    )
    pred_depth_mm = depth_result["pred_depth_mm"]
    print(f"  Depth prediction: rho={depth_result['rho']:.3f}, "
          f"MAE={depth_result['mae']:.1f}mm "
          f"({depth_result['rho']/depth_result['rho_oracle']*100:.0f}% of oracle)")

    # Mesh normals from deformed surface
    tris_all = _compact_tris(fwd_sphere["src"])
    _, pos_morphed = _cap_depth_to_skull(geo, pred_depth_mm)
    mesh_nn = compute_mesh_normals(pos_morphed, tris_all)

    fwd_depth = build_hybrid_forward(
        fwd_sphere, geo, pred_depth_mm, mesh_nn,
        sampling="ico3", verbose=0,
    )
    n_depth = fwd_depth["sol"]["data"].shape[1]
    print(f"  Depth mapping: {n_depth} dipoles")

    # ── 5. Build combined forward model (depth-informed template tuning) ──
    print("\n[5/8] Building combined model...")
    # Get fsaverage source positions for shape comparison
    src_fs = mne.setup_source_space(
        "fsaverage", spacing="ico3", surface="white",
        subjects_dir=SUBJECTS_DIR_FS, add_dist=False, verbose=0,
    )
    pos_fs = np.vstack([
        src_fs[0]["rr"][src_fs[0]["vertno"]],
        src_fs[1]["rr"][src_fs[1]["vertno"]],
    ])
    # Use common vertices (depth mapping may have dropped some)
    n_common = min(len(pos_morphed), len(pos_fs))
    stretch_est, rotation_est = extract_shape_params(
        pos_morphed[:n_common], pos_fs[:n_common],
    )
    print(f"  Depth-derived stretch: [{stretch_est[0]:.3f}, "
          f"{stretch_est[1]:.3f}, {stretch_est[2]:.3f}]")
    print(f"  Depth-derived rotation: [{rotation_est[0]:.1f}, "
          f"{rotation_est[1]:.1f}, {rotation_est[2]:.1f}] deg")

    fwd_combined = build_combined_fwd(
        info_fs, C_data, stretch_est, rotation_est,
    )
    n_combined = fwd_combined["sol"]["data"].shape[1]
    print(f"  Combined: {n_combined} dipoles")

    # ── 6. Build template tuning forward model ───────────────────────────
    print("\n[6/8] Building template tuning forward model...")
    fwd_tuning = build_template_tuning_fwd(info_fs, C_data)
    n_tuning = fwd_tuning["sol"]["data"].shape[1]
    print(f"  Template tuning: {n_tuning} dipoles")

    # ── 7. Evaluate all models ───────────────────────────────────────────
    print(f"\n[7/8] Evaluating all models ({EVAL_SOLVER}, {N_TRIALS} trials)...")
    eval_config = SimulationConfig(
        batch_size=N_TRIALS, n_sources=2, n_timepoints=200,
        snr_range=(50, 50), verbose=0, random_seed=RANDOM_SEED + 1,
    )
    eval_batch = next(SimulationGenerator(fwd_true, eval_config).generate())

    models = {
        "Vanilla template": fwd_vanilla,
        "Depth mapping": fwd_depth,
        "Combined": fwd_combined,
        "Template tuning": fwd_tuning,
    }

    results = {}
    for label, fwd_est in models.items():
        print(f"  {label}...")
        mles, emds = evaluate_model(
            fwd_est, fwd_true, eval_batch, info_mne, EVAL_SOLVER,
        )
        n_src = fwd_est["sol"]["data"].shape[1]
        results[label] = (mles, emds, n_src)

    # ── Riemannian covariance alignment ──────────────────────────────────
    print("  Riemannian alignment...")
    aligned_x = riemannian_align(C_data, fwd_vanilla, eval_batch)
    # Create a modified batch with aligned data
    class _AlignedBatch:
        pass
    aligned_batch = _AlignedBatch()
    aligned_batch.x = aligned_x
    aligned_batch.y = eval_batch.y
    mles_r, emds_r = evaluate_model(
        fwd_vanilla, fwd_true, aligned_batch, info_mne, EVAL_SOLVER,
    )
    results["Riemannian"] = (mles_r, emds_r, fwd_vanilla["sol"]["data"].shape[1])

    # ── Print results table ──────────────────────────────────────────────
    all_labels = list(models.keys()) + ["Riemannian"]
    print()
    print("=" * 74)
    print(f"{'Model':<22s} {'MLE med':>8s} {'±std':>6s} "
          f"{'EMD med':>8s} {'±std':>6s} {'n_src':>6s}")
    print("-" * 74)
    for label in all_labels:
        mles, emds, n = results[label]
        print(f"  {label:<20s} {np.median(mles):7.1f}  {np.std(mles):5.1f} "
              f"{np.median(emds):7.1f}  {np.std(emds):5.1f} {n:6d}")
    print("=" * 74)

    # Improvement
    van_mle = np.median(results["Vanilla template"][0])
    van_emd = np.median(results["Vanilla template"][1])
    print("\nImprovement vs vanilla:")
    for label in ["Depth mapping", "Combined", "Template tuning", "Riemannian"]:
        mles, emds, _ = results[label]
        dm = (1 - np.median(mles) / van_mle) * 100
        de = (1 - np.median(emds) / van_emd) * 100
        print(f"  {label:<20s}  MLE: {dm:+5.1f}%  EMD: {de:+5.1f}%")

    # ── Create FreeSurfer subject for depth mapping geometry ─────────────
    print("\nCreating depth mapping subject for visualization...")
    import tempfile
    depth_subjects_dir = tempfile.mkdtemp(prefix="depth_mapping_")
    depth_subject, remap_depth_stc = create_depth_mapping_subject(
        fwd_depth, "depth_mapped", depth_subjects_dir,
    )
    print(f"  Created: {depth_subjects_dir}/{depth_subject}/")

    print("\nDone (skipping visualization for speed).")
    return

    # ── Produce single-trial source reconstructions for figure ───────────
    print("\nPreparing visualization (single trial)...")
    viz_config = SimulationConfig(
        batch_size=1, n_sources=2, n_orders=(0, 3), n_timepoints=200,
        snr_range=(100, 100), verbose=0, random_seed=RANDOM_SEED + 2,
    )
    viz_batch = next(SimulationGenerator(fwd_true, viz_config).generate())
    evoked_viz = mne.EvokedArray(viz_batch.x[0], info_mne, tmin=0)

    # Ground truth STC
    verts_true = [fwd_true["src"][h]["vertno"] for h in range(2)]
    stc_true = mne.SourceEstimate(
        viz_batch.y[0], verts_true, tmin=0, tstep=1 / 250.0,
    )
    peak_time = stc_true.times[
        np.argmax(np.max(np.abs(stc_true.data), axis=0))
    ]
    print(f"  Peak time: {peak_time:.3f}s")

    # Invert with each model
    stcs = {}
    for label, fwd_est in models.items():
        s = Solver(EVAL_SOLVER)
        s.make_inverse_operator(fwd_est, evoked_viz, alpha="auto", verbose=0)
        stcs[label] = s.apply_inverse_operator(evoked_viz)

    # ── Multi-panel figure ───────────────────────────────────────────────
    import matplotlib.pyplot as plt

    print("\nRendering multi-panel figures...")

    # Remap the depth mapping STC to match the compact mesh
    stc_depth_remapped = remap_depth_stc(stcs["Depth mapping"])

    fig_args = dict(
        stc_true=stc_true,
        stc_vanilla=stcs["Vanilla template"],
        stc_depth=stc_depth_remapped,
        stc_tuning=stcs["Template tuning"],
        fwd_true=fwd_true,
        fwd_vanilla=fwd_vanilla,
        fwd_depth=fwd_depth,
        fwd_tuning=fwd_tuning,
        results=results,
        peak_time=peak_time,
        depth_subject=depth_subject,
        depth_subjects_dir=depth_subjects_dir,
    )

    for view_name in ["lateral", "dorsal"]:
        fig = make_comparison_figure(**fig_args, view=view_name)
        png_path = os.path.join(
            FIG_DIR, f"fig_comparison_{view_name}.png"
        )
        fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {png_path}")
        plt.close(fig)

    # Open interactive 3D brain viewers (one per model)
    print("\nOpening interactive brain viewers...")
    interactive_panels = [
        ("Ground truth (sample MRI)", stc_true, "sample", SUBJECTS_DIR_SAMPLE),
        ("Vanilla template", stcs["Vanilla template"], "fsaverage", SUBJECTS_DIR_FS),
        ("Depth mapping", stc_depth_remapped, depth_subject, depth_subjects_dir),
        ("Template tuning", stcs["Template tuning"], "fsaverage", SUBJECTS_DIR_FS),
    ]
    # Set subject attributes
    stc_true.subject = "sample"
    stcs["Vanilla template"].subject = "fsaverage"
    stc_depth_remapped.subject = depth_subject
    stcs["Template tuning"].subject = "fsaverage"

    brains = []
    for title, stc, subj, sdir in interactive_panels:
        brain = stc.plot(
            subjects_dir=sdir, subject=subj,
            surface="white", hemi="both", cortex="low_contrast",
            smoothing_steps=10, initial_time=peak_time,
            time_unit="s", size=(700, 500), verbose=0,
            background="white", show_traces=False,
            brain_kwargs={"title": title},
        )
        brain.show_view("lateral")
        brains.append(brain)
        print(f"  Opened: {title}")

    print("\nAll viewers open — rotate to inspect.")
    print("Close all windows or press Ctrl+C to exit.")
    try:
        from pyvistaqt import BackgroundPlotter
    except ImportError:
        pass
    # Keep the script alive so windows stay open
    try:
        input("Press Enter to exit...\n")
    except (EOFError, KeyboardInterrupt):
        pass

    print("\nDone.")


if __name__ == "__main__":
    main()
