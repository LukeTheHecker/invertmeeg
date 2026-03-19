"""Cross-subject hybrid morphing evaluation.

Tests the hybrid template morphing approach with a real different subject:
- F1 = MNE sample subject (individual, different cortical folding)
- F2 = fsaverage (template)

Compares forward model quality for:
1. Sphere model (no MRI at all)
2. Hybrid: EEG-morphed positions + mesh normals
3. Hybrid: EEG-morphed positions + template (fsaverage) normals
4. Oracle: true individual cortex (F1)

Usage:
    cd invert-package
    .venv/bin/python scripts/eval_hybrid_morphing.py
"""

import os

import mne
import numpy as np
from scipy.stats import spearmanr

from invert.forward.forward import _get_fsaverage_dir, get_info
from invert.cortical_folding.sh_analysis import (
    predict_depth_from_eeg,
    cart_to_sph,
    real_sh_matrix,
)


def build_sample_forward(sampling="ico3", verbose=0):
    """Build forward model for MNE sample subject with easycap-M10 layout.

    Returns
    -------
    fwd : mne.Forward
        Fixed-orientation forward model for sample subject.
    """
    sample_path = mne.datasets.sample.data_path()
    subjects_dir = os.path.join(sample_path, "subjects")
    trans = os.path.join(
        sample_path, "MEG", "sample", "sample_audvis_raw-trans.fif"
    )
    bem = os.path.join(
        subjects_dir, "sample", "bem", "sample-5120-5120-5120-bem-sol.fif"
    )
    info = get_info()  # easycap-M10, same as fsaverage experiments

    src = mne.setup_source_space(
        "sample",
        spacing=sampling,
        surface="white",
        subjects_dir=subjects_dir,
        add_dist=False,
        verbose=verbose,
    )

    fwd = mne.make_forward_solution(
        info,
        trans=trans,
        src=src,
        bem=bem,
        eeg=True,
        mindist=0.0,
        verbose=verbose,
    )
    fwd = mne.convert_forward_solution(
        fwd, surf_ori=True, force_fixed=True, use_cps=True, verbose=verbose
    )
    return fwd


def _find_sphere_center_and_radius(bem_dir, cortex_center):
    """Find the sphere center and radius that maximizes inscribed sphere.

    Starts from cortex centroid, iteratively moves toward the Chebyshev
    center (maximizes minimum distance to inner skull boundary).

    Returns
    -------
    center : (3,) array, sphere center in m
    radius : float, safe sphere radius in m
    """
    skull_pos = mne.read_surface(
        os.path.join(bem_dir, "inner_skull.surf")
    )[0] / 1000  # mm -> m

    center = cortex_center.copy()
    step = 0.002  # 2mm initial step
    best_min_r = 0.0
    for i in range(500):
        r = np.linalg.norm(skull_pos - center, axis=1)
        min_r = r.min()
        if min_r > best_min_r:
            best_min_r = min_r
            best_center = center.copy()

        # Move away from closest skull vertex
        closest = np.argmin(r)
        direction = center - skull_pos[closest]
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            break
        direction /= norm
        center = center + step * direction

        # Shrink step over time
        if i % 50 == 49:
            step *= 0.7

    return best_center, best_min_r - 0.002


def build_sphere_forward_for_subject(fwd_cortex, sampling="ico3", verbose=0):
    """Build sphere forward model using the subject's own BEM.

    Like experiment.create_sphere_forward but uses the subject's BEM/trans
    from the cortex forward model instead of fsaverage's.

    Returns
    -------
    fwd_sphere : mne.Forward
    geo : dict
    """
    # Use full ico3 vertex set (not the cortex fwd's reduced set) to avoid
    # BEM interior check issues with modified vertex subsets
    sample_path = mne.datasets.sample.data_path()
    subjects_dir = os.path.join(sample_path, "subjects")
    trans = os.path.join(
        sample_path, "MEG", "sample", "sample_audvis_raw-trans.fif"
    )
    bem = os.path.join(
        subjects_dir, "sample", "bem", "sample-5120-5120-5120-bem-sol.fif"
    )

    src = mne.setup_source_space(
        "sample",
        spacing=sampling,
        surface="white",
        subjects_dir=subjects_dir,
        add_dist=False,
        verbose=verbose,
    )

    # Compute sphere from full cortex positions
    pos_lh = src[0]["rr"][src[0]["vertno"]].copy()
    pos_rh = src[1]["rr"][src[1]["vertno"]].copy()
    pos_cortex = np.vstack([pos_lh, pos_rh])

    center = np.mean(pos_cortex, axis=0)
    radii = np.linalg.norm(pos_cortex - center, axis=1)
    radius = np.mean(radii)
    directions = (pos_cortex - center) / radii[:, None]
    pos_sphere = center + radius * directions

    if verbose:
        print(f"  Sphere radius: {radius*1000:.1f}mm "
              f"({len(pos_cortex)} vertices)")

    n_lh = len(src[0]["vertno"])
    for i, hemi in enumerate(src):
        verts = hemi["vertno"]
        sl = slice(0, n_lh) if i == 0 else slice(n_lh, None)
        hemi["rr"][verts] = pos_sphere[sl]
        hemi["nn"][verts] = directions[sl]

    fwd_sphere = mne.make_forward_solution(
        fwd_cortex["info"],
        trans=trans,
        src=src,
        bem=bem,
        eeg=True,
        mindist=0.0,
        verbose=verbose,
    )
    fwd_sphere = mne.convert_forward_solution(
        fwd_sphere,
        surf_ori=True,
        force_fixed=True,
        use_cps=True,
        verbose=verbose,
    )

    # Track which sources survived (a few may be outside BEM)
    vn_lh_surv = fwd_sphere["src"][0]["vertno"]
    vn_rh_surv = fwd_sphere["src"][1]["vertno"]
    n_surv = len(vn_lh_surv) + len(vn_rh_surv)
    n_total = len(pos_cortex)

    # Build survival mask for full vertex set
    lh_mask = np.isin(src[0]["vertno"], vn_lh_surv)
    rh_mask = np.isin(src[1]["vertno"], vn_rh_surv)
    surv_mask = np.concatenate([lh_mask, rh_mask])

    geo = {
        "center": center,
        "radius": radius,
        "pos_cortex": pos_cortex[surv_mask],
        "pos_sphere": pos_sphere[surv_mask],
        "radial_depth": (radii - radius)[surv_mask],
        "directions": directions[surv_mask],
        "surv_mask": surv_mask,
        "n_total": n_total,
    }
    return fwd_sphere, geo


def get_fsaverage_normals(vertno_lh, vertno_rh, sampling="ico3", verbose=0):
    """Get fsaverage cortex normals rotated into sample's surface RAS frame.

    Uses Talairach transforms to rotate normals from fsaverage's coordinate
    system to sample's coordinate system.
    """
    fs_dir = _get_fsaverage_dir(verbose=verbose)
    subjects_dir = os.path.dirname(fs_dir)

    src = mne.setup_source_space(
        "fsaverage",
        spacing=sampling,
        surface="white",
        subjects_dir=subjects_dir,
        add_dist=False,
        verbose=verbose,
    )

    nn_lh = src[0]["nn"][vertno_lh]
    nn_rh = src[1]["nn"][vertno_rh]
    nn = np.vstack([nn_lh, nn_rh])

    # Rotate from fsaverage surface RAS to sample surface RAS
    sample_path = mne.datasets.sample.data_path()
    sample_subjects_dir = os.path.join(sample_path, "subjects")
    tal_fs = mne.read_talxfm("fsaverage", sample_subjects_dir)
    tal_sample = mne.read_talxfm("sample", sample_subjects_dir)
    R = np.linalg.inv(tal_sample["trans"][:3, :3]) @ tal_fs["trans"][:3, :3]

    if verbose:
        angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        print(f"  Rotating normals by {angle:.1f} degrees "
              f"(fsaverage -> sample)")

    nn_rotated = nn @ R.T
    # Re-normalize
    norms = np.linalg.norm(nn_rotated, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return nn_rotated / norms


def _compact_tris(src):
    """Get triangulation with compact (0-based) vertex indices."""
    n_lh = len(src[0]["vertno"])
    all_tris = []
    offset = 0
    for hemi in src:
        vertno = hemi["vertno"]
        tris = hemi["use_tris"]
        # Check if tris use full-surface indices (max > nuse)
        if tris.max() >= len(vertno):
            # Remap from full-surface indices to compact indices
            vmap = np.full(vertno.max() + 1, -1, dtype=int)
            vmap[vertno] = np.arange(len(vertno))
            tris = vmap[tris]
            # Drop any triangles with missing vertices
            valid = np.all(tris >= 0, axis=1)
            tris = tris[valid]
        all_tris.append(tris + offset)
        offset += len(vertno)
    return np.vstack(all_tris)


def compute_mesh_normals(positions, tris):
    """Compute vertex normals from mesh geometry (area-weighted face normals)."""
    normals = np.zeros_like(positions)
    for tri in tris:
        v0, v1, v2 = positions[tri]
        face_normal = np.cross(v1 - v0, v2 - v0)  # area-weighted
        for idx in tri:
            normals[idx] += face_normal
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return normals / norms


def _cap_depth_to_skull(geo, pred_depth_mm):
    """Cap predicted depth per-direction so morphed positions stay inside the skull.

    Returns capped depth in meters and morphed positions.
    """
    sample_path = mne.datasets.sample.data_path()
    bem_dir = os.path.join(sample_path, "subjects", "sample", "bem")
    skull_pos = mne.read_surface(
        os.path.join(bem_dir, "inner_skull.surf")
    )[0] / 1000
    skull_rel = skull_pos - geo["center"]
    skull_r = np.linalg.norm(skull_rel, axis=1)
    skull_dirs = skull_rel / skull_r[:, None]
    cos_sim = geo["directions"] @ skull_dirs.T
    best_skull = np.argmax(cos_sim, axis=1)
    max_radius = skull_r[best_skull] - 0.005  # 5mm margin
    max_depth_m = max_radius - geo["radius"]
    depth_m = np.clip(pred_depth_mm / 1000, None, max_depth_m)
    pos_morphed = geo["pos_sphere"] + depth_m[:, None] * geo["directions"]
    return depth_m, pos_morphed


def build_hybrid_forward(
    fwd_sphere, geo, pred_depth_mm, normals=None, sampling="ico3",
    free_ori=False, verbose=0,
):
    """Build forward model with morphed positions and given normals.

    Uses the sphere fwd's vertex set (which may be a subset of the cortex
    fwd's vertices if some were dropped by the BEM).

    Parameters
    ----------
    normals : array or None
        Per-source normal vectors. Required when ``free_ori=False``.
        Ignored when ``free_ori=True``.
    free_ori : bool
        If True, return a free-orientation forward (3 components per source).
    """
    _, pos_morphed = _cap_depth_to_skull(geo, pred_depth_mm)

    if not free_ori and normals is None:
        raise ValueError("normals required when free_ori=False")

    src_sphere = fwd_sphere["src"]
    vertno_lh = src_sphere[0]["vertno"]
    vertno_rh = src_sphere[1]["vertno"]
    n_lh = len(vertno_lh)

    sample_path = mne.datasets.sample.data_path()
    subjects_dir = os.path.join(sample_path, "subjects")
    trans = os.path.join(
        sample_path, "MEG", "sample", "sample_audvis_raw-trans.fif"
    )
    bem = os.path.join(
        subjects_dir, "sample", "bem", "sample-5120-5120-5120-bem-sol.fif"
    )

    src = mne.setup_source_space(
        "sample",
        spacing=sampling,
        surface="white",
        subjects_dir=subjects_dir,
        add_dist=False,
        verbose=verbose,
    )

    src[0]["vertno"] = vertno_lh
    src[1]["vertno"] = vertno_rh
    for i, hemi in enumerate(src):
        verts = hemi["vertno"]
        sl = slice(0, n_lh) if i == 0 else slice(n_lh, None)
        hemi["rr"][verts] = pos_morphed[sl]
        if normals is not None:
            hemi["nn"][verts] = normals[sl]
        hemi["nuse"] = len(verts)
        hemi["inuse"][:] = 0
        hemi["inuse"][verts] = 1

    fwd = mne.make_forward_solution(
        fwd_sphere["info"],
        trans=trans,
        src=src,
        bem=bem,
        eeg=True,
        mindist=0.0,
        verbose=verbose,
    )
    fwd = mne.convert_forward_solution(
        fwd, surf_ori=True, force_fixed=not free_ori,
        use_cps=True, verbose=verbose,
    )
    return fwd


def evaluate_leadfield(G_true, G_test, pos_true):
    """Compare two leadfields column-wise.

    Returns
    -------
    result : dict
        mean_corr : mean column-wise Pearson correlation
        median_loc_err_mm : median peak localization error (mm)
        pct_within_20mm : fraction of sources localized within 20mm
    """
    n_dip = G_true.shape[1]

    # Column-wise correlation
    corrs = np.array([
        np.corrcoef(G_true[:, i], G_test[:, i])[0, 1]
        for i in range(n_dip)
    ])

    # Peak localization: for each true source, find which test source
    # has the most correlated leadfield pattern
    G_true_n = G_true / (np.linalg.norm(G_true, axis=0, keepdims=True) + 1e-30)
    G_test_n = G_test / (np.linalg.norm(G_test, axis=0, keepdims=True) + 1e-30)
    cross_corr = G_true_n.T @ G_test_n  # (n_dip, n_dip)
    best_match = np.argmax(np.abs(cross_corr), axis=1)
    loc_err = np.linalg.norm(pos_true - pos_true[best_match], axis=1) * 1000

    return {
        "mean_corr": np.mean(corrs),
        "median_corr": np.median(corrs),
        "median_loc_err_mm": np.median(loc_err),
        "mean_loc_err_mm": np.mean(loc_err),
        "pct_within_20mm": np.mean(loc_err < 20) * 100,
    }


def main():
    sampling = "ico3"
    l_max = 5
    n_samples = 5000

    print("=" * 70)
    print("CROSS-SUBJECT HYBRID MORPHING EVALUATION")
    print("F1 = MNE sample subject | F2 = fsaverage (template)")
    print("=" * 70)

    # --- Step 1: Build F1's forward models ---
    print("\n[1/5] Building sample subject forward model...")
    fwd_sample = build_sample_forward(sampling=sampling, verbose=0)
    n_ch = fwd_sample["sol"]["data"].shape[0]
    n_dip = fwd_sample["sol"]["data"].shape[1]
    print(f"  {n_ch} channels, {n_dip} dipoles")

    print("\n[2/5] Building sphere model in sample's coordinate system...")
    fwd_sphere, geo = build_sphere_forward_for_subject(
        fwd_sample, sampling=sampling, verbose=1
    )
    n_sphere = fwd_sphere["sol"]["data"].shape[1]
    print(f"  Sphere: {n_sphere} dipoles, radius={geo['radius']*1000:.1f}mm")
    if n_sphere != n_dip:
        print(f"  WARNING: {n_dip - n_sphere} sources dropped!")

    # --- Step 2: Predict depth from simulated EEG ---
    print(f"\n[3/5] Predicting depth (n_samples={n_samples}, l_max={l_max})...")
    depth_result = predict_depth_from_eeg(
        fwd_sample, fwd_sphere, geo,
        l_max=l_max, n_samples=n_samples, verbose=1,
    )

    # --- Step 3: Build hybrid models ---
    print("\n[4/5] Building hybrid forward models...")
    pred_depth_mm = depth_result["pred_depth_mm"]
    surv_mask = geo.get("surv_mask")

    # Get triangulation for mesh normals (using sphere fwd's vertex set)
    tris_all = _compact_tris(fwd_sphere["src"])

    # Capped morphed positions for mesh normal computation
    _, pos_morphed = _cap_depth_to_skull(geo, pred_depth_mm)

    # A) Radial normals (baseline — just the sphere directions)
    print("  Building hybrid with radial normals...")
    fwd_radial = build_hybrid_forward(
        fwd_sphere, geo, pred_depth_mm, geo["directions"],
        sampling=sampling, verbose=0,
    )

    # B) Mesh normals (computed from deformed surface)
    print("  Building hybrid with mesh normals...")
    mesh_nn = compute_mesh_normals(pos_morphed, tris_all)
    fwd_mesh = build_hybrid_forward(
        fwd_sphere, geo, pred_depth_mm, mesh_nn,
        sampling=sampling, verbose=0,
    )

    # C) Template normals (fsaverage cortex normals — the key test)
    print("  Building hybrid with template (fsaverage) normals...")
    vn_lh = fwd_sphere["src"][0]["vertno"]
    vn_rh = fwd_sphere["src"][1]["vertno"]
    template_nn = get_fsaverage_normals(
        vn_lh, vn_rh, sampling=sampling, verbose=1
    )
    fwd_template = build_hybrid_forward(
        fwd_sphere, geo, pred_depth_mm, template_nn,
        sampling=sampling, verbose=0,
    )

    # D) Oracle: true individual normals + predicted positions
    print("  Building hybrid with true (sample) normals...")
    src_sample = fwd_sample["src"]
    true_nn_lh = src_sample[0]["nn"][vn_lh]
    true_nn_rh = src_sample[1]["nn"][vn_rh]
    true_nn = np.vstack([true_nn_lh, true_nn_rh])
    fwd_true_nn = build_hybrid_forward(
        fwd_sphere, geo, pred_depth_mm, true_nn,
        sampling=sampling, verbose=0,
    )

    # --- Step 4: Evaluate ---
    print("\n[5/5] Evaluating leadfield quality...")

    # Find common vertices between cortex fwd and sphere fwd
    src_c = fwd_sample["src"]
    src_s = fwd_sphere["src"]
    common_lh = np.intersect1d(src_c[0]["vertno"], src_s[0]["vertno"])
    common_rh = np.intersect1d(src_c[1]["vertno"], src_s[1]["vertno"])

    # Build index maps into each fwd's dipole array
    cortex_idx_lh = np.searchsorted(src_c[0]["vertno"], common_lh)
    cortex_idx_rh = np.searchsorted(src_c[1]["vertno"], common_rh) + len(src_c[0]["vertno"])
    cortex_idx = np.concatenate([cortex_idx_lh, cortex_idx_rh])

    sphere_idx_lh = np.searchsorted(src_s[0]["vertno"], common_lh)
    sphere_idx_rh = np.searchsorted(src_s[1]["vertno"], common_rh) + len(src_s[0]["vertno"])
    sphere_idx = np.concatenate([sphere_idx_lh, sphere_idx_rh])

    G_true = fwd_sample["sol"]["data"][:, cortex_idx]
    pos_eval = geo["pos_cortex"][sphere_idx]  # use sphere's pos_cortex (which matches sphere fwd order)

    models = {
        "Sphere (no MRI)": fwd_sphere,
        "Hybrid + radial normals": fwd_radial,
        "Hybrid + mesh normals": fwd_mesh,
        "Hybrid + template normals": fwd_template,
        "Hybrid + TRUE normals": fwd_true_nn,
    }

    n_common = len(sphere_idx)
    print(f"\n  Comparing at {n_common} common vertices"
          f" (cortex: {n_dip}, sphere: {fwd_sphere['sol']['data'].shape[1]})")
    print("=" * 70)
    print(f"{'Model':<32s} {'LF corr':>8s} {'Loc err':>8s} {'<20mm':>6s}")
    print("-" * 70)

    for name, fwd in models.items():
        G_test = fwd["sol"]["data"][:, sphere_idx]
        ev = evaluate_leadfield(G_true, G_test, pos_eval)
        print(
            f"  {name:<30s} {ev['mean_corr']:8.3f} "
            f"{ev['median_loc_err_mm']:7.1f}mm {ev['pct_within_20mm']:5.0f}%"
        )

    print(f"  {'True individual (oracle)':<30s} {'1.000':>8s} {'0.0mm':>8s} {'100%':>6s}")
    print("=" * 70)

    # Depth prediction summary
    print(f"\nDepth prediction quality:")
    print(f"  rho={depth_result['rho']:.3f}, R2={depth_result['r2']:.3f}, "
          f"MAE={depth_result['mae']:.1f}mm")
    print(f"  Oracle (l<={l_max}): rho={depth_result['rho_oracle']:.3f}")
    print(f"  Signal recovered: {depth_result['rho'] / depth_result['rho_oracle'] * 100:.0f}%")


if __name__ == "__main__":
    main()
