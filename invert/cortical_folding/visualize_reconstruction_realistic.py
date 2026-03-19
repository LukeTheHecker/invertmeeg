"""Realistic cortical folding reconstruction — fully unsupervised.

Unlike visualize_reconstruction.py, this version does NOT use ground truth
depth labels for regression. The depth estimate comes purely from
accumulated EEG source power on a sphere model (physics-based proxy).

Pipeline (as if no individual MRI were available):
1. Build sphere forward model from fsaverage template
2. Simulate EEG from true cortex (stand-in for real EEG recordings)
3. Accumulate source power on sphere → estimate depth (unsupervised)
4. Build estimated forward model at predicted positions
5. Create FreeSurfer subject with estimated brain surface
6. Simulate test data with sparse sources on true brain
7. Invert test data with estimated forward model
8. Plot: ground truth on true brain vs estimate on estimated brain

Usage:
    cd invert-package
    .venv/bin/python -m invert.cortical_folding.visualize_reconstruction_realistic
"""

import os
import shutil
import tempfile

import mne
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from invert import Solver
from invert.forward import create_forward_model
from invert.forward.forward import _get_fsaverage_dir
from invert.simulate import SimulationConfig, SimulationGenerator
from invert.cortical_folding.experiment import create_sphere_forward
from invert.cortical_folding.sh_analysis import (
    estimate_depth_unsupervised,
    cart_to_sph,
    real_sh_matrix,
)


def _compact_tris(src):
    """Get triangulation with compact (0-based) vertex indices."""
    all_tris = []
    offset = 0
    for hemi in src:
        vertno = hemi["vertno"]
        tris = hemi["use_tris"]
        if tris.max() >= len(vertno):
            vmap = np.full(vertno.max() + 1, -1, dtype=int)
            vmap[vertno] = np.arange(len(vertno))
            tris = vmap[tris]
            valid = np.all(tris >= 0, axis=1)
            tris = tris[valid]
        all_tris.append(tris + offset)
        offset += len(vertno)
    return np.vstack(all_tris)


def _compute_mesh_normals(positions, tris):
    """Compute vertex normals from mesh geometry."""
    normals = np.zeros_like(positions)
    for tri in tris:
        v0, v1, v2 = positions[tri]
        face_normal = np.cross(v1 - v0, v2 - v0)
        for idx in tri:
            normals[idx] += face_normal
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return normals / norms


def build_estimated_forward(fwd_sphere, geo, pred_depth_mm, sampling="ico3", verbose=0):
    """Build forward model at estimated positions with mesh normals.

    Returns
    -------
    fwd_est : mne.Forward
        Fixed-orientation forward model at estimated positions.
    pos_est : (n_dip, 3) array
        Estimated source positions in meters.
    """
    fs_dir = _get_fsaverage_dir(verbose=0)
    subjects_dir = os.path.dirname(fs_dir)
    trans = os.path.join(fs_dir, "bem", "fsaverage-trans.fif")
    bem = os.path.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")

    # Morph sphere positions by predicted depth
    pos_est = geo["pos_sphere"] + (pred_depth_mm / 1000)[:, None] * geo["directions"]

    # Mesh normals from deformed surface
    tris = _compact_tris(fwd_sphere["src"])
    mesh_nn = _compute_mesh_normals(pos_est, tris)

    # Build source space
    src = mne.setup_source_space(
        "fsaverage", spacing=sampling, surface="white",
        subjects_dir=subjects_dir, add_dist=False, verbose=verbose,
    )
    n_lh = len(fwd_sphere["src"][0]["vertno"])
    for i, hemi in enumerate(src):
        verts = fwd_sphere["src"][i]["vertno"]
        hemi["vertno"] = verts
        sl = slice(0, n_lh) if i == 0 else slice(n_lh, None)
        hemi["rr"][verts] = pos_est[sl]
        hemi["nn"][verts] = mesh_nn[sl]
        hemi["nuse"] = len(verts)
        hemi["inuse"][:] = 0
        hemi["inuse"][verts] = 1

    fwd = mne.make_forward_solution(
        fwd_sphere["info"], trans=trans, src=src, bem=bem,
        eeg=True, mindist=0.0, verbose=verbose,
    )
    fwd = mne.convert_forward_solution(
        fwd, surf_ori=True, force_fixed=True, use_cps=True, verbose=verbose,
    )
    return fwd, pos_est


def create_estimated_subject(
    fwd_sphere, geo, pred_depth_mm, subject_name="estimated_brain", verbose=0
):
    """Create a FreeSurfer subject directory with estimated brain surface.

    Evaluates the SH depth coefficients at every full-resolution vertex
    to produce a smooth radial deformation of fsaverage's white surface.

    Returns
    -------
    subjects_dir : str
    """
    fs_dir = _get_fsaverage_dir(verbose=0)
    subjects_dir = os.path.dirname(fs_dir)
    surf_dir = os.path.join(subjects_dir, subject_name, "surf")
    os.makedirs(surf_dir, exist_ok=True)

    # Decompose ico3 depth estimate into SH coefficients
    center_mm = geo["center"] * 1000
    ico_pos_mm = geo["pos_sphere"] * 1000
    theta_ico, phi_ico = cart_to_sph(ico_pos_mm, center_mm)
    l_max = 5
    Y_ico = real_sh_matrix(theta_ico, phi_ico, l_max)
    from invert.cortical_folding.sh_analysis import sh_decompose
    depth_coeffs = sh_decompose(pred_depth_mm, Y_ico)

    for hi, hemi_name in enumerate(["lh", "rh"]):
        pos_full, faces = mne.read_surface(
            os.path.join(fs_dir, "surf", f"{hemi_name}.white")
        )

        # Spherical coords for full mesh
        rel = pos_full - center_mm
        r_full = np.linalg.norm(rel, axis=1)
        directions_full = rel / np.maximum(r_full, 1e-10)[:, None]
        theta_full, phi_full = cart_to_sph(pos_full, center_mm)

        # Evaluate SH depth at full resolution → smooth displacement
        Y_full = real_sh_matrix(theta_full, phi_full, l_max)
        depth_full_mm = Y_full @ depth_coeffs

        pos_deformed = pos_full + depth_full_mm[:, None] * directions_full
        mne.write_surface(
            os.path.join(surf_dir, f"{hemi_name}.white"),
            pos_deformed, faces, overwrite=True,
        )

        curv_src = os.path.join(fs_dir, "surf", f"{hemi_name}.curv")
        if os.path.exists(curv_src):
            shutil.copy2(curv_src, os.path.join(surf_dir, f"{hemi_name}.curv"))

    if verbose:
        print(f"  Created FreeSurfer subject: {subject_name}")
    return subjects_dir


def main():
    sampling = "ico3"
    l_max = 5
    n_depth_samples = 5000
    solver_name = "esmv"

    print("=" * 70)
    print("REALISTIC CORTICAL FOLDING RECONSTRUCTION (UNSUPERVISED)")
    print("No ground truth depth labels used in estimation.")
    print("=" * 70)

    # --- Step 1: True brain + sphere ---
    print("\n[1/6] Building forward models...")
    fwd_true = create_forward_model(sampling, fixed_ori=True, verbose=0)
    fwd_sphere, geo = create_sphere_forward(fwd_true, sampling=sampling, verbose=0)
    n_ch = fwd_true["sol"]["data"].shape[0]
    n_dip = fwd_true["sol"]["data"].shape[1]
    print(f"  True: {n_ch} channels, {n_dip} dipoles")
    print(f"  Sphere: {fwd_sphere['sol']['data'].shape[1]} dipoles, "
          f"radius={geo['radius']*1000:.1f}mm")

    # --- Step 2: Unsupervised depth estimation ---
    print(f"\n[2/6] Estimating depth from EEG (unsupervised, {n_depth_samples} samples)...")
    depth_result = estimate_depth_unsupervised(
        fwd_true, fwd_sphere, geo,
        l_max=l_max, n_samples=n_depth_samples, verbose=1,
    )
    pred_depth_mm = depth_result["pred_depth_mm"]

    # --- Step 3: Build estimated forward model ---
    print("\n[3/6] Building estimated forward model...")
    fwd_est, pos_est = build_estimated_forward(
        fwd_sphere, geo, pred_depth_mm, sampling=sampling, verbose=0,
    )
    print(f"  Estimated fwd: {fwd_est['sol']['data'].shape[1]} dipoles")

    # --- Step 4: Create FreeSurfer subject ---
    print("\n[4/6] Creating estimated brain surface...")
    est_subject = "estimated_brain"
    subjects_dir = create_estimated_subject(
        fwd_sphere, geo, pred_depth_mm, subject_name=est_subject, verbose=1,
    )

    # --- Step 5: Simulate test data ---
    print("\n[5/6] Simulating test data (2 sparse sources, SNR=10dB)...")
    test_config = SimulationConfig(
        batch_size=1,
        n_sources=2,
        n_timepoints=200,
        snr_range=(10, 10),
        verbose=0,
        random_seed=7,
    )
    test_gen = SimulationGenerator(fwd_true, test_config)
    test_batch = next(test_gen.generate())
    eeg_test = test_batch.x[0]
    src_test_true = test_batch.y[0]
    print(f"  {test_batch.info.iloc[0]['n_sources']} sources, "
          f"SNR={test_batch.info.iloc[0]['snr']:.0f}dB")

    # --- Step 6: Invert with estimated model ---
    print(f"\n[6/6] Inverting with {solver_name} on estimated forward model...")
    info_mne = mne.create_info(
        ch_names=fwd_true["info"]["ch_names"],
        sfreq=250.0,
        ch_types="eeg",
    )
    evoked_test = mne.EvokedArray(eeg_test, info_mne, tmin=0)

    solver = Solver(solver_name)
    solver.make_inverse_operator(fwd_est, evoked_test, alpha="auto", verbose=0)
    stc_est = solver.apply_inverse_operator(evoked_test)
    stc_est.subject = est_subject

    # Ground truth STC
    verts_true = [fwd_true["src"][0]["vertno"], fwd_true["src"][1]["vertno"]]
    stc_true = mne.SourceEstimate(
        src_test_true, verts_true, tmin=0, tstep=1 / 250.0, subject="fsaverage",
    )
    peak_time = stc_true.times[np.argmax(np.max(np.abs(stc_true.data), axis=0))]
    print(f"  Peak time: {peak_time:.3f}s")

    # --- Evaluation ---
    true_depth_mm = geo["radial_depth"] * 1000
    rho, _ = spearmanr(true_depth_mm, pred_depth_mm)
    mae = np.mean(np.abs(true_depth_mm - pred_depth_mm))
    print(f"\n{'='*70}")
    print(f"Depth estimation quality (unsupervised):")
    print(f"  rho={rho:.3f}, MAE={mae:.1f}mm")
    print(f"  Pred range: [{pred_depth_mm.min():.1f}, {pred_depth_mm.max():.1f}] mm")
    print(f"  True range: [{true_depth_mm.min():.1f}, {true_depth_mm.max():.1f}] mm")
    print(f"{'='*70}")

    # --- Plot ---
    base_kwargs = dict(
        subjects_dir=subjects_dir,
        surface="white",
        hemi="both",
        cortex="low_contrast",
        initial_time=peak_time,
        time_unit="s",
        size=(800, 400),
        smoothing_steps=10,
        verbose=0,
    )

    print("\nOpening brain viewers...")
    print("  Ground truth (fsaverage white surface)")
    brain_true = stc_true.plot(
        subject="fsaverage",
        brain_kwargs=dict(title="Ground Truth (true brain)"),
        **base_kwargs,
    )

    print(f"  {solver_name} estimate (estimated brain surface)")
    brain_est = stc_est.plot(
        subject=est_subject,
        brain_kwargs=dict(title=f"{solver_name} (estimated brain)"),
        **base_kwargs,
    )

    print("\nInteractive viewers — rotate to inspect, close windows when done.")
    # Keep viewers open until user closes them
    from pyvista import global_theme
    brain_true.show()
    brain_est.show()
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
