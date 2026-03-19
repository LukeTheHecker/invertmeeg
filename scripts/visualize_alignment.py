"""Interactive visualization of brain alignment: fsaverage vs sample subject.

Uses Talairach transforms to align both brains into MNI space for
proper anatomical overlay. This removes the MRI coordinate system
differences that make raw head-coordinate overlays misleading.

Usage:
    cd invert-package
    .venv/bin/python scripts/visualize_alignment.py
"""

import os

import mne
import nibabel.freesurfer as nfs
import numpy as np
import pyvista as pv
from mne.transforms import apply_trans

from invert.forward.forward import _get_fsaverage_dir


def transform_to_talairach(coords_mm, tal_trans):
    """Transform FreeSurfer surface coords (mm, tkRAS) to MNI space (mm)."""
    coords_m = coords_mm / 1000.0
    coords_h = np.c_[coords_m, np.ones(len(coords_m))]
    coords_mni_m = (tal_trans["trans"] @ coords_h.T).T[:, :3]
    return coords_mni_m * 1000.0


def load_surfaces(subject, subjects_dir, tal_trans):
    """Load white + BEM surfaces and transform to MNI space."""
    subj_dir = os.path.join(subjects_dir, subject)
    surfs = {}
    for hemi in ["lh", "rh"]:
        coords, faces = nfs.read_geometry(
            os.path.join(subj_dir, "surf", f"{hemi}.white"))
        surfs[f"cortex_{hemi}"] = (
            transform_to_talairach(coords, tal_trans), faces)
    for name in ["inner_skull", "outer_skin"]:
        path = os.path.join(subj_dir, "bem", f"{name}.surf")
        if os.path.exists(path):
            coords, faces = nfs.read_geometry(path)
            surfs[name] = (transform_to_talairach(coords, tal_trans), faces)
    return surfs


def make_mesh(pos_mm, tris, **scalars):
    n = len(tris)
    faces = np.column_stack([np.full(n, 3), tris]).ravel()
    mesh = pv.PolyData(pos_mm, faces)
    for k, v in scalars.items():
        mesh[k] = v
    return mesh


def depth_color(pos_mm):
    c = pos_mm.mean(axis=0)
    r = np.linalg.norm(pos_mm - c, axis=1)
    return r - r.mean()


def main():
    fs_dir = _get_fsaverage_dir(verbose=0)
    fs_subjects_dir = os.path.dirname(fs_dir)
    sample_path = mne.datasets.sample.data_path()
    sample_subjects_dir = os.path.join(sample_path, "subjects")

    # Talairach transforms → MNI space
    tal_fs = mne.read_talxfm("fsaverage", fs_subjects_dir)
    tal_sample = mne.read_talxfm("sample", sample_subjects_dir)

    print("Loading surfaces in MNI/Talairach space...")
    surfs_fs = load_surfaces("fsaverage", fs_subjects_dir, tal_fs)
    surfs_sample = load_surfaces("sample", sample_subjects_dir, tal_sample)

    # Verify alignment
    fs_cortex = np.vstack([surfs_fs["cortex_lh"][0], surfs_fs["cortex_rh"][0]])
    s_cortex = np.vstack([surfs_sample["cortex_lh"][0], surfs_sample["cortex_rh"][0]])
    offset = np.linalg.norm(fs_cortex.mean(0) - s_cortex.mean(0))
    print(f"  Centroid offset: {offset:.1f}mm (should be <5mm)")

    # ===== VIEWER =====
    print("\nOpening viewer...")
    fs_color = "steelblue"
    s_color = "darkorange"

    pl = pv.Plotter(shape=(2, 4), window_size=(2200, 1000), border=True)

    # Row 1: Individual views

    pl.subplot(0, 0)
    for h in ["lh", "rh"]:
        pos, tris = surfs_fs[f"cortex_{h}"]
        pl.add_mesh(make_mesh(pos, tris, depth=depth_color(pos)),
                     scalars="depth", cmap="RdBu_r", clim=[-20, 20],
                     smooth_shading=True, show_scalar_bar=False)
    pl.add_text("fsaverage cortex", font_size=11,
                position="upper_left", color="black")

    pl.subplot(0, 1)
    for h in ["lh", "rh"]:
        pos, tris = surfs_sample[f"cortex_{h}"]
        pl.add_mesh(make_mesh(pos, tris, depth=depth_color(pos)),
                     scalars="depth", cmap="RdBu_r", clim=[-20, 20],
                     smooth_shading=True, show_scalar_bar=False)
    pl.add_text("sample cortex", font_size=11,
                position="upper_left", color="black")

    pl.subplot(0, 2)
    if "outer_skin" in surfs_fs:
        pl.add_mesh(make_mesh(*surfs_fs["outer_skin"]), color="peachpuff",
                     opacity=0.15, smooth_shading=True)
    if "inner_skull" in surfs_fs:
        pl.add_mesh(make_mesh(*surfs_fs["inner_skull"]), color="lightyellow",
                     opacity=0.25, smooth_shading=True)
    for h in ["lh", "rh"]:
        pl.add_mesh(make_mesh(*surfs_fs[f"cortex_{h}"]),
                     color=fs_color, opacity=0.5, smooth_shading=True)
    pl.add_text("fsaverage\ncortex + skull + scalp",
                font_size=10, position="upper_left", color="black")

    pl.subplot(0, 3)
    if "outer_skin" in surfs_sample:
        pl.add_mesh(make_mesh(*surfs_sample["outer_skin"]), color="peachpuff",
                     opacity=0.15, smooth_shading=True)
    if "inner_skull" in surfs_sample:
        pl.add_mesh(make_mesh(*surfs_sample["inner_skull"]), color="lightyellow",
                     opacity=0.25, smooth_shading=True)
    for h in ["lh", "rh"]:
        pl.add_mesh(make_mesh(*surfs_sample[f"cortex_{h}"]),
                     color=s_color, opacity=0.5, smooth_shading=True)
    pl.add_text("sample\ncortex + skull + scalp",
                font_size=10, position="upper_left", color="black")

    # Row 2: Overlays

    pl.subplot(1, 0)
    for h in ["lh", "rh"]:
        pl.add_mesh(make_mesh(*surfs_fs[f"cortex_{h}"]),
                     color=fs_color, opacity=0.4, smooth_shading=True)
        pl.add_mesh(make_mesh(*surfs_sample[f"cortex_{h}"]),
                     color=s_color, opacity=0.4, smooth_shading=True)
    pl.add_text(f"CORTEX OVERLAY\noffset={offset:.1f}mm",
                font_size=10, position="upper_left", color="black")

    pl.subplot(1, 1)
    if "inner_skull" in surfs_fs and "inner_skull" in surfs_sample:
        pl.add_mesh(make_mesh(*surfs_fs["inner_skull"]),
                     color=fs_color, opacity=0.3, smooth_shading=True)
        pl.add_mesh(make_mesh(*surfs_sample["inner_skull"]),
                     color=s_color, opacity=0.3, smooth_shading=True)
    pl.add_text("SKULL OVERLAY",
                font_size=10, position="upper_left", color="black")

    pl.subplot(1, 2)
    if "outer_skin" in surfs_fs and "outer_skin" in surfs_sample:
        pl.add_mesh(make_mesh(*surfs_fs["outer_skin"]),
                     color=fs_color, opacity=0.3, smooth_shading=True)
        pl.add_mesh(make_mesh(*surfs_sample["outer_skin"]),
                     color=s_color, opacity=0.3, smooth_shading=True)
    pl.add_text("SCALP OVERLAY",
                font_size=10, position="upper_left", color="black")

    pl.subplot(1, 3)
    for h in ["lh", "rh"]:
        pl.add_mesh(make_mesh(*surfs_fs[f"cortex_{h}"]),
                     color=fs_color, opacity=0.3, smooth_shading=True)
        pl.add_mesh(make_mesh(*surfs_sample[f"cortex_{h}"]),
                     color=s_color, opacity=0.3, smooth_shading=True)
    if "inner_skull" in surfs_fs and "inner_skull" in surfs_sample:
        pl.add_mesh(make_mesh(*surfs_fs["inner_skull"]),
                     color=fs_color, opacity=0.12, smooth_shading=True)
        pl.add_mesh(make_mesh(*surfs_sample["inner_skull"]),
                     color=s_color, opacity=0.12, smooth_shading=True)
    pl.add_text("FULL OVERLAY\ncortex + skull",
                font_size=10, position="upper_left", color="black")

    pl.link_views()
    pl.background_color = "white"
    pl.camera.zoom(1.2)

    # Save a screenshot via off-screen render
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "alignment_pyvista.png")
    print("  Rotate to inspect, close when done.")
    pl.show()


if __name__ == "__main__":
    main()
