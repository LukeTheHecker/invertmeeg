"""Visualize BEM skull + brain deformations: vanilla, deformed, true geometry.

Produces a 3×2 figure:
  Columns: vanilla fsaverage | deformed fsaverage | true (sample) geometry
  Rows: top view (axial) | side view (sagittal)

Each panel shows inner skull surface (wireframe) + source positions (dots).
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mne
from scipy.spatial.transform import Rotation

from mne.bem import _check_complete_surface

from invert.cortical_folding.fast_bem import make_bem_solution_factored
from invert.cortical_folding.sh_analysis import cart_to_sph, real_sh_matrix
from invert.forward.forward import _get_fsaverage_dir, get_info

sys.stdout.reconfigure(line_buffering=True)

# ── Setup ──
fs_dir = _get_fsaverage_dir(verbose=0)
subjects_dir = os.path.dirname(fs_dir)
sample_path = mne.datasets.sample.data_path()
subjects_dir_sample = os.path.join(sample_path, 'subjects')
info = get_info()
trans_sample = os.path.join(sample_path, 'MEG', 'sample', 'sample_audvis_raw-trans.fif')

# ── Precomputed parameters ──
OFF9 = np.array([-0.0003, 0.0075, 0.0112])
STR9 = np.array([1.007, 0.987, 0.946])
ROT9 = np.array([-1.9, -1.4, -2.8])
SH_EXP3 = np.array([-0.0046, 0.0092, 0.0040, -0.0005, -0.0044,
                      0.0022, -0.0018, -0.0002, -0.0013, -0.0025, 0.0053, -0.0005])


def _apply_transform(pts, center, offset, stretch, rotation):
    p = pts - center
    if np.any(rotation != 0):
        R = Rotation.from_euler('XYZ', rotation, degrees=True).as_matrix()
        p = p @ R.T
    return p * stretch + center + offset


# ══════════════════════════════════════════════════════════════════════
# 1. VANILLA FSAVERAGE
# ══════════════════════════════════════════════════════════════════════
print('Loading vanilla fsaverage...')
bem_model_fs = mne.make_bem_model('fsaverage', ico=3, conductivity=(0.3, 0.006, 0.3),
    subjects_dir=subjects_dir, verbose=0)
skull_rr_vanilla = bem_model_fs[2]['rr'].copy()  # inner skull
skull_tris_vanilla = bem_model_fs[2]['tris'].copy()

src_fs = mne.setup_source_space('fsaverage', spacing='ico3', surface='white',
    subjects_dir=subjects_dir, add_dist=False, verbose=0)
pos_vanilla = np.vstack([src_fs[0]['rr'][src_fs[0]['vertno']].copy(),
                          src_fs[1]['rr'][src_fs[1]['vertno']].copy()])
n_lh = len(src_fs[0]['vertno'])

# ══════════════════════════════════════════════════════════════════════
# 2. DEFORMED FSAVERAGE (co-deform + SH)
# ══════════════════════════════════════════════════════════════════════
print('Computing deformed fsaverage...')
skull_center = skull_rr_vanilla.mean(axis=0)
src_center = pos_vanilla.mean(axis=0)

# SH deformation on skull
radii_base = np.linalg.norm(skull_rr_vanilla - skull_center, axis=1)
dirs_skull = (skull_rr_vanilla - skull_center) / radii_base[:, None]
theta_skull, phi_skull = cart_to_sph(skull_rr_vanilla, skull_center)
Y_skull = real_sh_matrix(theta_skull, phi_skull, l_max=3)

full_c = np.zeros(16)
full_c[4:16] = SH_EXP3
delta_r = Y_skull @ full_c
skull_rr_sh = skull_center + (radii_base + delta_r)[:, None] * dirs_skull

# Affine co-deform on SH-deformed skull
skull_rr_deformed = _apply_transform(skull_rr_sh, skull_center, OFF9, STR9, ROT9)

# Affine co-deform on sources
pos_deformed = _apply_transform(pos_vanilla, src_center, OFF9, STR9, ROT9)

# ══════════════════════════════════════════════════════════════════════
# 3. TRUE SAMPLE GEOMETRY
# ══════════════════════════════════════════════════════════════════════
print('Loading true (sample) geometry...')
bem_model_sample = mne.read_bem_surfaces(
    os.path.join(subjects_dir_sample, 'sample', 'bem', 'sample-5120-5120-5120-bem.fif'),
    verbose=0)
# Inner skull is the last surface (index 2)
skull_rr_true = bem_model_sample[2]['rr'].copy()
skull_tris_true = bem_model_sample[2]['tris'].copy()

# Transform to head coords using the trans file
trans = mne.read_trans(trans_sample)
# BEM surfaces are in MRI coords, need to transform to head coords
# trans is head -> MRI, we need MRI -> head
from mne.transforms import apply_trans, invert_transform
trans_inv = invert_transform(trans)
skull_rr_true = apply_trans(trans_inv, skull_rr_true)

# True source positions
fwd_true = mne.make_forward_solution(
    info, trans=trans_sample,
    src=mne.setup_source_space('sample', spacing='ico3', surface='white',
        subjects_dir=subjects_dir_sample, add_dist=False, verbose=0),
    bem=os.path.join(subjects_dir_sample, 'sample', 'bem',
        'sample-5120-5120-5120-bem-sol.fif'),
    eeg=True, mindist=0.0, verbose=0)
fwd_true = mne.convert_forward_solution(fwd_true, surf_ori=True, force_fixed=True, verbose=0)
pos_true = fwd_true['source_rr']
n_lh_true = len(fwd_true['src'][0]['vertno'])

print('All geometry loaded.')


# ══════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════

# Align true geometry to fsaverage by matching skull centroids.
# The two head coordinate systems are defined by different fiducials,
# so there's a translation offset that's not real anatomy.
skull_centroid_fs = skull_rr_vanilla.mean(axis=0)
skull_centroid_true = skull_rr_true.mean(axis=0)
centroid_shift = skull_centroid_fs - skull_centroid_true

skull_rr_true = skull_rr_true + centroid_shift
pos_true = pos_true + centroid_shift

# Convert to mm for readability
skull_vanilla_mm = skull_rr_vanilla * 1000
skull_deformed_mm = skull_rr_deformed * 1000
skull_true_mm = skull_rr_true * 1000
pos_vanilla_mm = pos_vanilla * 1000
pos_deformed_mm = pos_deformed * 1000
pos_true_mm = pos_true * 1000

print(f'Centroid alignment shift: [{centroid_shift[0]*1000:.1f}, '
      f'{centroid_shift[1]*1000:.1f}, {centroid_shift[2]*1000:.1f}] mm')

# SH deformation magnitude for color-coding the deformed skull
deform_mm = delta_r * 1000  # radial deformation in mm

datasets = [
    ('Vanilla fsaverage', skull_vanilla_mm, skull_tris_vanilla,
     pos_vanilla_mm, n_lh, None),
    ('Deformed fsaverage\n(co-deform + SH)', skull_deformed_mm, skull_tris_vanilla,
     pos_deformed_mm, n_lh, deform_mm),
    ('True (sample)', skull_true_mm, skull_tris_true,
     pos_true_mm, n_lh_true, None),
]

fig = plt.figure(figsize=(18, 12))

views = [
    ('Top view (axial)', 90, -90),     # looking down from above
    ('Side view (sagittal)', 0, 0),     # looking from right side
    ('Front view (coronal)', 0, -90),   # looking from front
]

for col, (title, skull_mm, tris, pos_mm, nlh, deform) in enumerate(datasets):
    for row, (view_name, elev, azim) in enumerate(views):
        ax = fig.add_subplot(3, 3, row * 3 + col + 1, projection='3d')

        # Draw skull as semi-transparent wireframe
        # Subsample triangles for clarity
        step = max(1, len(tris) // 800)
        for i in range(0, len(tris), step):
            tri = tris[i]
            verts = skull_mm[tri]
            # Color by deformation if available
            if deform is not None:
                mean_deform = np.mean(deform[tri])
                # Blue=negative (inward), red=positive (outward)
                val = np.clip(mean_deform / 10.0, -1, 1)  # normalize to [-1, 1]
                if val >= 0:
                    color = (0.5 + 0.5*val, 0.5 - 0.5*val, 0.5 - 0.5*val, 0.15)
                else:
                    color = (0.5 + 0.5*val, 0.5 + 0.5*val, 0.5 - 0.5*val, 0.15)
            else:
                color = (0.7, 0.7, 0.7, 0.1)

            poly = Poly3DCollection([verts], alpha=0.1,
                                     facecolor=color, edgecolor=(0.5, 0.5, 0.5, 0.15),
                                     linewidth=0.3)
            ax.add_collection3d(poly)

        # Draw source positions
        ax.scatter(pos_mm[:nlh, 0], pos_mm[:nlh, 1], pos_mm[:nlh, 2],
                   c='#2196F3', s=2, alpha=0.6, label='LH')
        ax.scatter(pos_mm[nlh:, 0], pos_mm[nlh:, 1], pos_mm[nlh:, 2],
                   c='#F44336', s=2, alpha=0.6, label='RH')

        # Set view
        ax.view_init(elev=elev, azim=azim)

        # Common axis limits (use true skull extent + padding)
        all_skull = np.vstack([skull_vanilla_mm, skull_deformed_mm, skull_true_mm])
        pad = 10
        for setter, idx in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
            setter(all_skull[:, idx].min() - pad, all_skull[:, idx].max() + pad)

        ax.set_aspect('equal')
        ax.set_xlabel('X (mm)', fontsize=7)
        ax.set_ylabel('Y (mm)', fontsize=7)
        ax.set_zlabel('Z (mm)', fontsize=7)
        ax.tick_params(labelsize=6)

        if row == 0:
            ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
        if col == 0:
            # Add view label on the left
            ax.text2D(-0.15, 0.5, view_name, transform=ax.transAxes,
                      fontsize=10, rotation=90, va='center', ha='center')

fig.suptitle('BEM Inner Skull + Cortical Sources: Vanilla → Deformed → True',
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0.02, 0.02, 1, 0.95])
out_path = os.path.join(os.path.dirname(__file__), 'bem_sh_deformation_comparison.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved to {out_path}')


# ══════════════════════════════════════════════════════════════════════
# OVERLAY PLOT: all three skulls + sources in one view
# ══════════════════════════════════════════════════════════════════════
from scipy.spatial import ConvexHull

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5.5))

projections = [
    ('Top view (axial)', 0, 1, 'X (mm)', 'Y (mm)'),      # XY
    ('Side view (sagittal)', 1, 2, 'Y (mm)', 'Z (mm)'),   # YZ
    ('Front view (coronal)', 0, 2, 'X (mm)', 'Z (mm)'),   # XZ
]

for ax, (view_name, xi, yi, xlabel, ylabel) in zip(axes2, projections):
    # Skull outlines via 2D convex hull of projected vertices
    for skull_mm, color, label, ls, lw in [
        (skull_vanilla_mm, '#9E9E9E', 'Vanilla fsaverage', '--', 1.5),
        (skull_deformed_mm, '#2196F3', 'Deformed fsaverage', '-', 2.0),
        (skull_true_mm, '#4CAF50', 'True (sample)', '-', 2.0),
    ]:
        pts_2d = skull_mm[:, [xi, yi]]
        hull = ConvexHull(pts_2d)
        hull_idx = np.append(hull.vertices, hull.vertices[0])
        ax.plot(pts_2d[hull_idx, 0], pts_2d[hull_idx, 1],
                color=color, linestyle=ls, linewidth=lw, label=label, alpha=0.9)

    # Source positions
    for pos_mm, color, alpha, zorder in [
        (pos_vanilla_mm, '#BDBDBD', 0.25, 1),
        (pos_true_mm, '#4CAF50', 0.3, 2),
        (pos_deformed_mm, '#2196F3', 0.3, 3),
    ]:
        ax.scatter(pos_mm[:, xi], pos_mm[:, yi], c=color, s=2, alpha=alpha,
                   zorder=zorder, edgecolors='none')

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(view_name, fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

axes2[0].legend(fontsize=9, loc='lower left', framealpha=0.9)
fig2.suptitle('Skull + Source Overlay: Vanilla (gray) vs Deformed (blue) vs True (green)',
              fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.93])
out_path2 = os.path.join(os.path.dirname(__file__), 'bem_sh_deformation_overlay.png')
plt.savefig(out_path2, dpi=150, bbox_inches='tight')
print(f'Saved to {out_path2}')


# ══════════════════════════════════════════════════════════════════════
# SH DEFORMATION MAP
# ══════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))

for ax, (view_name, xi, yi, xlabel, ylabel) in zip(axes3, projections):
    sc = ax.scatter(skull_deformed_mm[:, xi], skull_deformed_mm[:, yi],
                    c=deform_mm, cmap='RdBu_r', vmin=-10, vmax=10,
                    s=8, alpha=0.8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(view_name, fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

cbar = fig3.colorbar(sc, ax=axes3, shrink=0.8, pad=0.02)
cbar.set_label('Radial deformation (mm)', fontsize=10)
fig3.suptitle('SH Skull Deformation Map (blue=inward, red=outward)',
              fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 0.92, 0.93])
out_path3 = os.path.join(os.path.dirname(__file__), 'bem_sh_deformation_map.png')
plt.savefig(out_path3, dpi=150, bbox_inches='tight')
print(f'Saved to {out_path3}')

print('\nDone! Generated 3 figures.')
