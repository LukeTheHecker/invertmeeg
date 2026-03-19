"""Full cortical folding reconstruction and interactive 3D visualization.

Best approach: eLORETA + PSF features + SH-regularized regression.
At l≤5: ρ=0.509, R²=0.260, 96% of oracle (ρ=0.529).

Usage:
    cd invert-package
    .venv/bin/python -m invert.cortical_folding.visualize_reconstruction
"""

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from invert import Solver
from invert.forward import create_forward_model
from invert.cortical_folding.experiment import (
    create_sphere_forward,
    compute_geometry_metrics,
)
from invert.cortical_folding.sh_analysis import cart_to_sph, real_sh_matrix
from invert.simulate import SimulationConfig, SimulationGenerator


def build_hemi_mesh(positions_mm, tris):
    """Build a PyVista PolyData mesh from positions and triangle indices."""
    n_faces = len(tris)
    faces = np.column_stack([np.full(n_faces, 3), tris]).ravel()
    return pv.PolyData(positions_mm, faces)


def run_full_reconstruction(
    n_samples=5000,
    sampling="ico3",
    alpha=0.1,
    l_max=5,
    snr_range=(0, 5),
    n_sources=(1, 5),
    n_timepoints=20,
):
    """Run the full pipeline and return meshes + data for plotting.

    Parameters
    ----------
    n_samples : int
        Number of EEG samples to simulate.
    sampling : str
        Source space resolution ('ico3', 'ico4').
    alpha : float
        Regularization parameter for eLORETA.
    l_max : int
        Maximum SH order for regression.
    snr_range : tuple
        SNR range in dB. Use (500, 500) for near-noiseless development.
    n_sources : tuple
        Range of active sources per sample.
    n_timepoints : int
        Time points per EEG sample.
    """
    print("Creating forward models...")
    fwd_cortex = create_forward_model(sampling, fixed_ori=True, verbose=0)
    fwd_sphere, geo = create_sphere_forward(
        fwd_cortex, sampling=sampling, fixed_ori=True, verbose=0
    )
    metrics = compute_geometry_metrics(geo, fwd_cortex)

    src = fwd_cortex["src"]
    n_dip = fwd_sphere["sol"]["data"].shape[1]
    depth_mm = geo["radial_depth"] * 1000
    pos_cortex = geo["pos_cortex"] * 1000
    pos_sphere = geo["pos_sphere"] * 1000
    center = geo["center"] * 1000
    radius_mm = geo["radius"] * 1000
    directions = geo["directions"]

    # Triangulation
    tris_lh = src[0]["use_tris"]
    tris_rh = src[1]["use_tris"]
    n_lh = len(src[0]["vertno"])
    tris_all = np.vstack([tris_lh, tris_rh + n_lh])

    # KNN for local correlation
    tree = cKDTree(pos_sphere)
    _, knn = tree.query(pos_sphere, k=7)

    # --- eLORETA inverse ---
    print("Building eLORETA inverse operator...")
    solver = Solver("sr-eLORETA")
    solver.make_inverse_operator(fwd_sphere, alpha=alpha, verbose=0)
    K = solver.inverse_operators[0].data[0]  # (n_dip, n_ch)

    # --- Accumulate power + covariance ---
    print(f"Simulating and inverting {n_samples} EEG samples...")
    config = SimulationConfig(
        batch_size=500,
        n_sources=n_sources,
        n_timepoints=n_timepoints,
        snr_range=snr_range,
        verbose=0,
        random_seed=42,
    )
    gen = SimulationGenerator(fwd_cortex, config)

    C_src = np.zeros((n_dip, n_dip))
    acc_power = np.zeros(n_dip)
    n_proc = 0

    for b in range(n_samples // 500):
        batch = next(gen.generate())
        for s in range(batch.x.shape[0]):
            x_hat = K @ batch.x[s]
            C_src += x_hat @ x_hat.T / n_timepoints
            acc_power += np.mean(x_hat**2, axis=1)
            n_proc += 1
        print(f"  {n_proc}/{n_samples} samples")

    C_src /= n_proc
    acc_power /= n_proc

    # Source correlation matrix
    d = np.sqrt(np.diag(C_src))
    d[d == 0] = 1
    C_corr = C_src / np.outer(d, d)

    # --- Multi-scale features ---
    log_power = np.log10(acc_power + 1e-20)
    features = [acc_power, log_power]

    # Multi-scale local correlation at different neighborhood sizes
    for k in [6, 12, 20, 30, 50]:
        _, knn_k = tree.query(pos_sphere, k=k + 1)
        lc = np.array([np.mean(C_corr[i, knn_k[i, 1:]]) for i in range(n_dip)])
        features.append(lc)

    # Correlation at specific distance rings
    for r_min, r_max in [(0, 15), (15, 30), (30, 50), (50, 80), (80, 120)]:
        ring_corr = np.zeros(n_dip)
        for i in range(n_dip):
            dists = np.linalg.norm(pos_sphere - pos_sphere[i], axis=1)
            mask = (dists >= r_min) & (dists < r_max) & (np.arange(n_dip) != i)
            if mask.sum() > 0:
                ring_corr[i] = np.mean(C_corr[i, mask])
        features.append(ring_corr)

    # Directional PSF widths (per axis)
    for axis in range(3):
        dir_width = np.zeros(n_dip)
        for i in range(n_dip):
            delta = pos_sphere - pos_sphere[i]
            proj = np.abs(delta[:, axis])
            corrs = np.abs(C_corr[i, :].copy())
            corrs[i] = 0
            w_sum = corrs.sum()
            if w_sum > 0:
                dir_width[i] = np.sqrt(np.sum(corrs * proj**2) / w_sum)
        features.append(dir_width)

    # Correlation profile statistics
    corr_var = np.zeros(n_dip)
    corr_kurt = np.zeros(n_dip)
    corr_range_feat = np.zeros(n_dip)
    psf_width = np.zeros(n_dip)
    psf_skew = np.zeros(n_dip)
    psf_peak_dist = np.zeros(n_dip)
    for i in range(n_dip):
        c = C_corr[i, :].copy()
        c[i] = 0
        corr_var[i] = np.var(c)
        if np.std(c) > 0:
            corr_kurt[i] = np.mean((c - c.mean()) ** 4) / np.var(c) ** 2
        dists = np.linalg.norm(pos_sphere - pos_sphere[i], axis=1)
        above = C_corr[i, :] > 0.3
        if above.sum() > 1:
            corr_range_feat[i] = np.max(dists[above])
        w = np.abs(c)
        w_sum = w.sum()
        if w_sum > 0:
            mean_d = np.sum(w * dists) / w_sum
            psf_width[i] = np.sqrt(np.sum(w * dists**2) / w_sum)
            psf_peak_dist[i] = dists[np.argmax(c)]
            if psf_width[i] > 0:
                psf_skew[i] = np.sum(w * (dists - mean_d) ** 3) / (
                    w_sum * psf_width[i] ** 3
                )
    features.extend([corr_var, corr_kurt, corr_range_feat,
                     psf_width, psf_skew, psf_peak_dist])

    X_pw = np.column_stack(features + [np.ones(n_dip)])
    print(f"  {X_pw.shape[1]} features extracted")

    # --- SH-regularized regression ---
    print(f"SH regression (l≤{l_max})...")
    theta, phi = cart_to_sph(pos_sphere, center)
    Y = real_sh_matrix(theta, phi, l_max)
    Y_pinv = np.linalg.lstsq(Y, np.eye(n_dip), rcond=None)[0]

    X_sh = Y_pinv @ X_pw
    depth_sh = Y_pinv @ depth_mm
    beta_sh = np.linalg.lstsq(X_sh, depth_sh, rcond=None)[0]
    pred_sh_coeffs = X_sh @ beta_sh
    pred_depth = Y @ pred_sh_coeffs

    r_comb, _ = spearmanr(depth_mm, pred_depth)
    ss_res = np.sum((depth_mm - pred_depth) ** 2)
    ss_tot = np.sum((depth_mm - depth_mm.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    mae = np.mean(np.abs(depth_mm - pred_depth))

    # Oracle comparison
    depth_sh_true = np.linalg.lstsq(Y, depth_mm, rcond=None)[0]
    depth_oracle = Y @ depth_sh_true
    r_oracle, _ = spearmanr(depth_mm, depth_oracle)

    print(f"\nPrediction: ρ={r_comb:.3f}, R²={r2:.3f}, MAE={mae:.1f}mm")
    print(f"Oracle (true depth at l≤{l_max}): ρ={r_oracle:.3f}")
    print(f"Signal recovered: {r_comb / r_oracle * 100:.0f}% of oracle")
    print(f"True depth range: [{depth_mm.min():.0f}, {depth_mm.max():.0f}] mm")
    print(
        f"Predicted depth range: [{pred_depth.min():.1f}, {pred_depth.max():.1f}] mm"
    )

    # --- Build meshes ---
    mesh_cortex = build_hemi_mesh(pos_cortex, tris_all)
    mesh_cortex["depth"] = depth_mm

    mesh_sphere = build_hemi_mesh(pos_sphere, tris_all)
    mesh_sphere["depth"] = np.zeros(n_dip)

    # Mesh shape: smooth depth for vertex positions, raw depth for color.
    def _smooth(v, knn_idx, n_iter=5):
        for _ in range(n_iter):
            v_new = np.zeros_like(v)
            for i in range(len(v)):
                v_new[i] = 0.5 * v[i] + 0.5 * np.mean(v[knn_idx[i, 1:]])
            v = v_new
        return v

    def _make_mesh(depth_vals, shape_scale=1.0):
        shape_depth = _smooth(depth_vals, knn, n_iter=5)
        shape_centered = shape_depth - shape_depth.mean()
        r = radius_mm + shape_depth.mean() + shape_centered * shape_scale
        r = np.maximum(r, 0.3 * radius_mm)
        pos = center + r[:, None] * directions
        mesh = build_hemi_mesh(pos, tris_all)
        mesh["depth"] = depth_vals
        mesh = mesh.smooth(n_iter=30, relaxation_factor=0.1)
        return mesh

    mesh_predicted = _make_mesh(pred_depth)
    mesh_oracle = _make_mesh(depth_oracle)

    # Color-only meshes
    mesh_sphere_true = build_hemi_mesh(pos_sphere, tris_all)
    mesh_sphere_true["depth"] = depth_mm

    mesh_sphere_oracle = build_hemi_mesh(pos_sphere, tris_all)
    mesh_sphere_oracle["depth"] = depth_oracle

    mesh_sphere_pred = build_hemi_mesh(pos_sphere, tris_all)
    mesh_sphere_pred["depth"] = pred_depth

    return {
        "mesh_cortex": mesh_cortex,
        "mesh_sphere": mesh_sphere,
        "mesh_predicted": mesh_predicted,
        "mesh_oracle": mesh_oracle,
        "mesh_sphere_true": mesh_sphere_true,
        "mesh_sphere_oracle": mesh_sphere_oracle,
        "mesh_sphere_pred": mesh_sphere_pred,
        "depth_mm": depth_mm,
        "pred_depth": pred_depth,
        "r_comb": r_comb,
        "r2": r2,
        "mae": mae,
        "r_oracle": r_oracle,
        "l_max": l_max,
        "tris_all": tris_all,
        "center": center,
        "radius_mm": radius_mm,
        "directions": directions,
    }


def plot_interactive(data):
    """Open interactive PyVista viewer: true, oracle, predicted, overlay."""
    clim = [-35, 35]
    cmap = "RdBu_r"
    l = data["l_max"]
    stats = f"ρ={data['r_comb']:.2f}  R²={data['r2']:.3f}  MAE={data['mae']:.1f}mm"
    sbar = {"n_labels": 5, "position_x": 0.05, "width": 0.25}

    pl = pv.Plotter(shape=(2, 2), window_size=(1600, 1200), border=True)

    pl.subplot(0, 0)
    pl.add_mesh(
        data["mesh_cortex"],
        scalars="depth",
        cmap=cmap,
        clim=clim,
        smooth_shading=True,
        scalar_bar_args={**sbar, "title": "depth (mm)"},
    )
    pl.add_text("TRUE (fsaverage)", font_size=14, position="upper_left", color="black")

    pl.subplot(0, 1)
    pl.add_mesh(
        data["mesh_oracle"],
        scalars="depth",
        cmap=cmap,
        clim=clim,
        smooth_shading=True,
        scalar_bar_args={**sbar, "title": "depth (mm)"},
    )
    pl.add_text(
        f"ORACLE (true depth, SH l≤{l})\nρ={data['r_oracle']:.2f}",
        font_size=11,
        position="upper_left",
        color="black",
    )

    pl.subplot(1, 0)
    pl.add_mesh(
        data["mesh_predicted"],
        scalars="depth",
        cmap=cmap,
        clim=clim,
        smooth_shading=True,
        scalar_bar_args={**sbar, "title": "predicted (mm)"},
    )
    pl.add_text(
        f"PREDICTED (SH l≤{l}, eLORETA)\n{stats}",
        font_size=11,
        position="upper_left",
        color="black",
    )

    pl.subplot(1, 1)
    pl.add_mesh(
        data["mesh_sphere"],
        color="whitesmoke",
        smooth_shading=True,
        opacity=0.3,
    )
    pl.add_mesh(
        data["mesh_predicted"],
        scalars="depth",
        cmap=cmap,
        clim=clim,
        smooth_shading=True,
        opacity=0.85,
        scalar_bar_args={**sbar, "title": "predicted (mm)"},
    )
    pl.add_text(
        "SPHERE + PREDICTED", font_size=14, position="upper_left", color="black"
    )

    pl.link_views()
    pl.background_color = "white"
    pl.camera.zoom(1.3)
    print("Interactive viewer — rotate to inspect, close window when done.")
    pl.show()


def plot_interactive_flat(data):
    """Color-only comparison on sphere mesh (no deformation artifacts)."""
    clim = [-35, 35]
    cmap = "RdBu_r"
    l = data["l_max"]
    stats = f"ρ={data['r_comb']:.2f}  R²={data['r2']:.3f}  MAE={data['mae']:.1f}mm"
    sbar = {"n_labels": 5, "position_x": 0.05, "width": 0.25}

    pl = pv.Plotter(shape=(2, 2), window_size=(1600, 1200), border=True)

    pl.subplot(0, 0)
    pl.add_mesh(
        data["mesh_cortex"],
        scalars="depth",
        cmap=cmap,
        clim=clim,
        smooth_shading=True,
        scalar_bar_args={**sbar, "title": "depth (mm)"},
    )
    pl.add_text("TRUE (cortex mesh)", font_size=14, position="upper_left", color="black")

    pl.subplot(0, 1)
    pl.add_mesh(
        data["mesh_sphere_true"],
        scalars="depth",
        cmap=cmap,
        clim=clim,
        smooth_shading=True,
        scalar_bar_args={**sbar, "title": "depth (mm)"},
    )
    pl.add_text(
        "TRUE depth (on sphere)", font_size=14, position="upper_left", color="black"
    )

    pl.subplot(1, 0)
    pl.add_mesh(
        data["mesh_sphere_pred"],
        scalars="depth",
        cmap=cmap,
        clim=clim,
        smooth_shading=True,
        scalar_bar_args={**sbar, "title": "predicted (mm)"},
    )
    pl.add_text(
        f"PREDICTED depth (on sphere)\nSH l≤{l}, eLORETA  {stats}",
        font_size=11,
        position="upper_left",
        color="black",
    )

    pl.subplot(1, 1)
    pl.add_mesh(
        data["mesh_sphere_oracle"],
        scalars="depth",
        cmap=cmap,
        clim=clim,
        smooth_shading=True,
        scalar_bar_args={**sbar, "title": "depth (mm)"},
    )
    pl.add_text(
        f"ORACLE (true SH l≤{l}, on sphere)\nρ={data['r_oracle']:.2f}",
        font_size=11,
        position="upper_left",
        color="black",
    )

    pl.link_views()
    pl.background_color = "white"
    pl.camera.zoom(1.3)
    print("Color-only viewer — rotate to inspect, close window when done.")
    pl.show()


if __name__ == "__main__":
    data = run_full_reconstruction(n_samples=5000, l_max=8)

    print("\n--- Opening color-only viewer (artifact-free) ---")
    plot_interactive_flat(data)

    print("\n--- Opening deformed-mesh viewer ---")
    plot_interactive(data)
