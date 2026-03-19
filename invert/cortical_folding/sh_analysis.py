"""Spherical harmonics analysis for cortical folding estimation.

Decomposes depth and feature maps into SH basis, checks per-order
prediction quality, and reconstructs smoothed depth estimates.
"""

import numpy as np
from scipy.special import sph_harm_y
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from invert.forward.forward import _get_fsaverage_dir


def cart_to_sph(pos, center):
    """Convert Cartesian positions to spherical (theta, phi).

    Returns
    -------
    theta : polar angle [0, pi] (0 = north pole)
    phi : azimuthal angle [0, 2pi)
    """
    rel = pos - center
    r = np.linalg.norm(rel, axis=1)
    theta = np.arccos(np.clip(rel[:, 2] / r, -1, 1))
    phi = np.arctan2(rel[:, 1], rel[:, 0]) % (2 * np.pi)
    return theta, phi


def real_sh_matrix(theta, phi, l_max):
    """Build the real spherical harmonics design matrix.

    Returns Y of shape (n_points, n_coeffs) where n_coeffs = (l_max+1)^2.
    Column order: l=0 m=0, l=1 m=-1, l=1 m=0, l=1 m=1, l=2 m=-2, ...

    Uses real SH convention:
        Y_l^m     = sqrt(2) * Re(Y_l^|m|)  for m > 0
        Y_l^0     = Y_l^0 (real)
        Y_l^{-m}  = sqrt(2) * Im(Y_l^|m|)  for m < 0
    """
    n = len(theta)
    n_coeffs = (l_max + 1) ** 2
    Y = np.zeros((n, n_coeffs))

    col = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            ylm = sph_harm_y(l, abs(m), theta, phi)
            if m > 0:
                Y[:, col] = np.sqrt(2) * np.real(ylm)
            elif m < 0:
                Y[:, col] = np.sqrt(2) * np.imag(ylm)
            else:
                Y[:, col] = np.real(ylm)
            col += 1

    return Y


def sh_decompose(values, Y):
    """Least-squares SH decomposition: find coeffs such that Y @ coeffs ≈ values."""
    coeffs, _, _, _ = np.linalg.lstsq(Y, values, rcond=None)
    return coeffs


def sh_reconstruct(coeffs, Y, l_max_cut=None, l_max_full=None):
    """Reconstruct from SH coefficients, optionally truncated at l_max_cut."""
    if l_max_cut is None or l_max_full is None:
        return Y @ coeffs
    # Zero out coefficients above l_max_cut
    coeffs_trunc = coeffs.copy()
    col = 0
    for l in range(l_max_full + 1):
        for m in range(-l, l + 1):
            if l > l_max_cut:
                coeffs_trunc[col] = 0
            col += 1
    return Y @ coeffs_trunc


def sh_order_indices(l_max):
    """Return list of (start, end) index slices for each SH order l."""
    indices = []
    col = 0
    for l in range(l_max + 1):
        n_m = 2 * l + 1
        indices.append((col, col + n_m))
        col += n_m
    return indices


def run_sh_analysis(
    fwd_cortex,
    fwd_sphere,
    geo,
    metrics,
    l_max=15,
    n_samples=5000,
    alpha=0.1,
    solvers=("mne",),
    batch_size=500,
    verbose=1,
):
    """Full SH analysis: decompose features, predict depth per SH order.

    Parameters
    ----------
    solvers : tuple of str
        Solver names to compare.
    l_max : int
        Maximum SH order.

    Returns
    -------
    results : dict with per-solver and per-order prediction quality.
    """
    from invert import Solver
    from invert.cortical_folding.experiment import _build_tikhonov_kernel
    from invert.simulate import SimulationConfig, SimulationGenerator

    n_dip = fwd_sphere["sol"]["data"].shape[1]
    depth_mm = geo["radial_depth"] * 1000
    pos_sphere = geo["pos_sphere"] * 1000
    center = geo["center"] * 1000

    # Build SH basis
    theta, phi = cart_to_sph(pos_sphere, center)
    Y = real_sh_matrix(theta, phi, l_max)
    n_coeffs = Y.shape[1]
    order_idx = sh_order_indices(l_max)

    # Decompose true depth
    depth_coeffs = sh_decompose(depth_mm, Y)
    depth_reconstructed = Y @ depth_coeffs

    # Check SH reconstruction quality (how well does the basis fit?)
    r_fit, _ = spearmanr(depth_mm, depth_reconstructed)
    if verbose:
        print(f"SH basis fit (l_max={l_max}): ρ={r_fit:.3f}")
        # Power spectrum of true depth
        print("True depth SH power spectrum:")
        for l in range(min(l_max + 1, 16)):
            s, e = order_idx[l]
            power = np.sum(depth_coeffs[s:e] ** 2)
            print(f"  l={l:2d}: power={power:8.1f}  ({2*l+1} coeffs)")

    # --- Simulate EEG ---
    config = SimulationConfig(
        batch_size=batch_size,
        n_sources=(1, 5),
        n_timepoints=20,
        snr_range=(0, 5),
        verbose=0,
        random_seed=42,
    )
    gen = SimulationGenerator(fwd_cortex, config)

    if verbose:
        print(f"\nSimulating {n_samples} EEG samples...")

    # Pre-generate all EEG data
    all_eeg = []
    n_batches = n_samples // batch_size
    for b in range(n_batches):
        batch = next(gen.generate())
        all_eeg.append(batch.x)
        if verbose and (b + 1) % 5 == 0:
            print(f"  {(b+1)*batch_size}/{n_samples}")
    all_eeg = np.concatenate(all_eeg, axis=0)

    # --- Per-solver analysis ---
    tree = cKDTree(pos_sphere)
    _, knn = tree.query(pos_sphere, k=7)

    solver_results = {}

    for solver_name in solvers:
        if verbose:
            print(f"\n=== Solver: {solver_name} ===")

        # Build inverse kernel
        solver = Solver(solver_name)
        if solver_name == "music":
            # MUSIC needs data object; skip for now, use kernel-based solvers
            if verbose:
                print("  Skipping MUSIC (needs per-sample mne_obj)")
            continue
        solver.make_inverse_operator(fwd_sphere, alpha=alpha, verbose=0)
        K = solver.inverse_operators[0].data[0]  # (n_dip, n_ch)

        # Accumulate power + covariance
        C_src = np.zeros((n_dip, n_dip))
        acc_power = np.zeros(n_dip)

        for i in range(all_eeg.shape[0]):
            x_hat = K @ all_eeg[i]
            C_src += x_hat @ x_hat.T / 20
            acc_power += np.mean(x_hat**2, axis=1)

        C_src /= all_eeg.shape[0]
        acc_power /= all_eeg.shape[0]

        # Correlation matrix
        d = np.sqrt(np.diag(C_src))
        d[d == 0] = 1
        C_corr = C_src / np.outer(d, d)

        # Features
        local_corr = np.array(
            [np.mean(C_corr[i, knn[i, 1:]]) for i in range(n_dip)]
        )
        corr_range = np.zeros(n_dip)
        for i in range(n_dip):
            dists_i = np.linalg.norm(pos_sphere - pos_sphere[i], axis=1)
            above = C_corr[i, :] > 0.3
            if above.sum() > 1:
                corr_range[i] = np.max(dists_i[above])

        # --- Pointwise prediction (baseline) ---
        log_power = np.log10(acc_power + 1e-20)
        X_pw = np.column_stack(
            [local_corr, corr_range, acc_power, log_power, np.ones(n_dip)]
        )
        beta_pw = np.linalg.lstsq(X_pw, depth_mm, rcond=None)[0]
        pred_pw = X_pw @ beta_pw
        r_pw, _ = spearmanr(depth_mm, pred_pw)

        # --- SH-domain prediction ---
        # Decompose each feature into SH
        features = {
            "power": acc_power,
            "log_power": log_power,
            "local_corr": local_corr,
            "corr_range": corr_range,
        }
        feat_coeffs = {}
        for fname, fvals in features.items():
            feat_coeffs[fname] = sh_decompose(fvals, Y)

        # Build SH-domain feature matrix: for each SH coeff, use the
        # corresponding coefficients from all features
        X_sh = np.column_stack(
            [feat_coeffs[f] for f in features] + [np.ones(n_coeffs)]
        )

        # Predict depth SH coefficients from feature SH coefficients
        beta_sh = np.linalg.lstsq(X_sh, depth_coeffs, rcond=None)[0]
        pred_depth_coeffs = X_sh @ beta_sh

        # Per-order prediction quality
        per_order = []
        for l in range(l_max + 1):
            s, e = order_idx[l]
            true_c = depth_coeffs[s:e]
            pred_c = pred_depth_coeffs[s:e]
            if np.std(true_c) > 1e-10 and len(true_c) > 1:
                r_l, _ = spearmanr(true_c, pred_c)
            elif len(true_c) == 1:
                # Single coefficient — compare values directly
                r_l = 1.0 if np.sign(true_c[0]) == np.sign(pred_c[0]) else -1.0
            else:
                r_l = 0.0
            true_power = np.sum(true_c**2)
            pred_power = np.sum(pred_c**2)
            per_order.append({
                "l": l,
                "r": r_l,
                "true_power": true_power,
                "pred_power": pred_power,
                "n_coeffs": e - s,
            })

        # Reconstruct with different truncation levels
        truncation_results = []
        for l_cut in range(1, l_max + 1):
            pred_trunc = sh_reconstruct(
                pred_depth_coeffs, Y, l_max_cut=l_cut, l_max_full=l_max
            )
            r_trunc, _ = spearmanr(depth_mm, pred_trunc)
            ss_res = np.sum((depth_mm - pred_trunc) ** 2)
            ss_tot = np.sum((depth_mm - depth_mm.mean()) ** 2)
            r2_trunc = 1 - ss_res / ss_tot
            truncation_results.append({
                "l_max": l_cut,
                "r": r_trunc,
                "r2": r2_trunc,
            })

        # Best truncation
        best = max(truncation_results, key=lambda x: x["r"])

        if verbose:
            print(f"  Pointwise: ρ={r_pw:.3f}")
            print(f"  Best SH (l_max={best['l_max']}): ρ={best['r']:.3f}, R²={best['r2']:.3f}")
            print(f"  Per-order prediction quality:")
            for po in per_order[:16]:
                bar = "█" * max(0, int(po["r"] * 20)) if po["r"] > 0 else ""
                print(
                    f"    l={po['l']:2d}: ρ={po['r']:+.2f}  "
                    f"true_pow={po['true_power']:8.1f}  {bar}"
                )

        solver_results[solver_name] = {
            "r_pointwise": r_pw,
            "per_order": per_order,
            "truncation": truncation_results,
            "best_truncation": best,
            "pred_depth_coeffs": pred_depth_coeffs,
            "features": features,
            "feat_coeffs": feat_coeffs,
            "pred_pw": pred_pw,
        }

    return {
        "solver_results": solver_results,
        "Y": Y,
        "depth_coeffs": depth_coeffs,
        "depth_mm": depth_mm,
        "l_max": l_max,
        "order_idx": order_idx,
        "pos_sphere": pos_sphere,
        "center": center,
    }


def extract_multiscale_features(C_corr, acc_power, pos_sphere_mm):
    """Extract multi-scale features from source correlation matrix.

    Parameters
    ----------
    C_corr : (n_dip, n_dip) array
        Source correlation matrix.
    acc_power : (n_dip,) array
        Accumulated source power.
    pos_sphere_mm : (n_dip, 3) array
        Sphere positions in mm.

    Returns
    -------
    X : (n_dip, n_features+1) array
        Feature matrix with intercept column appended.
    """
    n_dip = len(acc_power)
    tree = cKDTree(pos_sphere_mm)

    log_power = np.log10(acc_power + 1e-20)
    features = [acc_power, log_power]

    # Multi-scale local correlation at different neighborhood sizes
    for k in [6, 12, 20, 30, 50]:
        _, knn_k = tree.query(pos_sphere_mm, k=k + 1)
        lc = np.array([np.mean(C_corr[i, knn_k[i, 1:]]) for i in range(n_dip)])
        features.append(lc)

    # Correlation at specific distance rings
    for r_min, r_max in [(0, 15), (15, 30), (30, 50), (50, 80), (80, 120)]:
        ring_corr = np.zeros(n_dip)
        for i in range(n_dip):
            dists = np.linalg.norm(pos_sphere_mm - pos_sphere_mm[i], axis=1)
            mask = (dists >= r_min) & (dists < r_max) & (np.arange(n_dip) != i)
            if mask.sum() > 0:
                ring_corr[i] = np.mean(C_corr[i, mask])
        features.append(ring_corr)

    # Directional PSF widths (per axis)
    for axis in range(3):
        dir_width = np.zeros(n_dip)
        for i in range(n_dip):
            delta = pos_sphere_mm - pos_sphere_mm[i]
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
        dists = np.linalg.norm(pos_sphere_mm - pos_sphere_mm[i], axis=1)
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

    return np.column_stack(features + [np.ones(n_dip)])


def predict_depth_from_eeg(
    fwd_cortex,
    fwd_sphere,
    geo,
    l_max=5,
    n_samples=5000,
    alpha=0.1,
    snr_range=(0, 5),
    n_sources=(1, 5),
    n_timepoints=20,
    batch_size=500,
    verbose=1,
):
    """Full pipeline: simulate EEG, invert on sphere, predict radial depth.

    Parameters
    ----------
    fwd_cortex : mne.Forward
        Ground-truth cortical forward model (EEG is simulated from this).
    fwd_sphere : mne.Forward
        Spherical forward model (used for inversion).
    geo : dict
        Geometry dict from create_sphere_forward (center, radius, directions, etc).
    l_max : int
        Maximum SH order for regression.
    n_samples : int
        Number of EEG samples to simulate.

    Returns
    -------
    result : dict
        pred_depth_mm : predicted radial depth per vertex (mm)
        true_depth_mm : true radial depth (mm)
        rho : Spearman correlation of prediction
        r2 : R-squared
        mae : mean absolute error (mm)
        rho_oracle : Spearman correlation of oracle (true depth at l<=l_max)
    """
    from invert import Solver
    from invert.simulate import SimulationConfig, SimulationGenerator

    n_dip = fwd_sphere["sol"]["data"].shape[1]
    depth_mm = geo["radial_depth"] * 1000
    pos_sphere_mm = geo["pos_sphere"] * 1000
    center_mm = geo["center"] * 1000

    # eLORETA inverse
    if verbose:
        print("Building eLORETA inverse operator...")
    solver = Solver("sr-eLORETA")
    solver.make_inverse_operator(fwd_sphere, alpha=alpha, verbose=0)
    K = solver.inverse_operators[0].data[0]

    # Simulate and accumulate
    if verbose:
        print(f"Simulating and inverting {n_samples} EEG samples...")
    config = SimulationConfig(
        batch_size=batch_size,
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

    for b in range(n_samples // batch_size):
        batch = next(gen.generate())
        for s in range(batch.x.shape[0]):
            x_hat = K @ batch.x[s]
            C_src += x_hat @ x_hat.T / n_timepoints
            acc_power += np.mean(x_hat**2, axis=1)
            n_proc += 1
        if verbose:
            print(f"  {n_proc}/{n_samples} samples")

    C_src /= n_proc
    acc_power /= n_proc

    # Source correlation matrix
    d = np.sqrt(np.diag(C_src))
    d[d == 0] = 1
    C_corr = C_src / np.outer(d, d)

    # Multi-scale features
    X_pw = extract_multiscale_features(C_corr, acc_power, pos_sphere_mm)
    if verbose:
        print(f"  {X_pw.shape[1]} features extracted")

    # SH-regularized regression
    if verbose:
        print(f"SH regression (l<={l_max})...")
    theta, phi = cart_to_sph(pos_sphere_mm, center_mm)
    Y = real_sh_matrix(theta, phi, l_max)
    Y_pinv = np.linalg.lstsq(Y, np.eye(n_dip), rcond=None)[0]

    X_sh = Y_pinv @ X_pw
    depth_sh = Y_pinv @ depth_mm
    beta_sh = np.linalg.lstsq(X_sh, depth_sh, rcond=None)[0]
    pred_sh_coeffs = X_sh @ beta_sh
    pred_depth = Y @ pred_sh_coeffs

    rho, _ = spearmanr(depth_mm, pred_depth)
    ss_res = np.sum((depth_mm - pred_depth) ** 2)
    ss_tot = np.sum((depth_mm - depth_mm.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    mae = np.mean(np.abs(depth_mm - pred_depth))

    # Oracle
    depth_sh_true = np.linalg.lstsq(Y, depth_mm, rcond=None)[0]
    depth_oracle = Y @ depth_sh_true
    rho_oracle, _ = spearmanr(depth_mm, depth_oracle)

    if verbose:
        print(f"Prediction: rho={rho:.3f}, R2={r2:.3f}, MAE={mae:.1f}mm")
        print(f"Oracle (true depth at l<={l_max}): rho={rho_oracle:.3f}")
        print(f"Signal recovered: {rho / rho_oracle * 100:.0f}% of oracle")

    return {
        "pred_depth_mm": pred_depth,
        "true_depth_mm": depth_mm,
        "rho": rho,
        "r2": r2,
        "mae": mae,
        "rho_oracle": rho_oracle,
        "C_corr": C_corr,
        "acc_power": acc_power,
    }


def _leadfield_data_fit(G, C_data):
    """Per-source Rayleigh quotient: g.T @ C @ g / ||g||^2."""
    CG = C_data @ G
    num = np.sum(G * CG, axis=0)
    den = np.sum(G * G, axis=0)
    den[den == 0] = 1
    return num / den


def _build_forward_at_radius(fwd_ref, geo, offset_mm):
    """Build forward model at a specific sphere radius offset (mm)."""
    import os
    fs_dir = _get_fsaverage_dir(verbose=0)
    subjects_dir = os.path.dirname(fs_dir)
    trans = os.path.join(fs_dir, "bem", "fsaverage-trans.fif")
    bem_path = os.path.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")

    import mne as _mne
    center = geo["center"]
    r = geo["radius"] + offset_mm / 1000
    src = _mne.setup_source_space(
        "fsaverage", spacing="ico3", surface="white",
        subjects_dir=subjects_dir, add_dist=False, verbose=0,
    )
    pos_all = np.vstack([
        src[0]["rr"][src[0]["vertno"]],
        src[1]["rr"][src[1]["vertno"]],
    ])
    radii = np.linalg.norm(pos_all - center, axis=1)
    dirs = (pos_all - center) / radii[:, None]
    pos_r = center + r * dirs
    n_lh = len(src[0]["vertno"])
    for i, hemi in enumerate(src):
        sl = slice(0, n_lh) if i == 0 else slice(n_lh, None)
        hemi["rr"][hemi["vertno"]] = pos_r[sl]
        hemi["nn"][hemi["vertno"]] = dirs[sl]
    fwd = _mne.make_forward_solution(
        fwd_ref["info"], trans=trans, src=src, bem=bem_path,
        eeg=True, mindist=0.0, verbose=0,
    )
    return _mne.convert_forward_solution(
        fwd, surf_ori=True, force_fixed=True, use_cps=True, verbose=0,
    )


def estimate_depth_unsupervised(
    fwd_cortex,
    fwd_sphere,
    geo,
    l_max=5,
    n_samples=5000,
    alpha=0.1,
    snr_range=(0, 5),
    n_sources=(1, 5),
    n_timepoints=20,
    batch_size=500,
    C_data=None,
    verbose=1,
):
    """Estimate radial depth from EEG without ground truth labels.

    Combines two complementary feature sets via PCA:

    1. **Inverse-derived features** (21 features): accumulated source power,
       local correlations, PSF widths, correlation profile statistics —
       extracted from eLORETA inversion on the sphere model.

    2. **Leadfield consistency features** (5 features): per-source Rayleigh
       quotient (g.T @ C_data @ g / ||g||^2) at multiple sphere radii,
       measuring how well each candidate position's leadfield matches the
       observed sensor covariance.

    PCA on all 26 features (PC1-4) captures depth-related variation.
    No regression against true depth is performed.

    Parameters
    ----------
    fwd_cortex : mne.Forward | None
        Forward model used to simulate EEG.  Can be ``None`` when
        ``C_data`` is provided.
    fwd_sphere : mne.Forward
        Spherical forward model (used for inversion).
    geo : dict
        Geometry dict from ``create_sphere_forward`` or
        ``create_sphere_forward_template``.
    l_max : int
        Maximum SH order for smoothing.
    n_samples : int
        Number of EEG samples to simulate (ignored when ``C_data`` is given).
    C_data : (n_ch, n_ch) array or None
        Pre-computed sensor covariance.  When provided, the simulation step
        is skipped and source features are computed analytically as
        ``C_src = K @ C_data @ K.T``.

    Returns
    -------
    result : dict
        pred_depth_mm : estimated radial depth per vertex (mm)
        acc_power : raw accumulated source power
        C_corr : source correlation matrix
    """
    from invert import Solver

    n_dip = fwd_sphere["sol"]["data"].shape[1]
    pos_sphere_mm = geo["pos_sphere"] * 1000
    center_mm = geo["center"] * 1000

    # eLORETA inverse on sphere
    if verbose:
        print("Building eLORETA inverse operator...")
    solver = Solver("sr-eLORETA")
    solver.make_inverse_operator(fwd_sphere, alpha=alpha, verbose=0)
    K = solver.inverse_operators[0].data[0]

    if C_data is not None:
        # Analytic path: compute source covariance from sensor covariance
        if verbose:
            print("Computing source features from sensor covariance (analytic)...")
        C_src = K @ C_data @ K.T
        acc_power = np.diag(C_src).copy()
    else:
        # Simulation path (backward-compatible)
        from invert.simulate import SimulationConfig, SimulationGenerator

        if verbose:
            print(f"Simulating and inverting {n_samples} EEG samples...")
        config = SimulationConfig(
            batch_size=batch_size,
            n_sources=n_sources,
            n_timepoints=n_timepoints,
            snr_range=snr_range,
            verbose=0,
            random_seed=42,
        )
        gen = SimulationGenerator(fwd_cortex, config)

        n_ch = fwd_sphere["sol"]["data"].shape[0]
        C_src = np.zeros((n_dip, n_dip))
        C_data = np.zeros((n_ch, n_ch))
        acc_power = np.zeros(n_dip)
        n_proc = 0
        for b in range(n_samples // batch_size):
            batch = next(gen.generate())
            for s in range(batch.x.shape[0]):
                x = batch.x[s]
                x_hat = K @ x
                C_src += x_hat @ x_hat.T / n_timepoints
                C_data += x @ x.T / n_timepoints
                acc_power += np.mean(x_hat**2, axis=1)
                n_proc += 1
            if verbose:
                print(f"  {n_proc}/{n_samples} samples")
        C_src /= n_proc
        C_data /= n_proc
        acc_power /= n_proc

    d = np.sqrt(np.diag(C_src))
    d[d == 0] = 1
    C_corr = C_src / np.outer(d, d)

    # --- Inverse-derived features ---
    if verbose:
        print("Extracting multi-scale features...")
    features = extract_multiscale_features(C_corr, acc_power, pos_sphere_mm)

    # --- Leadfield consistency features ---
    if verbose:
        print("Computing leadfield consistency at multiple radii...")
    offsets_mm = np.arange(-25, 7.5, 2.5)
    fit_per_radius = {}
    for offset in offsets_mm:
        fwd_r = _build_forward_at_radius(fwd_sphere, geo, offset)
        if fwd_r["sol"]["data"].shape[1] != n_dip:
            continue
        fit_per_radius[offset] = _leadfield_data_fit(
            fwd_r["sol"]["data"], C_data,
        )
        if verbose:
            print(f"  offset={offset:+.1f}mm ok")

    valid_offsets = sorted(fit_per_radius.keys())

    if len(valid_offsets) >= 2:
        # fit_matrix: (n_dip, n_radii) — each column is fit at one radius
        fit_matrix = np.column_stack([fit_per_radius[o] for o in valid_offsets])
        offsets_arr = np.array(valid_offsets)

        # Leadfield consistency features
        # Normalize over radii (axis=1) for each source
        fit_norm = fit_matrix - fit_matrix.min(axis=1, keepdims=True)
        fit_norm = fit_norm / (fit_norm.sum(axis=1, keepdims=True) + 1e-30)
        weighted_offset = (fit_norm * offsets_arr[None, :]).sum(axis=1)
        best_offset = offsets_arr[np.argmax(fit_matrix, axis=1)]
        fit_at_sphere = _leadfield_data_fit(fwd_sphere["sol"]["data"], C_data)
        fit_max = fit_matrix.max(axis=1)
        fit_range = fit_matrix.max(axis=1) - fit_matrix.min(axis=1)

        lf_features = [weighted_offset, best_offset, fit_at_sphere,
                        fit_max, fit_range]
        n_lf = 5
    else:
        if verbose:
            print("  No valid radii for leadfield consistency — using "
                  "inverse features only")
        lf_features = []
        n_lf = 0

    # --- Combine all features and run PCA ---
    n_feat = 21 + n_lf
    if verbose:
        print(f"PCA depth estimation ({n_feat} features, PC1-4)...")
    X = np.column_stack([features[:, :-1]] + lf_features)

    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1
    X_std = (X - mu) / sigma

    U, S, Vt = np.linalg.svd(X_std, full_matrices=False)
    composite = sum(S[k] * U[:, k] for k in range(4))

    # Sign: composite should anti-correlate with log-power
    log_power = X_std[:, 1]
    if np.corrcoef(composite, log_power)[0, 1] > 0:
        composite = -composite

    # SH-smooth and scale to cortex-like depth range
    theta, phi = cart_to_sph(pos_sphere_mm, center_mm)
    Y = real_sh_matrix(theta, phi, l_max)
    proxy = Y @ sh_decompose(composite, Y)
    proxy = proxy - np.mean(proxy)
    current_std = np.std(proxy)
    if current_std > 0:
        proxy = proxy * (8.0 / current_std)

    if verbose:
        print(f"Estimated depth range: [{proxy.min():.1f}, {proxy.max():.1f}] mm")
        print(f"Estimated depth std: {np.std(proxy):.1f} mm")
        true_depth_mm = geo["radial_depth"] * 1000
        rho, _ = spearmanr(true_depth_mm, proxy)
        print(f"Correlation with true depth: rho={rho:.3f} (for reference only)")

    return {
        "pred_depth_mm": proxy,
        "acc_power": acc_power,
        "C_corr": C_corr,
    }


if __name__ == "__main__":
    from invert.forward import create_forward_model
    from invert.cortical_folding.experiment import (
        create_sphere_forward,
        compute_geometry_metrics,
    )

    fwd_cortex = create_forward_model("ico3", fixed_ori=True, verbose=0)
    fwd_sphere, geo = create_sphere_forward(
        fwd_cortex, sampling="ico3", fixed_ori=True, verbose=0
    )
    metrics = compute_geometry_metrics(geo, fwd_cortex)

    results = run_sh_analysis(
        fwd_cortex,
        fwd_sphere,
        geo,
        metrics,
        l_max=15,
        n_samples=5000,
        solvers=("mne", "eloreta", "sloreta", "dspm", "laura"),
        verbose=1,
    )

    # Summary table
    print("\n" + "=" * 70)
    print("SOLVER COMPARISON SUMMARY")
    print("=" * 70)
    for name, sr in results["solver_results"].items():
        b = sr["best_truncation"]
        print(
            f"  {name:12s}  pointwise ρ={sr['r_pointwise']:.3f}  "
            f"best SH (l≤{b['l_max']:2d}) ρ={b['r']:.3f}  R²={b['r2']:.3f}"
        )
