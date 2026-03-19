"""Physics-based covariance fitting for cortical folding estimation.

Directly fits cortical geometry (SH coefficients) to minimize the mismatch
between observed and predicted sensor covariance — no source inversion needed.

The forward model is linearized around the sphere:
    G(a) ≈ G_sphere + Σ_lm a_lm J_lm
    C(a) ≈ C_sphere + Σ_lm a_lm M_lm

where M_lm = G₀ @ J_lm^T + J_lm @ G₀^T. This gives a linear least-squares
problem: 36 SH unknowns from 1891 covariance entries (53x overdetermined).
"""

import os

import mne
import numpy as np
from scipy.stats import spearmanr

from invert.forward.forward import _get_fsaverage_dir
from invert.cortical_folding.sh_analysis import cart_to_sph, real_sh_matrix


def compute_jacobian(fwd_sphere, geo, l_max=5, eps_mm=0.5, verbose=1):
    """Compute Jacobian of the leadfield w.r.t. SH depth coefficients.

    For each SH basis function Y_lm, perturbs all source positions by
    ±eps along the radial direction weighted by Y_lm, and measures the
    change in the leadfield via finite differences.

    Returns
    -------
    J : list of ndarray
        J[k] has shape (n_ch, n_dip) = ∂G/∂a_k for the k-th SH coefficient.
    G0 : ndarray
        The sphere leadfield (n_ch, n_dip).
    """
    fs_dir = _get_fsaverage_dir(verbose=0)
    subjects_dir = os.path.dirname(fs_dir)
    trans = os.path.join(fs_dir, "bem", "fsaverage-trans.fif")
    bem = mne.read_bem_solution(
        os.path.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif"),
        verbose=0,
    )

    pos_sphere = geo["pos_sphere"]  # (n_dip, 3) in meters
    directions = geo["directions"]
    center = geo["center"]
    n_dip = len(pos_sphere)
    n_lh = n_dip // 2

    # SH basis on the sphere
    pos_mm = pos_sphere * 1000
    center_mm = center * 1000
    theta, phi = cart_to_sph(pos_mm, center_mm)
    Y = real_sh_matrix(theta, phi, l_max)
    n_sh = Y.shape[1]

    # Reference leadfield
    G0 = fwd_sphere["sol"]["data"].copy()
    eps_m = eps_mm / 1000.0

    def _make_fwd(positions_m):
        src = mne.setup_source_space(
            "fsaverage", spacing="ico3", surface="white",
            subjects_dir=subjects_dir, add_dist=False, verbose=0,
        )
        for i, hemi in enumerate(src):
            sl = slice(0, n_lh) if i == 0 else slice(n_lh, None)
            hemi["rr"][hemi["vertno"]] = positions_m[sl]
            hemi["nn"][hemi["vertno"]] = directions[sl]
        fwd = mne.make_forward_solution(
            fwd_sphere["info"], trans=trans, src=src, bem=bem,
            eeg=True, mindist=0.0, verbose=0,
        )
        fwd = mne.convert_forward_solution(
            fwd, surf_ori=True, force_fixed=True, use_cps=True, verbose=0,
        )
        return fwd["sol"]["data"]

    J = []
    for k in range(n_sh):
        # Perturbation: move each source radially by eps * Y_lm(theta_i, phi_i)
        delta = eps_m * Y[:, k][:, None] * directions  # (n_dip, 3)

        G_plus = _make_fwd(pos_sphere + delta)
        G_minus = _make_fwd(pos_sphere - delta)

        J_k = (G_plus - G_minus) / (2 * eps_m)
        J.append(J_k)

        if verbose:
            norm = np.linalg.norm(J_k)
            print(f"  J[{k:2d}]: ||∂G/∂a|| = {norm:.1f}")

    return J, G0


def fit_covariance(C_obs, G0, J, l_max=5):
    """Fit SH depth coefficients to match observed sensor covariance.

    Linearized model:
        C(a) ≈ C₀ + Σ_k a_k M_k
    where M_k = G₀ @ J_k^T + J_k @ G₀^T

    Solves: a* = argmin ||vec_upper(C_obs - C₀) - A @ a||²

    Returns
    -------
    a_fit : ndarray of SH coefficients
    C_pred : predicted sensor covariance
    """
    n_ch = G0.shape[0]
    n_sh = len(J)

    C0 = G0 @ G0.T  # sphere covariance

    # Scale C_obs to match C0 (absorb unknown source variance)
    scale = np.trace(C_obs) / np.trace(C0)
    C0_scaled = C0 * scale
    # Also scale J
    scale_sqrt = np.sqrt(scale)

    # Build the linear system using upper-triangle vectorization
    triu_idx = np.triu_indices(n_ch)
    n_obs = len(triu_idx[0])

    residual = C_obs - C0_scaled
    b = residual[triu_idx]

    A = np.zeros((n_obs, n_sh))
    for k in range(n_sh):
        M_k = scale_sqrt * (G0 @ J[k].T + J[k] @ G0.T) * scale_sqrt
        A[:, k] = M_k[triu_idx]

    # Solve least squares
    a_fit, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # Predicted covariance
    C_pred = C0_scaled.copy()
    for k in range(n_sh):
        M_k = scale_sqrt * (G0 @ J[k].T + J[k] @ G0.T) * scale_sqrt
        C_pred += a_fit[k] * M_k

    return a_fit, C_pred


def run_covariance_fitting(
    fwd_cortex, fwd_sphere, geo, l_max=5, n_samples=5000, verbose=1,
):
    """Full pipeline: simulate EEG, compute Jacobian, fit geometry.

    Returns
    -------
    results : dict with fitted SH coefficients, predicted depth, metrics.
    """
    from invert.simulate import SimulationConfig, SimulationGenerator
    from invert.cortical_folding.sh_analysis import sh_decompose

    n_dip = fwd_sphere["sol"]["data"].shape[1]
    depth_mm = geo["radial_depth"] * 1000
    pos_sphere_mm = geo["pos_sphere"] * 1000
    center_mm = geo["center"] * 1000

    # SH basis
    theta, phi = cart_to_sph(pos_sphere_mm, center_mm)
    Y = real_sh_matrix(theta, phi, l_max)

    # True depth SH coefficients (for comparison)
    depth_coeffs_true = sh_decompose(depth_mm, Y)

    # --- Step 1: Simulate EEG and compute sensor covariance ---
    if verbose:
        print(f"Simulating {n_samples} EEG samples...")
    config = SimulationConfig(
        batch_size=500, n_sources=(1, 5), n_timepoints=20,
        snr_range=(500, 500), verbose=0, random_seed=42,
    )
    gen = SimulationGenerator(fwd_cortex, config)

    n_ch = fwd_cortex["sol"]["data"].shape[0]
    C_obs = np.zeros((n_ch, n_ch))
    n_proc = 0
    for _ in range(n_samples // 500):
        batch = next(gen.generate())
        for s in range(batch.x.shape[0]):
            data = batch.x[s]  # (n_ch, n_tp)
            C_obs += data @ data.T / data.shape[1]
            n_proc += 1
    C_obs /= n_proc
    if verbose:
        print(f"  Sensor covariance: {n_ch}×{n_ch}, rank={np.linalg.matrix_rank(C_obs)}")

    # --- Step 2: Compute Jacobian ---
    if verbose:
        print(f"Computing Jacobian ({(l_max+1)**2} SH coefficients)...")
    J, G0 = compute_jacobian(fwd_sphere, geo, l_max=l_max, verbose=verbose)

    # --- Step 3: Fit SH coefficients ---
    if verbose:
        print("Fitting SH coefficients to covariance...")
    a_fit, C_pred = fit_covariance(C_obs, G0, J, l_max=l_max)

    # --- Step 4: Reconstruct depth ---
    pred_depth = Y @ a_fit  # predicted depth in mm

    # Metrics
    r, _ = spearmanr(depth_mm, pred_depth)
    ss_res = np.sum((depth_mm - pred_depth) ** 2)
    ss_tot = np.sum((depth_mm - depth_mm.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    mae = np.mean(np.abs(depth_mm - pred_depth))

    # Oracle
    depth_oracle = Y @ depth_coeffs_true
    r_oracle, _ = spearmanr(depth_mm, depth_oracle)

    # Covariance fit quality
    cov_corr = np.corrcoef(C_obs.ravel(), C_pred.ravel())[0, 1]

    if verbose:
        print(f"\nCovariance fit: r={cov_corr:.4f}")
        print(f"Depth prediction: ρ={r:.3f}, R²={r2:.3f}, MAE={mae:.1f}mm")
        print(f"Oracle (l≤{l_max}): ρ={r_oracle:.3f}")
        print(f"Signal recovered: {r/r_oracle*100:.0f}% of oracle")

        # Per-coefficient comparison
        print(f"\nSH coefficient comparison (true vs fitted):")
        col = 0
        for l in range(l_max + 1):
            n_m = 2 * l + 1
            true_pow = np.sum(depth_coeffs_true[col:col+n_m] ** 2)
            fit_pow = np.sum(a_fit[col:col+n_m] ** 2)
            if n_m > 1:
                r_l, _ = spearmanr(depth_coeffs_true[col:col+n_m], a_fit[col:col+n_m])
            else:
                r_l = np.sign(depth_coeffs_true[col]) * np.sign(a_fit[col])
            print(f"  l={l}: true_pow={true_pow:8.1f}  fit_pow={fit_pow:8.1f}  ρ={r_l:+.2f}")
            col += n_m

    return {
        "a_fit": a_fit,
        "pred_depth": pred_depth,
        "depth_coeffs_true": depth_coeffs_true,
        "C_obs": C_obs,
        "C_pred": C_pred,
        "cov_corr": cov_corr,
        "r": r,
        "r2": r2,
        "mae": mae,
        "r_oracle": r_oracle,
        "Y": Y,
    }


if __name__ == "__main__":
    from invert.forward import create_forward_model
    from invert.cortical_folding.experiment import (
        create_sphere_forward, compute_geometry_metrics,
    )

    fwd_cortex = create_forward_model("ico3", fixed_ori=True, verbose=0)
    fwd_sphere, geo = create_sphere_forward(
        fwd_cortex, sampling="ico3", fixed_ori=True, verbose=0
    )

    results = run_covariance_fitting(
        fwd_cortex, fwd_sphere, geo, l_max=5, n_samples=5000,
    )
