"""Core functions for cortical folding estimation experiment.

Approach:
1. Create a spherical source model by projecting fsaverage cortex onto a sphere
2. Simulate EEG from real cortex sources (fsaverage)
3. Solve using the spherical forward model
4. Accumulate source power across many samples
5. The accumulated power pattern on the sphere should reflect cortical geometry
"""

import os

import mne
import numpy as np

from invert.forward.forward import _get_fsaverage_dir
from invert.forward import create_forward_model


def _make_sphere_src_and_geo(cortex_fwd, sampling, verbose):
    """Shared helper: build sphere source space and geometry dict."""
    fs_dir = _get_fsaverage_dir(verbose=verbose)
    subjects_dir = os.path.dirname(fs_dir)
    trans = os.path.join(fs_dir, "bem", "fsaverage-trans.fif")
    bem = os.path.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")

    src = mne.setup_source_space(
        "fsaverage",
        spacing=sampling,
        surface="white",
        subjects_dir=subjects_dir,
        add_dist=False,
        verbose=verbose,
    )

    pos_lh = src[0]["rr"][src[0]["vertno"]].copy()
    pos_rh = src[1]["rr"][src[1]["vertno"]].copy()
    pos_cortex = np.vstack([pos_lh, pos_rh])

    center = np.mean(pos_cortex, axis=0)
    radii = np.linalg.norm(pos_cortex - center, axis=1)
    radius = np.mean(radii)

    directions = (pos_cortex - center) / radii[:, None]
    pos_sphere = center + radius * directions

    n_lh = len(src[0]["vertno"])
    for i, hemi in enumerate(src):
        sl = slice(0, n_lh) if i == 0 else slice(n_lh, None)
        hemi["rr"][hemi["vertno"]] = pos_sphere[sl]
        hemi["nn"][hemi["vertno"]] = directions[sl]

    geo = {
        "center": center,
        "radius": radius,
        "pos_cortex": pos_cortex,
        "pos_sphere": pos_sphere,
        "radial_depth": radii - radius,
        "directions": directions,
    }
    return src, trans, bem, geo


def create_sphere_forward(cortex_fwd, sampling="ico3", fixed_ori=True, verbose=0):
    """Create forward model with sources projected onto a sphere.

    Parameters
    ----------
    cortex_fwd : mne.Forward
        The cortical forward model (ground truth).
    sampling : str
        Source space resolution (must match cortex_fwd).
    fixed_ori : bool
        If True, fixed radial orientation (1 component per source).
        If False, free orientation (3 components per source).
    verbose : int
        MNE verbosity level.

    Returns
    -------
    fwd_sphere : mne.Forward
        Forward model with spherical source positions.
    geo : dict
        Geometry info: center, radius, cortex/sphere positions, radial_depth.
    """
    src, trans, bem, geo = _make_sphere_src_and_geo(cortex_fwd, sampling, verbose)

    fwd_sphere = mne.make_forward_solution(
        cortex_fwd["info"],
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
        force_fixed=fixed_ori,
        use_cps=True,
        verbose=verbose,
    )

    n_cortex = cortex_fwd["sol"]["data"].shape[1]
    n_sphere_expected = n_cortex if fixed_ori else 3 * n_cortex
    n_sphere = fwd_sphere["sol"]["data"].shape[1]
    if n_sphere != n_sphere_expected:
        n_dropped = n_cortex - n_sphere // (1 if fixed_ori else 3)
        print(
            f"Warning: {n_dropped} sources dropped (too close to BEM). "
            f"Got {n_sphere}, expected {n_sphere_expected}."
        )

    return fwd_sphere, geo


def create_sphere_forward_template(info, sampling="ico3", fixed_ori=True, verbose=0):
    """Create sphere forward model from fsaverage template + electrode info.

    Unlike :func:`create_sphere_forward`, this does **not** need a cortical
    forward model.  The sphere centre / radius come from fsaverage's white-
    surface positions at the requested sampling.

    Parameters
    ----------
    info : mne.Info
        Electrode montage information (channel names + positions).
    sampling : str
        Source-space resolution (e.g. ``'ico3'``).
    fixed_ori : bool
        Fixed radial orientation if True.
    verbose : int
        MNE verbosity level.

    Returns
    -------
    fwd_sphere : mne.Forward
    geo : dict
    """
    src, trans, bem, geo = _make_sphere_src_and_geo(None, sampling, verbose)

    fwd_sphere = mne.make_forward_solution(
        info, trans=trans, src=src, bem=bem,
        eeg=True, mindist=0.0, verbose=verbose,
    )
    fwd_sphere = mne.convert_forward_solution(
        fwd_sphere, surf_ori=True, force_fixed=fixed_ori,
        use_cps=True, verbose=verbose,
    )
    return fwd_sphere, geo


def compute_geometry_metrics(geo, fwd_cortex):
    """Compute cortical geometry metrics for analysis.

    Parameters
    ----------
    geo : dict
        Output from create_sphere_forward.
    fwd_cortex : mne.Forward
        Cortical forward model (for normals).

    Returns
    -------
    metrics : dict
        radial_depth : signed radial displacement (m), +ve = gyral
        orientation_deviation : angle between cortex normal and radial (rad)
        displacement : 3D distance between cortex and sphere positions (m)
    """
    # Cortex normals (fixed orientation)
    src = fwd_cortex["src"]
    nn_lh = src[0]["nn"][src[0]["vertno"]]
    nn_rh = src[1]["nn"][src[1]["vertno"]]
    normals_cortex = np.vstack([nn_lh, nn_rh])

    # Angle between cortex normal and radial direction
    cos_angle = np.sum(normals_cortex * geo["directions"], axis=1)
    cos_angle = np.clip(cos_angle, -1, 1)
    orientation_deviation = np.arccos(np.abs(cos_angle))  # 0 = radial, pi/2 = tangential

    # 3D displacement between cortex and sphere positions
    displacement = np.linalg.norm(geo["pos_cortex"] - geo["pos_sphere"], axis=1)

    return {
        "radial_depth": geo["radial_depth"],
        "orientation_deviation": orientation_deviation,
        "displacement": displacement,
        "normals_cortex": normals_cortex,
    }


def run_accumulation(
    fwd_cortex,
    fwd_sphere,
    n_samples=1000,
    n_timepoints=20,
    snr_range=(0, 5),
    n_sources=(1, 5),
    solver_name="mne",
    alpha=0.1,
    batch_size=200,
    verbose=0,
):
    """Simulate EEG from cortex, solve on sphere, accumulate power.

    Parameters
    ----------
    fwd_cortex : mne.Forward
        Ground truth cortical forward model for simulation.
    fwd_sphere : mne.Forward
        Spherical forward model for inversion.
    n_samples : int
        Total number of EEG samples to process.
    n_timepoints : int
        Time points per sample.
    snr_range : tuple
        SNR range in dB.
    n_sources : tuple
        Range of active sources per sample.
    solver_name : str
        Solver to use ('mne', 'dspm', 'eloreta').
    alpha : float
        Regularization parameter.
    batch_size : int
        Samples per simulation batch.
    verbose : int
        Verbosity level.

    Returns
    -------
    results : dict
        accumulated_power : mean source power at each sphere dipole
        accumulated_abs : mean |source estimate| at each sphere dipole
        n_processed : number of samples processed
    """
    from invert import Solver
    from invert.simulate import SimulationConfig, SimulationGenerator

    config = SimulationConfig(
        batch_size=batch_size,
        n_sources=n_sources,
        n_timepoints=n_timepoints,
        snr_range=snr_range,
        verbose=verbose,
    )

    gen = SimulationGenerator(fwd_cortex, config)

    # Create solver with sphere forward model
    solver = Solver(solver_name)
    solver.make_inverse_operator(fwd_sphere, alpha=alpha, verbose=verbose)

    n_dipoles = fwd_sphere["sol"]["data"].shape[1]
    accumulated_power = np.zeros(n_dipoles)
    accumulated_abs = np.zeros(n_dipoles)
    n_processed = 0

    n_batches = max(1, n_samples // batch_size)
    for batch_idx in range(n_batches):
        batch = next(gen.generate())
        x = batch.x  # (batch_size, n_channels, n_timepoints)

        for i in range(x.shape[0]):
            x_hat = solver.inverse_operators[0].apply(x[i])  # (n_dipoles, n_tp)
            power = np.mean(x_hat**2, axis=1)
            accumulated_power += power
            accumulated_abs += np.mean(np.abs(x_hat), axis=1)
            n_processed += 1

        if verbose:
            print(f"  Batch {batch_idx + 1}/{n_batches}: {n_processed} samples done")

    accumulated_power /= n_processed
    accumulated_abs /= n_processed

    return {
        "accumulated_power": accumulated_power,
        "accumulated_abs": accumulated_abs,
        "n_processed": n_processed,
    }


# ---------------------------------------------------------------------------
# Approach 1: Free-orientation inversion → recover source orientations
# ---------------------------------------------------------------------------


def _build_tikhonov_kernel(G, alpha=0.1):
    """Build Tikhonov inverse kernel with CAR and relative regularisation."""
    G = G - G.mean(axis=0, keepdims=True)  # CAR
    n_ch = G.shape[0]
    GGt = G @ G.T
    alpha_abs = alpha * np.trace(GGt) / n_ch
    K = G.T @ np.linalg.solve(GGt + alpha_abs * np.eye(n_ch), np.eye(n_ch))
    return K  # (n_src_cols, n_ch)


def run_orientation_analysis(
    fwd_cortex,
    fwd_sphere_free,
    n_samples=2000,
    n_timepoints=20,
    snr_range=(0, 5),
    n_sources=(1, 5),
    alpha=0.1,
    batch_size=500,
    verbose=0,
):
    """Free-orientation inversion to estimate source orientations.

    Uses 3-component sources on the sphere.  For each dipole the ratio
    of tangential to total power reveals whether the true cortex source
    is radially or tangentially oriented.

    Returns
    -------
    results : dict
        tangential_ratio : fraction of power in tangential components
        dominant_ori : dominant orientation in surf_ori frame (n_dip, 3)
        orientation_strength : fraction of total variance in dominant direction
        normal_power, tangential_power : per-component power
    """
    from invert.simulate import SimulationConfig, SimulationGenerator

    G_free = fwd_sphere_free["sol"]["data"]  # (n_ch, 3*n_dip)
    n_ch, n_3dip = G_free.shape
    n_dip = n_3dip // 3

    K = _build_tikhonov_kernel(G_free, alpha)  # (3*n_dip, n_ch)

    config = SimulationConfig(
        batch_size=batch_size,
        n_sources=n_sources,
        n_timepoints=n_timepoints,
        snr_range=snr_range,
        verbose=verbose,
    )
    gen = SimulationGenerator(fwd_cortex, config)

    orient_tensor = np.zeros((n_dip, 3, 3))
    normal_power = np.zeros(n_dip)
    tangential_power = np.zeros(n_dip)
    n_processed = 0

    n_batches = max(1, n_samples // batch_size)
    for batch_idx in range(n_batches):
        batch = next(gen.generate())
        x = batch.x

        for i in range(x.shape[0]):
            x_hat = K @ x[i]  # (3*n_dip, n_tp)
            x_3d = x_hat.reshape(n_dip, 3, -1)

            # Accumulate second-moment tensor (n_dip, 3, 3)
            orient_tensor += np.einsum("dit,djt->dij", x_3d, x_3d)
            normal_power += np.sum(x_3d[:, 0, :] ** 2, axis=1)
            tangential_power += np.sum(
                x_3d[:, 1, :] ** 2 + x_3d[:, 2, :] ** 2, axis=1
            )
            n_processed += 1

        if verbose:
            print(f"  Orientation batch {batch_idx + 1}/{n_batches}: {n_processed}")

    orient_tensor /= n_processed
    normal_power /= n_processed
    tangential_power /= n_processed

    total_power = normal_power + tangential_power
    tangential_ratio = tangential_power / (total_power + 1e-30)

    # Dominant orientation via eigendecomposition of the 3x3 tensor
    dominant_ori = np.zeros((n_dip, 3))
    orientation_strength = np.zeros(n_dip)
    for i in range(n_dip):
        evals, evecs = np.linalg.eigh(orient_tensor[i])
        dominant_ori[i] = evecs[:, -1]
        orientation_strength[i] = evals[-1] / (evals.sum() + 1e-30)

    return {
        "tangential_ratio": tangential_ratio,
        "dominant_ori": dominant_ori,
        "orientation_strength": orientation_strength,
        "normal_power": normal_power,
        "tangential_power": tangential_power,
        "total_power": total_power,
        "n_processed": n_processed,
    }


# ---------------------------------------------------------------------------
# Approach 2: Source covariance / resolution matrix analysis
# ---------------------------------------------------------------------------


def compute_resolution_matrix(fwd_cortex, fwd_sphere, alpha=0.1):
    """Compute R = K_sphere @ G_cortex (analytical resolution matrix).

    R[i, j] = how much cortex source j contributes to sphere estimate i.
    """
    G_cortex = fwd_cortex["sol"]["data"].copy()
    K_sphere = _build_tikhonov_kernel(fwd_sphere["sol"]["data"].copy(), alpha)
    # CAR the cortex leadfield the same way
    G_cortex = G_cortex - G_cortex.mean(axis=0, keepdims=True)
    return K_sphere @ G_cortex  # (n_dip, n_dip)


def run_covariance_analysis(
    fwd_cortex,
    fwd_sphere,
    n_samples=2000,
    n_timepoints=20,
    snr_range=(0, 5),
    n_sources=(1, 5),
    alpha=0.1,
    batch_size=500,
    verbose=0,
):
    """Accumulate source-source covariance from sphere inversions.

    Returns
    -------
    results : dict
        C_corr : (n_dip, n_dip) correlation matrix of source estimates
        local_corr : mean correlation with k-nearest neighbors on the sphere
        psf_width : point-spread-function width from resolution matrix
        peak_displacement : where each cortex source maps on the sphere
    """
    from scipy.spatial import cKDTree
    from invert.simulate import SimulationConfig, SimulationGenerator
    from invert.util import pos_from_forward

    G_sphere = fwd_sphere["sol"]["data"]
    n_ch, n_dip = G_sphere.shape
    K = _build_tikhonov_kernel(G_sphere, alpha)

    # --- Simulation-based covariance ---
    config = SimulationConfig(
        batch_size=batch_size,
        n_sources=n_sources,
        n_timepoints=n_timepoints,
        snr_range=snr_range,
        verbose=verbose,
    )
    gen = SimulationGenerator(fwd_cortex, config)

    C_src = np.zeros((n_dip, n_dip))
    n_processed = 0

    n_batches = max(1, n_samples // batch_size)
    for batch_idx in range(n_batches):
        batch = next(gen.generate())
        x = batch.x

        for i in range(x.shape[0]):
            x_hat = K @ x[i]  # (n_dip, n_tp)
            C_src += x_hat @ x_hat.T / n_timepoints
            n_processed += 1

        if verbose:
            print(f"  Covariance batch {batch_idx + 1}/{n_batches}: {n_processed}")

    C_src /= n_processed

    # Normalize to correlation
    d = np.sqrt(np.diag(C_src))
    d[d == 0] = 1.0
    C_corr = C_src / np.outer(d, d)

    # --- Resolution matrix (analytical) ---
    R = compute_resolution_matrix(fwd_cortex, fwd_sphere, alpha)

    # PSF width: for each cortex source j, how spread is column R[:, j]?
    pos_sphere = pos_from_forward(fwd_sphere)
    psf_width = np.zeros(n_dip)
    peak_indices = np.argmax(np.abs(R), axis=0)
    peak_displacement = np.linalg.norm(
        pos_sphere[peak_indices] - pos_sphere, axis=1
    )
    for j in range(n_dip):
        col = np.abs(R[:, j])
        col_norm = col / (col.sum() + 1e-30)
        dists = np.linalg.norm(pos_sphere - pos_sphere[peak_indices[j]], axis=1)
        psf_width[j] = np.sum(col_norm * dists)

    # Local correlation: mean correlation with k nearest neighbours
    tree = cKDTree(pos_sphere)
    k = 6
    _, indices = tree.query(pos_sphere, k=k + 1)
    local_corr = np.zeros(n_dip)
    for i in range(n_dip):
        neighbors = indices[i, 1:]
        local_corr[i] = np.mean(C_corr[i, neighbors])

    return {
        "C_corr": C_corr,
        "local_corr": local_corr,
        "R": R,
        "psf_width": psf_width,
        "peak_displacement": peak_displacement,
        "peak_indices": peak_indices,
        "n_processed": n_processed,
    }
