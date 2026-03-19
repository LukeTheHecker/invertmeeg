"""High-level API for cortical geometry estimation from EEG.

Usage::

    from invert.cortical_folding.estimator import CorticalFoldingEstimator

    est = CorticalFoldingEstimator()
    est.fit(fwd_true, method="unsupervised")

    fwd_est = est.forward
    est.create_subject("my_estimated_brain")
    metrics = est.evaluate()
"""

import os
import shutil
import tempfile

import mne
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from invert.forward import create_forward_model
from invert.forward.forward import _get_fsaverage_dir
from invert.cortical_folding.experiment import (
    create_sphere_forward,
    create_sphere_forward_template,
)
from invert.cortical_folding.sh_analysis import (
    estimate_depth_unsupervised,
    predict_depth_from_eeg,
)


def _compact_tris(src):
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
    normals = np.zeros_like(positions)
    for tri in tris:
        v0, v1, v2 = positions[tri]
        fn = np.cross(v1 - v0, v2 - v0)
        for idx in tri:
            normals[idx] += fn
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return normals / norms


class CorticalFoldingEstimator:
    """Estimate cortical geometry from EEG and build a forward model.

    Parameters
    ----------
    sampling : str
        Source space resolution (default ``'ico3'``).

    Attributes (available after :meth:`fit`)
    -----------------------------------------
    forward : mne.Forward
        Estimated forward model at predicted source positions.
    fwd_true : mne.Forward
        The true (input) forward model.
    fwd_sphere : mne.Forward
        Sphere forward model used internally.
    geo : dict
        Sphere geometry (center, radius, positions, depth).
    pred_depth_mm : array
        Estimated radial depth per source (mm).
    method : str
        Estimation method used (``'supervised'`` or ``'unsupervised'``).
    subjects_dir : str
        FreeSurfer subjects directory.
    """

    def __init__(self, sampling="ico3"):
        self.sampling = sampling
        self._fs_dir = _get_fsaverage_dir(verbose=0)
        self.subjects_dir = os.path.dirname(self._fs_dir)
        self.forward = None
        self.fwd_true = None
        self.fwd_sphere = None
        self.geo = None
        self.pred_depth_mm = None
        self.method = None
        self._pos_est = None

    def fit(self, fwd_true, method="unsupervised", n_samples=5000, l_max=5,
            verbose=1):
        """Estimate cortical depth and build forward model.

        Parameters
        ----------
        fwd_true : mne.Forward
            The true forward model (used to simulate EEG).
        method : ``'unsupervised'`` | ``'supervised'``
            ``'unsupervised'``: PCA on EEG-derived features + leadfield
            consistency. No ground truth depth labels used.
            ``'supervised'``: SH-regularized regression against true depth.
            Leaks ground truth — use for comparison only.
        n_samples : int
            Number of EEG samples to simulate for depth estimation.
        l_max : int
            Maximum spherical harmonics order for smoothing.
        verbose : int
            Verbosity level.
        """
        self.fwd_true = fwd_true
        self.method = method

        if verbose:
            print(f"Building sphere model...")
        self.fwd_sphere, self.geo = create_sphere_forward(
            fwd_true, sampling=self.sampling, verbose=0,
        )
        if verbose:
            n_dip = fwd_true["sol"]["data"].shape[1]
            r = self.geo["radius"] * 1000
            print(f"  {n_dip} dipoles, sphere radius={r:.1f}mm")

        if verbose:
            print(f"Estimating depth (method={method}, {n_samples} samples)...")
        if method == "unsupervised":
            result = estimate_depth_unsupervised(
                fwd_true, self.fwd_sphere, self.geo,
                l_max=l_max, n_samples=n_samples, verbose=verbose,
            )
        elif method == "supervised":
            result = predict_depth_from_eeg(
                fwd_true, self.fwd_sphere, self.geo,
                l_max=l_max, n_samples=n_samples, verbose=verbose,
            )
        else:
            raise ValueError(f"method must be 'unsupervised' or 'supervised', "
                             f"got {method!r}")

        self.pred_depth_mm = result["pred_depth_mm"]
        self._build_forward(verbose=verbose)

    def fit_from_data(self, sensor_cov, info, l_max=5, verbose=1):
        """Estimate cortical geometry from pre-computed sensor covariance.

        This is the "no ground truth" path: only the sensor covariance
        and electrode montage are needed.

        Parameters
        ----------
        sensor_cov : (n_ch, n_ch) array
            Sensor-space covariance matrix from EEG recordings.
        info : mne.Info
            Electrode montage information.
        l_max : int
            Maximum SH order for smoothing.
        verbose : int
            Verbosity level.
        """
        self.method = "unsupervised"
        self.fwd_true = None

        if verbose:
            print("Building sphere model from template...")
        self.fwd_sphere, self.geo = create_sphere_forward_template(
            info, sampling=self.sampling, verbose=0,
        )
        if verbose:
            n_dip = self.fwd_sphere["sol"]["data"].shape[1]
            r = self.geo["radius"] * 1000
            print(f"  {n_dip} dipoles, sphere radius={r:.1f}mm")

        if verbose:
            print("Estimating depth (unsupervised, from sensor covariance)...")
        result = estimate_depth_unsupervised(
            None, self.fwd_sphere, self.geo,
            l_max=l_max, C_data=sensor_cov, verbose=verbose,
        )
        self.pred_depth_mm = result["pred_depth_mm"]
        self._build_forward(info=info, verbose=verbose)

    def _build_forward(self, info=None, verbose=0):
        """Build estimated forward model at predicted positions."""
        info = info if info is not None else self.fwd_true["info"]
        trans = os.path.join(self._fs_dir, "bem", "fsaverage-trans.fif")
        bem = os.path.join(self._fs_dir, "bem",
                           "fsaverage-5120-5120-5120-bem-sol.fif")

        depth_m = self.pred_depth_mm / 1000
        self._pos_est = (self.geo["pos_sphere"]
                         + depth_m[:, None] * self.geo["directions"])

        tris = _compact_tris(self.fwd_sphere["src"])
        mesh_nn = _compute_mesh_normals(self._pos_est, tris)

        src = mne.setup_source_space(
            "fsaverage", spacing=self.sampling, surface="white",
            subjects_dir=self.subjects_dir, add_dist=False, verbose=0,
        )
        n_lh = len(self.fwd_sphere["src"][0]["vertno"])
        for i, hemi in enumerate(src):
            verts = self.fwd_sphere["src"][i]["vertno"]
            hemi["vertno"] = verts
            sl = slice(0, n_lh) if i == 0 else slice(n_lh, None)
            hemi["rr"][verts] = self._pos_est[sl]
            hemi["nn"][verts] = mesh_nn[sl]
            hemi["nuse"] = len(verts)
            hemi["inuse"][:] = 0
            hemi["inuse"][verts] = 1

        if verbose:
            print("Building estimated forward model...")
        fwd = mne.make_forward_solution(
            info, trans=trans, src=src, bem=bem,
            eeg=True, mindist=0.0, verbose=0,
        )
        self.forward = mne.convert_forward_solution(
            fwd, surf_ori=True, force_fixed=True, use_cps=True, verbose=0,
        )
        if verbose:
            print(f"  {self.forward['sol']['data'].shape[1]} dipoles")

    def evaluate(self):
        """Compare estimated depth against true depth.

        Returns
        -------
        dict with rho (Spearman), mae (mm), pred/true ranges.
        """
        true = self.geo["radial_depth"] * 1000
        pred = self.pred_depth_mm
        rho, _ = spearmanr(true, pred)
        mae = np.mean(np.abs(true - pred))
        return {
            "rho": rho,
            "mae": mae,
            "pred_range": (pred.min(), pred.max()),
            "true_range": (true.min(), true.max()),
        }

    def create_subject(self, name="estimated_brain"):
        """Create a FreeSurfer subject directory with the estimated surface.

        The depth estimate is SH l<=5 by construction. We evaluate the SH
        coefficients at every full-resolution vertex position to get a
        perfectly smooth radial displacement, then apply it to fsaverage's
        white surface.

        Parameters
        ----------
        name : str
            Subject name (directory will be created under subjects_dir).

        Returns
        -------
        name : str
            The subject name, for use with ``stc.plot(subject=name)``.
        """
        from invert.cortical_folding.sh_analysis import (
            cart_to_sph, real_sh_matrix, sh_decompose,
        )

        surf_dir = os.path.join(self.subjects_dir, name, "surf")
        os.makedirs(surf_dir, exist_ok=True)

        # Decompose the ico3-level depth estimate into SH coefficients
        # using the ico3 sphere positions, then evaluate at full resolution.
        center_mm = self.geo["center"] * 1000
        ico_pos_mm = self.geo["pos_sphere"] * 1000
        theta_ico, phi_ico = cart_to_sph(ico_pos_mm, center_mm)
        l_max = 5
        Y_ico = real_sh_matrix(theta_ico, phi_ico, l_max)
        depth_coeffs = sh_decompose(self.pred_depth_mm, Y_ico)

        for hi, hemi_name in enumerate(["lh", "rh"]):
            # Read full-resolution fsaverage surface (mm)
            pos_full, faces = mne.read_surface(
                os.path.join(self._fs_dir, "surf", f"{hemi_name}.white")
            )

            # Compute spherical coords for every full-mesh vertex
            # relative to the same center used for ico3
            rel = pos_full - center_mm
            r_full = np.linalg.norm(rel, axis=1)
            directions_full = rel / np.maximum(r_full, 1e-10)[:, None]
            theta_full, phi_full = cart_to_sph(pos_full, center_mm)

            # Evaluate SH depth at full resolution
            Y_full = real_sh_matrix(theta_full, phi_full, l_max)
            depth_full_mm = Y_full @ depth_coeffs

            # Apply radial displacement
            pos_deformed = pos_full + depth_full_mm[:, None] * directions_full

            mne.write_surface(
                os.path.join(surf_dir, f"{hemi_name}.white"),
                pos_deformed, faces, overwrite=True,
            )
            curv = os.path.join(self._fs_dir, "surf", f"{hemi_name}.curv")
            if os.path.exists(curv):
                shutil.copy2(curv, os.path.join(surf_dir, f"{hemi_name}.curv"))

        return name

    def plot_stc(self, stc, subject=None, title=None, **kwargs):
        """Plot a SourceEstimate and return the screenshot as PNG bytes.

        Parameters
        ----------
        stc : mne.SourceEstimate
        subject : str or None
            FreeSurfer subject. If None, uses 'fsaverage'.
        title : str or None
        **kwargs
            Passed to ``stc.plot()``.

        Returns
        -------
        png_path : str
            Path to the saved screenshot.
        """
        plot_kw = dict(
            subjects_dir=self.subjects_dir,
            subject=subject or "fsaverage",
            surface="white",
            hemi="both",
            cortex="low_contrast",
            time_unit="s",
            size=(800, 400),
            smoothing_steps=10,
            verbose=0,
        )
        plot_kw.update(kwargs)
        brain_kw = {"title": title} if title else {}
        brain = stc.plot(brain_kwargs=brain_kw, **plot_kw)
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        brain.save_image(f.name)
        brain.close()
        return f.name
