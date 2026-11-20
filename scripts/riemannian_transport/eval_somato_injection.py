"""Inject isolated dipoles into measured poststimulus gradiometer backgrounds."""

import argparse
import json
from pathlib import Path

import mne
import nibabel as nib
import numpy as np
from eval_atlas_gain_fusion import atlas_gain
from eval_spatial_calibration import (
    build_target,
    compute_transport_matrix,
    predict,
    rigid,
)
from scipy.spatial import cKDTree


def fiducial_pose(info, fs):
    fid, _ = mne.io.read_fiducials(
        fs / "fsaverage/bem/fsaverage-fiducials.fif", verbose=False
    )
    measured = {p["ident"]: p["r"] for p in info["dig"] if p["kind"] == 1}
    return rigid(
        np.array([measured[p["ident"]] for p in fid]), np.array([p["r"] for p in fid])
    )


def xyz_transform(source, destination):
    # Both coordinate arrays belong to the same known donor or template.
    return np.linalg.lstsq(
        np.column_stack((source, np.ones(len(source)))), destination, rcond=None
    )[0]


def score(model, observations, scale, true_mm):
    gain, position = model
    fit = predict(gain / scale[:, None], position, observations, True)
    error = np.linalg.norm(fit - true_mm, axis=1)
    return {
        "mean_mm": float(error.mean()),
        "median_mm": float(np.median(error)),
        "p90_mm": float(np.percentile(error, 90)),
        "errors_mm": error.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("subjects", "fs", "prepared", "output"):
        parser.add_argument(arg, type=Path)
    a = parser.parse_args()
    ev = mne.read_evokeds(a.prepared / "evoked-ave.fif", 0, verbose=False)
    individual = mne.read_forward_solution(
        a.prepared / "forward-fwd.fif", verbose=False
    )
    assert individual["sol"]["row_names"] == ev.ch_names
    generators = mne.convert_forward_solution(
        individual, surf_ori=True, force_fixed=True, verbose=False
    )["sol"]["data"].astype(float)
    truth_mm = 1000 * individual["source_rr"]
    cov = mne.read_cov(a.prepared / "noise-cov.fif", verbose=False)
    assert cov.ch_names == ev.ch_names
    scale = np.sqrt(cov.data.diagonal())
    # Use measured fiducials to initialize, then fit the scalp digitization.
    initial = fiducial_pose(ev.info, a.fs)
    tpl = build_target(a.subjects, a.fs, "fsaverage", "meg", ev.info, initial, True)
    fid_only = build_target(a.subjects, a.fs, "fsaverage", "meg", ev.info, initial)
    head_to_atlas = xyz_transform(tpl[4], tpl[2])
    landmarks = np.vstack([np.zeros(3), 100 * np.eye(3), -100 * np.eye(3)])
    atlas_mm = np.column_stack([landmarks, np.ones(len(landmarks))]) @ head_to_atlas
    donors = []
    for subj in ("100206", "100307", "100408"):
        tal = mne.read_talxfm(subj, a.subjects)
        mri = mne.transforms.apply_trans(np.linalg.inv(tal["trans"]), atlas_mm / 1000)
        donors.append(
            build_target(
                a.subjects,
                a.fs,
                subj,
                "meg",
                ev.info,
                rigid(landmarks / 1000, mri),
                True,
            )
        )
    assert tpl[3] == ev.ch_names
    fused_head = np.column_stack([tpl[2], np.ones(len(tpl[2]))]) @ np.mean(
        [xyz_transform(d[2], d[4]) for d in donors], axis=0
    )
    prestim = ev.data[:, ev.times < 0]
    cpre = np.cov(prestim)
    models = {
        "fs_fiducials_only": (fid_only[1], fid_only[4]),
        "fs_untransported": (tpl[1], tpl[4]),
        "fs_prestim_rt": (
            np.linalg.solve(compute_transport_matrix(cpre, tpl[0]), tpl[1]),
            tpl[4],
        ),
        "individual_reference": (individual["sol"]["data"], truth_mm),
    }
    for use_rt in (False, True):
        fields = [atlas_gain(d, tpl, cpre, use_rt) for d in donors]
        models["donor_prestim_rt" if use_rt else "donor_untransported"] = (
            np.mean(fields, axis=0).reshape(len(scale), -1),
            fused_head,
        )

    # Assess achieved alignment using physical scalp samples in head coordinates.
    tal = mne.read_talxfm("fsaverage", a.fs)["trans"]
    scalp, _ = nib.freesurfer.read_geometry(a.fs / "fsaverage/bem/outer_skin.surf")
    surface = mne.transforms.apply_trans(tal, scalp / 1000) * 1000
    surface_head = np.column_stack([surface, np.ones(len(surface))]) @ xyz_transform(
        tpl[2], tpl[4]
    )
    dig = np.array([d["r"] for d in ev.info["dig"] if d["kind"] == 4]) * 1000
    # Also retain the explicit metric definition for reproducing this QC.
    distances = cKDTree(surface_head).query(dig)[0]
    output = {
        "n_dig": len(dig),
        "alignment_metric": "distance to nearest scalp mesh vertex (mm)",
        "scalp_median_mm": float(np.median(distances)),
        "scalp_p90_mm": float(np.percentile(distances, 90)),
        "prestim_samples": prestim.shape[1],
        "prestim_cov_rank": int(np.linalg.matrix_rank(cpre)),
        "n_channels": len(scale),
        "nave": ev.nave,
        "snr_results": {},
    }
    # Matched-filter amplitude of a known waveform in ten real disjoint windows.
    length = 12
    post = (ev.data[:, ev.times >= 0] - prestim.mean(axis=1, keepdims=True)) / scale[
        :, None
    ]
    count = post.shape[1] // length
    post = post[:, : count * length].reshape(len(scale), count, length)
    waveform = np.sin(np.pi * (np.arange(length) + 0.5) / length)
    waveform /= np.linalg.norm(waveform)
    rng = np.random.default_rng(21112026)
    n = 512
    sites = rng.choice(len(truth_mm), n, replace=False)
    windows = rng.permutation(np.arange(n) % count)
    directions = (generators[:, sites] / scale[:, None]) * rng.choice([-1, 1], n)
    directions /= np.linalg.norm(directions, axis=0)
    bg_norm = np.linalg.norm(post, axis=(0, 2))[windows]
    output.update(
        n_sites=n,
        n_windows=count,
        samples_per_window=length,
        sites=sites.tolist(),
        window_indices=windows.tolist(),
    )
    for level in (0, 10, 20, "clean"):
        if level == "clean":
            y = directions
        else:
            # Ratio of signal to background energy in the full sensor-time window.
            amplitude = bg_norm * 10 ** (level / 20)
            injected = (directions * amplitude)[:, :, None] * waveform
            y = (post[:, windows, :] + injected) @ waveform
        scores = {
            name: score(model, y, scale, truth_mm[sites])
            for name, model in models.items()
        }
        output["snr_results"][str(level)] = scores
        print(
            level,
            {key: round(v["mean_mm"], 2) for key, v in scores.items()},
            flush=True,
        )
    a.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
