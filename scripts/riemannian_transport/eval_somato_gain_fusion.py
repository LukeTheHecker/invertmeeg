"""Measured 40 ms somatosensory topography; predict unseen gradiometer pairs."""

import argparse
import json
from pathlib import Path

import mne
import numpy as np
from eval_atlas_gain_fusion import atlas_gain
from eval_spatial_calibration import (
    build_target,
    compute_transport_matrix,
    extract,
    rigid,
)


def fit(gain, y, train):
    g = gain[train].transpose(1, 0, 2)
    moments = np.linalg.pinv(g) @ y[train]
    pred = (g @ moments[:, :, None])[:, :, 0]
    idx = np.argmin(np.sum((pred - y[train]) ** 2, axis=1))
    return idx, gain[:, idx] @ moments[idx]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("subjects", "fs_subjects", "prepared", "output"):
        p.add_argument(key, type=Path)
    args = p.parse_args()
    e = mne.read_evokeds(args.prepared / "evoked-ave.fif", 0, verbose=False)
    src = mne.setup_source_space(
        "fsaverage",
        spacing="ico4",
        subjects_dir=args.fs_subjects,
        add_dist=False,
        verbose=False,
    )
    bem = args.fs_subjects / "fsaverage/bem"
    f = mne.make_forward_solution(
        e.info,
        bem / "fsaverage-trans.fif",
        src,
        bem / "fsaverage-5120-5120-5120-bem-sol.fif",
        eeg=False,
        meg=True,
        mindist=0,
        verbose=False,
    )
    template = extract(f, args.subjects, args.fs_subjects, "fsaverage", "meg")
    donors = [
        build_target(args.subjects, args.fs_subjects, s, "meg", info=e.info)
        for s in ("100206", "100307", "100408")
    ]
    fixed, free, xyz, names, head = template
    assert names == e.ch_names
    ev = e.copy().apply_baseline((None, 0))
    cal = ev.data[:, (ev.times >= 0.080) & (ev.times <= 0.300)]
    cov = np.cov(cal)
    y = ev.data[:, (ev.times >= 0.035) & (ev.times <= 0.045)].mean(axis=1)
    scale = ev.data[:, ev.times < 0].std(axis=1)
    cols = len(fixed)
    warp = np.linalg.inv(compute_transport_matrix(cov, fixed))
    individual = mne.read_forward_solution(
        args.prepared / "forward-fwd.fif", verbose=False
    )
    assert individual["sol"]["row_names"] == names
    models = {
        "individual_free": (
            individual["sol"]["data"].reshape(cols, -1, 3),
            1000 * individual["source_rr"],
        ),
        "fs_free": (free.reshape(cols, -1, 3), head),
        "fs_rt_free": ((warp @ free).reshape(cols, -1, 3), head),
    }
    # Map the fused anatomical coordinates back to the mean donor head frame
    # for the proxy-distance readout; sensor prediction needs no such mapping.
    augmented = np.column_stack((xyz, np.ones(len(xyz))))
    maps = [
        np.linalg.lstsq(np.column_stack((d[2], np.ones(len(d[2])))), d[4], rcond=None)[
            0
        ]
        for d in donors
    ]
    for rt in (False, True):
        fields = [atlas_gain(d, template, cov, rt) for d in donors]
        models["fusion_rt" if rt else "fusion"] = (
            np.mean(fields, axis=0),
            augmented @ np.mean(maps, axis=0),
        )
    # Independent pose prior: express the nominal fsaverage head frame in
    # each donor anatomy, then refine with this recording's scalp digitization.
    frame = np.linalg.lstsq(
        np.column_stack((head, np.ones(len(head)))), xyz, rcond=None
    )[0]
    landmarks = np.vstack([np.zeros(3), 100 * np.eye(3), -100 * np.eye(3)])
    atlas_mm = np.column_stack((landmarks, np.ones(len(landmarks)))) @ frame
    adjusted = []
    for subject in ("100206", "100307", "100408", "fsaverage"):
        tal = mne.read_talxfm(
            subject, args.fs_subjects if subject == "fsaverage" else args.subjects
        )
        donor_mri = mne.transforms.apply_trans(
            np.linalg.inv(tal["trans"]), atlas_mm / 1000
        )
        pose = rigid(landmarks / 1000, donor_mri)
        d = build_target(
            args.subjects, args.fs_subjects, subject, "meg", e.info, pose, True
        )
        if subject == "fsaverage":
            models["fs_digitized"] = (d[1].reshape(cols, -1, 3), d[4])
        else:
            adjusted.append(d)
    maps = [
        np.linalg.lstsq(np.column_stack((d[2], np.ones(len(d[2])))), d[4], rcond=None)[
            0
        ]
        for d in adjusted
    ]
    for rt in (False, True):
        fields = [atlas_gain(d, template, cov, rt) for d in adjusted]
        models["digitized_fusion_rt" if rt else "digitized_fusion"] = (
            np.mean(fields, axis=0),
            augmented @ np.mean(maps, axis=0),
        )
    reference = json.loads((args.prepared / "meta.json").read_text())[
        "reference_peak_pos_mm"
    ]
    groups = np.array([name[:-1] for name in names])
    folds = np.array_split(np.random.default_rng(214).permutation(np.unique(groups)), 6)
    rows = {}
    for name, (g, coords) in models.items():
        g = g / scale[:, None, None]
        data = y / scale
        prediction = np.zeros_like(data)
        for fold in folds:
            hold = np.isin(groups, fold)
            _, pred = fit(g, data, ~hold)
            prediction[hold] = pred[hold]
        idx, _ = fit(g, data, np.ones(cols, dtype=bool))
        rows[name] = dict(
            cv_r2=float(1 - np.sum((prediction - data) ** 2) / sum(data**2)),
            correlation=float(np.corrcoef(data, prediction)[0, 1]),
            head_mm=coords[idx].tolist(),
            distance_to_esmv_mm=float(np.linalg.norm(coords[idx] - reference)),
        )
    print(rows, flush=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
