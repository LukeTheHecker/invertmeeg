"""Pose-only control with unchanged fsaverage anatomy and discrete pose selection."""

import argparse
import json
from pathlib import Path

import mne
import numpy as np
from eval_atlas_gain_fusion import score, simulate_case
from eval_spatial_calibration import (
    build_target,
    cached_forward,
    compute_transport_matrix,
    extract,
    load_forward,
    rigid,
    sensor_info,
)


def pose_bank(subjects, fs, modality, donors):
    reference = cached_forward(subjects, "fsaverage", modality)
    # Explicitly account for possible differences between head axes and atlas axes.
    landmarks = np.vstack([np.zeros(3), 100 * np.eye(3), -100 * np.eye(3)])
    augmented = np.column_stack((landmarks, np.ones(len(landmarks))))
    tal = mne.read_talxfm("fsaverage", fs)
    trans = []
    for d in donors:
        affine = np.linalg.lstsq(
            np.column_stack((d[4], np.ones(len(d[4])))), d[2], rcond=None
        )[0]
        guess = mne.transforms.apply_trans(
            np.linalg.inv(tal["trans"]), augmented @ affine / 1000
        )
        trans.append(rigid(landmarks / 1000, guess))
    targets = [mne.transforms.apply_trans(t, landmarks / 1000) for t in trans]
    trans.append(rigid(landmarks / 1000, np.mean(targets, axis=0)))
    src = mne.setup_source_space(
        "fsaverage", spacing="ico4", subjects_dir=fs, add_dist=False, verbose=False
    )
    conductivity = (0.3, 0.006, 0.3) if modality == "eeg" else (0.3,)
    model = mne.make_bem_model(
        "fsaverage", ico=4, conductivity=conductivity, subjects_dir=fs, verbose=False
    )
    bem = mne.make_bem_solution(model, verbose=False)
    info = sensor_info(reference, modality)
    # Recompute the nominal candidate through the identical BEM path.
    pose_list = [np.linalg.inv(reference["mri_head_t"]["trans"]), *trans]
    bank = []
    for pose in pose_list:
        f = mne.make_forward_solution(
            info,
            mne.transforms.Transform("head", "mri", pose),
            src,
            bem,
            eeg=modality == "eeg",
            meg=modality == "meg",
            mindist=0,
            verbose=False,
        )
        bank.append(extract(f, subjects, fs, "fsaverage", modality))
    return bank


def covariance_cost(cov, fixed):
    # Profile out the unknown amplitude by trace normalization. Compare in
    # log-eigenvalue space to preserve weaker identifiable sensor modes.
    def log_cov(c):
        eig, vec = np.linalg.eigh(c / (np.trace(c) / len(c)))
        return (vec * np.log(np.maximum(eig, 0) + 0.1)) @ vec.T

    return np.linalg.norm(log_cov(cov) - log_cov(fixed @ fixed.T))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("subjects", "fs_subjects", "output"):
        p.add_argument(key, type=Path)
    args = p.parse_args()
    results = {}
    for modality in ("eeg", "meg"):
        donors = [
            load_forward(args.subjects, args.fs_subjects, s, modality)
            for s in ("100206", "100307", "100408")
        ]
        bank = pose_bank(args.subjects, args.fs_subjects, modality, donors)
        for subject in (
            "100610",
            "101006",
            "101107",
            "101309",
            "101915",
            "102109",
            "102311",
        ):
            target = build_target(args.subjects, args.fs_subjects, subject, modality)
            cov, bases, truths = simulate_case(
                target, int(subject) + 811, modality == "eeg"
            )
            costs = [covariance_cost(cov, d[0]) for d in bank]
            rows = {
                "chosen_index": int(np.argmin(costs)),
                "covariance_cost": np.array(costs).tolist(),
            }
            for label, idx in [
                ("nominal", 0),
                ("mean_pose", 4),
                ("selected", rows["chosen_index"]),
            ]:
                fixed, free, xyz = bank[idx][:3]
                for rt in (False, True):
                    g = (
                        np.linalg.solve(compute_transport_matrix(cov, fixed), free)
                        if rt
                        else free
                    )
                    rows[label + ("_rt" if rt else "")] = score(
                        g.reshape(len(g), -1, 3), bases, xyz, truths
                    )
            results[modality + "_" + subject] = rows
            print(
                modality,
                subject,
                rows["chosen_index"],
                {
                    k: round(v["mean_mm"], 1)
                    for k, v in rows.items()
                    if isinstance(v, dict)
                },
                flush=True,
            )
            args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
