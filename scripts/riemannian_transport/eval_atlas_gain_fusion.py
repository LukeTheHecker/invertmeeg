"""Fit a common free-orientation gain field using independent anatomical donors.

Test on simultaneous cortical neighborhoods after mixed-source calibration.
All anatomical inputs are local. Test-subject geometry is simulation/scoring only.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from eval_spatial_calibration import (
    build_target,
    compute_transport_matrix,
    load_forward,
)
from scipy.spatial import cKDTree


def unit(x):
    return x / np.linalg.norm(x, axis=0)


def atlas_gain(donor, template, cov, transport):
    fixed, free, xyz, names, head = donor
    assert names == template[3]
    # Map anatomical dipole components to head axes. Strip affine scale/shear.
    linear = np.linalg.lstsq(head - head.mean(0), xyz - xyz.mean(0), rcond=None)[0]
    u, _, vt = np.linalg.svd(linear)
    rotation = u @ vt
    if transport:
        free = np.linalg.solve(compute_transport_matrix(cov, fixed), free)
    blocks = free.reshape(len(free), len(xyz), 3) @ rotation
    distance, neighbors = cKDTree(xyz).query(template[2], k=4)
    weights = 1 / np.maximum(0.01, distance) ** 2
    weights /= weights.sum(axis=1, keepdims=True)
    gain = np.einsum("cvko,vk->cvo", blocks[:, neighbors], weights)
    return gain / np.sqrt(np.mean(gain**2))


def score(gain, subspace, xyz, truth):
    blocks = gain.transpose(1, 0, 2)
    q = np.linalg.qr(blocks)[0]
    matches = []
    for basis, targets in zip(subspace, truth, strict=True):
        # Largest canonical correlation of each dipole with signal subspace.
        proj = q.transpose(0, 2, 1) @ basis
        aa, bb = (proj**2).sum(axis=1).T
        ab = (proj[:, :, 0] * proj[:, :, 1]).sum(axis=1)
        power = aa + bb + np.sqrt((aa - bb) ** 2 + 4 * ab**2)
        first = xyz[power.argmax()]
        power[np.linalg.norm(xyz - first, axis=1) < 25] = -np.inf
        pair = np.stack([first, xyz[power.argmax()]])
        direct = np.linalg.norm(pair - targets, axis=1)
        reverse = np.linalg.norm(pair[::-1] - targets, axis=1)
        matches.append(direct if direct.sum() < reverse.sum() else reverse)
    errors = np.asarray(matches)
    return dict(
        mean_mm=errors.mean().item(),
        p90_mm=np.quantile(errors, 0.9).item(),
        both_within_15mm=np.mean(errors.max(axis=1) <= 15).item(),
    )


def simulate_case(target, seed, eeg):
    gain, _, xyz, _, _ = target
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(xyz))
    calibration_sites, test_sites = np.array_split(order, 2)
    # Independent resting windows, twelve active dipoles plus sensor noise.
    cov = np.zeros((len(gain), len(gain)))
    for _ in range(128):
        a = gain[:, rng.choice(calibration_sites, 12, replace=False)]
        y = a @ rng.normal(size=(12, 32))
        noise = rng.normal(size=y.shape)
        if eeg:
            noise -= noise.mean(axis=0)
        y += noise * (0.32 * np.linalg.norm(y) / np.linalg.norm(noise))
        cov += y @ y.T
    tree = cKDTree(xyz[test_sites])
    bases, truths = [], []
    for _ in range(128):
        first = rng.choice(test_sites)
        distant = test_sites[np.linalg.norm(xyz[test_sites] - xyz[first], axis=1) > 45]
        centers = [first, rng.choice(distant)]
        _, neighbors = tree.query(xyz[centers], k=5)
        # Align neighboring normal polarities before coherent patch summation.
        patches = []
        for center, group in zip(centers, test_sites[neighbors], strict=True):
            patch = gain[:, group]
            patches.append(patch @ np.sign(gain[:, center] @ patch))
        time = rng.normal(size=(2, 64))
        for i in range(1, 64):
            time[:, i] += 0.8 * time[:, i - 1]
        y = unit(np.array(patches).T) @ (time / time.std(axis=1, keepdims=True))
        other = gain[:, rng.choice(calibration_sites, 12)] @ rng.normal(size=(12, 64))
        noise = rng.normal(size=y.shape)
        if eeg:
            noise -= noise.mean(axis=0)
        y += np.linalg.norm(y) * (
            0.4 * other / np.linalg.norm(other) + 0.2 * noise / np.linalg.norm(noise)
        )
        bases.append(
            np.linalg.svd(y - y.mean(axis=1, keepdims=True), full_matrices=False)[0][
                :, :2
            ]
        )
        truths.append(xyz[centers])
    return cov, bases, truths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subjects", type=Path)
    parser.add_argument("fs_subjects", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    cached = ("100206", "100307", "100408")
    heldout = ("100610", "101006", "101107", "101309", "101915", "102109", "102311")
    results = {}
    for modality in ("eeg", "meg"):
        library = {
            s: load_forward(args.subjects, args.fs_subjects, s, modality)
            for s in ("fsaverage", *cached)
        }
        template = library["fsaverage"]
        fixed, free, xyz = template[:3]
        for subject in heldout if args.confirm else cached:
            target = (
                build_target(args.subjects, args.fs_subjects, subject, modality)
                if args.confirm
                else library[subject]
            )
            donors = [library[s] for s in cached if s != subject]
            cov, bases, truths = simulate_case(
                target, int(subject) + 811, modality == "eeg"
            )
            warp = np.linalg.inv(compute_transport_matrix(cov, fixed))
            rows = {}
            for name, gain, coords in [
                ("individual_fixed", target[0][:, :, None], target[2]),
                ("individual_free", target[1].reshape(len(free), -1, 3), target[2]),
                ("fixed", fixed[:, :, None], xyz),
                ("rt_fixed", (warp @ fixed)[:, :, None], xyz),
                ("free", free.reshape(len(free), -1, 3), xyz),
                ("rt_free", (warp @ free).reshape(len(free), -1, 3), xyz),
            ]:
                rows[name] = score(gain, bases, coords, truths)
            for use_rt in (False, True):
                fields = [atlas_gain(d, template, cov, use_rt) for d in donors]
                name = "rt_donor" if use_rt else "donor"
                for index, field in enumerate(fields):
                    rows[f"{name}_{index + 1}"] = score(field, bases, xyz, truths)
                rows[name + "_mean"] = score(
                    np.mean(fields, axis=0), bases, xyz, truths
                )
            results[modality + "_" + subject] = rows
            print(
                modality,
                subject,
                {n: round(v["mean_mm"], 1) for n, v in rows.items()},
                flush=True,
            )
            args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
