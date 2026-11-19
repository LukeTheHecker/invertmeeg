"""Unlabeled spatial calibration of isolated-dipole topographies, offline only."""

import argparse
import json
import re
from pathlib import Path

import mne
import numpy as np
from scipy.spatial import cKDTree

from invert.cortical_folding.riemannian_transport import compute_transport_matrix


def cached_forward(subjects, subject, modality):
    # The original cache keys start with the sensor hash. Other variants insert
    # conductivity or an orientation label before the hash.
    prefix = f"{subject}_ico4_{modality}_"
    cache = subjects / "_forward_cache"
    files = [
        p
        for p in cache.glob(prefix + "*-fwd.fif")
        if re.match(r"[0-9a-f]{16}_", p.name[len(prefix) :])
    ]
    if len(files) != 1:
        raise ValueError(
            f"Expected one original {prefix} forward in {cache}, found {len(files)}"
        )
    return mne.read_forward_solution(files[0], verbose=False)


def sensor_info(reference, modality):
    sensor = reference["info"]
    info = mne.create_info(
        sensor["ch_names"], 1000, "eeg" if modality == "eeg" else "mag"
    )
    with info._unlock():
        info["chs"] = sensor["chs"]
        info["dev_head_t"] = sensor["dev_head_t"]
    return info


def load_forward(subjects, fs, subject, modality):
    fwd = cached_forward(subjects, subject, modality)
    return extract(fwd, subjects, fs, subject, modality)


def extract(fwd, subjects, fs, subject, modality):
    tal = mne.read_talxfm(subject, fs if subject == "fsaverage" else subjects)
    transform = tal["trans"] @ np.linalg.inv(fwd["mri_head_t"]["trans"])
    xyz = 1000 * mne.transforms.apply_trans(transform, fwd["source_rr"])
    if fwd["surf_ori"] or fwd["sol"]["ncol"] != 3 * len(xyz):
        raise ValueError("Expected a Cartesian free-orientation forward")
    free = fwd["sol"]["data"].astype(float)
    fixed = mne.convert_forward_solution(
        fwd, surf_ori=True, force_fixed=True, verbose=False
    )["sol"]["data"].astype(float)
    if modality == "eeg":
        fixed -= fixed.mean(axis=0)
        free -= free.mean(axis=0)
    return fixed, free, xyz, fwd["sol"]["row_names"], 1000 * fwd["source_rr"]


def power(matrix, exponent):
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * values**exponent) @ vectors.T


def simulate(gain, vertices, snr, rng, eeg):
    y = gain[:, vertices].copy()
    noise = rng.standard_normal(y.shape)
    if eeg:
        noise -= noise.mean(axis=0)
    y += noise * (
        np.linalg.norm(y, axis=0) / np.linalg.norm(noise, axis=0) * 10 ** (-snr / 20)
    )
    return y


def predict(gain, xyz, observations, free):
    # Fit the dipole moment at each location by least squares.
    if free:
        blocks = gain.T.reshape(len(xyz), 3, -1).transpose(0, 2, 1)
        basis = np.linalg.qr(blocks)[0].transpose(0, 2, 1).reshape(-1, len(gain))
    else:
        basis = (gain / np.linalg.norm(gain, axis=0)).T
    estimates = []
    for batch in np.array_split(
        observations, max(1, observations.shape[1] // 64), axis=1
    ):
        scores = (basis @ batch) ** 2
        if free:
            scores = scores.reshape(len(xyz), 3, -1).sum(axis=1)
        estimates.append(xyz[scores.argmax(axis=0)])
    return np.vstack(estimates)


def fit_envelope(points, reference):
    """Match mean projected extent of the outer 3% along fixed directions."""
    poles = np.arange(48) + 0.5
    z = 1 - 2 * poles / len(poles)
    phi = poles * np.pi * (3 - np.sqrt(5))
    radii = np.sqrt(1 - z * z)
    axes = np.c_[radii * np.cos(phi), radii * np.sin(phi), z]
    # Center and scale the regressors to keep translation and strain comparable.
    center = reference.mean(axis=0)
    scale = np.std(reference, axis=0).mean()
    basis = np.column_stack(((points - center) / scale, np.ones(len(points))))
    transform = np.zeros((4, 3))

    def outer_indices(pos):
        k = max(1, int(0.03 * len(pos)))
        return np.argpartition(pos @ axes.T, -k, axis=0)[-k:].T

    targets = (reference[outer_indices(reference)].mean(axis=1) * axes).sum(axis=1)
    for _ in range(4):
        warped = points + basis @ transform
        outer = outer_indices(warped)
        residual = targets - (warped[outer].mean(axis=1) * axes).sum(axis=1)
        jac = (basis[outer].mean(axis=1)[:, :, None] * axes[:, None, :]).reshape(-1, 12)
        # Ridge limits poorly identified shear and rotation.
        step = np.linalg.solve(
            jac.T @ jac + np.eye(12), jac.T @ residual - transform.ravel()
        )
        transform += step.reshape(4, 3)
    return center, scale, transform


def evaluate(
    template,
    target,
    rng,
    modality,
    snr,
    biased=False,
    interference=False,
    donors=(),
    n_cal=1024,
):
    fixed, free, xyz, names, head = template
    gain, _, truth, target_names, true_head = target
    assert names == target_names
    order = rng.permutation(len(truth))
    test = order[:512]
    pool = order[512:]
    if biased:
        # Stress focal sampling: fourfold probability in one hemisphere.
        axis = 1 if biased == 2 else 0
        weights = np.where(truth[pool, axis] < np.median(truth[pool, axis]), 4.0, 1.0)
        pool = rng.choice(pool, n_cal, replace=False, p=weights / weights.sum())
    else:
        pool = pool[:n_cal]
    cal = simulate(gain, pool, snr, rng, modality == "eeg")
    y = simulate(gain, test, snr, rng, modality == "eeg")
    if interference:
        # Independent weaker dipole in each map, including calibration.
        # Keep all nuisance vertices disjoint from both sets of primary sites.
        nuisance = np.setdiff1d(order, np.r_[test, pool])
        for maps in (cal, y):
            other = gain[:, rng.choice(nuisance, maps.shape[1])]
            maps += other * (
                rng.choice([-1, 1], maps.shape[1])
                * 0.4
                * np.linalg.norm(maps, axis=0)
                / np.linalg.norm(other, axis=0)
            )
    cov = cal @ cal.T
    warp = np.linalg.inv(compute_transport_matrix(cov, fixed, alpha=0.8))
    tree = cKDTree(xyz)
    rows = {}

    def record(name, pos, head_pos=None):
        err = np.linalg.norm(pos - truth[test], axis=1)
        if head_pos is None:
            head_pos = head[tree.query(pos)[1]]
        head_err = np.linalg.norm(head_pos - true_head[test], axis=1)
        rows[name] = dict(
            head_mean=float(head_err.mean()),
            mean=float(err.mean()),
            median=float(np.median(err)),
            p90=float(np.quantile(err, 0.9)),
        )

    oracle = predict(gain, truth, y, False)
    record("individual_fixed", oracle, true_head[cKDTree(truth).query(oracle)[1]])
    for use_rt in (False, True):
        pooled, pooled_head = [], []
        for index, (
            donor_fixed,
            donor_free,
            donor_xyz,
            donor_names,
            donor_head,
        ) in enumerate(donors):
            assert donor_names == names
            gain_model = donor_free
            if use_rt:
                gain_model = np.linalg.solve(
                    compute_transport_matrix(cov, donor_fixed), donor_free
                )
            fit = predict(gain_model, donor_xyz, y, True)
            fitted_head = donor_head[cKDTree(donor_xyz).query(fit)[1]]
            prefix = "donor" if use_rt else "raw_donor"
            record(f"{prefix}_{index + 1}", xyz[tree.query(fit)[1]], fitted_head)
            pooled.append(fit)
            pooled_head.append(fitted_head)
        if pooled:
            # Every estimate uses the same output cortical grid.
            mean_pos = np.mean(pooled, axis=0)
            record(
                prefix + "_mean",
                xyz[tree.query(mean_pos)[1]],
                np.mean(pooled_head, axis=0),
            )
    for label, dictionary, flexible in [
        ("fixed", fixed, False),
        ("rt_fixed", warp @ fixed, False),
        ("free", free, True),
        ("rt_free", warp @ free, True),
    ]:
        fits = predict(dictionary, xyz, np.column_stack((cal, y)), flexible)
        calibration, raw = np.split(fits, [len(pool)])
        record(label, raw)
        # Estimate the affine shift/stretch from apparent dipole locations
        # to the template's cortical distribution, then snap to its surface.
        center = calibration.mean(axis=0)
        corrected = (raw - center) @ power(np.cov(calibration.T), -0.5)
        corrected = corrected @ power(np.cov(xyz.T), 0.5) + xyz.mean(axis=0)
        record(label + "_spatial", xyz[tree.query(corrected)[1]])
        # A hemispheric activity imbalance should not imply a lateral head shift.
        # Equalize the two halves when estimating shared spatial moments.
        weights = np.ones(len(calibration))
        left = calibration[:, 0] < 0
        if left.any() and (~left).any():
            weights[left] /= left.sum()
            weights[~left] /= (~left).sum()
        centered = raw - np.average(calibration, weights=weights, axis=0)
        mapped = centered @ power(np.cov(calibration.T, aweights=weights), -0.5)
        mapped = mapped @ power(np.cov(xyz.T), 0.5) + xyz.mean(axis=0)
        record(label + "_balanced", xyz[tree.query(mapped)[1]])
        if label == "rt_free":
            origin, scale, deform = fit_envelope(calibration, xyz)
            basis = np.column_stack(((raw - origin) / scale, np.ones(len(raw))))
            record("rt_free_envelope", xyz[tree.query(raw + basis @ deform)[1]])
    return rows


def scalp_landmarks(scalp):
    x, y, z = scalp.T
    mid = scalp[(abs(x) < 10) & (z > -20) & (z < 30)]
    nose = mid[mid[:, 1].argmax()]
    ears = scalp[(abs(y) < 20) & (z > -25) & (z < 20)]
    left, right = ears[ears[:, 0].argmin()], ears[ears[:, 0].argmax()]
    origin = (left + right) / 2
    lateral = right - left
    lateral /= np.linalg.norm(lateral)
    superior = np.cross(lateral, nose - origin)
    superior /= np.linalg.norm(superior)
    trans = np.eye(4)
    trans[:3, :3] = np.column_stack([lateral, np.cross(superior, lateral), superior])
    trans[:3, 3] = origin / 1000
    return trans


def rigid(source, dest):
    u, _, vt = np.linalg.svd((source - source.mean(0)).T @ (dest - dest.mean(0)))
    sign = np.linalg.det(u @ vt)
    rot = (u * [1, 1, sign]) @ vt
    transform = np.eye(4)
    transform[:3, :3] = rot.T
    transform[:3, 3] = dest.mean(0) - source.mean(0) @ rot
    return transform


def build_target(subjects, fs, subject, modality, info=None, pose=None, digitize=False):
    """Build the forward in RAM. Optional pose maps head to surface RAS (meters)."""
    import nibabel as nib

    directory = fs if subject == "fsaverage" else subjects
    scalp, _ = nib.freesurfer.read_geometry(directory / subject / "bem/outer_skin.surf")
    trans = scalp_landmarks(scalp) if pose is None else pose
    if digitize:
        points = np.array([d["r"] for d in info["dig"] if d["kind"] == 4])
        fid = np.array([d["r"] for d in info["dig"] if d["kind"] == 1])
        anchors = mne.transforms.apply_trans(trans, fid)
        shell = scalp / 1000
        tree = cKDTree(shell)
        for _ in range(30):
            _, pairs = tree.query(mne.transforms.apply_trans(trans, points))
            trans = rigid(
                np.vstack([points, fid, fid]),
                np.vstack([shell[pairs], anchors, anchors]),
            )
    if info is None:
        info = sensor_info(cached_forward(subjects, "fsaverage", modality), modality)
    src = mne.setup_source_space(
        subject, spacing="ico4", subjects_dir=directory, add_dist=False, verbose=False
    )
    conductivity = (0.3, 0.006, 0.3) if modality == "eeg" else (0.3,)
    model = mne.make_bem_model(
        subject,
        ico=4 if subject == "fsaverage" else None,
        conductivity=conductivity,
        subjects_dir=directory,
        verbose=False,
    )
    solution = mne.make_bem_solution(model, verbose=False)
    fwd = mne.make_forward_solution(
        info,
        mne.transforms.Transform("head", "mri", trans),
        src,
        solution,
        meg=modality == "meg",
        eeg=modality == "eeg",
        mindist=0,
        verbose=False,
    )
    return extract(fwd, subjects, fs, subject, modality)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subjects", type=Path)
    parser.add_argument("fs_subjects", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--atlas", default="fsaverage")
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    results = {}
    for modality in ("eeg", "meg"):
        template = load_forward(args.subjects, args.fs_subjects, args.atlas, modality)
        subjects = (
            ("100610", "101006", "101107", "101309", "101915", "102109", "102311")
            if args.confirm
            else ("100206", "100307", "100408")
        )
        for subject in subjects:
            target = (build_target if args.confirm else load_forward)(
                args.subjects, args.fs_subjects, subject, modality
            )
            donors = (
                [
                    load_forward(args.subjects, args.fs_subjects, donor, modality)
                    for donor in ("100206", "100307", "100408")
                    if donor != subject
                ]
                if args.controls
                else []
            )
            cases = [
                (20, 0, False, 1024),
                (5, 0, False, 1024),
                (20, 1, False, 1024),
                (20, 2, False, 1024),
                (10, 0, True, 1024),
            ]
            if args.controls:
                cases.append((20, 0, False, 64))
            for snr, biased, interference, n_cal in cases:
                key = f"{modality}_{subject}_{snr}_{int(biased)}_{int(interference)}_{n_cal}"
                results[key] = evaluate(
                    template,
                    target,
                    np.random.default_rng(int(subject)),
                    modality,
                    snr,
                    biased,
                    interference,
                    donors,
                    n_cal,
                )
                print(
                    key,
                    {k: round(v["mean"], 1) for k, v in results[key].items()},
                    flush=True,
                )
            args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
