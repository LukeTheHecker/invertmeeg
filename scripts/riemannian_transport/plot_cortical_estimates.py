"""Render head 100610's replication scans on inflated fsaverage (offline)."""

import argparse
import json
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np
from eval_atlas_gain_fusion import atlas_gain, simulate_case
from eval_spatial_calibration import (
    build_target,
    cached_forward,
    compute_transport_matrix,
    load_forward,
)
from scipy.spatial import cKDTree

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D

METHODS = {
    "fixed": "Fixed",
    "free": "Free",
    "rt_fixed": "Fixed + RT",
    "rt_free": "Free + RT",
    "rt_donor_mean": "Donors + RT",
}


def scan(gain, bases, xyz, truth):
    q = np.linalg.qr(gain.transpose(1, 0, 2))[0].transpose(0, 2, 1)
    scores, peaks, errors = [], [], []
    for basis, centers in zip(bases, truth, strict=True):
        projection = q @ basis
        aa, bb = (projection**2).sum(axis=1).T
        ab = np.prod(projection, axis=2).sum(axis=1)
        # Squared maximum canonical correlation, on an absolute 0–1 scale.
        value = (aa + bb + np.sqrt((aa - bb) ** 2 + 4 * ab**2)) / 2
        first = value.argmax()
        second = np.where(
            np.linalg.norm(xyz - xyz[first], axis=1) >= 25, value, -np.inf
        ).argmax()
        pair = xyz[[first, second]]
        direct = np.linalg.norm(pair - centers, axis=1).mean()
        reverse = np.linalg.norm(pair[::-1] - centers, axis=1).mean()
        scores.append(value)
        peaks.append([first, second])
        errors.append(min(direct, reverse))
    return np.array(scores), np.array(peaks), np.array(errors)


def surface(fs, src, hemi, offset):
    # Use the true folding pattern from the full-resolution cortex. Map scores
    # to it by nearest neighbors on this hemisphere's registration sphere.
    prefix = fs / "fsaverage" / "surf" / hemi
    points, faces = nib.freesurfer.read_geometry(str(prefix) + ".inflated")
    sphere, _ = nib.freesurfer.read_geometry(str(prefix) + ".sphere")
    sulc = nib.freesurfer.read_morph_data(str(prefix) + ".sulc")
    vertices = src["vertno"]
    neighbors = cKDTree(sphere[vertices]).query(sphere, k=3)
    weights = 1 / np.maximum(neighbors[0], 1e-6) ** 2
    weights /= weights.sum(axis=1, keepdims=True)
    # Sort the front-facing triangles once per view for painter-order rendering.
    panels = []
    for side in ("lateral", "medial"):
        direction = (-1 if hemi == "lh" else 1) * (1 if side == "lateral" else -1)
        plane = points[:, 1:] * [-direction, 1]
        depth = points[:, 0] * direction
        normal = np.cross(
            points[faces[:, 1]] - points[faces[:, 0]],
            points[faces[:, 2]] - points[faces[:, 0]],
        )
        normal /= np.maximum(1e-12, np.linalg.norm(normal, axis=1, keepdims=True))
        visible = np.where(normal[:, 0] * direction > 0)[0]
        visible = visible[np.argsort(depth[faces[visible]].mean(axis=1))]
        light = np.clip(normal[visible] @ [direction * 0.86, -0.12, 0.48], 0, 1)
        gray = (
            0.59 + 0.18 * light - 0.17 * np.tanh(sulc[faces[visible]].mean(axis=1))
        )[:, None]
        height = np.full(len(points), False)
        height[np.unique(faces[visible])] = True
        # Determine marker occlusion from the foremost face at its projected point.
        # A local tree bounds the search; exact barycentric interpolation gives depth.
        projected = plane[faces[visible]]
        centroids = projected.mean(axis=1)
        radius = np.linalg.norm(projected - centroids[:, None, :], axis=2).max() + 1e-6
        tree = cKDTree(centroids)
        front = []
        for vertex in vertices:
            if not height[vertex]:
                front.append(False)
                continue
            candidates = visible[tree.query_ball_point(plane[vertex], radius)]
            weights2 = tri_weights(plane[faces[candidates]], plane[vertex])
            inside = np.all(weights2 >= -1e-7, axis=1)
            max_depth = np.max(
                (depth[faces[candidates[inside]]] * weights2[inside]).sum(1),
                initial=-np.inf,
            )
            front.append(max_depth - depth[vertex] <= 1.0)
        panels.append(
            (projected, faces[visible], gray, light, plane[vertices], np.array(front))
        )
    return offset, neighbors[1], weights, panels


def tri_weights(triangles, point):
    a, b, c = triangles.transpose(1, 0, 2)

    def det(u, v):
        return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    denom = det(b - a, c - a)
    w = det(point - a, c - a) / denom
    v = det(b - a, point - a) / denom
    return np.column_stack((1 - w - v, w, v))


def draw(output, modality, trial, surfaces, estimates, truth_idx):
    fig, axes = plt.subplots(4, len(METHODS), figsize=(12.5, 7.4))
    fig.subplots_adjust(
        left=0.105, right=0.985, top=0.82, bottom=0.105, wspace=0.045, hspace=0.02
    )
    colors = plt.get_cmap("inferno")
    for j, (key, label) in enumerate(METHODS.items()):
        values, peaks, errors = estimates[key]
        axes[0, j].set_title(f"{label}\n{errors[trial]:.1f} mm", fontsize=12)
        for h, (offset, neighbor, weight, panels) in enumerate(surfaces):
            field = (values[trial, neighbor + offset] * weight).sum(axis=1)
            for v, (poly, face, gray, light, coords, front) in enumerate(panels):
                ax = axes[2 * h + v, j]
                value = field[face].mean(axis=1)
                strength = np.clip((value - 0.6) / 0.4, 0, 1)
                rgb = colors(strength)[:, :3] * (0.88 + 0.12 * light[:, None])
                rgb = np.where((value >= 0.6)[:, None], rgb, gray)
                ax.add_collection(
                    PolyCollection(
                        poly, facecolors=rgb, edgecolors="none", antialiased=False
                    )
                )
                for indices, mark, col, size in [
                    (truth_idx, "o", "#26e5de", 80),
                    (peaks[trial], "+", "#1ea3ff", 105),
                ]:
                    ids = indices - offset
                    ids = ids[(ids >= 0) & (ids < len(coords))]
                    ids = ids[front[ids]]
                    if len(ids):
                        x, y = coords[ids].T
                        if mark == "o":
                            ax.scatter(
                                x,
                                y,
                                s=size,
                                facecolors="none",
                                edgecolors=col,
                                linewidths=2.0,
                            )
                        else:
                            ax.scatter(
                                x, y, s=size, marker=mark, color=col, linewidths=2.3
                            )
                ax.autoscale_view()
                ax.set_aspect("equal")
                ax.axis("off")
                if j == 0:
                    ax.text(
                        -0.04,
                        0.5,
                        ("Left", "Right")[h] + "\n" + ("lateral", "medial")[v],
                        ha="right",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=10,
                    )
    fig.suptitle(
        f"{modality.upper()}  •  Two simultaneous cortical sources", fontsize=20, y=0.98
    )
    fig.text(
        0.5,
        0.915,
        f"Held-out head 100610   /   trial {trial + 1}   /   mean error of the two peaks",
        ha="center",
        fontsize=11,
    )
    legend = [
        Line2D(
            [],
            [],
            marker="o",
            markerfacecolor="none",
            markeredgecolor="#009b96",
            linestyle="none",
            label="True center (projected)",
        ),
        Line2D(
            [],
            [],
            marker="+",
            color="#1ea3ff",
            linestyle="none",
            label="Estimated peak",
        ),
    ]
    fig.legend(
        handles=legend,
        loc="lower left",
        bbox_to_anchor=(0.09, 0.006),
        ncol=2,
        frameon=False,
    )
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(0.6, 1), cmap=colors),
        cax=fig.add_axes([0.64, 0.065, 0.30, 0.016]),
        orientation="horizontal",
    )
    cb.set_label("Squared subspace correlation", fontsize=9)
    fig.savefig(output, dpi=145)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("subjects", "fs", "output"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    subject = "100610"
    benchmark = json.loads(
        (
            Path(__file__).parent / "spatial_calibration/patch_replication.json"
        ).read_text()
    )
    src = cached_forward(args.subjects, "fsaverage", "eeg")["src"]
    surfaces = [
        surface(args.fs, src[0], "lh", 0),
        surface(args.fs, src[1], "rh", len(src[0]["vertno"])),
    ]
    meg_src = cached_forward(args.subjects, "fsaverage", "meg")["src"]
    for eeg_hemi, meg_hemi in zip(src, meg_src, strict=True):
        np.testing.assert_array_equal(eeg_hemi["vertno"], meg_hemi["vertno"])
    for modality in ("meg", "eeg"):
        template = load_forward(args.subjects, args.fs, "fsaverage", modality)
        fixed, free, xyz = template[:3]
        target = build_target(args.subjects, args.fs, subject, modality)
        cov, bases, truth = simulate_case(target, int(subject) + 811, modality == "eeg")
        warp = np.linalg.inv(compute_transport_matrix(cov, fixed))
        donors = [
            load_forward(args.subjects, args.fs, s, modality)
            for s in ("100206", "100307", "100408")
        ]
        gain = {
            "fixed": fixed[:, :, None],
            "free": free.reshape(len(free), -1, 3),
            "rt_fixed": (warp @ fixed)[:, :, None],
            "rt_free": (warp @ free).reshape(len(free), -1, 3),
            "rt_donor_mean": np.mean(
                [atlas_gain(d, template, cov, True) for d in donors], axis=0
            ),
        }
        results = {k: scan(g, bases, xyz, truth) for k, g in gain.items()}
        for k, (_, _, errors) in results.items():
            np.testing.assert_allclose(
                errors.mean(),
                benchmark[f"{modality}_{subject}"][k]["mean_mm"],
                atol=1e-4,
            )
        indices = cKDTree(xyz).query(truth)[1]
        bilateral = (indices < len(src[0]["vertno"])).sum(axis=1) == 1
        delta = results["free"][2] - results["rt_donor_mean"][2]
        choices = {"typical": np.median(delta), "lower_tail": np.percentile(delta, 10)}
        for kind, desired in choices.items():
            trial = np.argmin(np.where(bilateral, np.abs(delta - desired), np.inf))
            print(
                modality,
                kind,
                trial + 1,
                delta[trial],
                {k: round(v[2][trial], 2) for k, v in results.items()},
                flush=True,
            )
            draw(
                args.output / f"cortex_{modality}_{kind}.png",
                modality,
                trial,
                surfaces,
                results,
                indices[trial],
            )


if __name__ == "__main__":
    main()
