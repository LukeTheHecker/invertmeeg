"""Regenerate the paired-subject comparison from the adjacent result files."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def main():
    point = json.loads((ROOT / "replication.json").read_text())
    patch = json.loads((ROOT / "patch_replication.json").read_text())
    figure, axes = plt.subplots(2, 2, figsize=(8.1, 5.3), sharey="row")
    for column, modality in enumerate(("eeg", "meg")):
        axes[0, column].set_title(modality.upper(), fontweight="bold")
        for row, (data, metric, methods) in enumerate(
            [
                (point, "mean", ["rt_fixed", "rt_free", "donor_mean"]),
                (patch, "mean_mm", ["rt_fixed", "rt_free", "rt_donor_mean"]),
            ]
        ):
            rows = [
                v
                for k, v in sorted(data.items())
                if k.startswith(modality + "_")
                and (row == 1 or k.endswith("_20_0_0_1024"))
            ]
            if len(rows) != 7:
                raise ValueError("All seven replication heads are required")
            values = np.array([[s[m][metric] for m in methods] for s in rows])
            ax = axes[row, column]
            ax.plot(np.arange(3), values.T, color="#a3b5ba", lw=0.9, marker="o", ms=3)
            ax.plot(
                np.arange(3),
                values.mean(axis=0),
                color="#00515f",
                lw=2.5,
                marker="o",
                ms=5,
            )
            ax.set_xticks(
                range(3), ["Fixed\nfsaverage", "Free\nfsaverage", "Three\ndonors"]
            )
            ax.set_xlim(-0.15, 2.15)
            ax.set_ylim(bottom=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("Single dipole\nmean error (mm)")
    axes[1, 0].set_ylabel("Two patches\nmean error (mm)")
    figure.suptitle("Localization with covariance transport on seven unseen anatomies")
    figure.tight_layout()
    figure.savefig(ROOT / "paired_errors.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
