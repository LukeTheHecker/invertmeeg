import json
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from .runner import SOLVER_CATEGORIES, BenchmarkResult

METRIC_LABELS = {
    "mean_localization_error": "Mean Localization Error (m)",
    "emd": "Earth Mover's Distance",
    "spatial_dispersion": "Spatial Dispersion",
    "average_precision": "Average Precision",
    "correlation": "Correlation",
}

CATEGORY_COLORS = {
    "minimum_norm": "#1f77b4",
    "loreta": "#2ca02c",
    "beamformer": "#d62728",
    "empirical_bayes": "#9467bd",
    "sparse_bayesian": "#ff7f0e",
    "music": "#8c564b",
    "matching_pursuit": "#e377c2",
    "other": "#7f7f7f",
    "baseline": "#bcbd22",
}


def _solver_to_category(solver_name: str) -> str:
    for cat, members in SOLVER_CATEGORIES.items():
        if solver_name in members:
            return cat
    return "other"


def visualize_results(
    results_path_or_data: Union[str, Path, list[BenchmarkResult]],
    metrics: Optional[list[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> list[plt.Figure]:
    if isinstance(results_path_or_data, (str, Path)):
        path = Path(results_path_or_data)
        data = json.loads(path.read_text())
        results = [BenchmarkResult(**r) for r in data["results"]]
    else:
        results = results_path_or_data

    if metrics is None:
        metrics = list(METRIC_LABELS.keys())

    datasets = sorted(set(r.dataset_name for r in results))
    solvers = sorted(set(r.solver_name for r in results))

    lookup: dict[tuple[str, str], BenchmarkResult] = {}
    for r in results:
        lookup[(r.dataset_name, r.solver_name)] = r

    # Sort solvers by category for visual grouping
    solvers.sort(key=lambda s: (_solver_to_category(s), s))

    # Build color map: solvers in the same category share a base hue with
    # slight lightness variation so individual bars are distinguishable.
    solver_colors: dict[str, str] = {}
    cat_groups: dict[str, list[str]] = {}
    for s in solvers:
        cat = _solver_to_category(s)
        cat_groups.setdefault(cat, []).append(s)
    for cat, members in cat_groups.items():
        base = np.array(
            plt.matplotlib.colors.to_rgb(CATEGORY_COLORS.get(cat, "#7f7f7f"))
        )
        n = len(members)
        for i, s in enumerate(members):
            # vary lightness: blend toward white for later members
            t = 0.15 * (i / max(n - 1, 1))  # 0 to 0.15
            solver_colors[s] = tuple((base * (1 - t) + t).tolist())  # type: ignore[assignment]

    n_solvers = len(solvers)
    n_datasets = len(datasets)
    fig_width = max(10, 1.0 * n_solvers * n_datasets + 2)
    fig_width = min(fig_width, 40)
    bar_width = min(0.8 / n_solvers, 0.15)

    figures = []
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(fig_width, 6))
        x = np.arange(n_datasets)

        for j, solver in enumerate(solvers):
            means = []
            stds = []
            for ds in datasets:
                entry = lookup.get((ds, solver))
                if entry and metric in entry.metrics:
                    means.append(entry.metrics[metric].mean)
                    stds.append(entry.metrics[metric].std)
                else:
                    means.append(0.0)
                    stds.append(0.0)
            offset = (j - n_solvers / 2 + 0.5) * bar_width
            ax.bar(
                x + offset,
                means,
                bar_width,
                yerr=stds,
                capsize=2,
                color=solver_colors[solver],
                edgecolor="white",
                linewidth=0.3,
                label=solver,
            )

        ax.set_xlabel("Dataset")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=15, ha="right")

        # Build legend grouped by category, placed below the plot
        handles, labels = ax.get_legend_handles_labels()
        label_to_handle = dict(zip(labels, handles))

        legend_handles = []
        legend_labels = []
        for cat, members in cat_groups.items():
            # Category header as invisible patch
            legend_handles.append(
                plt.matplotlib.patches.Patch(
                    facecolor="none",
                    edgecolor="none",
                )
            )
            legend_labels.append(f"$\\bf{{{cat.replace('_', ' ')}}}$")
            for s in members:
                if s in label_to_handle:
                    legend_handles.append(label_to_handle[s])
                    legend_labels.append(s)

        ncol = max(1, n_solvers // 8)
        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=ncol,
            fontsize="small",
            frameon=False,
            columnspacing=1.0,
            handletextpad=0.4,
        )
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.3)
        figures.append(fig)

        if save_path:
            sp = Path(save_path)
            sp.mkdir(parents=True, exist_ok=True)
            fig.savefig(sp / f"{metric}.png", dpi=150, bbox_inches="tight")

    return figures
