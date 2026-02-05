"""SPM-style glass brain visualization of MNE SourceEstimate objects.

Projects source activations onto three orthogonal planes (sagittal, coronal,
axial) as scatter plots, giving a "see through the brain" view.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.spatial import ConvexHull


def _get_source_positions(src):
    """Extract 3D positions for all source-space vertices.

    Parameters
    ----------
    src : mne.SourceSpaces
        The source space (forward['src']).

    Returns
    -------
    pos : ndarray, shape (n_sources, 3)
        All source positions in metres.
    """
    positions = []
    for hemi in src:
        positions.append(hemi["rr"][hemi["vertno"]])
    return np.concatenate(positions, axis=0)


def plot_glass_brain(
    stc,
    src,
    time_idx=None,
    threshold=0.2,
    cmap="magma",
    alpha=0.8,
    marker_size=15,
    brain_alpha=0.1,
    brain_color="0.7",
    figsize=(12, 4),
    title=None,
    ax=None,
):
    """Plot a glass-brain projection of a SourceEstimate.

    Shows a faint outline of all source positions (the brain shape) with
    supra-threshold activations overlaid in colour.

    Parameters
    ----------
    stc : mne.SourceEstimate
        Source estimate returned by a solver.
    src : mne.SourceSpaces
        Source space, typically ``forward['src']``.
    time_idx : int | None
        Time sample index to plot. If None, plots the time point with
        maximum activation. If the data is a single time point, uses that.
    threshold : float
        Fraction of the peak value below which sources are hidden.
        0.0 shows all sources, 0.5 hides the bottom 50 %.
    cmap : str
        Matplotlib colormap name.
    alpha : float
        Marker transparency for active sources.
    marker_size : float
        Scatter marker size for active sources.
    brain_alpha : float
        Transparency for the brain outline.
    brain_color : str
        Color for the brain outline.
    figsize : tuple
        Figure size (width, height) in inches.
    title : str | None
        Figure title.
    ax : array of Axes | None
        Three matplotlib Axes to draw into. If None, a new figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    """
    all_pos = _get_source_positions(src)  # all source space vertices
    all_pos_mm = all_pos * 1e3

    # Select time point
    data = stc.data
    if data.ndim == 1:
        values = data
    else:
        if time_idx is None:
            time_idx = np.argmax(np.max(np.abs(data), axis=0))
        values = data[:, time_idx]

    values = np.abs(values)

    # Active source positions (same ordering as stc.data rows)
    active_pos = []
    for i, hemi in enumerate(src):
        active_pos.append(hemi["rr"][stc.vertices[i]])
    active_pos = np.concatenate(active_pos, axis=0) * 1e3

    # Apply threshold
    peak = values.max()
    if peak > 0:
        mask = values >= threshold * peak
    else:
        mask = np.ones(len(values), dtype=bool)

    active_pos = active_pos[mask]
    values = values[mask]

    # Sort for MIP effect (higher values on top)
    idx = np.argsort(values)
    active_pos = active_pos[idx]
    values = values[idx]

    # Normalise for colormap
    norm = Normalize(vmin=0, vmax=peak if peak > 0 else 1)

    # Projection planes: (sagittal=YZ, coronal=XZ, axial=XY)
    planes = [
        ("Sagittal", 1, 2, "Y (mm)", "Z (mm)"),
        ("Coronal", 0, 2, "X (mm)", "Z (mm)"),
        ("Axial", 0, 1, "X (mm)", "Y (mm)"),
    ]

    if ax is None:
        fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    else:
        axes = np.asarray(ax).ravel()
        fig = axes[0].figure

    for ax_i, (label, xi, yi, _xlabel, _ylabel) in zip(axes, planes):
        # 1. Draw brain outline using ConvexHull of projected points
        points_2d = all_pos_mm[:, [xi, yi]]
        try:
            hull = ConvexHull(points_2d)
            for simplex in hull.simplices:
                ax_i.plot(
                    points_2d[simplex, 0],
                    points_2d[simplex, 1],
                    color=brain_color,
                    alpha=0.3,
                    linewidth=1.5,
                )
        except Exception:
            # Fallback to scatter if ConvexHull fails
            ax_i.scatter(
                points_2d[:, 0],
                points_2d[:, 1],
                c=brain_color,
                s=1,
                alpha=0.05,
                edgecolors="none",
            )

        # 2. Draw faint cloud of all sources for depth
        ax_i.scatter(
            points_2d[:, 0],
            points_2d[:, 1],
            c=brain_color,
            s=1,
            alpha=brain_alpha,
            edgecolors="none",
        )

        # 3. Overlay active sources
        sc = ax_i.scatter(
            active_pos[:, xi],
            active_pos[:, yi],
            c=values,
            cmap=cmap,
            norm=norm,
            s=marker_size,
            alpha=alpha,
            edgecolors="none",
            zorder=10,
        )

        # 4. Styling
        ax_i.set_aspect("equal")
        ax_i.axis("off")

        # Add anatomical labels
        xlim = ax_i.get_xlim()
        ylim = ax_i.get_ylim()
        pad_x = (xlim[1] - xlim[0]) * 0.05
        pad_y = (ylim[1] - ylim[0]) * 0.05

        if label == "Sagittal":  # Y, Z
            ax_i.text(xlim[1] + pad_x, (ylim[0] + ylim[1]) / 2, "A", va="center")
            ax_i.text(xlim[0] - pad_x, (ylim[0] + ylim[1]) / 2, "P", va="center")
            ax_i.text((xlim[0] + xlim[1]) / 2, ylim[1] + pad_y, "S", ha="center")
        elif label == "Coronal":  # X, Z
            ax_i.text(xlim[1] + pad_x, (ylim[0] + ylim[1]) / 2, "R", va="center")
            ax_i.text(xlim[0] - pad_x, (ylim[0] + ylim[1]) / 2, "L", va="center")
            ax_i.text((xlim[0] + xlim[1]) / 2, ylim[1] + pad_y, "S", ha="center")
        elif label == "Axial":  # X, Y
            ax_i.text(xlim[1] + pad_x, (ylim[0] + ylim[1]) / 2, "R", va="center")
            ax_i.text(xlim[0] - pad_x, (ylim[0] + ylim[1]) / 2, "L", va="center")
            ax_i.text((xlim[0] + xlim[1]) / 2, ylim[1] + pad_y, "A", ha="center")

        ax_i.set_title(label, fontweight="bold", pad=15)

    # Single colorbar for all subplots
    cbar = fig.colorbar(
        sc, ax=axes, orientation="vertical", fraction=0.02, pad=0.04, aspect=30
    )
    cbar.set_label("Activation", fontweight="bold")
    cbar.outline.set_visible(False)

    if title is not None:
        fig.suptitle(title, fontsize=16, fontweight="bold")

    return fig
