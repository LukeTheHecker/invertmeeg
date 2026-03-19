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
    cmap="Reds",
    alpha=0.8,
    marker_size=15,
    brain_alpha=0.1,
    brain_color="0.7",
    figsize=(12, 4),
    title=None,
    ax=None,
    true_sources=None,
    true_source_marker="+",
    true_source_color="green",
    true_source_size=None,
    true_source_alpha=1.0,
    show_colorbar=True,
    show_view_titles=True,
    show_direction_labels=True,
    direction_label_fontsize=10,
    direction_label_pad=0.05,
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
    true_sources : array-like | None
        Optional ground-truth source indices in full source-space ordering
        (matching ``_get_source_positions(src)``), or a boolean mask of length
        ``n_sources``.
    true_source_marker : str
        Matplotlib marker used to draw true sources. Default ``"+"``.
    true_source_color : str
        Marker color for true sources.
    true_source_size : float | None
        Marker size for true sources. If None, uses ``marker_size * 1.8``.
    true_source_alpha : float
        Marker alpha for true sources.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    """
    all_pos = _get_source_positions(src)  # all source space vertices
    all_pos_mm = all_pos * 1e3
    n_all_sources = int(all_pos_mm.shape[0])

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

    # Optional overlay of known true source locations (in full-source indexing).
    true_pos = None
    if true_sources is not None:
        true_sources_arr = np.asarray(true_sources)
        if true_sources_arr.dtype == bool:
            if true_sources_arr.ndim != 1 or true_sources_arr.shape[0] != n_all_sources:
                raise ValueError(
                    "Boolean true_sources must be 1D with length equal to "
                    f"number of sources ({n_all_sources}), got {true_sources_arr.shape}."
                )
            true_indices = np.flatnonzero(true_sources_arr)
        else:
            true_indices = np.asarray(true_sources_arr, dtype=int).ravel()
            if true_indices.size:
                if np.any(true_indices < 0) or np.any(true_indices >= n_all_sources):
                    raise ValueError(
                        "true_sources contains out-of-range indices for source "
                        f"space size {n_all_sources}."
                    )
                true_indices = np.unique(true_indices)
        if true_indices.size:
            true_pos = all_pos_mm[true_indices]

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

    for ax_i, (label, xi, yi, _xlabel, _ylabel) in zip(axes, planes, strict=False):
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
        if true_pos is not None:
            ts_size = float(marker_size) * 1.8 if true_source_size is None else true_source_size
            ax_i.scatter(
                true_pos[:, xi],
                true_pos[:, yi],
                c=true_source_color,
                marker=true_source_marker,
                s=ts_size,
                alpha=true_source_alpha,
                linewidths=1.5,
                zorder=20,
            )

        # 4. Styling
        ax_i.set_aspect("equal")
        ax_i.axis("off")

        if show_direction_labels:
            # Anatomical direction labels placed in axes-relative coords
            # so they stay inside each panel and don't bleed into neighbours.
            # Top labels (S/A) are omitted — the view titles already convey
            # the orientation.
            fs = direction_label_fontsize
            kw = dict(fontsize=fs, fontweight="bold", color="0.35",
                      transform=ax_i.transAxes, clip_on=False)

            if label == "Sagittal":  # Y, Z — left=P, right=A
                ax_i.text(-0.03, 0.50, "P", va="center", ha="right", **kw)
                ax_i.text(1.03, 0.50, "A", va="center", ha="left", **kw)
            elif label == "Coronal":  # X, Z — left=L, right=R
                ax_i.text(-0.02, 0.50, "L", va="center", ha="right", **kw)
                ax_i.text(1.02, 0.50, "R", va="center", ha="left", **kw)
            elif label == "Axial":  # X, Y — left=L, right=R
                ax_i.text(-0.02, 0.50, "L", va="center", ha="right", **kw)
                ax_i.text(1.02, 0.50, "R", va="center", ha="left", **kw)

        if show_view_titles:
            ax_i.set_title(label, fontweight="bold", pad=8)

    if show_colorbar:
        # Single colorbar for all subplots
        cbar = fig.colorbar(
            sc, ax=axes, orientation="vertical", fraction=0.02, pad=0.04, aspect=30
        )
        cbar.set_label("Activation", fontweight="bold")
        cbar.outline.set_visible(False)

    if title is not None:
        fig.suptitle(title, fontsize=16, fontweight="bold")

    return fig
