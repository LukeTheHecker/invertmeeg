"""3D-like glass brain visualization using Matplotlib with depth cues.

Simulates a 3D view of the brain by projecting sources to 2D and scaling
markers based on their depth (distance to viewer).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.spatial import ConvexHull


def _get_source_positions(src):
    """Extract 3D positions for all source-space vertices."""
    positions = []
    for hemi in src:
        positions.append(hemi["rr"][hemi["vertno"]])
    return np.concatenate(positions, axis=0)


def _rotate_points(points, azimuth, elevation):
    """Rotate points by azimuth (around Z) and elevation (around X)."""
    # Convert to radians
    az = np.radians(azimuth)
    el = np.radians(elevation)

    # Rotation around Z axis (Azimuth)
    Rz = np.array([
        [np.cos(az), -np.sin(az), 0],
        [np.sin(az), np.cos(az), 0],
        [0, 0, 1]
    ])

    # Rotation around X axis (Elevation)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(el), -np.sin(el)],
        [0, np.sin(el), np.cos(el)]
    ])

    # Apply rotations: R = Rx @ Rz
    # Points are (N, 3). We want (R @ P.T).T = P @ R.T
    R = Rx @ Rz
    return points @ R.T


def plot_3d_glass_brain(
    stc,
    src,
    time_idx=None,
    threshold=0.2,
    cmap="magma",
    alpha=0.9,
    marker_size=30,
    brain_alpha=0.05,
    brain_color="0.7",
    figsize=(8, 8),
    title=None,
    ax=None,
    view=(90, 10),  # Azimuth, Elevation
    depth_scale=True,
):
    """Plot a 3D-like glass brain visualization.

    Parameters
    ----------
    stc : mne.SourceEstimate
        Source estimate.
    src : mne.SourceSpaces
        Source space.
    time_idx : int | None
        Time index to plot.
    threshold : float
        Threshold fraction (0.0 to 1.0) of peak activation.
    cmap : str
        Colormap.
    alpha : float
        Opacity of active sources.
    marker_size : float
        Base marker size.
    brain_alpha : float
        Opacity of the inactive source cloud.
    brain_color : str
        Color of the brain outline/cloud.
    figsize : tuple
        Figure size.
    title : str | None
        Title.
    ax : matplotlib.axes.Axes | None
        Axes to plot on.
    view : tuple
        (azimuth, elevation) in degrees.
    depth_scale : bool
        Whether to scale marker size by depth.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    """
    all_pos = _get_source_positions(src)
    all_pos_mm = all_pos * 1e3

    # Rotate all points
    az, el = view
    rotated_pos = _rotate_points(all_pos_mm, az, el)

    # Extract data
    data = stc.data
    if data.ndim == 1:
        values = data
    else:
        if time_idx is None:
            time_idx = np.argmax(np.max(np.abs(data), axis=0))
        values = data[:, time_idx]
    values = np.abs(values)

    # Get active positions
    active_pos_list = []
    for i, hemi in enumerate(src):
        active_pos_list.append(hemi["rr"][stc.vertices[i]])
    active_pos = np.concatenate(active_pos_list, axis=0) * 1e3

    # Rotate active positions using same rotation
    active_pos_rot = _rotate_points(active_pos, az, el)

    # Threshold
    peak = values.max()
    if peak > 0:
        mask = values >= threshold * peak
    else:
        mask = np.ones(len(values), dtype=bool)

    active_pos_rot = active_pos_rot[mask]
    values = values[mask]

    # Sort by depth (Y coordinate after rotation? Or Z?
    # In matplotlib 2D, we plot X vs Y (or Z).
    # Let's project to X-Z plane (coronal-like) or X-Y (axial-like).
    # If we rotate such that the viewer is looking down the Y axis (or Z axis).
    # Let's assume we plot X (horizontal) and Z (vertical) of the rotated points.
    # The depth is Y (going into the screen).
    # Standard: X=Right, Y=Anterior, Z=Superior.
    # After rotation, we plot X' and Z'. Y' is depth.
    # If Y' is depth, larger Y' means closer or further depending on convention.
    # Let's assume positive Y is "into screen" or "away".
    # Actually, in right-handed system:
    # If we look from +Y towards origin, then X is right, Z is up.
    # So Y is depth. Larger Y = closer to viewer if we look from +inf Y.

    # Let's define projection:
    x_proj = rotated_pos[:, 0]
    y_proj = rotated_pos[:, 2] # Plot Z on vertical axis
    depth = rotated_pos[:, 1]  # Y is depth

    # Active sources projection
    act_x = active_pos_rot[:, 0]
    act_y = active_pos_rot[:, 2]
    act_depth = active_pos_rot[:, 1]

    # Sort active sources by depth (furthest first -> smallest Y first if looking from +Y)
    # If looking from +Y, larger Y is closer. We want to plot furthest (small Y) first.
    sort_idx = np.argsort(act_depth)
    act_x = act_x[sort_idx]
    act_y = act_y[sort_idx]
    act_depth = act_depth[sort_idx]
    values = values[sort_idx]

    # Setup plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # 1. Draw brain outline (Convex Hull of projected points)
    points_2d = np.column_stack((x_proj, y_proj))
    try:
        hull = ConvexHull(points_2d)
        for simplex in hull.simplices:
            ax.plot(
                points_2d[simplex, 0],
                points_2d[simplex, 1],
                color=brain_color,
                alpha=0.5,
                linewidth=1.5,
            )
    except Exception:
        pass

    # 2. Draw faint cloud of all sources
    ax.scatter(
        x_proj,
        y_proj,
        c=brain_color,
        s=1,
        alpha=brain_alpha,
        edgecolors="none",
        zorder=1,
    )

    # 3. Draw active sources
    if len(values) > 0:
        norm = Normalize(vmin=0, vmax=peak if peak > 0 else 1)

        sizes = marker_size
        if depth_scale:
            # Scale size by depth.
            # Normalize depth to [0, 1] or similar.
            # Larger depth (closer) -> larger size.
            # act_depth range:
            d_min, d_max = depth.min(), depth.max()
            if d_max > d_min:
                d_norm = (act_depth - d_min) / (d_max - d_min) # 0 to 1
                # Scale factor: e.g. 0.5 to 2.0
                scale = 0.5 + 2.0 * d_norm
                sizes = marker_size * (scale ** 2) # Area scales with square

        ax.scatter(
            act_x,
            act_y,
            c=values,
            cmap=cmap,
            norm=norm,
            s=sizes,
            alpha=alpha,
            edgecolors="none",
            zorder=10,
        )

        # Add colorbar if this is the only axis or requested
        # (Handling colorbar is tricky with subplots, usually done outside)

    # Styling
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontweight="bold")

    return fig
