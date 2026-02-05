"""Cortical surface visualization of MNE SourceEstimate objects using PyVista.

Renders source activations on the FreeSurfer cortical mesh for
publication-ready 3D figures.
"""

from __future__ import annotations

import numpy as np


def _build_mesh(stc, src, time_idx=0):
    """Build a PyVista mesh from source space with stc data mapped on.

    Parameters
    ----------
    stc : mne.SourceEstimate
        Source estimate.
    src : mne.SourceSpaces
        The source space (forward['src']).
    time_idx : int
        Which time index to read from stc.data.

    Returns
    -------
    meshes : list of pyvista.PolyData
        One mesh per hemisphere with 'activation' scalars set.
    """
    import pyvista as pv

    meshes = []
    offset = 0  # running offset into stc.data rows
    for i, hemi in enumerate(src):
        rr = hemi["rr"] * 1e3  # metres -> mm
        tris = hemi["use_tris"]
        n_faces = len(tris)
        faces = np.column_stack([np.full(n_faces, 3), tris]).ravel()

        mesh = pv.PolyData(rr[hemi["vertno"]], faces)

        # Map stc data onto mesh vertices
        n_stc_verts = len(stc.vertices[i])
        act = np.zeros(len(hemi["vertno"]), dtype=float)

        # Build lookup: vertex number -> index in hemi["vertno"]
        vert_to_idx = {v: idx for idx, v in enumerate(hemi["vertno"])}

        for j, sv in enumerate(stc.vertices[i]):
            if sv in vert_to_idx:
                val = stc.data[offset + j]
                if val.ndim >= 1:
                    val = val[time_idx]
                act[vert_to_idx[sv]] = val

        offset += n_stc_verts
        mesh["activation"] = np.abs(act)
        meshes.append(mesh)

    return meshes


def plot_surface(
    stc,
    src,
    time_idx=None,
    threshold=0.2,
    cmap="magma",
    background="white",
    views=None,
    figsize=(1200, 400),
    title=None,
    screenshot_path=None,
    show=True,
):
    """Plot source activations on the cortical surface mesh.

    Parameters
    ----------
    stc : mne.SourceEstimate
        Source estimate returned by a solver.
    src : mne.SourceSpaces
        Source space, typically ``forward['src']``.
    time_idx : int | None
        Time sample index to plot. If None, uses the time of peak activation.
    threshold : float
        Fraction of peak below which activations are hidden (transparent).
    cmap : str
        Colormap for activations.
    background : str
        Background color of the rendering window.
    views : list of str | None
        Camera views to show. Defaults to ``['lateral', 'medial', 'dorsal']``.
        Options: 'lateral', 'medial', 'dorsal', 'ventral', 'anterior',
        'posterior'.
    figsize : tuple
        Window size (width, height) in pixels.
    title : str | None
        Window title.
    screenshot_path : str | None
        If provided, saves a screenshot to this path and returns the image
        array instead of showing the interactive window.
    show : bool
        Whether to display the interactive window (ignored if
        screenshot_path is set).

    Returns
    -------
    plotter : pyvista.Plotter
        The plotter instance.
    """
    import pyvista as pv

    if views is None:
        views = ["lateral", "medial", "dorsal"]

    # Select time point
    data = stc.data
    if data.ndim == 2 and data.shape[1] > 1:
        if time_idx is None:
            time_idx = np.argmax(np.max(np.abs(data), axis=0))
    else:
        time_idx = 0

    meshes = _build_mesh(stc, src, time_idx=time_idx)

    # Camera presets
    camera_positions = {
        "lateral": [(400, 0, 0), (0, 0, 0), (0, 0, 1)],
        "medial": [(-400, 0, 0), (0, 0, 0), (0, 0, 1)],
        "dorsal": [(0, 0, 400), (0, 0, 0), (0, 1, 0)],
        "ventral": [(0, 0, -400), (0, 0, 0), (0, 1, 0)],
        "anterior": [(0, 400, 0), (0, 0, 0), (0, 0, 1)],
        "posterior": [(0, -400, 0), (0, 0, 0), (0, 0, 1)],
    }

    n_views = len(views)
    shape = (1, n_views)

    off_screen = screenshot_path is not None
    plotter = pv.Plotter(
        shape=shape,
        window_size=figsize,
        off_screen=off_screen,
    )
    plotter.set_background(background)

    # Determine global clim across hemispheres
    all_act = np.concatenate([m["activation"] for m in meshes])
    peak = all_act.max() if all_act.size > 0 else 0
    clim = [threshold * peak, peak] if peak > 0 else [0, 1]

    for vi, view_name in enumerate(views):
        plotter.subplot(0, vi)
        for mesh in meshes:
            # Add the base brain (grey)
            plotter.add_mesh(
                mesh,
                color="lightgrey",
                opacity=1.0,
                smooth_shading=True,
            )
            # Add the activation overlay
            # We use an opacity map: 0 below threshold, 1 above
            # This is a bit complex in PyVista, simpler to just use clim
            # and a colormap that starts with transparency if supported,
            # but standard way is to add mesh twice or use scalars.

            if peak > 0:
                plotter.add_mesh(
                    mesh,
                    scalars="activation",
                    cmap=cmap,
                    clim=clim,
                    show_scalar_bar=False,
                    smooth_shading=True,
                    opacity="linear",  # Fades out low values
                )

        # Add a final scalar bar to the last subplot
        if vi == n_views - 1 and peak > 0:
            plotter.add_scalar_bar(
                title="Activation",
                label_font_size=12,
                title_font_size=14,
                n_labels=5,
            )

        cam = camera_positions.get(view_name)
        if cam is not None:
            plotter.camera_position = cam

        plotter.add_text(
            view_name.capitalize(),
            font_size=12,
            position="upper_left",
            color="black" if background == "white" else "white",
        )

    if title:
        plotter.add_text(title, font_size=16, position="upper_edge")

    if screenshot_path is not None:
        plotter.screenshot(screenshot_path)
        plotter.close()
    elif show:
        plotter.show()

    return plotter
