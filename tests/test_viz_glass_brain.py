import matplotlib
import numpy as np

from invert.viz.glass_brain import plot_glass_brain

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


def test_plot_glass_brain_true_sources_overlay(forward_model, simulated_stc):
    stc = simulated_stc
    src = forward_model["src"]

    fig_plain = plot_glass_brain(stc, src, threshold=0.0, marker_size=8)
    plain_counts = [len(ax.collections) for ax in fig_plain.axes[:3]]
    plt.close(fig_plain)

    n_lh = len(src[0]["vertno"])
    true_sources = [0, n_lh]
    fig_overlay = plot_glass_brain(
        stc,
        src,
        threshold=0.0,
        marker_size=8,
        true_sources=true_sources,
        true_source_marker="+",
    )
    overlay_counts = [len(ax.collections) for ax in fig_overlay.axes[:3]]
    plt.close(fig_overlay)

    np.testing.assert_array_equal(
        np.asarray(overlay_counts, dtype=int),
        np.asarray(plain_counts, dtype=int) + 1,
    )
