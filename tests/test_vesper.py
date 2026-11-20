import mne
import numpy as np

from invert import Solver


def test_vesper_whitens_and_recovers_a_noiseless_focal_source(
    forward_model, sensor_info, leadfield
):
    rng = np.random.default_rng(2026)
    vertex = leadfield.shape[1] // 2
    waveform = rng.normal(size=80)
    evoked = mne.EvokedArray(
        leadfield[:, vertex, None] * waveform, sensor_info, verbose=False
    )
    evoked.info["bads"] = [evoked.ch_names[1]]
    channels = evoked.ch_names[::-1]
    covariance = mne.Covariance(
        np.diag(np.linspace(1, 2, len(channels))), channels, [], [], nfree=200
    )

    original_channels = list(forward_model.ch_names)
    solver = Solver("vesper")
    estimate = solver.make_inverse_operator(
        forward_model, evoked, noise_cov=covariance
    ).apply_inverse_operator(evoked)
    assert forward_model.ch_names == original_channels
    assert np.argmax(np.linalg.norm(estimate.data, axis=1)) == vertex
    assert np.corrcoef(estimate.data[vertex], waveform)[0, 1] > 0.995


def test_vesper_pca_orientation_accepts_empty_signal(
    forward_model_free_surface, simulated_evoked_free_surface
):
    solver = Solver("vesper", orientation="pca")
    evoked = simulated_evoked_free_surface
    solver.make_inverse_operator(forward_model_free_surface, evoked)
    estimate = solver.apply_inverse_operator(evoked)
    n_vertices = int(forward_model_free_surface["nsource"])
    assert estimate.data.shape == (n_vertices, evoked.data.shape[1])
    assert np.all(np.isfinite(estimate.data))
    assert np.any(estimate.data)

    zeros = mne.EvokedArray(np.zeros_like(evoked.data), evoked.info, verbose=False)
    zero_estimate = solver.apply_inverse_operator(zeros)
    assert zero_estimate.data.shape == estimate.data.shape
    assert not np.any(zero_estimate.data)
