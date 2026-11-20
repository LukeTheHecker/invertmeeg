import mne
import numpy as np
import pytest

from invert import Solver
from invert.solvers.bayesian.moraine import _PatchEvidence


def test_profile_evidence_matches_explicit_projection():
    rng = np.random.default_rng(703)
    fields = rng.normal(size=(25, 7))
    fields[3] = fields[0]
    fields[4] = 0
    transforms = rng.normal(size=(2, 7, 7))
    data = rng.normal(size=(2, 7, 12))
    cov = data @ data.transpose(0, 2, 1) / 12
    sizes = np.array([7, 17])
    model = _PatchEvidence(fields, transforms, cov, sizes)
    scores, _, _ = model.scan([0, 1], len(fields))
    expected = np.zeros(len(fields))
    eligible = np.ones(len(fields), dtype=bool)
    for b, transform in enumerate(transforms):
        whitened = fields @ transform.T
        q = np.linalg.qr(whitened[:2].T)[0]
        projected = whitened @ (np.eye(7) - q @ q.T)
        norm2 = np.sum(projected**2, axis=1)
        eligible &= norm2 > 0.015 * np.sum(whitened**2, axis=1)
        power = np.maximum(
            np.sum((projected @ model.cov[b]) * projected, axis=1)
            / np.maximum(norm2, 1e-20),
            1,
        )
        expected += sizes[b] * (power - 1 - np.log(power))
    np.testing.assert_allclose(scores[eligible], expected[eligible], rtol=1e-10)
    assert np.all(np.isneginf(scores[~eligible]))


def test_noiseless_focal_source_with_covariance_channel_alignment(
    forward_model, sensor_info, leadfield
):
    source = leadfield.shape[1] // 2
    waveform = np.random.default_rng(285).normal(size=81)
    signal = leadfield[:, source, None] * waveform
    evoked = mne.EvokedArray(signal, sensor_info, verbose=False)
    evoked.info["bads"] = [evoked.ch_names[2]]
    names = evoked.ch_names[::-1]
    cov = mne.Covariance(np.diag(np.linspace(1, 3, len(names))), names, [], [], 100)
    original_channels = list(forward_model.ch_names)
    model = Solver("moraine").make_inverse_operator(
        forward_model, evoked, noise_cov=cov
    )
    estimate = model.apply_inverse_operator(evoked)
    assert forward_model.ch_names == original_channels
    assert np.argmax(np.linalg.norm(estimate.data, axis=1)) == source
    np.testing.assert_allclose(estimate.data[source], waveform, atol=1e-5)
    reordered = evoked.copy().reorder_channels(names)
    np.testing.assert_allclose(
        model.apply_inverse_operator(reordered).data, estimate.data
    )


@pytest.mark.parametrize("n_times", [1, 9, 80])
def test_short_epochs_and_dc_control(forward_model, sensor_info, leadfield, n_times):
    data = np.tile(leadfield[:, 4, None], (1, n_times))
    evoked = mne.EvokedArray(data, sensor_info, verbose=False)
    model = Solver("moraine").make_inverse_operator(forward_model, evoked)
    result = model.apply_inverse_operator(evoked).data
    assert np.isfinite(result).all()
    assert np.argmax(np.linalg.norm(result, axis=1)) == 4
    ac = Solver("moraine", include_dc=False).make_inverse_operator(
        forward_model, evoked
    )
    np.testing.assert_allclose(ac.apply_inverse_operator(evoked).data, 0, atol=1e-13)


def test_pca_orientation_and_empty_recording(
    forward_model_free_surface, simulated_evoked_free_surface
):
    evoked = simulated_evoked_free_surface
    model = Solver("moraine", orientation="pca")
    model.make_inverse_operator(forward_model_free_surface, evoked)
    data = model.apply_inverse_operator(evoked).data
    assert data.shape == (forward_model_free_surface["nsource"], evoked.data.shape[1])
    assert np.isfinite(data).all() and np.any(data)
    zeros = mne.EvokedArray(np.zeros_like(evoked.data), evoked.info, verbose=False)
    assert not np.any(model.apply_inverse_operator(zeros).data)
