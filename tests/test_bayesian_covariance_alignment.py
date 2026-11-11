"""Bayesian fitting must use the channel intersection established by whitening."""

import mne
import numpy as np
import pytest

from invert.solvers.bayesian.vb_sbl import SolverVBSBL


@pytest.fixture(scope="module")
def discrete_forward_data():
    # Offline conductor/source model: no fsaverage download or cached assets.
    info = mne.create_info(
        ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "O1", "O2"], 200.0, "eeg"
    )
    info.set_montage("standard_1020")
    src = mne.setup_volume_source_space(
        pos={
            "rr": np.array(
                [[0.02, 0.01, 0.04], [-0.02, 0.02, 0.05], [0.01, -0.02, 0.04]]
            ),
            "nn": np.tile([0.0, 0.0, 1.0], (3, 1)),
        },
        sphere=(0.0, 0.0, 0.0, 0.08),
        verbose=False,
    )
    sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.0), head_radius=0.09, verbose=False)
    fwd = mne.make_forward_solution(
        info, trans=None, src=src, bem=sphere, eeg=True, meg=False, verbose=False
    )
    fwd = mne.convert_forward_solution(
        fwd, surf_ori=True, force_fixed=True, use_cps=False, verbose=False
    )
    rng = np.random.default_rng(41)
    data = fwd["sol"]["data"] @ rng.normal(size=(3, 24)) + 0.1 * rng.normal(
        size=(8, 24)
    )
    return fwd, mne.EvokedArray(data, info, verbose=False)


@pytest.mark.parametrize("reverse_covariance", [False, True])
def test_covariance_subset_matches_explicit_channel_selection(
    discrete_forward_data, reverse_covariance
):
    fwd, evoked = discrete_forward_data
    names = ["Fp1", "F3", "C3", "C4", "O2"]
    covariance_names = names[::-1] if reverse_covariance else names
    rng = np.random.default_rng(72)
    a = rng.normal(size=(len(names), len(names)))
    matrix = a @ a.T + np.eye(len(names))
    if reverse_covariance:
        matrix = matrix[::-1, ::-1]
    covariance = mne.Covariance(
        matrix, names=covariance_names, bads=[], projs=[], nfree=24
    )
    automatic = SolverVBSBL(n_reg_params=1)
    explicit = SolverVBSBL(n_reg_params=1)
    params = dict(
        alpha=0.1,
        noise_cov=covariance,
        max_iter=30,
        prune=False,
        convergence_criterion=0,
    )
    automatic.make_inverse_operator(fwd, evoked, **params)
    selected_evoked = evoked.copy().pick(names)
    explicit.make_inverse_operator(
        fwd.copy().pick_channels(names, ordered=True), selected_evoked, **params
    )
    auto_stc = automatic.apply_inverse_operator(evoked)
    explicit_stc = explicit.apply_inverse_operator(selected_evoked)
    assert np.isfinite(auto_stc.data).all()
    assert automatic.inverse_operators[0].data[0].shape == (3, len(names))
    np.testing.assert_allclose(auto_stc.data, explicit_stc.data, rtol=1e-11, atol=1e-11)
