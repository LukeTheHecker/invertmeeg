from copy import deepcopy

import mne
import numpy as np


def test_dspm_noise_cov_matches_manual_prewhitened_excluding_bads(
    sensor_info, forward_model
):
    """Noise-cov whitening path should match explicit pre-whitening.

    Regression test for a subtle but important detail: when selecting channels
    by *type* (e.g. ['eeg', 'meg']), MNE's .pick(...) keeps bad channels unless
    exclude='bads' is explicitly provided. If a solver keeps bad channels, its
    internal whitening/inverse can be noticeably worse than a "manual"
    pre-whitened pipeline that excludes them.
    """
    from invert.solvers.minimum_norm.dspm import SolverDSPM

    rng = np.random.RandomState(0)
    fwd = deepcopy(forward_model)
    G = np.asarray(fwd["sol"]["data"], dtype=float)
    n_chans, n_src = G.shape
    n_times = 20

    sources = rng.standard_normal((n_src, n_times))
    data = G @ sources

    bad_ch = fwd.ch_names[0]
    bad_idx = fwd.ch_names.index(bad_ch)
    data[bad_idx] = 50.0 * rng.standard_normal(n_times)  # obvious artifact

    evoked = mne.EvokedArray(data, sensor_info.copy(), verbose=0)
    evoked.info["bads"] = [bad_ch]
    fwd["info"]["bads"] = [bad_ch]

    cov = mne.Covariance(
        data=np.eye(n_chans, dtype=float),
        names=list(fwd.ch_names),
        bads=[],
        projs=[],
        nfree=1,
    )

    solver_cov = SolverDSPM(n_reg_params=1)
    solver_cov.make_inverse_operator(deepcopy(fwd), evoked, alpha=0.1, noise_cov=cov)
    stc_cov = solver_cov.apply_inverse_operator(evoked)

    good_chs = [ch for ch in fwd.ch_names if ch != bad_ch]
    ev_good = evoked.copy().pick(good_chs)
    fwd_good = deepcopy(fwd).pick_channels(good_chs, ordered=True)
    cov_good = mne.cov.pick_channels_cov(cov, good_chs)
    whitener, _ = mne.cov.compute_whitener(cov_good, ev_good.info, rank="info")

    ev_white = ev_good.copy()
    ev_white.data = whitener @ ev_white.data
    fwd_white = deepcopy(fwd_good)
    fwd_white["sol"]["data"] = whitener @ fwd_white["sol"]["data"]

    solver_manual = SolverDSPM(n_reg_params=1)
    solver_manual.make_inverse_operator(fwd_white, ev_white, alpha=0.1)
    stc_manual = solver_manual.apply_inverse_operator(ev_white)

    np.testing.assert_allclose(stc_cov.data, stc_manual.data, atol=1e-10, rtol=1e-10)
