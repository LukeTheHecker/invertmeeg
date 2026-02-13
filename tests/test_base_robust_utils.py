import mne
import numpy as np
import pytest

from invert.solvers.base import BaseSolver, InverseOperator
from invert.solvers.minimum_norm.dspm_mne import SolverDSPMMNE


def _as_mne_cov(cov: np.ndarray, ch_names: list[str], nfree: int = 1) -> mne.Covariance:
    return mne.Covariance(
        data=np.asarray(cov, dtype=float),
        names=list(ch_names),
        bads=[],
        projs=[],
        nfree=int(max(nfree, 1)),
    )


def test_coerce_diag_source_prior_accepts_scalar_vector_and_diag():
    n_sources = 3

    scalar = BaseSolver.coerce_diag_source_prior(2.5, n_sources)
    np.testing.assert_allclose(scalar, [2.5, 2.5, 2.5])

    vector = BaseSolver.coerce_diag_source_prior(np.array([1.0, 2.0, 3.0]), n_sources)
    np.testing.assert_allclose(vector, [1.0, 2.0, 3.0])

    matrix = BaseSolver.coerce_diag_source_prior(np.diag([4.0, 5.0, 6.0]), n_sources)
    np.testing.assert_allclose(matrix, [4.0, 5.0, 6.0])


def test_coerce_diag_source_prior_rejects_full_matrix():
    cov = np.eye(3)
    cov[0, 1] = 0.1
    cov[1, 0] = 0.1
    with pytest.raises(ValueError, match="non-diagonal"):
        BaseSolver.coerce_diag_source_prior(cov, n_sources=3)


def test_compute_sensor_projector_identity_without_projs():
    solver = BaseSolver()
    info_like = {
        "projs": [],
        "bads": [],
        "ch_names": ["EEG001", "EEG002", "EEG003"],
    }
    projector = solver.compute_sensor_projector(forward_or_info=info_like, n_chans=3)
    np.testing.assert_allclose(projector, np.eye(3))


def test_compute_sensor_whitener_whitens_retained_subspace():
    noise_cov = np.diag([4.0, 1.0, 0.0])
    whitener = BaseSolver.compute_sensor_whitener(noise_cov, rank_tol=1e-12, eps=1e-15)
    assert whitener.shape == (2, 3)
    whitened_cov = whitener @ noise_cov @ whitener.T
    np.testing.assert_allclose(whitened_cov, np.eye(2), atol=1e-12)


def test_compute_depth_prior_whitened_depth_zero_returns_inverse_sensitivity():
    G_white = np.array([[1.0, 0.5, 2.0], [0.0, 0.5, 0.0]])
    prior = BaseSolver.compute_depth_prior_whitened(
        G_white, depth=0.0, depth_limit=10.0, eps=1e-15
    )
    expected = 1.0 / np.sum(G_white * G_white, axis=0)
    np.testing.assert_allclose(prior, expected)


def test_compute_depth_prior_whitened_clipping_with_limit_one():
    G_white = np.array([[1e-3, 1.0, 2.0]])
    prior = BaseSolver.compute_depth_prior_whitened(
        G_white, depth=1.0, depth_limit=1.0, eps=1e-15
    )
    np.testing.assert_allclose(prior, np.full(3, prior[0]))


def test_trace_normalize_operator_sets_target_trace():
    A = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, -0.5]])
    A_norm, scale = BaseSolver.trace_normalize_operator(A, target_rank=2, eps=1e-15)
    assert scale > 0
    np.testing.assert_allclose(np.sum(A_norm * A_norm), 2.0, atol=1e-12)


def test_solve_tikhonov_svd_matches_closed_form():
    rng = np.random.RandomState(0)
    A = rng.randn(4, 6)
    lambda2 = 0.2
    left_scale = np.abs(rng.randn(6)) + 0.1

    K = BaseSolver.solve_tikhonov_svd(A, lambda2, left_scale=left_scale, eps=1e-15)
    K_expected = (
        np.diag(left_scale)
        @ A.T
        @ np.linalg.inv(A @ A.T + lambda2 * np.eye(A.shape[0]))
    )
    np.testing.assert_allclose(K, K_expected, atol=1e-10)


def test_noise_normalize_rows_returns_finite_unit_norm_rows():
    K_white = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
    K_norm, noise_std = BaseSolver.noise_normalize_rows(K_white, eps=1e-12)
    np.testing.assert_allclose(noise_std[0], 5.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(K_norm[0]), 1.0, atol=1e-12)
    np.testing.assert_allclose(K_norm[1], 0.0, atol=1e-12)
    assert np.all(np.isfinite(K_norm))


def test_dspm_mne_uses_base_tikhonov_solver(monkeypatch, forward_model):
    calls = {"n": 0}
    original = BaseSolver.solve_tikhonov_svd

    def _wrapped(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(SolverDSPMMNE, "solve_tikhonov_svd", staticmethod(_wrapped))

    solver = SolverDSPMMNE(n_reg_params=3)
    solver.r_values = np.asarray([1e-3, 1e-2, 1e-1], dtype=float)
    n_chans = int(forward_model["sol"]["data"].shape[0])
    noise_cov = _as_mne_cov(np.eye(n_chans, dtype=float), forward_model.ch_names)
    solver.make_inverse_operator(
        forward_model,
        alpha="auto",
        noise_cov=noise_cov,
    )

    assert calls["n"] == len(solver.alphas)
    kernel = solver.inverse_operators[0].data[0]
    assert kernel.shape == (solver.leadfield.shape[1], solver.leadfield.shape[0])
    assert np.all(np.isfinite(kernel))


def test_align_channel_sets_preserves_forward_order():
    solver = BaseSolver()
    report = solver.align_channel_sets(
        forward_ch_names=["A", "B", "C", "D"],
        data_ch_names=["C", "B", "X", "A"],
        noise_cov_ch_names=["B", "A", "C", "Y"],
        context="test",
    )
    assert report.kept_ch_names == ["A", "B", "C"]
    assert report.dropped_from_forward == ["D"]
    assert report.dropped_from_data == ["X"]
    assert report.dropped_from_noise_cov == ["Y"]


def test_prepare_whitened_forward_reorders_named_covariance(forward_model):
    solver_a = BaseSolver(n_reg_params=1)
    solver_a.make_inverse_operator(forward_model, alpha=0.1)
    n_chans = int(solver_a.leadfield.shape[0])
    base_cov = np.diag(np.linspace(1.0, 2.0, n_chans))
    cov_obj_aligned = _as_mne_cov(base_cov, forward_model.ch_names)
    wf_aligned = solver_a.prepare_whitened_forward(noise_cov=cov_obj_aligned)

    solver_b = BaseSolver(n_reg_params=1)
    solver_b.make_inverse_operator(forward_model, alpha=0.1)
    perm = np.arange(n_chans)[::-1]
    perm_cov = base_cov[np.ix_(perm, perm)]
    perm_names = [forward_model.ch_names[idx] for idx in perm]
    cov_obj = _as_mne_cov(perm_cov, perm_names)
    wf_obj = solver_b.prepare_whitened_forward(noise_cov=cov_obj)

    np.testing.assert_allclose(
        wf_obj.sensor_transform, wf_aligned.sensor_transform, atol=1e-12
    )


def test_prepare_whitened_forward_logs_dropped_cov_and_forward_channels(
    forward_model, caplog
):
    solver = BaseSolver(n_reg_params=1)
    solver.make_inverse_operator(forward_model, alpha=0.1)

    keep_names = list(forward_model.ch_names[:-1])
    cov_names = keep_names + ["EOG999"]
    cov = np.eye(len(cov_names), dtype=float)
    cov_obj = mne.Covariance(
        data=cov,
        names=cov_names,
        bads=[],
        projs=[],
        nfree=1,
    )

    caplog.set_level("INFO")
    solver.prepare_whitened_forward(noise_cov=cov_obj)

    assert "prepare_whitened_forward channel alignment" in caplog.text
    assert "dropped_from_forward=1" in caplog.text
    assert "dropped_from_noise_cov=1" in caplog.text
    assert solver.forward.ch_names == keep_names


def test_unpack_data_obj_logs_dropped_extra_data_channels(
    forward_model, simulated_evoked, caplog
):
    solver = BaseSolver(n_reg_params=1)
    solver.make_inverse_operator(forward_model, alpha=0.1)

    data = np.vstack(
        [simulated_evoked.data, np.zeros((1, simulated_evoked.data.shape[1]))]
    )
    ch_names = list(simulated_evoked.ch_names) + ["EOG999"]
    ch_types = ["eeg"] * len(simulated_evoked.ch_names) + ["eog"]
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=float(simulated_evoked.info["sfreq"]),
        ch_types=ch_types,
    )
    evoked = mne.EvokedArray(data, info, tmin=float(simulated_evoked.tmin), verbose=0)

    caplog.set_level("INFO")
    unpacked = solver.unpack_data_obj(evoked)

    assert "unpack_data_obj type filter" in caplog.text
    assert "dropped=1" in caplog.text
    assert unpacked.shape[0] == len(forward_model.ch_names)


def test_apply_raises_clear_error_when_data_channels_missing(
    forward_model, simulated_evoked
):
    solver = SolverDSPMMNE(n_reg_params=1)
    n_chans = int(forward_model["sol"]["data"].shape[0])
    noise_cov = _as_mne_cov(np.eye(n_chans, dtype=float), forward_model.ch_names)
    solver.make_inverse_operator(
        forward_model,
        simulated_evoked,
        alpha=0.1,
        noise_cov=noise_cov,
    )

    evoked_missing = simulated_evoked.copy().pick(simulated_evoked.ch_names[:-1])
    with pytest.raises(
        ValueError,
        match="Recompute the inverse operator on the current data channel set",
    ):
        solver.apply_inverse_operator(evoked_missing)


def test_apply_raises_clear_error_when_data_channels_include_extra_meeg(
    forward_model, simulated_evoked
):
    solver = SolverDSPMMNE(n_reg_params=1)
    n_chans = int(forward_model["sol"]["data"].shape[0])
    noise_cov = _as_mne_cov(np.eye(n_chans, dtype=float), forward_model.ch_names)
    solver.make_inverse_operator(
        forward_model,
        simulated_evoked,
        alpha=0.1,
        noise_cov=noise_cov,
    )

    data = np.vstack(
        [simulated_evoked.data, np.zeros((1, simulated_evoked.data.shape[1]))]
    )
    ch_names = list(simulated_evoked.ch_names) + ["EEG999"]
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=float(simulated_evoked.info["sfreq"]),
        ch_types=["eeg"] * len(ch_names),
    )
    evoked_extra = mne.EvokedArray(
        data, info, tmin=float(simulated_evoked.tmin), verbose=0
    )

    with pytest.raises(ValueError, match="Extra in data: 1"):
        solver.apply_inverse_operator(evoked_extra)


def test_log_channel_alignment_lists_all_dropped_channels(caplog):
    solver = BaseSolver()
    forward_ch_names = [f"EEG{idx:03d}" for idx in range(12)]
    report = solver.align_channel_sets(
        forward_ch_names=forward_ch_names,
        data_ch_names=forward_ch_names[:2],
        context="test_log_channel_alignment",
    )

    caplog.set_level("INFO")
    solver.log_channel_alignment(report)

    assert "test_log_channel_alignment channel alignment" in caplog.text
    for ch_name in forward_ch_names[2:]:
        assert ch_name in caplog.text
    assert "..." not in caplog.text


def test_validate_compatibility_works_without_precomputed_operator_matrix():
    solver = BaseSolver()
    solver._operator_ch_names = ("EEG001", "EEG002", "EEG003")
    solver._last_input_data_ch_names = ("EEG001", "EEG002")
    solver.inverse_operators = []

    data = np.zeros((2, 5), dtype=float)
    with pytest.raises(ValueError, match="operator expects 3, data has 2"):
        solver.validate_operator_data_compatibility(data)


@pytest.mark.parametrize(
    ("inverse_mats", "expected_idx", "edge_label"),
    [
        ([np.zeros((2, 2)), 0.5 * np.eye(2)], 0, "lowest"),
        ([0.5 * np.eye(2), np.zeros((2, 2))], 1, "highest"),
    ],
)
def test_regularise_product_warns_when_alpha_choice_is_on_edge(
    inverse_mats, expected_idx, edge_label, caplog
):
    solver = BaseSolver(n_reg_params=2)
    solver.leadfield = np.eye(2)
    solver.alphas = [1e-6, 1e3]
    solver.inverse_operators = [
        InverseOperator(mat, "test_product") for mat in inverse_mats
    ]

    caplog.set_level("WARNING")
    _, optimum_idx = solver.regularise_product(np.eye(2))

    assert optimum_idx == expected_idx
    assert "Product selected the" in caplog.text
    assert f"{edge_label} regularization parameter" in caplog.text


def test_regularise_gcv_does_not_warn_when_only_one_operator_is_tested(caplog):
    solver = BaseSolver(n_reg_params=4)
    solver.leadfield = np.eye(2)
    solver.alphas = [1e-9, 1e-6, 1e-3, 1e0]
    solver.inverse_operators = [InverseOperator(np.eye(2), "test_single")]

    caplog.set_level("WARNING")
    _, optimum_idx = solver.regularise_gcv(np.eye(2), gamma=1.0, method="gcv")

    assert optimum_idx == 0
    assert "selected the" not in caplog.text
