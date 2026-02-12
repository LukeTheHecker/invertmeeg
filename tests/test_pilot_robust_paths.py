import mne
import numpy as np
import pytest

from invert.solvers.bayesian.champagne import SolverChampagne
from invert.solvers.beamformers.esmv import SolverESMV
from invert.solvers.beamformers.lcmv import SolverLCMV
from invert.solvers.minimum_norm.dspm import SolverDSPM
from invert.solvers.minimum_norm.dspm_mne import SolverDSPMMNE
from invert.solvers.minimum_norm.eloreta import SolverELORETA
from invert.solvers.minimum_norm.laura import SolverLAURA
from invert.solvers.minimum_norm.loreta import SolverLORETA
from invert.solvers.minimum_norm.mne import SolverMNE
from invert.solvers.minimum_norm.sloreta import SolverSLORETA
from invert.solvers.minimum_norm.smap import SolverSMAP
from invert.solvers.minimum_norm.wmne import SolverWMNE


def _kernel(solver):
    return solver.inverse_operators[0].data[0]


def _as_mne_cov(cov: np.ndarray, ch_names: list[str]) -> mne.Covariance:
    return mne.Covariance(
        data=np.asarray(cov, dtype=float),
        names=list(ch_names),
        bads=[],
        projs=[],
        nfree=1,
    )


def _as_bad_mne_cov(cov: np.ndarray, ch_names: list[str]) -> mne.Covariance:
    return mne.Covariance(
        data=np.asarray(cov, dtype=float),
        names=list(ch_names),
        bads=[],
        projs=[],
        nfree=1,
    )


def test_dspm_vector_prior_matches_diag_matrix(forward_model, simulated_evoked):
    n_sources = int(forward_model["sol"]["data"].shape[1])
    prior_vec = np.linspace(0.5, 1.5, n_sources)
    prior_mat = np.diag(prior_vec)
    n_chans = int(forward_model["sol"]["data"].shape[0])
    noise_cov = _as_mne_cov(np.eye(n_chans, dtype=float), forward_model.ch_names)

    solver_vec = SolverDSPM(n_reg_params=1)
    solver_vec.make_inverse_operator(
        forward_model,
        simulated_evoked,
        alpha=0.1,
        noise_cov=noise_cov,
        source_cov=prior_vec,
    )

    solver_mat = SolverDSPM(n_reg_params=1)
    solver_mat.make_inverse_operator(
        forward_model,
        simulated_evoked,
        alpha=0.1,
        noise_cov=noise_cov,
        source_cov=prior_mat,
    )

    np.testing.assert_allclose(_kernel(solver_vec), _kernel(solver_mat), atol=1e-10)


def test_dspm_gcv_selects_alpha_on_mne_kernel(forward_model, simulated_evoked):
    """dSPM regularization selection should run on the underlying MNE kernel.

    The dSPM kernel is a noise-normalized statistical map and can yield an
    influence matrix with trace(H) > n_channels, which makes plain GCV return
    only ``inf`` values. The solver should therefore choose the regularization
    parameter based on the unnormalized MNE kernel instead.
    """
    n_chans = int(forward_model["sol"]["data"].shape[0])
    rng = np.random.RandomState(0)
    A = rng.randn(n_chans, n_chans)
    noise_cov = _as_mne_cov(A @ A.T, forward_model.ch_names)

    solver = SolverDSPM(n_reg_params=7, regularisation_method="GCV")
    solver.r_values = np.logspace(-6, 0, 7)
    solver.make_inverse_operator(
        forward_model,
        simulated_evoked,
        alpha="auto",
        noise_cov=noise_cov,
    )

    stc = solver.apply_inverse_operator(simulated_evoked)
    assert np.all(np.isfinite(stc.data))

    # Recompute the GCV values the solver should have used (on MNE kernels).
    data = solver.unpack_data_obj(simulated_evoked)
    L = solver.leadfield
    gamma = float(solver.gcv_gamma)
    gcv_values = []
    for op in solver._mne_inverse_operators:
        W = op.data[0]
        x = W @ data
        M_hat = L @ x
        trace_H = float(np.sum(L * W.T))
        residual = data - M_hat
        residual_ss = float(np.linalg.norm(np.asarray(residual)) ** 2)
        effective_dof = L.shape[0] - gamma * trace_H
        if effective_dof <= 0 or abs(effective_dof) < 1e-10:
            gcv_values.append(np.inf)
        else:
            gcv_values.append(residual_ss / (effective_dof**2))

    gcv_values = np.asarray(gcv_values, dtype=float)
    assert np.any(np.isfinite(gcv_values))

    valid = np.isfinite(gcv_values)
    valid_positions = np.where(valid)[0]
    assert int(solver.last_reg_idx) in set(valid_positions.tolist())


def test_esmv_robust_covariance_mode_accepts_noise_cov(forward_model, simulated_evoked):
    n_chans = int(forward_model["sol"]["data"].shape[0])
    noise_cov = _as_mne_cov(np.eye(n_chans, dtype=float), forward_model.ch_names)

    solver = SolverESMV(use_robust_covariance=True, n_reg_params=1)
    solver.make_inverse_operator(
        forward_model,
        simulated_evoked,
        alpha=0.1,
        noise_cov=noise_cov,
    )

    kernel = _kernel(solver)
    assert np.all(np.isfinite(kernel))
    assert kernel.shape[0] == solver.leadfield.shape[1]
    assert kernel.shape[1] == solver.leadfield.shape[0]


def test_esmv_robust_covariance_mode_maps_back_to_raw_sensors(
    forward_model, simulated_evoked
):
    """If noise_cov is rank-deficient, the whitening path must still return an
    operator that can be applied to raw sensor data (n_chans columns)."""
    n_chans = int(forward_model["sol"]["data"].shape[0])
    diag = np.ones(n_chans, dtype=float)
    diag[-1] = 0.0  # Force rank drop in the whitener.
    noise_cov = _as_mne_cov(np.diag(diag), forward_model.ch_names)

    solver = SolverESMV(use_robust_covariance=True, n_reg_params=1)
    solver.make_inverse_operator(
        forward_model,
        simulated_evoked,
        alpha=0.1,
        noise_cov=noise_cov,
    )

    kernel = _kernel(solver)
    assert kernel.shape[1] == n_chans
    stc = solver.apply_inverse_operator(simulated_evoked)
    assert np.all(np.isfinite(stc.data))


def test_lcmv_robust_covariance_mode_maps_back_to_raw_sensors(
    forward_model, simulated_evoked
):
    """LCMV whitening path must return an operator applicable to raw sensors."""
    n_chans = int(forward_model["sol"]["data"].shape[0])
    diag = np.ones(n_chans, dtype=float)
    diag[-1] = 0.0  # Force rank drop in the whitener.
    noise_cov = _as_mne_cov(np.diag(diag), forward_model.ch_names)

    solver = SolverLCMV(use_robust_covariance=True, n_reg_params=1)
    solver.make_inverse_operator(
        forward_model,
        simulated_evoked,
        alpha=0.1,
        noise_cov=noise_cov,
        weight_norm=False,
    )

    kernel = _kernel(solver)
    assert kernel.shape[1] == n_chans
    stc = solver.apply_inverse_operator(simulated_evoked)
    assert np.all(np.isfinite(stc.data))


def test_champagne_validates_noise_cov_shape(forward_model, simulated_evoked):
    n_chans = int(forward_model["sol"]["data"].shape[0])
    bad_cov = _as_bad_mne_cov(
        np.eye(max(1, n_chans - 1), dtype=float),
        list(forward_model.ch_names),
    )
    solver = SolverChampagne(n_reg_params=1)
    with pytest.raises(
        ValueError, match="channel-name length does not match covariance shape"
    ):
        solver.make_inverse_operator(
            forward_model,
            simulated_evoked,
            alpha=0.1,
            max_iter=5,
            noise_cov=bad_cov,
        )


def test_eloreta_robust_whitener_mode_accepts_noise_cov(
    forward_model, simulated_evoked
):
    n_chans = int(forward_model["sol"]["data"].shape[0])
    noise_cov = _as_mne_cov(np.eye(n_chans, dtype=float), forward_model.ch_names)
    solver = SolverELORETA(use_noise_whitener=True, n_reg_params=1)
    solver.make_inverse_operator(
        forward_model,
        simulated_evoked,
        alpha=0.1,
        noise_cov=noise_cov,
        max_iter=10,
    )
    kernel = _kernel(solver)
    assert np.all(np.isfinite(kernel))
    assert kernel.shape[0] == solver.leadfield.shape[1]
    assert kernel.shape[1] == solver.leadfield.shape[0]


def test_eloreta_robust_whitener_mode_validates_noise_cov_shape(
    forward_model, simulated_evoked
):
    n_chans = int(forward_model["sol"]["data"].shape[0])
    bad_cov = _as_bad_mne_cov(
        np.eye(max(1, n_chans - 1), dtype=float),
        list(forward_model.ch_names),
    )
    solver = SolverELORETA(use_noise_whitener=True, n_reg_params=1)
    with pytest.raises(
        ValueError, match="channel-name length does not match covariance shape"
    ):
        solver.make_inverse_operator(
            forward_model,
            simulated_evoked,
            alpha=0.1,
            noise_cov=bad_cov,
            max_iter=5,
        )


@pytest.mark.parametrize(
    "solver_cls",
    [
        SolverMNE,
        SolverWMNE,
        SolverSLORETA,
        SolverLORETA,
        SolverSMAP,
        SolverLAURA,
    ],
)
def test_minimum_norm_whitener_mode_maps_back_to_raw_sensors(
    solver_cls, forward_model, simulated_evoked
):
    n_chans = int(forward_model["sol"]["data"].shape[0])
    diag = np.ones(n_chans, dtype=float)
    diag[-1] = 0.0  # Force rank drop in the whitener.
    noise_cov = _as_mne_cov(np.diag(diag), forward_model.ch_names)

    solver = solver_cls(use_noise_whitener=True, n_reg_params=1)
    solver.make_inverse_operator(
        forward_model,
        simulated_evoked,
        alpha=0.1,
        noise_cov=noise_cov,
    )

    kernel = _kernel(solver)
    assert kernel.shape[1] == n_chans
    stc = solver.apply_inverse_operator(simulated_evoked)
    assert np.all(np.isfinite(stc.data))


@pytest.mark.parametrize(
    "solver_cls",
    [
        SolverMNE,
        SolverWMNE,
        SolverSLORETA,
        SolverLORETA,
        SolverSMAP,
    ],
)
def test_minimum_norm_whitener_mode_validates_noise_cov_shape(
    solver_cls, forward_model, simulated_evoked
):
    n_chans = int(forward_model["sol"]["data"].shape[0])
    bad_cov = _as_bad_mne_cov(
        np.eye(max(1, n_chans - 1), dtype=float),
        list(forward_model.ch_names),
    )
    solver = solver_cls(use_noise_whitener=True, n_reg_params=1)
    with pytest.raises(
        ValueError, match="channel-name length does not match covariance shape"
    ):
        solver.make_inverse_operator(
            forward_model,
            simulated_evoked,
            alpha=0.1,
            noise_cov=bad_cov,
        )


def test_laura_validates_noise_cov_shape(forward_model, simulated_evoked):
    n_chans = int(forward_model["sol"]["data"].shape[0])
    bad_cov = _as_bad_mne_cov(
        np.eye(max(1, n_chans - 1), dtype=float),
        list(forward_model.ch_names),
    )
    solver = SolverLAURA(n_reg_params=1)
    with pytest.raises(
        ValueError, match="channel-name length does not match covariance shape"
    ):
        solver.make_inverse_operator(
            forward_model,
            simulated_evoked,
            alpha=0.1,
            noise_cov=bad_cov,
        )


@pytest.mark.parametrize("solver_cls", [SolverSLORETA, SolverELORETA])
def test_projected_alpha_scaling_uses_transformed_leadfield(
    solver_cls, forward_model, simulated_evoked, monkeypatch
):
    n_chans = int(forward_model["sol"]["data"].shape[0])
    projector = np.eye(n_chans, dtype=float)
    projector[-1, -1] = 0.0

    solver = solver_cls(
        n_reg_params=4,
        use_noise_whitener=False,
        use_trace_normalization=False,
    )

    def _projector(*args, **kwargs):  # noqa: ARG001
        return projector

    monkeypatch.setattr(solver, "compute_sensor_projector", _projector)
    solver.make_inverse_operator(
        forward_model,
        simulated_evoked,
        alpha="auto",
    )

    leadfield_eff = projector @ solver.leadfield
    max_eig = float(
        np.linalg.svd(leadfield_eff @ leadfield_eff.T, compute_uv=False).max()
    )
    expected_alphas = np.asarray(max_eig * solver.r_values, dtype=float)
    np.testing.assert_allclose(np.asarray(solver.alphas, dtype=float), expected_alphas)


@pytest.mark.parametrize(
    "solver_cls,extra_kwargs",
    [
        (SolverSLORETA, {}),
        (SolverELORETA, {"max_iter": 10}),
        (SolverLORETA, {}),
        (SolverSMAP, {}),
        (SolverLAURA, {}),
        (SolverDSPMMNE, {}),
    ],
)
def test_minimum_norm_robust_whitener_handles_zero_noise_cov(
    solver_cls, extra_kwargs, forward_model, simulated_evoked
):
    n_chans = int(forward_model["sol"]["data"].shape[0])
    noise_cov = _as_mne_cov(
        np.zeros((n_chans, n_chans), dtype=float), forward_model.ch_names
    )

    solver = solver_cls(use_noise_whitener=True, n_reg_params=1)
    solver.make_inverse_operator(
        forward_model,
        simulated_evoked,
        alpha=0.1,
        noise_cov=noise_cov,
        **extra_kwargs,
    )

    kernel = _kernel(solver)
    assert kernel.shape[1] == n_chans
    assert np.all(np.isfinite(kernel))


@pytest.mark.parametrize(
    "solver_cls,extra_kwargs",
    [
        (SolverSLORETA, {}),
        (SolverELORETA, {"max_iter": 10}),
        (SolverLORETA, {}),
        (SolverSMAP, {}),
        (SolverLAURA, {}),
    ],
)
def test_trace_normalization_applies_unit_trace_constraint(
    solver_cls, extra_kwargs, forward_model, simulated_evoked, monkeypatch
):
    n_chans = int(forward_model["sol"]["data"].shape[0])
    noise_cov = _as_mne_cov(np.eye(n_chans, dtype=float), forward_model.ch_names)

    solver = solver_cls(
        use_noise_whitener=True,
        use_trace_normalization=True,
        n_reg_params=1,
    )
    calls = {}
    original = solver.trace_normalize_operator

    def wrapped(A, target_rank, *, eps=1e-15):
        A_norm, scale = original(A, target_rank=target_rank, eps=eps)
        calls["target_rank"] = int(target_rank)
        calls["trace_after"] = float(np.sum(A_norm * A_norm))
        calls["scale"] = float(scale)
        return A_norm, scale

    monkeypatch.setattr(solver, "trace_normalize_operator", wrapped)
    solver.make_inverse_operator(
        forward_model,
        simulated_evoked,
        alpha=0.1,
        noise_cov=noise_cov,
        **extra_kwargs,
    )

    assert "target_rank" in calls
    assert np.isfinite(calls["scale"])
    np.testing.assert_allclose(
        calls["trace_after"],
        float(calls["target_rank"]),
        rtol=1e-10,
        atol=1e-10,
    )
