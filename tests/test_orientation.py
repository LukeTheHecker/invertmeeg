import mne
import numpy as np
import pytest

from invert.solvers.base import BaseSolver
from invert.solvers.orientation import estimate_orientation_pca
from invert.solvers.minimum_norm.loreta import SolverLORETA


def test_surface_anatomical_matches_mne_fixed_conversion(forward_model_free_surface):
    fwd_fixed = mne.convert_forward_solution(
        forward_model_free_surface,
        surf_ori=True,
        force_fixed=True,
        use_cps=True,
        verbose=0,
    )

    solver = BaseSolver(orientation="fixed", n_reg_params=1)
    solver.make_inverse_operator(forward_model_free_surface, alpha=0.1)

    np.testing.assert_allclose(
        solver.leadfield,
        np.asarray(fwd_fixed["sol"]["data"], dtype=float),
        atol=1e-10,
    )


def test_pca_orientation_recovers_known_q_on_discrete(forward_model_discrete_free):
    G_free = np.asarray(forward_model_discrete_free["sol"]["data"], dtype=float)
    n_locs = int(G_free.shape[1] // 3)
    n_time = 50
    rng = np.random.RandomState(0)

    q_true = rng.randn(n_locs, 3)
    q_true /= np.maximum(np.linalg.norm(q_true, axis=1, keepdims=True), 1e-15)
    x = np.zeros((n_locs, n_time), dtype=float)
    active = int(min(7, max(0, n_locs - 1)))
    x[active] = rng.randn(n_time)

    J_true = (q_true[:, :, np.newaxis] * x[:, np.newaxis, :]).reshape(n_locs * 3, n_time)
    Y = G_free @ J_true

    q_est = estimate_orientation_pca(
        G_free,
        Y,
        reg=0.01,
        deterministic_sign=True,
    )

    np.testing.assert_allclose(np.linalg.norm(q_est, axis=1), 1.0, atol=1e-6)
    align = np.abs(np.sum(q_est * q_true, axis=1))
    assert float(align[active]) > 0.95


def test_free_orientation_errors_for_scalar_solver(
    forward_model_free_surface, simulated_evoked_free_surface
):
    solver = SolverLORETA(orientation="free", n_reg_params=1)
    with pytest.raises(ValueError, match=r"orientation='free'"):
        solver.make_inverse_operator(
            forward_model_free_surface, simulated_evoked_free_surface, alpha=0.1
        )
