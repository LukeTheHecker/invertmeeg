"""Independent numerical checks for retained Bayesian audit corrections."""

import numpy as np
import pytest

from invert.solvers.bayesian.cmem import (
    _mem_dual_obj_grad,
    _optimize_lambda_batch_lbfgs,
)


def _identity_mem_components():
    return [
        {
            "vertices": np.array([0, 1]),
            "Sigma_k": np.eye(2),
            "mu_k": np.zeros(2),
            "alpha_k": 0.5,
        }
    ]


@pytest.mark.parametrize(
    "lam", [np.array([0.4, 0.1]), np.array([40.0, 1.0]), np.array([80.0, -2.0])]
)
def test_cmem_dual_gradient_matches_objective(lam):
    def evaluate(v):
        return _mem_dual_obj_grad(
            v, np.zeros(2), np.eye(2), _identity_mem_components(), np.zeros(2)
        )

    objective, gradient = evaluate(lam)
    energy = 0.5 * np.dot(lam, lam)
    expected = energy + np.logaddexp(np.log(0.5), np.log(0.5) + energy)
    assert objective == pytest.approx(expected, rel=1e-12)
    h = 1e-4
    finite_difference = [
        (evaluate(lam + h * e)[0] - evaluate(lam - h * e)[0]) / (2 * h)
        for e in np.eye(2)
    ]
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-7, atol=1e-7)


def test_cmem_high_energy_optimizer_solves_stationary_posterior():
    # At this energy, p(active) rounds to one. The stationary condition for
    # identity gain, source covariance and noise covariance is 2*lambda = y.
    y = np.array([[80.0], [2.0]])
    components = _identity_mem_components()
    result = _optimize_lambda_batch_lbfgs(y, np.eye(2), components, max_iter=100)
    np.testing.assert_allclose(result, y / 2, rtol=1e-6, atol=1e-5)
    objective, gradient = _mem_dual_obj_grad(
        result[:, 0], y[:, 0], np.eye(2), components, np.zeros(2)
    )
    assert np.linalg.norm(gradient) < 1e-5
    assert np.isfinite(objective)
