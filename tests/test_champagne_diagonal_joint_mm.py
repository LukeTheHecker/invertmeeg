"""Independent Gaussian-posterior calculation for simultaneous diagonal MM."""

import numpy as np
import pytest

from invert.solvers.bayesian.champagne import SolverChampagne


@pytest.mark.parametrize("mode", ["diagonal", "precision"])
@pytest.mark.parametrize("rule", ["Convexity", "MM"])
@pytest.mark.parametrize("prune", [False, True])
@pytest.mark.parametrize("amplitude", [1e-10, 1.0, 20.0, 1e5])
def test_simultaneous_diagonal_mm_reference(mode, rule, prune, amplitude):
    rng = np.random.default_rng(20261107)
    L = rng.normal(size=(6, 13))
    L /= np.linalg.norm(L, axis=0)
    Y = amplitude * rng.normal(size=(6, 24))
    Y -= Y.mean(axis=0)
    alpha = 0.4
    m, n = L.shape
    S = np.linalg.inv(np.eye(n) + L.T @ L / alpha)
    mu = S @ L.T @ Y / alpha
    Si = np.eye(m) / alpha - L @ S @ L.T / alpha**2
    gamma = np.sqrt(np.mean(mu**2, axis=1) / (1 - np.diag(S)))
    residual = Y - L @ mu
    # Existing modes retain their tiny 1e-10 denominator stabilizers.
    # Compare against the exact MM equation, within their bounded roundoff effect.
    noise = np.sqrt(np.mean(residual**2, axis=1) / np.diag(Si))
    active = np.flatnonzero(gamma > 0.1 * gamma.max()) if prune else np.arange(n)
    ga = gamma[active]
    La = L[:, active]
    expected = np.zeros((n, m))
    expected[active] = (
        ga[:, None] * La.T @ np.linalg.inv(np.diag(noise) + (La * ga) @ La.T)
    )
    s = SolverChampagne(
        update_rule=rule, noise_learning="learn", noise_learning_mode=mode
    )
    s.leadfield = L
    s.noise_cov = np.eye(m)
    s.max_iter = 1
    s.prune = prune
    s.pruning_thresh = 0.1
    s.convergence_criterion = 0
    actual = s.make_champagne(Y, alpha, pruning_thresh=0.1)
    np.testing.assert_allclose(s.learned_noise_cov, np.diag(noise), rtol=1e-9, atol=0)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("mode", ["diagonal", "precision"])
@pytest.mark.parametrize("amplitude", [1e-8, 1.0, 1e5])
def test_joint_mm_evidence_does_not_increase(mode, amplitude):
    rng = np.random.default_rng(107)
    L = rng.normal(size=(6, 13))
    L /= np.linalg.norm(L, axis=0)
    Y = amplitude * rng.normal(size=(6, 24))
    Y -= Y.mean(axis=0)
    losses = []

    class Recorded(SolverChampagne):
        def _update_noise_covariance_pruned(self, residuals, L_act, gammas, Si, N, T):
            losses.append(-np.linalg.slogdet(Si)[1] + np.trace(Si @ Y @ Y.T) / T)
            return super()._update_noise_covariance_pruned(
                residuals, L_act, gammas, Si, N, T
            )

    s = Recorded(
        update_rule="Convexity", noise_learning="learn", noise_learning_mode=mode
    )
    s.leadfield = L
    s.noise_cov = np.eye(6)
    s.max_iter = 30
    s.prune = False
    s.pruning_thresh = 0
    s.convergence_criterion = 0
    s.make_champagne(Y, 0.4)
    assert len(losses) > 1
    assert np.max(np.diff(losses)) <= 1e-9 * max(1.0, max(abs(np.array(losses))))
