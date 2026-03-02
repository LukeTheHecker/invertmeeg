from __future__ import annotations

import numpy as np

from invert import Solver


def test_random_noise_seed_42_is_reproducible(forward_model, simulated_evoked):
    solver = Solver("random-noise", random_seed=42)
    solver.make_inverse_operator(forward_model, simulated_evoked, alpha=0.1)
    stc_a = solver.apply_inverse_operator(simulated_evoked)
    stc_b = solver.apply_inverse_operator(simulated_evoked)
    np.testing.assert_allclose(stc_a.data, stc_b.data, atol=0.0, rtol=0.0)


def test_random_noise_none_seed_is_not_forced_reproducible(
    forward_model, simulated_evoked
):
    solver = Solver("random-noise", random_seed=None)
    solver.make_inverse_operator(forward_model, simulated_evoked, alpha=0.1)
    stc_a = solver.apply_inverse_operator(simulated_evoked)
    stc_b = solver.apply_inverse_operator(simulated_evoked)
    assert not np.allclose(stc_a.data, stc_b.data)


def test_rembo_seed_42_is_reproducible(forward_model, simulated_evoked):
    solver = Solver("rembo", random_seed=42)
    solver.make_inverse_operator(forward_model, simulated_evoked, alpha=0.1)
    stc_a = solver.apply_inverse_operator(simulated_evoked, max_boost_iter=10)
    stc_b = solver.apply_inverse_operator(simulated_evoked, max_boost_iter=10)
    np.testing.assert_allclose(stc_a.data, stc_b.data, atol=0.0, rtol=0.0)
