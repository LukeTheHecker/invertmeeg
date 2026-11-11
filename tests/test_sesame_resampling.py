"""Regression for SESAME particle bookkeeping through resampling.

Tests execute the actual solver routines. SESAME's MNE I/O is replaced with
identity preprocessing so its particle state can be checked independently of a
particular anatomical forward model.
"""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from invert.solvers.base import BaseSolver
from invert.solvers.dipoles.sesame import SolverSESAME


def test_sesame_resampling_keeps_best_fit_bookkeeping_consistent(monkeypatch):
    rng = np.random.RandomState(80)
    leadfield = rng.randn(8, 24)
    leadfield -= leadfield.mean(axis=0)
    leadfield /= np.linalg.norm(leadfield, axis=0)
    sources = np.zeros((24, 30))
    sources[[2, 11]] = rng.randn(2, 30)
    data = leadfield @ sources + 0.5 * rng.randn(8, 30)
    data -= data.mean(axis=0)

    solver = SolverSESAME()
    solver.leadfield = leadfield
    monkeypatch.setattr(BaseSolver, "make_inverse_operator", lambda *a, **k: None)
    monkeypatch.setattr(
        BaseSolver,
        "prepare_whitened_forward",
        lambda *a, **k: SimpleNamespace(sensor_transform=np.eye(8)),
    )
    monkeypatch.setattr(BaseSolver, "unpack_data_obj", lambda *a, **k: data)

    state = {}
    target_code = SolverSESAME.make_inverse_operator.__code__
    previous_trace = sys.gettrace()

    def capture_final_state(frame, event, arg):
        if frame.f_code is target_code and event == "return":
            local = frame.f_locals
            state["recorded_rss"] = float(local["best_rss"])
            state["actual_rss"] = float(np.sum(local["residual_best"] ** 2))
            state["best_dipoles"] = local["best_dipoles"].copy()
            state["ess"] = float(local["ess"])
        return capture_final_state

    sys.settrace(capture_final_state)
    try:
        solver.make_inverse_operator(
            None,
            None,
            n=2,
            n_particles=16,
            resample_threshold=1.0,
            random_state=0,
        )
    finally:
        sys.settrace(previous_trace)

    assert state["ess"] < 16  # the resampling branch was actually taken
    assert state["recorded_rss"] == pytest.approx(state["actual_rss"], rel=1e-12)
    np.testing.assert_array_equal(solver.selected_dipoles, state["best_dipoles"])
    kernel = solver.inverse_operators[0].data[0]
    actual_output_rss = np.sum((data - leadfield @ kernel @ data) ** 2)
    assert state["recorded_rss"] == pytest.approx(actual_output_rss, rel=1e-12)
