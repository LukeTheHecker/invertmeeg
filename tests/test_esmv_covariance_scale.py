"""Covariance weighting is dimensionless, conditional on a stable subspace.

Fixed-rank cases isolate the weighting bug. Automatic-rank cases cover ordinary
rescalings; extreme auto-rank failures remain recorded in the audit artifact.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from invert.solvers.base import BaseSolver
from invert.solvers.beamformers.esmv_mvpure import SolverESMVMVPURE
from invert.solvers.beamformers.iresmv import SolverIRESMV
from invert.solvers.beamformers.ssp_esmv import SolverSSPESMV
from invert.solvers.beamformers.ssp_iresmv import SolverSSPIRESMV


@pytest.mark.parametrize(
    "cls", [SolverIRESMV, SolverSSPESMV, SolverSSPIRESMV, SolverESMVMVPURE]
)
@pytest.mark.parametrize(
    "scale,fixed_rank",
    [
        (1e-9, True),
        (1e-6, True),
        (1e-3, True),
        (0.1, True),
        (10.0, True),
        (1e3, True),
        (0.1, False),
        (10.0, False),
    ],
)
def test_esmv_output_scales_linearly_with_data(cls, scale, fixed_rank, monkeypatch):
    rng = np.random.default_rng(4)
    L = rng.normal(size=(8, 24))
    L -= L.mean(axis=0)
    y = rng.normal(size=(8, 100))
    y -= y.mean(axis=0)

    def solve(B):
        solver = cls()
        if fixed_rank:
            solver.select_signal_subspace = lambda C: BaseSolver.select_signal_subspace(
                solver, C, rank=3
            )
        solver.leadfield = L.copy()
        solver.leadfield_original = L.copy()
        solver.leadfield_normed = L / np.linalg.norm(L, axis=0)
        if cls is not SolverESMVMVPURE:
            return solver._solve(B.copy())
        solver.alpha = 0.1
        monkeypatch.setattr(BaseSolver, "make_inverse_operator", lambda *a, **k: None)
        monkeypatch.setattr(
            BaseSolver,
            "prepare_whitened_forward",
            lambda *a, **k: SimpleNamespace(
                G_white=L.copy(), sensor_transform=np.eye(8)
            ),
        )
        monkeypatch.setattr(BaseSolver, "unpack_data_obj", lambda *a, **k: B.copy())
        solver.make_inverse_operator(None, None, alpha=0.1)
        return solver.inverse_operators[0].data[0] @ B

    baseline = solve(y)
    scaled = solve(scale * y)
    np.testing.assert_allclose(scaled, scale * baseline, rtol=1e-9, atol=1e-12 * scale)
