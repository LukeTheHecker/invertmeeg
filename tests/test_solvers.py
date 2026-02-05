"""Smoke tests for all solvers: correct shape, finite values, non-trivial output."""

import numpy as np
import pytest

from invert import Solver
from invert.config import all_solvers

# Solvers that need data passed to make_inverse_operator
DATA_DEPENDENT_SOLVERS = {
    "MVAB",
    "LCMV",
    "SMV",
    "WNMV",
    "HOCMV",
    "ESMV",
    "MCMV",
    "HOCMCMV",
    "ReciPSIICOS",
    "SAM",
    "EBB",
    "MUSIC",
    "RAP-MUSIC",
    "TRAP-MUSIC",
    "FLEX-MUSIC",
    "FLEX-AP",
    "AP",
    "FLEX-SSM",
    "SSM",
}

# Solvers that require tensorflow + SimulationConfig — cannot be smoke-tested
# with a plain Evoked object; they need a dedicated test with proper setup.
NN_SOLVERS = {"FC", "CovCNN", "LSTM", "CNN"}

# Solvers that don't accept alpha="auto"
NO_AUTO_ALPHA = {"sLORETA"}

# Solvers that are listed in config but not wired up in Solver()
KNOWN_MISSING = {
    "HS-Champagne",
    "Patch-Champagne",
    "Hierarchical-Patch-Champagne",
    "FUN",
    "wMNE",
}

# MUSIC-family solvers need a lower stop_crit for small test data
MUSIC_SOLVERS = {"MUSIC", "RAP-MUSIC", "TRAP-MUSIC", "FLEX-MUSIC"}


def _solver_ids():
    return [s for s in all_solvers if s not in KNOWN_MISSING and s not in NN_SOLVERS]


@pytest.mark.parametrize("solver_name", _solver_ids())
def test_solver_smoke(solver_name, forward_model, simulated_evoked, leadfield):
    """Each solver should produce finite output with the correct shape."""
    n_chans, n_dipoles = leadfield.shape
    n_time = simulated_evoked.data.shape[1]

    alpha = 0.1 if solver_name in NO_AUTO_ALPHA else 0.1
    solver = Solver(solver_name)

    extra_kwargs = dict(alpha=alpha, epochs=1, n=2, k=2)
    if solver_name in MUSIC_SOLVERS:
        extra_kwargs["stop_crit"] = 0.1

    if solver_name in DATA_DEPENDENT_SOLVERS:
        solver.make_inverse_operator(forward_model, simulated_evoked, **extra_kwargs)
    else:
        solver.make_inverse_operator(forward_model, simulated_evoked, **extra_kwargs)

    stc = solver.apply_inverse_operator(simulated_evoked)
    data = stc.data

    # Shape check
    assert data.shape[0] == n_dipoles, (
        f"{solver_name}: expected {n_dipoles} dipoles, got {data.shape[0]}"
    )
    assert data.shape[1] == n_time, (
        f"{solver_name}: expected {n_time} time points, got {data.shape[1]}"
    )

    # Finite check
    assert np.all(np.isfinite(data)), f"{solver_name}: output contains NaN or Inf"

    # Non-trivial check
    assert np.any(data != 0), f"{solver_name}: output is all zeros"
