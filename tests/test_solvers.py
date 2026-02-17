"""Smoke tests for all solvers: correct shape, finite values, non-trivial output."""

import mne
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

# Solvers that support orientation="free" (true vector/free orientation)
# These are: solvers with SUPPORTS_VECTOR_ORIENTATION = True + all beamformers
FREE_ORIENTATION_SOLVERS = {
    # SUPPORTS_VECTOR_ORIENTATION = True (Backus-Gilbert excluded: free-ori bug)
    "MNE",
    "eLORETA",
    "EPIFOCUS",
    "LCMV",
    # All beamformers (auto-detected via module path)
    "Adapt-Flex-ESMV",
    "DICS",
    "Deblur-Flex-ESMV",
    "ESMV",
    "ESMV-MVPURE",
    "Flex-ESMV",
    "Flex-ESMV-MVPURE",
    "Flex-ESMV2",
    "HOCMCMV",
    "HOCMV",
    "IR-ESMV",
    "LCMV-MVPURE",
    "MCMV",
    "MVAB",
    "ReciPSIICOS",
    "SAM",
    "SMV",
    "SSP-ESMV",
    "SSP-IR-ESMV",
    "Safe-Flex-ESMV",
    "Sharp-Flex-ESMV",
    "Sharp-Flex-ESMV2",
    "Unit-Noise-Gain",
    "WNMV",
}


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


@pytest.mark.parametrize("solver_name", _solver_ids())
def test_solver_fixed_orientation_smoke(
    solver_name, forward_model_free_surface, simulated_evoked_free_surface
):
    """Every solver should work with orientation='fixed' on a free-orientation forward."""
    n_dipoles = int(forward_model_free_surface["nsource"])
    n_time = simulated_evoked_free_surface.data.shape[1]

    solver = Solver(solver_name, orientation="fixed")

    extra_kwargs = dict(alpha=0.1, epochs=1, n=2, k=2)
    if solver_name in MUSIC_SOLVERS:
        extra_kwargs["stop_crit"] = 0.1

    solver.make_inverse_operator(
        forward_model_free_surface, simulated_evoked_free_surface, **extra_kwargs
    )
    stc = solver.apply_inverse_operator(simulated_evoked_free_surface)
    data = stc.data

    assert data.shape[0] == n_dipoles, (
        f"{solver_name}: expected {n_dipoles} dipoles, got {data.shape[0]}"
    )
    assert data.shape[1] == n_time, (
        f"{solver_name}: expected {n_time} time points, got {data.shape[1]}"
    )
    assert np.all(np.isfinite(data)), f"{solver_name}: output contains NaN or Inf"
    assert np.any(data != 0), f"{solver_name}: output is all zeros"


@pytest.mark.parametrize("solver_name", _solver_ids())
def test_solver_pca_orientation_smoke(
    solver_name, forward_model_free_surface, simulated_evoked_free_surface
):
    """Every solver should work with orientation='pca' on a free-orientation forward."""
    n_dipoles = int(forward_model_free_surface["nsource"])
    n_time = simulated_evoked_free_surface.data.shape[1]

    solver = Solver(solver_name, orientation="pca")

    extra_kwargs = dict(alpha=0.1, epochs=1, n=2, k=2)
    if solver_name in MUSIC_SOLVERS:
        extra_kwargs["stop_crit"] = 0.1

    solver.make_inverse_operator(
        forward_model_free_surface, simulated_evoked_free_surface, **extra_kwargs
    )
    stc = solver.apply_inverse_operator(simulated_evoked_free_surface)
    data = stc.data

    assert data.shape[0] == n_dipoles, (
        f"{solver_name}: expected {n_dipoles} dipoles, got {data.shape[0]}"
    )
    assert data.shape[1] == n_time, (
        f"{solver_name}: expected {n_time} time points, got {data.shape[1]}"
    )
    assert np.all(np.isfinite(data)), f"{solver_name}: output contains NaN or Inf"
    assert np.any(data != 0), f"{solver_name}: output is all zeros"


def _free_orientation_solver_ids():
    return [s for s in _solver_ids() if s in FREE_ORIENTATION_SOLVERS]


@pytest.mark.parametrize("solver_name", _free_orientation_solver_ids())
def test_solver_free_orientation_smoke(
    solver_name, forward_model_free_surface, simulated_evoked_free_surface
):
    """Solvers supporting free orientation should produce correct scalar output."""
    n_dipoles = int(forward_model_free_surface["nsource"])
    n_time = simulated_evoked_free_surface.data.shape[1]

    solver = Solver(solver_name, orientation="free")

    extra_kwargs = dict(alpha=0.1, epochs=1, n=2, k=2)
    if solver_name in MUSIC_SOLVERS:
        extra_kwargs["stop_crit"] = 0.1

    solver.make_inverse_operator(
        forward_model_free_surface, simulated_evoked_free_surface, **extra_kwargs
    )
    stc = solver.apply_inverse_operator(simulated_evoked_free_surface)
    data = stc.data

    # Free orientation collapses vector to scalar via norm → n_dipoles, not 3*n_dipoles
    assert data.shape[0] == n_dipoles, (
        f"{solver_name}: expected {n_dipoles} dipoles, got {data.shape[0]}"
    )
    assert data.shape[1] == n_time, (
        f"{solver_name}: expected {n_time} time points, got {data.shape[1]}"
    )
    assert np.all(np.isfinite(data)), f"{solver_name}: output contains NaN or Inf"
    assert np.any(data != 0), f"{solver_name}: output is all zeros"


@pytest.mark.parametrize("solver_name", ["loreta", "champagne", "subsmp"])
def test_discrete_forward_scalar_solver_runs_with_pca(
    solver_name, forward_model_discrete_free, simulated_evoked_discrete_free
):
    """A discrete free-orientation forward should run via PCA reduction on scalar solvers."""
    solver = Solver(solver_name, orientation="pca", n_reg_params=1)
    solver.make_inverse_operator(
        forward_model_discrete_free,
        simulated_evoked_discrete_free,
        alpha=0.1,
        max_iter=3,
        epochs=1,
        n=2,
        k=2,
    )

    # PCA mode reduces 3*n -> n.
    assert solver.leadfield.shape[1] == int(forward_model_discrete_free["nsource"])

    stc = solver.apply_inverse_operator(simulated_evoked_discrete_free)
    assert stc.data.shape[0] == int(forward_model_discrete_free["nsource"])
    assert stc.data.shape[1] == simulated_evoked_discrete_free.data.shape[1]
    assert np.all(np.isfinite(stc.data))
    assert np.any(stc.data != 0)


def test_mne_free_orientation_vector_output(
    forward_model_free_surface, simulated_evoked_free_surface
):
    """Minimum-norm solvers should support true free orientation + vector STC output."""
    solver = Solver("mne", orientation="free", n_reg_params=1)
    solver.make_inverse_operator(
        forward_model_free_surface,
        simulated_evoked_free_surface,
        alpha=0.1,
    )

    stc_scalar = solver.apply_inverse_operator(simulated_evoked_free_surface)
    assert isinstance(stc_scalar, mne.SourceEstimate)
    assert stc_scalar.data.shape[0] == int(forward_model_free_surface["nsource"])
    assert stc_scalar.data.shape[1] == simulated_evoked_free_surface.data.shape[1]

    stc_vec = solver.apply_inverse_operator_vector(simulated_evoked_free_surface)
    assert isinstance(stc_vec, mne.VectorSourceEstimate)
    assert stc_vec.data.shape == (
        int(forward_model_free_surface["nsource"]),
        3,
        simulated_evoked_free_surface.data.shape[1],
    )
    assert np.all(np.isfinite(stc_vec.data))
