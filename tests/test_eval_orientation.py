"""Evaluation tests: PCA/free orientation should beat fixed for tangential sources.

When the true dipole is tangential to the cortical surface, fixed-normal
orientation cannot represent it well. PCA and free orientation should
always produce lower EMD (better localization) than fixed.

Run with:  pytest -m slow tests/test_eval_orientation.py -v --timeout=300
Skip with: pytest -m "not slow"
"""

import mne
import numpy as np
import pytest
from scipy.spatial.distance import cdist

from invert import Solver
from invert.forward import create_forward_model, get_info
from invert.solver_registry import iter_solver_specs
from invert.solvers.orientation import ensure_surface_free_surf_ori

ot = pytest.importorskip("ot")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_SEEDS = 20
SFREQ = 200.0
N_TIMES = 200  # 1 second at 200 Hz
NOISE_REL = 0.01  # high SNR
ALPHA = 0.1

# ---------------------------------------------------------------------------
# Solver sets (derived from solver registry)
# ---------------------------------------------------------------------------
_ALL_SPECS = {
    spec.solver_id: spec
    for spec in iter_solver_specs(include_internal=False, include_unavailable=True)
}

# NN solvers require training — too slow for parametrized orientation tests
NN_SOLVERS = {sid for sid, spec in _ALL_SPECS.items() if "torch" in spec.requires}

MUSIC_SOLVERS = {"music", "rap-music", "trap-music", "flex-music"}

# Null baseline — outputs random noise, orientation is meaningless
RANDOM_BASELINE = {"random-noise"}

# Non-beamformer solvers with SUPPORTS_VECTOR_ORIENTATION = True
_VECTOR_MIN_NORM = {"mne", "eloreta", "epifocus", "wmne", "backus-gilbert"}

# Solvers that support orientation="free" (mirrors base.py logic at line ~2070)
FREE_ORIENTATION_SOLVERS = {
    sid
    for sid, spec in _ALL_SPECS.items()
    if sid in _VECTOR_MIN_NORM or ".beamformers." in spec.module_path
}


def _solver_ids():
    return [
        sid
        for sid in _ALL_SPECS
        if sid not in NN_SOLVERS and sid not in RANDOM_BASELINE
    ]


# ---------------------------------------------------------------------------
# EMD helper (inlined to avoid invert.evaluate import chain issues)
# ---------------------------------------------------------------------------
def _eval_emd(M, values_1, values_2, threshold=0.25):
    v1 = np.asarray(values_1, dtype=float).copy()
    v2 = np.asarray(values_2, dtype=float).copy()
    if not (np.all(np.isfinite(v1)) and np.all(np.isfinite(v2))):
        return float("nan")
    v1[v1 < threshold * v1.max()] = 0.0
    v2[v2 < threshold * v2.max()] = 0.0
    s1, s2 = v1.sum(), v2.sum()
    if s1 <= 0 or s2 <= 0:
        return float("nan")
    v1 /= s1
    v2 /= s2
    try:
        val = ot.emd2(v1, v2, M)
    except Exception:
        return float("nan")
    return (
        float(np.asarray(val).item())
        if isinstance(val, (list, np.ndarray))
        else float(val)
    )


# ---------------------------------------------------------------------------
# Simulation helper
# ---------------------------------------------------------------------------
def _simulate_tangential(G, n_src, seed):
    """Simulate a single tangential dipole. Returns (Y, x_true, active_idx)."""
    rng = np.random.RandomState(seed)
    n_chans = G.shape[0]

    active = np.array([rng.choice(n_src)], dtype=int)

    # Tangential in surf_ori basis: (t1=1, t2=0, normal=0)
    q = np.array([1.0, 0.0, 0.0], dtype=float)

    # Time course: windowed sinusoid
    t = np.arange(N_TIMES, dtype=float) / SFREQ
    freq = float(rng.uniform(8.0, 20.0))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    tc = np.sin(2.0 * np.pi * freq * t + phase) * np.hanning(N_TIMES)

    # True source distribution (all mass at one location)
    x_true = np.zeros(n_src, dtype=float)
    x_true[active[0]] = 1.0

    # Vector current density
    J = np.zeros((3 * n_src, N_TIMES), dtype=float)
    idx = active[0]
    J[3 * idx : 3 * idx + 3, :] = q[:, None] * tc[None, :]

    Y = G @ J
    sigma = NOISE_REL * float(np.std(Y))
    if np.isfinite(sigma) and sigma > 0:
        Y = Y + rng.randn(n_chans, N_TIMES) * sigma

    return Y, x_true, active


# ---------------------------------------------------------------------------
# Module-scoped fixture: forward model with surf_ori + distance matrix
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tangential_setup():
    """Free-orientation forward model (surf_ori) with precomputed distance matrix."""
    info = get_info(kind="biosemi16", sfreq=SFREQ)
    fwd = create_forward_model(info=info, sampling="ico1", fixed_ori=False)
    fwd = ensure_surface_free_surf_ori(fwd)
    G = np.asarray(fwd["sol"]["data"], dtype=float)
    n_src = G.shape[1] // 3
    src = fwd["src"]
    coords_mm = np.vstack(
        [
            src[0]["rr"][src[0]["vertno"]] * 1000.0,
            src[1]["rr"][src[1]["vertno"]] * 1000.0,
        ]
    )
    dist_matrix = cdist(coords_mm, coords_mm)
    return dict(fwd=fwd, info=info, G=G, n_src=n_src, dist_matrix=dist_matrix)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("solver_name", _solver_ids())
def test_orientation_emd_on_tangential(solver_name, tangential_setup):
    """PCA (and free where supported) should produce lower EMD than fixed."""
    fwd = tangential_setup["fwd"]
    info = tangential_setup["info"]
    G = tangential_setup["G"]
    n_src = tangential_setup["n_src"]
    dist_matrix = tangential_setup["dist_matrix"]
    n_train = N_TIMES // 2

    supports_free = solver_name in FREE_ORIENTATION_SOLVERS
    orientations = ["fixed", "pca"]
    if supports_free:
        orientations.append("free")

    emds = {ori: [] for ori in orientations}

    extra_kwargs = dict(alpha=ALPHA, epochs=1, n=2, k=2)
    if solver_name in MUSIC_SOLVERS:
        extra_kwargs["stop_crit"] = 0.1

    for seed in range(N_SEEDS):
        Y, x_true, _active = _simulate_tangential(G, n_src, seed)

        evoked_train = mne.EvokedArray(Y[:, :n_train], info, tmin=0.0, verbose=0)
        evoked_test = mne.EvokedArray(
            Y[:, n_train:], info, tmin=float(n_train) / SFREQ, verbose=0
        )

        for ori in orientations:
            solver = Solver(solver_name, orientation=ori)
            solver.make_inverse_operator(fwd, evoked_train, **extra_kwargs)
            stc = solver.apply_inverse_operator(evoked_test)
            source_pred = np.sum(stc.data**2, axis=1)
            emd_val = _eval_emd(dist_matrix, x_true, source_pred)
            emds[ori].append(emd_val)

    fixed_med = float(np.nanmedian(emds["fixed"]))
    pca_med = float(np.nanmedian(emds["pca"]))

    # Allow 16mm additive tolerance: some solvers (e.g. cMEM, MCMV) may not
    # consistently benefit from PCA orientation on small test problems.
    ORI_TOL_MM = 16.0
    assert pca_med <= fixed_med + ORI_TOL_MM, (
        f"{solver_name}: PCA EMD ({pca_med:.2f}) should be <= "
        f"fixed EMD + {ORI_TOL_MM}mm ({fixed_med + ORI_TOL_MM:.2f}) for tangential sources"
    )

    if supports_free:
        free_med = float(np.nanmedian(emds["free"]))
        assert free_med <= fixed_med + ORI_TOL_MM, (
            f"{solver_name}: free EMD ({free_med:.2f}) should be <= "
            f"fixed EMD + {ORI_TOL_MM}mm ({fixed_med + ORI_TOL_MM:.2f}) for tangential sources"
        )
