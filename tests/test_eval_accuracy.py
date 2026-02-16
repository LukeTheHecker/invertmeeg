"""Accuracy regression test: every solver must localize a focal source at high SNR.

Each solver runs on 10 seeds with a single focal source at ~40dB SNR.
Metrics: EMD (localization) and R² (data fit).

Assertion tiers:
1. Universal floor (only when no baseline exists): median_emd < 65mm
2. Per-solver baseline (from solver_baselines.json): 10% tolerance on both EMD and R²

Run with:  pytest -m slow tests/test_eval_accuracy.py -v --timeout=600
Update baselines: pytest -m slow tests/test_eval_accuracy.py --update-baselines
"""

import json
import pathlib

import mne
import numpy as np
import pytest
from scipy.spatial.distance import cdist

from invert import Solver
from invert.forward import create_forward_model, get_info
from invert.solver_registry import iter_solver_specs

ot = pytest.importorskip("ot")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_SEEDS = 10
SFREQ = 200.0
N_TIMES = 200  # 1 second at 200 Hz
NOISE_REL = 0.01  # ~40dB SNR
MAX_EMD = 65.0  # mm — universal floor (only when no baseline exists)
TOLERANCE = 0.10  # 10% regression tolerance
EMD_ABS_TOLERANCE = 2.0  # mm — additive tolerance for near-zero EMD baselines

BASELINES_PATH = pathlib.Path(__file__).parent / "solver_baselines.json"

# ---------------------------------------------------------------------------
# Solver selection
# ---------------------------------------------------------------------------
_ALL_SPECS = {
    spec.solver_id: spec
    for spec in iter_solver_specs(include_internal=False, include_unavailable=True)
}

NN_SOLVERS = {sid for sid, spec in _ALL_SPECS.items() if "torch" in spec.requires}
MUSIC_SOLVERS = {"music", "rap-music", "trap-music", "flex-music"}
EXCLUDED = {"random-noise", "cmem"}  # random-noise: by design random; cmem: too slow


def _solver_ids():
    return [
        sid
        for sid in _ALL_SPECS
        if sid not in NN_SOLVERS and sid not in EXCLUDED
    ]


# ---------------------------------------------------------------------------
# EMD helper (same as test_eval_orientation — inlined to avoid import issues)
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
    return float(np.asarray(val).item()) if isinstance(val, (list, np.ndarray)) else float(val)


# ---------------------------------------------------------------------------
# Simulation helper
# ---------------------------------------------------------------------------
def _simulate_focal_source(leadfield, n_dipoles, info, seed):
    """Simulate a single focal source with windowed sinusoid + Gaussian noise.

    Returns (evoked, y_true) where y_true is n_dipoles with 1.0 at the active vertex.
    """
    rng = np.random.RandomState(seed)
    n_chans = leadfield.shape[0]

    active = rng.choice(n_dipoles)

    # Time course: windowed sinusoid
    t = np.arange(N_TIMES, dtype=float) / SFREQ
    freq = float(rng.uniform(8.0, 20.0))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    tc = np.sin(2.0 * np.pi * freq * t + phase) * np.hanning(N_TIMES)

    # Ground truth
    y_true = np.zeros(n_dipoles, dtype=float)
    y_true[active] = 1.0

    # Source matrix (single active dipole)
    S = np.zeros((n_dipoles, N_TIMES), dtype=float)
    S[active, :] = tc

    # Sensor data
    Y = leadfield @ S
    sigma = NOISE_REL * float(np.std(Y))
    if np.isfinite(sigma) and sigma > 0:
        Y = Y + rng.randn(n_chans, N_TIMES) * sigma

    evoked = mne.EvokedArray(Y, info, tmin=0.0, verbose=0)
    return evoked, y_true


# ---------------------------------------------------------------------------
# R² helper
# ---------------------------------------------------------------------------
def _compute_r2(evoked_data, leadfield, stc_data):
    """Compute % variance explained: 1 - ||Y - L @ S||² / ||Y||²."""
    residual = evoked_data - leadfield @ stc_data
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum(evoked_data ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def _load_baselines():
    if BASELINES_PATH.exists():
        return json.loads(BASELINES_PATH.read_text())
    return {}


def _save_baselines(baselines):
    # Replace NaN/Inf with None for valid JSON
    def _clean(v):
        if v is None:
            return None
        return v if np.isfinite(v) else None

    clean = {}
    for k, v in baselines.items():
        clean[k] = {mk: _clean(mv) for mk, mv in v.items()}
    BASELINES_PATH.write_text(json.dumps(clean, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Module-scoped fixture: forward model + distance matrix
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def accuracy_setup():
    """Fixed-orientation forward model with precomputed distance matrix."""
    info = get_info(kind="biosemi16", sfreq=SFREQ)
    fwd = create_forward_model(info=info, sampling="ico1")
    leadfield = np.asarray(fwd["sol"]["data"], dtype=float)
    n_dipoles = leadfield.shape[1]
    src = fwd["src"]
    coords_mm = np.vstack([
        src[0]["rr"][src[0]["vertno"]] * 1000.0,
        src[1]["rr"][src[1]["vertno"]] * 1000.0,
    ])
    dist_matrix = cdist(coords_mm, coords_mm)
    return dict(
        fwd=fwd, info=info, leadfield=leadfield,
        n_dipoles=n_dipoles, dist_matrix=dist_matrix,
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("solver_name", _solver_ids())
def test_solver_localization(solver_name, accuracy_setup, update_baselines):
    fwd = accuracy_setup["fwd"]
    info = accuracy_setup["info"]
    leadfield = accuracy_setup["leadfield"]
    n_dipoles = accuracy_setup["n_dipoles"]
    dist_matrix = accuracy_setup["dist_matrix"]

    extra_kwargs = dict(alpha=0.1, epochs=1, n=2, k=2)
    if solver_name in MUSIC_SOLVERS:
        extra_kwargs["stop_crit"] = 0.1

    emds = []
    r2s = []

    for seed in range(N_SEEDS):
        evoked, y_true = _simulate_focal_source(leadfield, n_dipoles, info, seed)

        try:
            solver = Solver(solver_name)
            solver.make_inverse_operator(fwd, evoked, **extra_kwargs)
            stc = solver.apply_inverse_operator(evoked)
        except Exception:
            emds.append(float("nan"))
            r2s.append(float("nan"))
            continue

        # EMD: power-based source distribution
        source_power = np.sum(stc.data ** 2, axis=1)
        emd_val = _eval_emd(dist_matrix, y_true, source_power)
        emds.append(emd_val)

        # R²: data fit
        r2_val = _compute_r2(evoked.data, leadfield, stc.data)
        r2s.append(r2_val)

    median_emd = float(np.nanmedian(emds))
    median_r2 = float(np.nanmedian(r2s))

    # --- Update mode: save baselines and skip assertions ---
    if update_baselines:
        baselines = _load_baselines()
        baselines[solver_name] = {"emd": round(median_emd, 4), "r2": round(median_r2, 6)}
        _save_baselines(baselines)
        pytest.skip(f"Baseline saved: EMD={median_emd:.2f}, R²={median_r2:.4f}")

    # --- Assert mode ---
    baselines = _load_baselines()

    if solver_name in baselines:
        bl = baselines[solver_name]
        bl_emd = bl.get("emd")
        bl_r2 = bl.get("r2")

        # EMD regression: allow 10% relative + 2mm absolute (for near-zero baselines)
        if bl_emd is not None:
            emd_limit = bl_emd * (1 + TOLERANCE) + EMD_ABS_TOLERANCE
            assert median_emd < emd_limit, (
                f"{solver_name}: EMD regressed: {median_emd:.2f}mm > "
                f"limit {emd_limit:.2f}mm (baseline {bl_emd:.2f}mm)"
            )

        # R² regression: use absolute difference to handle negative R² correctly
        if bl_r2 is not None:
            r2_limit = bl_r2 - abs(bl_r2) * TOLERANCE
            assert median_r2 >= r2_limit, (
                f"{solver_name}: R² regressed: {median_r2:.4f} < "
                f"limit {r2_limit:.4f} (baseline {bl_r2:.4f})"
            )
    else:
        # Universal floor — only when no baseline exists (new solvers)
        assert median_emd < MAX_EMD, (
            f"{solver_name}: median EMD {median_emd:.2f}mm exceeds floor {MAX_EMD}mm"
        )
