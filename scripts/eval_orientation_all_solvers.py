#!/usr/bin/env python
"""Evaluate all solvers: fixed vs PCA vs free orientation on tangential sources.

Hypothesis: PCA and free orientation should produce lower EMD than fixed
when the true dipole is tangential to the cortical surface normal.

Run:
  cd invert-package && .venv/bin/python scripts/eval_orientation_all_solvers.py
"""

from __future__ import annotations

import time

import mne
import numpy as np
from scipy.spatial.distance import cdist

from invert import Solver
from invert.forward import create_forward_model, get_info
from invert.solver_registry import iter_solver_specs
from invert.solvers.orientation import ensure_surface_free_surf_ori

try:
    import ot
except ImportError:
    raise SystemExit("This script requires the POT library: pip install POT")


def eval_emd(M: np.ndarray, values_1: np.ndarray, values_2: np.ndarray,
             threshold: float = 0.25) -> float:
    """Compute Earth Mover's Distance between two distributions."""
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
# Solver sets (derived from solver registry)
# ---------------------------------------------------------------------------
_ALL_SPECS = {spec.solver_id: spec
              for spec in iter_solver_specs(include_internal=False, include_unavailable=True)}

# NN solvers require training — too slow for orientation eval
NN_SOLVERS = {sid for sid, spec in _ALL_SPECS.items() if "torch" in spec.requires}

MUSIC_SOLVERS = {"music", "rap-music", "trap-music", "flex-music"}

# Non-beamformer solvers with SUPPORTS_VECTOR_ORIENTATION = True
_VECTOR_MIN_NORM = {"mne", "eloreta", "epifocus", "wmne", "backus-gilbert"}

# Solvers that support orientation="free" (mirrors base.py logic at line ~2070)
FREE_ORIENTATION_SOLVERS = {
    sid for sid, spec in _ALL_SPECS.items()
    if sid in _VECTOR_MIN_NORM or ".beamformers." in spec.module_path
}


def _solver_ids():
    return [sid for sid in _ALL_SPECS if sid not in NN_SOLVERS]


def _simulate_tangential(
    *,
    G: np.ndarray,
    n_src: int,
    sfreq: float,
    n_times: int,
    noise_rel: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a single tangential dipole and return (Y, x_true, active_idx)."""
    rng = np.random.RandomState(seed)
    n_chans = G.shape[0]

    active = np.array([rng.choice(n_src)], dtype=int)

    # Tangential orientation in surf_ori basis: (t1=1, t2=0, normal=0)
    q = np.array([1.0, 0.0, 0.0], dtype=float)

    # Time course: windowed sinusoid
    t = np.arange(n_times, dtype=float) / sfreq
    freq = float(rng.uniform(8.0, 20.0))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    tc = np.sin(2.0 * np.pi * freq * t + phase) * np.hanning(n_times)

    # True source power (for EMD: all mass at one location)
    x_true = np.zeros(n_src, dtype=float)
    x_true[active[0]] = 1.0

    # Build vector current density J (3*n_src, n_times)
    J = np.zeros((3 * n_src, n_times), dtype=float)
    idx = active[0]
    J[3 * idx : 3 * idx + 3, :] = q[:, None] * tc[None, :]

    # Measurements
    Y = G @ J
    sigma = noise_rel * float(np.std(Y))
    if np.isfinite(sigma) and sigma > 0:
        Y = Y + rng.randn(n_chans, n_times) * sigma

    return Y, x_true, active


def main() -> int:
    # ------------------------------------------------------------------
    # Parameters (fast: 16ch, ico1, 20 seeds, high SNR)
    # ------------------------------------------------------------------
    sfreq = 200.0
    n_times = 200  # 1 second
    n_seeds = 20
    noise_rel = 0.01  # high SNR
    n_train = n_times // 2  # 50/50 train/test split
    alpha = 0.1

    # ------------------------------------------------------------------
    # Forward model
    # ------------------------------------------------------------------
    info = get_info(kind="biosemi16", sfreq=sfreq)
    fwd = create_forward_model(info=info, sampling="ico1", fixed_ori=False)
    fwd = ensure_surface_free_surf_ori(fwd)

    G = np.asarray(fwd["sol"]["data"], dtype=float)
    n_chans, n_cols = G.shape
    n_src = n_cols // 3

    # Source coordinates and distance matrix for EMD
    src = fwd["src"]
    coords_mm = np.vstack([
        src[0]["rr"][src[0]["vertno"]] * 1000.0,
        src[1]["rr"][src[1]["vertno"]] * 1000.0,
    ])
    dist_matrix = cdist(coords_mm, coords_mm)

    solvers = _solver_ids()

    print("Orientation EMD evaluation — tangential sources")
    print(f"biosemi16 ({n_chans}ch), ico1 ({n_src} sources), "
          f"{n_seeds} seeds, noise_rel={noise_rel}, alpha={alpha}")
    print()
    header = (f"{'Solver':<25s} {'Fix EMD':>8s} {'PCA EMD':>8s} "
              f"{'Free EMD':>9s}  {'PCA<Fix':>7s} {'Free<Fix':>8s}")
    print(header)
    print("-" * len(header))

    results = []
    t0 = time.time()

    for solver_name in solvers:
        supports_free = solver_name in FREE_ORIENTATION_SOLVERS
        orientations = ["fixed", "pca"]
        if supports_free:
            orientations.append("free")

        emd_by_ori: dict[str, list[float]] = {ori: [] for ori in orientations}

        for seed in range(n_seeds):
            Y, x_true, active = _simulate_tangential(
                G=G, n_src=n_src, sfreq=sfreq, n_times=n_times,
                noise_rel=noise_rel, seed=seed,
            )

            # Train/test split (avoid leakage for PCA orientation)
            evoked_train = mne.EvokedArray(
                Y[:, :n_train], info, tmin=0.0, verbose=0
            )
            evoked_test = mne.EvokedArray(
                Y[:, n_train:], info, tmin=float(n_train) / sfreq, verbose=0
            )

            extra_kwargs = dict(alpha=alpha, epochs=1, n=2, k=2)
            if solver_name in MUSIC_SOLVERS:
                extra_kwargs["stop_crit"] = 0.1

            for ori in orientations:
                try:
                    solver = Solver(solver_name, orientation=ori)
                    solver.make_inverse_operator(fwd, evoked_train, **extra_kwargs)
                    stc = solver.apply_inverse_operator(evoked_test)
                    source_pred = np.sum(stc.data ** 2, axis=1)
                    emd_val = eval_emd(dist_matrix, x_true, source_pred)
                    emd_by_ori[ori].append(float(emd_val))
                except Exception as e:
                    emd_by_ori[ori].append(float("nan"))

        # Compute medians
        med = {ori: float(np.nanmedian(vals)) for ori, vals in emd_by_ori.items()}

        fixed_emd = med["fixed"]
        pca_emd = med["pca"]
        free_emd = med.get("free", float("nan"))

        pca_wins = pca_emd < fixed_emd
        free_wins = free_emd < fixed_emd if supports_free else None

        pca_label = "YES" if pca_wins else "no"
        free_label = "YES" if free_wins else ("n/a" if free_wins is None else "no")
        free_str = f"{free_emd:9.2f}" if supports_free else f"{'n/a':>9s}"

        print(f"{solver_name:<25s} {fixed_emd:8.2f} {pca_emd:8.2f} "
              f"{free_str}  {pca_label:>7s} {free_label:>8s}")

        results.append(dict(
            solver=solver_name,
            fixed=fixed_emd, pca=pca_emd, free=free_emd,
            pca_wins=pca_wins, free_wins=free_wins,
            supports_free=supports_free,
        ))

    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_total = len(results)
    n_pca_wins = sum(1 for r in results if r["pca_wins"])
    n_free_tested = sum(1 for r in results if r["supports_free"])
    n_free_wins = sum(1 for r in results if r["free_wins"] is True)

    print()
    print(f"Completed in {elapsed:.1f}s")
    print(f"PCA < Fixed:  {n_pca_wins}/{n_total} solvers")
    print(f"Free < Fixed: {n_free_wins}/{n_free_tested} solvers (of {n_free_tested} tested)")

    # List solvers where hypothesis fails
    pca_failures = [r["solver"] for r in results if not r["pca_wins"]]
    free_failures = [r["solver"] for r in results
                     if r["supports_free"] and r["free_wins"] is False]

    if pca_failures:
        print(f"\nPCA did NOT beat fixed for: {', '.join(pca_failures)}")
    if free_failures:
        print(f"Free did NOT beat fixed for: {', '.join(free_failures)}")
    if not pca_failures and not free_failures:
        print("\nHypothesis confirmed for all solvers.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
