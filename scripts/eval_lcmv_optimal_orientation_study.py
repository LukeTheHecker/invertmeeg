#!/usr/bin/env python
"""Study: LCMV orientation handling (fixed vs PCA vs free vs free+optimal collapse).

This is designed to stress exactly the failure mode of fixed-normal forward models:
when the true dipole is tangential (orthogonal to the cortical normal), fixed-normal
LCMV cannot represent it well.

We evaluate on held-out time samples to avoid leakage for PCA-based orientation.

Run:
  uv run --project invert-package --extra dev python scripts/eval_lcmv_optimal_orientation_study.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import mne
import numpy as np

from invert import Solver
from invert.forward import create_forward_model, get_info
from invert.solvers.orientation import ensure_surface_free_surf_ori


@dataclass(frozen=True)
class Scenario:
    name: str
    n_active: int
    correlated: bool
    orient: str  # "tangential" | "normal" | "mixed"


def _coords_mm(forward: mne.Forward) -> np.ndarray:
    src = forward["src"]
    if not (len(src) == 2 and all(s.get("type") == "surf" for s in src)):
        raise ValueError("Expected surface forward.")
    coords = []
    for hemi in src:
        vertno = np.asarray(hemi["vertno"], dtype=int)
        rr = np.asarray(hemi["rr"], dtype=float)
        coords.append(rr[vertno])
    return np.vstack(coords) * 1000.0


def _make_burst(
    *,
    sfreq: float,
    n_times: int,
    freq_hz: float,
    phase: float,
) -> np.ndarray:
    t = np.arange(n_times, dtype=float) / float(sfreq)
    x = np.sin(2.0 * np.pi * float(freq_hz) * t + float(phase))
    x *= np.hanning(n_times)
    return x.astype(float)


def _pick_active(rng: np.random.RandomState, n_src: int, n_active: int) -> np.ndarray:
    if n_active <= 0:
        raise ValueError("n_active must be > 0")
    if n_active > n_src:
        raise ValueError("n_active must be <= n_src")
    return np.asarray(rng.choice(n_src, n_active, replace=False), dtype=int)


def _simulate(
    *,
    forward_free_surf: mne.Forward,
    sfreq: float,
    n_times: int,
    scenario: Scenario,
    noise_rel: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Y, x_true, active_idx)."""
    rng = np.random.RandomState(int(seed))
    G = np.asarray(forward_free_surf["sol"]["data"], dtype=float)
    n_chans, n_cols = G.shape
    n_src = int(n_cols // 3)

    active = _pick_active(rng, n_src, scenario.n_active)

    # Orientation in surf_ori basis: (t1, t2, n).
    if scenario.orient == "tangential":
        q = np.array([1.0, 0.0, 0.0], dtype=float)
    elif scenario.orient == "normal":
        q = np.array([0.0, 0.0, 1.0], dtype=float)
    elif scenario.orient == "mixed":
        q = np.array([1.0, 0.0, 1.0], dtype=float)
        q /= np.linalg.norm(q)
    else:
        raise ValueError(f"Unknown orient {scenario.orient!r}")

    # Time courses.
    x_true = np.zeros((n_src, n_times), dtype=float)
    if scenario.correlated:
        base = _make_burst(
            sfreq=sfreq,
            n_times=n_times,
            freq_hz=float(rng.uniform(8.0, 20.0)),
            phase=float(rng.uniform(0.0, 2.0 * np.pi)),
        )
        for idx in active:
            x_true[idx] = float(rng.uniform(0.8, 1.2)) * base
    else:
        for idx in active:
            x_true[idx] = _make_burst(
                sfreq=sfreq,
                n_times=n_times,
                freq_hz=float(rng.uniform(8.0, 20.0)),
                phase=float(rng.uniform(0.0, 2.0 * np.pi)),
            )

    # Build J_true (3*n_src, T).
    J = np.zeros((3 * n_src, n_times), dtype=float)
    for idx in active:
        # q is fixed per scenario; scale/time course is x_true[idx].
        J[3 * idx + 0, :] = q[0] * x_true[idx]
        J[3 * idx + 1, :] = q[1] * x_true[idx]
        J[3 * idx + 2, :] = q[2] * x_true[idx]

    Y = G @ J
    if noise_rel > 0:
        sigma = float(noise_rel) * float(np.std(Y))
        if np.isfinite(sigma) and sigma > 0:
            Y = Y + rng.randn(n_chans, n_times) * sigma
    return Y, x_true, active


def _match_loc_error(coords_mm: np.ndarray, true_idx: np.ndarray, est_idx: np.ndarray) -> float:
    """Greedy matching for small k: mean distance between matched pairs."""
    true_idx = np.asarray(true_idx, dtype=int).ravel()
    est_idx = np.asarray(est_idx, dtype=int).ravel()
    if true_idx.size == 0 or est_idx.size == 0:
        return float("nan")

    remaining_true = list(true_idx.tolist())
    remaining_est = list(est_idx.tolist())
    dists = []
    while remaining_true and remaining_est:
        best = None
        best_pair = None
        for ti in remaining_true:
            for ei in remaining_est:
                d = float(np.linalg.norm(coords_mm[ti] - coords_mm[ei]))
                if best is None or d < best:
                    best = d
                    best_pair = (ti, ei)
        assert best_pair is not None
        ti, ei = best_pair
        remaining_true.remove(ti)
        remaining_est.remove(ei)
        dists.append(best)
    return float(np.mean(dists)) if dists else float("nan")


def _evaluate_stc(
    *,
    stc_data: np.ndarray,  # (n_src, T)
    coords_mm: np.ndarray,
    active: np.ndarray,
    top_k: int,
) -> dict:
    power = np.sum(np.asarray(stc_data, dtype=float) ** 2, axis=1)
    est_idx = np.argsort(power)[::-1][: int(max(1, top_k))]
    loc_err = _match_loc_error(coords_mm, active, est_idx)
    power_sum = float(np.sum(power))
    true_power = float(np.sum(power[active])) if active.size else 0.0
    frac = true_power / power_sum if power_sum > 0 else float("nan")
    return dict(loc_err_mm=float(loc_err), true_power_frac=float(frac))


def _run_variant(
    *,
    solver_name: str,
    fwd: mne.Forward,
    evoked_train: mne.Evoked,
    evoked_test: mne.Evoked,
    orientation: str,
    collapse: str,
    alpha: float,
    n_reg_params: int,
) -> np.ndarray:
    solver_name = str(solver_name).strip()
    kwargs = dict(orientation=orientation, n_reg_params=int(n_reg_params))
    if solver_name.lower() == "lcmv":
        kwargs["free_orientation_collapse"] = collapse
    solver = Solver(solver_name, **kwargs)
    solver.make_inverse_operator(fwd, evoked_train, alpha=float(alpha))
    stc = solver.apply_inverse_operator(evoked_test)
    return np.asarray(stc.data, dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver",
        type=str,
        default="lcmv",
        help="Beamformer solver name (e.g., lcmv, esmv-mvpure, lcmv-mvpure).",
    )
    parser.add_argument(
        "--montage",
        type=str,
        default="biosemi16",
        help="MNE standard montage name (e.g., biosemi64, standard_1020).",
    )
    parser.add_argument("--sampling", type=str, default="ico1")
    parser.add_argument("--sfreq", type=float, default=200.0)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--noise-rel", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--n-reg-params", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=25)
    args = parser.parse_args()

    sfreq = float(args.sfreq)
    n_times = int(max(20, round(float(args.duration) * sfreq)))
    frac = float(args.train_frac)
    if not (0.05 <= frac <= 0.95):
        raise SystemExit("--train-frac must be in [0.05, 0.95]")
    n_train = int(max(2, min(n_times - 2, round(frac * n_times))))

    info = get_info(kind=str(args.montage), sfreq=sfreq)
    fwd = create_forward_model(info=info, sampling=args.sampling, fixed_ori=False)
    fwd = ensure_surface_free_surf_ori(fwd)
    coords = _coords_mm(fwd)

    scenarios = [
        Scenario(name="single_tangential", n_active=1, correlated=False, orient="tangential"),
        Scenario(name="single_normal", n_active=1, correlated=False, orient="normal"),
        Scenario(name="two_uncorr_tangential", n_active=2, correlated=False, orient="tangential"),
        Scenario(name="two_corr_tangential", n_active=2, correlated=True, orient="tangential"),
        Scenario(name="two_uncorr_mixed", n_active=2, correlated=False, orient="mixed"),
    ]

    solver_name = str(args.solver).strip()
    variants = [
        ("fixed", "amplitude"),
        ("pca", "amplitude"),
        ("free", "amplitude"),
    ]
    if solver_name.lower() == "lcmv":
        variants.append(("free", "optimal"))

    print(
        "Beamformer orientation study (metrics evaluated on held-out test segment)\n"
        f"solver={solver_name}\n"
        f"sampling={args.sampling}  sfreq={sfreq:g}  n_times={n_times}  train={n_train}  "
        f"noise_rel={float(args.noise_rel):g}  alpha={float(args.alpha):g}\n"
        f"variants: {', '.join([f'{o}({c})' for o, c in variants])}\n"
    )

    for sc in scenarios:
        rows = {v: [] for v in variants}
        for seed in range(int(args.seeds)):
            Y, _x_true, active = _simulate(
                forward_free_surf=fwd,
                sfreq=sfreq,
                n_times=n_times,
                scenario=sc,
                noise_rel=float(args.noise_rel),
                seed=seed,
            )
            evoked_train = mne.EvokedArray(Y[:, :n_train], info, tmin=0.0, verbose=0)
            evoked_test = mne.EvokedArray(
                Y[:, n_train:],
                info,
                tmin=float(n_train) / sfreq,
                verbose=0,
            )

            for v in variants:
                orientation, collapse = v
                stc = _run_variant(
                    solver_name=solver_name,
                    fwd=fwd,
                    evoked_train=evoked_train,
                    evoked_test=evoked_test,
                    orientation=orientation,
                    collapse=collapse,
                    alpha=float(args.alpha),
                    n_reg_params=int(args.n_reg_params),
                )
                rows[v].append(
                    _evaluate_stc(
                        stc_data=stc,
                        coords_mm=coords,
                        active=active,
                        top_k=sc.n_active,
                    )
                )

        print(f"Scenario: {sc.name} (k={sc.n_active}, correlated={sc.correlated}, orient={sc.orient})")
        for v in variants:
            items = rows[v]
            loc = np.array([it["loc_err_mm"] for it in items], dtype=float)
            frac = np.array([it["true_power_frac"] for it in items], dtype=float)
            loc_med = float(np.nanmedian(loc))
            frac_med = float(np.nanmedian(frac))
            label = f"{v[0]}({v[1]})"
            print(f"  {label:<15s} loc_err_mm median={loc_med:6.1f}  true_power_frac median={frac_med:6.3f}")
        print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
