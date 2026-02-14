#!/usr/bin/env python
"""Benchmark free-orientation support vs fixed-normal (orthogonal) forward models.

Design:
- Build a small *surface* free-orientation forward model (ico1).
- Convert to surf_ori=True so the 3 columns per vertex are (tangent, tangent, normal).
- Simulate a single focal *tangential* source. This is representable in free orientation,
  but poorly representable with a fixed-normal (orthogonal-to-cortex) model.
- To avoid optimistic bias for PCA orientation estimation, split time into
  train/test segments:
  - Train is used only to estimate q (orientation="pca") and build the operator.
  - Test is used for evaluation (metrics are computed on test only).
- For each solver:
  - "free" condition:
    - vector-capable solvers: orientation="free" (true 3-component model)
    - scalar solvers: orientation="pca" (data-driven reduction free->scalar)
  - "fixed" condition: orientation="fixed" (cortical normal)

Metrics:
- rel_resid: relative residual in the solver's operator space (after projector/whitener).
  Note: for standardized solvers (e.g., sLORETA), this is not always a meaningful
  “forward fit” metric, so assertions focus on source-space metrics.
- loc_err_mm: distance between argmax source power and the true source location.
- corr_at_true: correlation between estimated and true time course at the true vertex
  (amplitude-only for vector estimates).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import mne
import numpy as np

from invert import Solver, list_solvers
from invert.forward import create_forward_model, get_info
from invert.solvers.orientation import ensure_surface_free_surf_ori


@dataclass(frozen=True)
class SimSpec:
    true_idx: int
    x_true: np.ndarray  # (T,)
    y: np.ndarray  # (n_chans, T)


def _coords_from_surface_forward(forward: mne.Forward) -> np.ndarray:
    """Return source coordinates (n_sources, 3) in millimeters for surface forwards."""
    src = forward["src"]
    if not (len(src) == 2 and all(s.get("type") == "surf" for s in src)):
        raise ValueError("Expected a surface forward (two hemis, type='surf').")
    coords = []
    for hemi in src:
        vertno = np.asarray(hemi["vertno"], dtype=int)
        rr = np.asarray(hemi["rr"], dtype=float)
        coords.append(rr[vertno])
    return np.vstack(coords) * 1000.0


def _simulate_single_tangential_source(
    forward_free_surf: mne.Forward,
    *,
    true_idx: int,
    sfreq: float,
    duration_s: float,
    freq_hz: float,
    noise_rel: float,
    seed: int,
) -> SimSpec:
    """Simulate Y = G_free J_true with a single tangential component active."""
    rng = np.random.RandomState(int(seed))
    G = np.asarray(forward_free_surf["sol"]["data"], dtype=float)
    n_chans, n_cols = G.shape
    if n_cols % 3 != 0:
        raise ValueError(f"Expected free-orientation leadfield with 3*n cols, got {n_cols}")
    n_src = n_cols // 3
    if not (0 <= true_idx < n_src):
        raise ValueError(f"true_idx must be in [0, {n_src-1}], got {true_idx}")

    n_times = int(round(float(duration_s) * float(sfreq)))
    n_times = max(n_times, 5)
    t = np.arange(n_times, dtype=float) / float(sfreq)

    # Smooth, narrowband burst.
    x = np.sin(2.0 * np.pi * float(freq_hz) * t)
    win = np.hanning(n_times)
    x = (x * win).astype(float)

    J = np.zeros((3 * n_src, n_times), dtype=float)
    # Tangential component (index 0 in (t1, t2, n) basis when surf_ori=True).
    J[3 * true_idx + 0, :] = x

    Y = G @ J
    if noise_rel > 0:
        sigma = float(noise_rel) * float(np.std(Y))
        if np.isfinite(sigma) and sigma > 0:
            Y = Y + rng.randn(*Y.shape) * sigma

    return SimSpec(true_idx=int(true_idx), x_true=x, y=Y)


def _corr_abs(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size != b.size or a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return float("nan")
    return float(abs(np.dot(a, b) / denom))


def _run_one(
    *,
    solver,
    evoked: mne.Evoked,
    coords_mm: np.ndarray,
    true_idx: int,
    x_true: np.ndarray,
    want_vector: bool,
) -> dict:
    data = solver.unpack_data_obj(evoked)  # (n_chans, T) in original sensor space
    sensor_transform = getattr(solver, "_sensor_transform", None)
    if sensor_transform is None:
        Y_eff = np.asarray(data, dtype=float)
    else:
        Y_eff = np.asarray(sensor_transform @ np.asarray(data, dtype=float), dtype=float)

    if want_vector:
        stc_vec = solver.apply_inverse_operator_vector(evoked)
        vec = np.asarray(stc_vec.data, dtype=float)  # (n_src, 3, T)
        n_src, n_orient, n_times = vec.shape
        assert n_orient == 3
        J_hat = vec.reshape(n_src * 3, n_times)
        power = np.sum(vec**2, axis=(1, 2))
        x_hat_true = np.linalg.norm(vec[true_idx, :, :], axis=0)
        x_target = np.abs(np.asarray(x_true, dtype=float))
        corr_at_true = _corr_abs(x_hat_true, x_target)
        Y_hat_eff = np.asarray(solver.leadfield @ J_hat, dtype=float)
    else:
        stc = solver.apply_inverse_operator(evoked)
        X_hat = np.asarray(stc.data, dtype=float)  # (n_src, T)
        power = np.sum(X_hat**2, axis=1)
        corr_at_true = _corr_abs(X_hat[true_idx, :], x_true)
        Y_hat_eff = np.asarray(solver.leadfield @ X_hat, dtype=float)

    est_idx = int(np.argmax(power))
    loc_err_mm = float(np.linalg.norm(coords_mm[est_idx] - coords_mm[true_idx]))

    resid = float(np.linalg.norm(Y_eff - Y_hat_eff))
    denom = float(np.linalg.norm(Y_eff))
    rel_resid = resid / max(denom, np.finfo(float).eps)

    power_sum = float(np.sum(power))
    true_frac = float(power[true_idx] / power_sum) if power_sum > 0 else float("nan")

    out = dict(
        solver=getattr(solver, "name", solver.__class__.__name__),
        orientation=getattr(solver, "orientation", "unknown"),
        want_vector=bool(want_vector),
        n_src=int(coords_mm.shape[0]),
        rel_resid=float(rel_resid),
        loc_err_mm=float(loc_err_mm),
        corr_at_true=float(corr_at_true),
        true_power_frac=float(true_frac),
    )

    # If PCA reduction was used, report alignment for the true location if available.
    q = getattr(solver, "_orientation_q", None)
    if isinstance(q, np.ndarray) and q.ndim == 2 and q.shape[1] == 3 and q.shape[0] > true_idx:
        # True orientation is tangent1 => [1,0,0] in surf_ori basis.
        out["pca_q_t1_align"] = float(abs(q[true_idx, 0]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solvers",
        type=str,
        default="champagne,sloreta,mne",
        help="Comma-separated solver names (as accepted by invert.Solver).",
    )
    parser.add_argument("--sampling", type=str, default="ico1")
    parser.add_argument("--sfreq", type=float, default=200.0)
    parser.add_argument("--duration", type=float, default=0.4)
    parser.add_argument("--freq", type=float, default=12.0)
    parser.add_argument("--noise-rel", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--n-reg-params", type=int, default=1)
    parser.add_argument("--true-idx", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.5,
        help="Fraction of time samples used for PCA orientation estimation (train).",
    )
    parser.add_argument(
        "--assert-improves",
        action="store_true",
        help=(
            "Exit nonzero if the free condition does not improve source-space metrics "
            "(either lower loc_err_mm or higher true_power_frac) vs fixed."
        ),
    )
    args = parser.parse_args()

    solver_names = [s.strip() for s in str(args.solvers).split(",") if s.strip()]
    known = set(list_solvers())
    solver_names = [s for s in solver_names if s.lower() in known]
    if not solver_names:
        raise SystemExit("No valid solvers selected.")

    info = get_info(kind="biosemi16", sfreq=float(args.sfreq))
    info = info.copy()

    fwd_free = create_forward_model(info=info, sampling=args.sampling, fixed_ori=False)
    fwd_free = ensure_surface_free_surf_ori(fwd_free)
    G_free = np.asarray(fwd_free["sol"]["data"], dtype=float)
    if G_free.shape[1] != int(fwd_free["nsource"]) * 3:
        raise RuntimeError("Unexpected leadfield shape for free surface forward.")

    coords_mm = _coords_from_surface_forward(fwd_free)
    sim = _simulate_single_tangential_source(
        fwd_free,
        true_idx=int(args.true_idx),
        sfreq=float(args.sfreq),
        duration_s=float(args.duration),
        freq_hz=float(args.freq),
        noise_rel=float(args.noise_rel),
        seed=int(args.seed),
    )

    # --- Train/test split to avoid using evaluation data to estimate PCA orientation.
    Y = np.asarray(sim.y, dtype=float)
    x_true = np.asarray(sim.x_true, dtype=float)
    n_times = int(Y.shape[1])
    frac = float(args.train_frac)
    if not (0.05 <= frac <= 0.95):
        raise SystemExit("--train-frac must be in [0.05, 0.95]")
    n_train = int(max(2, min(n_times - 2, round(frac * n_times))))
    Y_train, Y_test = Y[:, :n_train], Y[:, n_train:]
    x_train, x_test = x_true[:n_train], x_true[n_train:]

    evoked_train = mne.EvokedArray(Y_train, info, tmin=0.0, verbose=0)
    evoked_test = mne.EvokedArray(Y_test, info, tmin=float(n_train) / float(args.sfreq), verbose=0)

    # Small per-solver kwargs to keep runtime down and avoid signature mismatches.
    extra_by_solver = {
        "champagne": dict(max_iter=200, prune=True, convergence_criterion=1e-8),
    }

    rows: list[dict] = []
    for s in solver_names:
        key = s.lower()
        extra = dict(extra_by_solver.get(key, {}))

        # Fixed-normal (orthogonal-to-cortex) baseline: build on train, evaluate on test.
        solver_fixed = Solver(s, orientation="fixed", n_reg_params=int(args.n_reg_params))
        solver_fixed.make_inverse_operator(
            fwd_free, evoked_train, alpha=float(args.alpha), **extra
        )
        fixed = _run_one(
            solver=solver_fixed,
            evoked=evoked_test,
            coords_mm=coords_mm,
            true_idx=sim.true_idx,
            x_true=x_test,
            want_vector=False,
        )

        # Free condition: true free for vector solvers, PCA-reduced for scalar solvers.
        supports_vector = bool(
            getattr(Solver(s), "SUPPORTS_VECTOR_ORIENTATION", False)  # instantiate cheaply
        )
        if supports_vector:
            solver_free = Solver(s, orientation="free", n_reg_params=int(args.n_reg_params))
            solver_free.make_inverse_operator(
                fwd_free, evoked_train, alpha=float(args.alpha), **extra
            )
            free = _run_one(
                solver=solver_free,
                evoked=evoked_test,
                coords_mm=coords_mm,
                true_idx=sim.true_idx,
                x_true=x_test,
                want_vector=True,
            )
        else:
            solver_pca = Solver(s, orientation="pca", n_reg_params=int(args.n_reg_params))
            solver_pca.make_inverse_operator(
                fwd_free, evoked_train, alpha=float(args.alpha), **extra
            )
            free = _run_one(
                solver=solver_pca,
                evoked=evoked_test,
                coords_mm=coords_mm,
                true_idx=sim.true_idx,
                x_true=x_test,
                want_vector=False,
            )

        rows.extend([fixed, free])

        ratio = float(free["rel_resid"] / max(fixed["rel_resid"], np.finfo(float).eps))
        print(
            f"{s:12s}  rel_resid fixed={fixed['rel_resid']:.4f}  free={free['rel_resid']:.4f}  ratio={ratio:.3f}  "
            f"loc_err_mm fixed={fixed['loc_err_mm']:.1f} free={free['loc_err_mm']:.1f}  "
            f"corr_at_true fixed={fixed['corr_at_true']:.3f} free={free['corr_at_true']:.3f}"
        )
        if args.assert_improves:
            loc_improves = bool(free["loc_err_mm"] < fixed["loc_err_mm"])
            pow_improves = bool(free["true_power_frac"] > fixed["true_power_frac"])
            if not (loc_improves or pow_improves):
                print(
                    f"ASSERT FAILED: {s} did not improve source-space metrics "
                    f"(loc_err_mm {fixed['loc_err_mm']:.1f}->{free['loc_err_mm']:.1f}, "
                    f"true_power_frac {fixed['true_power_frac']:.3f}->{free['true_power_frac']:.3f})."
                )
                return 2

    # Compact summary table.
    print("\nSummary (lower is better):")
    print("solver        condition   rel_resid   loc_err_mm   corr_at_true   true_power_frac")
    for r in rows:
        cond = "fixed" if r["orientation"] == "fixed" else ("free" if r["orientation"] == "free" else "pca")
        print(
            f"{r['solver']:<12s} {cond:<10s} {r['rel_resid']:>8.4f} {r['loc_err_mm']:>11.1f} "
            f"{r['corr_at_true']:>12.3f} {r['true_power_frac']:>15.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
