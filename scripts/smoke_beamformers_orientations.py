#!/usr/bin/env python
"""Smoke-test all beamformer solvers for orientation modes: fixed, pca, free.

This is a *runtime* verification script (not a pytest) meant to catch:
- beamformer solvers that forgot to pass mne_obj into BaseSolver.make_inverse_operator
  (breaks orientation='pca')
- solvers that assume fixed-orientation leadfields internally (breaks orientation='free')

Run:
  uv run --project invert-package --extra dev python scripts/smoke_beamformers_orientations.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import mne
import numpy as np

from invert import Solver
from invert.invert import _get_registry
from invert.forward import create_forward_model, get_info
from invert.solvers.orientation import ensure_surface_free_surf_ori


@dataclass(frozen=True)
class Case:
    key: str
    cls: type


def _unique_beamformer_cases() -> list[Case]:
    reg = _get_registry()
    seen: set[type] = set()
    cases: list[Case] = []
    for key, ctor in reg.items():
        cls = ctor
        if not isinstance(cls, type):
            continue
        if ".beamformers." not in str(getattr(cls, "__module__", "")):
            continue
        if cls in seen:
            continue
        seen.add(cls)
        cases.append(Case(key=key, cls=cls))
    cases.sort(key=lambda c: c.key)
    return cases


def _simulate_tangential(fwd_free_surf: mne.Forward, sfreq: float, n_times: int) -> np.ndarray:
    G = np.asarray(fwd_free_surf["sol"]["data"], dtype=float)
    n_src = int(fwd_free_surf["nsource"])
    T = int(n_times)
    true_idx = min(10, n_src - 1)
    t = np.arange(T, dtype=float) / float(sfreq)
    x = np.sin(2.0 * np.pi * 12.0 * t) * np.hanning(T)
    J = np.zeros((3 * n_src, T), dtype=float)
    J[3 * true_idx + 0, :] = x  # tangential1 in surf_ori basis
    return G @ J


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--montage", type=str, default="biosemi16")
    parser.add_argument("--sampling", type=str, default="ico1")
    parser.add_argument("--sfreq", type=float, default=200.0)
    parser.add_argument("--n-times", type=int, default=120)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    info = get_info(kind=str(args.montage), sfreq=float(args.sfreq))
    fwd = create_forward_model(info=info, sampling=str(args.sampling), fixed_ori=False)
    fwd = ensure_surface_free_surf_ori(fwd)
    Y = _simulate_tangential(fwd, sfreq=float(args.sfreq), n_times=int(args.n_times))
    n_train = int(Y.shape[1] // 2)
    evoked_train = mne.EvokedArray(Y[:, :n_train], info, tmin=0.0, verbose=0)
    evoked_test = mne.EvokedArray(
        Y[:, n_train:], info, tmin=float(n_train) / float(args.sfreq), verbose=0
    )

    cases = _unique_beamformer_cases()
    if not cases:
        print("No beamformer solvers found.")
        return 0

    modes = ["fixed", "pca", "free"]
    failures: list[str] = []

    for case in cases:
        for mode in modes:
            try:
                solver = Solver(case.key, orientation=mode, n_reg_params=1)
                solver.make_inverse_operator(fwd, evoked_train, alpha=float(args.alpha))
                stc = solver.apply_inverse_operator(evoked_test)
                data = np.asarray(stc.data, dtype=float)
                if data.ndim != 2 or data.shape[1] != evoked_test.data.shape[1]:
                    raise RuntimeError(f"unexpected stc shape {data.shape}")
                if not np.isfinite(data).all():
                    raise RuntimeError("output contains NaN/Inf")
            except Exception as e:
                failures.append(f"{case.key} mode={mode}: {type(e).__name__}: {e}")
                break
        print(f"{case.key}: OK" if not any(f.startswith(f"{case.key} ") for f in failures) else f"{case.key}: FAIL")

    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  {f}")
        return 2

    print(f"\nAll {len(cases)} beamformer solvers passed modes={modes}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

