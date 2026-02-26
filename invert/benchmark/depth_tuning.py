from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from functools import cache
from typing import Any

from invert.solver_registry import (
    get_solver_class,
    get_solver_spec,
    iter_solver_specs,
)

RECOMMENDED_DEPTH_BY_CATEGORY: dict[str, float] = {
    "minimum_norm": 0.5,
    "loreta": 0.0,
    "bayesian": 0.0,
    "music": 0.5,
}

RECOMMENDED_DEPTH_BY_SOLVER: dict[str, float] = {
    "backus-gilbert": 0.0,
    "gbf": 1.0,
    "gft-l1": 0.0,
    "mne": 1.0,
    "s-map": 0.0,
    "tv": 0.0,
}


def _infer_category(module_path: str, solver_id: str) -> str:
    parts = module_path.split(".")
    if len(parts) < 3 or parts[0] != "invert" or parts[1] != "solvers":
        return "other"
    directory = parts[2]
    if directory == "beamformers":
        return "beamformer"
    if directory == "bayesian":
        return "bayesian"
    if directory == "minimum_norm":
        if "loreta" in solver_id:
            return "loreta"
        return "minimum_norm"
    if directory == "music":
        return "music"
    if directory == "matching_pursuit":
        return "matching_pursuit"
    if directory == "neural_networks":
        return "neural_networks"
    return "other"


@cache
def infer_depth_strategy(solver_name: str) -> str | None:
    """Infer how depth should be set for a solver.

    Returns one of:
    - ``"base"``: BaseSolver depth weighting (``use_depth_weighting`` + ``depth_weighting``)
    - ``"dspm-depth"``: dSPM-style depth prior via ``depth`` with ``use_depth_weighting=True``
    - ``"laura-depth"``: LAURA-specific ``depth_weight``
    - ``"make-depth-weighting"``: make-time ``depth_weighting`` argument
    - ``None``: no detectable depth control
    """
    spec = get_solver_spec(solver_name)
    solver_cls = get_solver_class(spec.solver_id)

    init_sig = inspect.signature(solver_cls.__init__)
    make_sig = inspect.signature(solver_cls.make_inverse_operator)
    init_params = init_sig.parameters
    make_params = make_sig.parameters

    if "depth_weight" in init_params:
        return "laura-depth"
    if "depth" in init_params:
        return "dspm-depth"
    if "use_depth_weighting" in init_params or "depth_weighting" in init_params:
        return "base"

    accepts_init_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in init_params.values()
    )
    if accepts_init_kwargs:
        try:
            solver = solver_cls(**dict(spec.default_kwargs))
        except Exception:
            solver = None

        if solver is not None and bool(getattr(solver, "prep_leadfield", True)):
            return "base"

    if "depth_weighting" in make_params:
        return "make-depth-weighting"
    return None


def build_depth_params_for_solver(solver_name: str, depth: float) -> dict[str, Any]:
    """Build benchmark ``solver_params`` entries for a solver at one depth."""
    strategy = infer_depth_strategy(solver_name)
    if strategy is None:
        return {}
    if strategy == "laura-depth":
        return {"init__depth_weight": float(depth)}
    if strategy == "dspm-depth":
        return {"init__use_depth_weighting": True, "init__depth": float(depth)}
    if strategy == "make-depth-weighting":
        return {"make__depth_weighting": float(depth)}
    return {"init__use_depth_weighting": True, "init__depth_weighting": float(depth)}


def scan_depth_effective_solvers(
    *,
    solver_names: Sequence[str] | None = None,
    include_unavailable: bool = False,
) -> dict[str, dict[str, str]]:
    """Return solver -> metadata for solvers where depth has a detectable effect."""
    if solver_names is None:
        names = [
            spec.solver_id
            for spec in iter_solver_specs(
                include_internal=False,
                include_unavailable=include_unavailable,
            )
        ]
    else:
        names = [get_solver_spec(name).solver_id for name in solver_names]

    out: dict[str, dict[str, str]] = {}
    for solver_name in names:
        strategy = infer_depth_strategy(solver_name)
        if strategy is None:
            continue
        spec = get_solver_spec(solver_name)
        out[solver_name] = {
            "category": _infer_category(spec.module_path, spec.solver_id),
            "strategy": strategy,
            "module_path": spec.module_path,
        }
    return out


def build_depth_solver_params(
    *,
    solvers: Sequence[str],
    depth_by_category: Mapping[str, float] | None,
    depth_by_solver: Mapping[str, float] | None = None,
    base_solver_params: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compose per-solver depth params from category+solver defaults + overrides.

    Precedence:
    1) category depth defaults
    2) per-solver depth overrides
    3) explicit keys in ``base_solver_params``
    """
    merged: dict[str, dict[str, Any]] = {}
    selected_solver_ids = {get_solver_spec(name).solver_id for name in solvers}

    if depth_by_category:
        for solver_name in solvers:
            sid = get_solver_spec(solver_name).solver_id
            spec = get_solver_spec(sid)
            category = _infer_category(spec.module_path, sid)
            if category not in depth_by_category:
                continue
            depth = float(depth_by_category[category])
            depth_params = build_depth_params_for_solver(sid, depth)
            if depth_params:
                merged.setdefault(sid, {}).update(depth_params)

    if depth_by_solver:
        for solver_name, depth in depth_by_solver.items():
            sid = get_solver_spec(solver_name).solver_id
            if sid not in selected_solver_ids:
                continue
            depth_params = build_depth_params_for_solver(sid, float(depth))
            if depth_params:
                merged.setdefault(sid, {}).update(depth_params)

    if base_solver_params:
        for solver_name, params in base_solver_params.items():
            sid = get_solver_spec(solver_name).solver_id
            merged.setdefault(sid, {}).update(dict(params))

    return merged
