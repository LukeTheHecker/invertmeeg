"""Generate benchmark dashboard assets for MkDocs.

This script:
- scans results/release/*.json first, then falls back to results/benchmark_results*.json
- sanitizes NaN -> null (strict JSON)
- adds an aggregate dataset "all" by averaging per-dataset mean/median
- emits:
    - assets/benchmarks/manifest.json
    - assets/benchmarks/<run-id>.json
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mkdocs_gen_files

_RE_NAN = re.compile(r"\bNaN\b")

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("MNE_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "mne-config"))

FOLDER_TO_CATEGORY: dict[str, str] = {
    "minimum_norm": "Minimum Norm",
    "bayesian": "Bayesian",
    "beamformers": "Beamformers",
    "music": "Subspace Methods",
    "matching_pursuit": "Matching Pursuit",
    "dipoles": "Dipole Fitting",
    "neural_networks": "Neural Networks",
    "hybrids": "Hybrid",
}


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    source_path: Path
    timestamp: str | None
    n_samples: int | None
    datasets: list[str]
    metrics: list[str]
    name: str | None
    description: str | None


def _read_json_sanitized(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = _RE_NAN.sub("null", raw)
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict JSON in {path}, got {type(obj).__name__}")
    return obj


def _list_runs(results_dir: Path) -> list[Path]:
    release_dir = results_dir / "release"
    if release_dir.is_dir():
        release_runs = sorted(release_dir.glob("*.json"))
        if release_runs:
            return release_runs
    # Legacy fallback
    return sorted(results_dir.glob("benchmark_results*.json"))


def _get_default_run_id(paths: list[Path]) -> str | None:
    # Prefer the "leaderboard" stem
    for p in paths:
        if p.stem == "leaderboard":
            return p.stem
    for p in paths:
        if p.name == "benchmark_results.json":
            return p.stem
    return paths[0].stem if paths else None


def _compute_all_dataset_rows(data: dict[str, Any]) -> None:
    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError("Expected top-level 'results' list")

    # Group by solver and accumulate per-metric stats across datasets.
    acc: dict[tuple[str, str, str], list[dict[str, float | None]]] = {}
    meta_by_solver: dict[str, dict[str, Any]] = {}

    for row in results:
        if not isinstance(row, dict):
            continue
        solver = row.get("solver_name")
        dataset = row.get("dataset_name")
        category = row.get("category") or "other"
        metrics = row.get("metrics") or {}

        if not isinstance(solver, str) or not isinstance(dataset, str):
            continue
        if dataset == "all":
            continue
        if not isinstance(metrics, dict):
            continue

        meta_by_solver.setdefault(
            solver,
            {
                "solver_name": solver,
                "category": category,
            },
        )

        for metric_name, stats in metrics.items():
            if not isinstance(metric_name, str) or not isinstance(stats, dict):
                continue
            key = (solver, category, metric_name)
            acc.setdefault(key, []).append(
                {
                    "mean": stats.get("mean"),
                    "median": stats.get("median"),
                }
            )

    all_rows: list[dict[str, Any]] = []
    for (solver, category, metric_name), stats_list in acc.items():
        means = [s.get("mean") for s in stats_list if isinstance(s.get("mean"), (int, float))]
        medians = [
            s.get("median") for s in stats_list if isinstance(s.get("median"), (int, float))
        ]
        if not means and not medians:
            continue

        mean_mean = sum(means) / len(means) if means else None
        median_mean = sum(medians) / len(medians) if medians else None

        # We'll assemble full per-solver rows below; here we aggregate per metric.
        # Use std=None explicitly to avoid implying a statistical combination.
        meta_by_solver.setdefault(
            solver,
            {
                "solver_name": solver,
                "category": category,
            },
        )
        meta_by_solver[solver].setdefault("metrics", {})
        meta_by_solver[solver]["metrics"][metric_name] = {
            "mean": mean_mean,
            "median": median_mean,
            "std": None,
        }

    for solver, base in meta_by_solver.items():
        metrics = base.get("metrics") or {}
        if not metrics:
            continue
        all_rows.append(
            {
                "solver_name": solver,
                "dataset_name": "all",
                "category": base.get("category") or "other",
                "metrics": metrics,
                "samples": [],
            }
        )

    if all_rows:
        results.extend(all_rows)

    # Provide a dataset-rank mapping for "all" that mirrors global_ranks if available.
    ranks = data.get("ranks")
    global_ranks = data.get("global_ranks")
    if isinstance(ranks, dict) and isinstance(global_ranks, dict):
        ranks.setdefault("all", global_ranks)

def _ensure_dataset_configs(data: dict[str, Any]) -> None:
    """Backfill dataset configs for older benchmark_results*.json files."""
    existing = data.get("datasets")
    if isinstance(existing, dict) and existing:
        return
    try:
        repo_root = Path(__file__).resolve().parents[2]
        ds_path = repo_root / "invert" / "benchmark" / "datasets.py"
        spec = importlib.util.spec_from_file_location("invert.benchmark._datasets", str(ds_path))
        if spec is None or spec.loader is None:
            return
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        datasets = getattr(mod, "BENCHMARK_DATASETS", None)
        if isinstance(datasets, dict) and datasets:
            data["datasets"] = {k: v.model_dump() for k, v in datasets.items()}
    except Exception:
        # Non-fatal: the dashboard can operate without dataset config metadata.
        return


def _extract_run_info(run_id: str, data: dict[str, Any], source_path: Path) -> RunInfo:
    results = data.get("results") or []
    datasets = sorted({r.get("dataset_name") for r in results if isinstance(r, dict) and r.get("dataset_name")})
    if "all" not in datasets:
        datasets.append("all")
    metrics = sorted(
        {
            metric
            for r in results
            if isinstance(r, dict)
            for metric in (r.get("metrics") or {}).keys()
            if isinstance(metric, str)
        }
    )
    md = data.get("metadata") or {}
    ts = md.get("timestamp") if isinstance(md, dict) else None
    n_samples = md.get("n_samples") if isinstance(md, dict) else None
    run_name = md.get("name") if isinstance(md, dict) else None
    run_description = md.get("description") if isinstance(md, dict) else None
    return RunInfo(
        run_id=run_id,
        source_path=source_path,
        timestamp=ts if isinstance(ts, str) else None,
        n_samples=n_samples if isinstance(n_samples, int) else None,
        datasets=datasets,
        metrics=metrics,
        name=run_name if isinstance(run_name, str) else None,
        description=run_description if isinstance(run_description, str) else None,
    )

def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value or "solver"


def _literal_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Str):
        return node.s
    return None


def _extract_meta_from_class(node: ast.ClassDef) -> dict[str, Any] | None:
    for stmt in node.body:
        target = None
        value = None
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            value = stmt.value
        elif isinstance(stmt, ast.AnnAssign):
            target = stmt.target
            value = stmt.value
        if not isinstance(target, ast.Name) or target.id != "meta":
            continue
        if not isinstance(value, ast.Call):
            continue
        func = value.func
        func_name = None
        if isinstance(func, ast.Name):
            func_name = func.id
        elif isinstance(func, ast.Attribute):
            func_name = func.attr
        if func_name != "SolverMeta":
            continue

        meta: dict[str, Any] = {}
        for kw in value.keywords:
            if not kw.arg:
                continue
            if kw.arg in {"slug", "acronym", "full_name", "category", "internal"}:
                meta[kw.arg] = _literal_value(kw.value)
        return meta
    return None


def _extract_meta_from_file(path: Path) -> dict[str, dict[str, Any]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    found: dict[str, dict[str, Any]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        meta = _extract_meta_from_class(node)
        if meta:
            found[node.name] = meta
    return found


def _build_solver_pages_map(only_solvers: set[str] | None = None) -> dict[str, str]:
    """Return mapping: benchmark solver short name -> docs-relative solver page URL."""
    repo_root = Path(__file__).resolve().parents[2]
    runner_path = repo_root / "invert" / "benchmark" / "runner.py"
    try:
        runner_tree = ast.parse(runner_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    solver_registry: dict[str, tuple[str, str]] = {}
    for node in runner_tree.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "_SOLVER_REGISTRY" not in targets:
            continue
        try:
            literal = ast.literal_eval(node.value)
        except Exception:
            continue
        if not isinstance(literal, dict):
            continue
        for key, value in literal.items():
            if not isinstance(key, str):
                continue
            if (
                isinstance(value, tuple)
                and len(value) == 2
                and isinstance(value[0], str)
                and isinstance(value[1], str)
            ):
                solver_registry[key] = (value[0], value[1])
        break

    if not solver_registry:
        return {}

    repo_root = Path(__file__).resolve().parents[2]
    file_cache: dict[Path, dict[str, dict[str, Any]]] = {}
    mapping: dict[str, str] = {}

    for solver_name, (module_path, class_name) in solver_registry.items():
        if only_solvers is not None and solver_name not in only_solvers:
            continue
        if "._old" in module_path or module_path.endswith("._old"):
            continue

        parts = module_path.split(".")
        module_fs_path = repo_root / Path(*parts)
        candidate_files: list[Path] = []

        py_path = module_fs_path.with_suffix(".py")
        if py_path.exists():
            candidate_files.append(py_path)
        elif module_fs_path.is_dir():
            candidate_files.extend(sorted(module_fs_path.glob("*.py")))

        meta_info = None
        for file_path in candidate_files:
            if file_path not in file_cache:
                file_cache[file_path] = _extract_meta_from_file(file_path)
            meta_info = file_cache[file_path].get(class_name)
            if meta_info:
                break

        if not meta_info:
            continue
        if meta_info.get("internal"):
            continue

        category = meta_info.get("category")
        if not isinstance(category, str) or not category:
            folder = ""
            if len(parts) >= 3 and parts[0] == "invert" and parts[1] == "solvers":
                folder = parts[2]
            category = FOLDER_TO_CATEGORY.get(folder, "Baseline")
        cat_slug = _slugify(category)

        solver_slug = meta_info.get("slug")
        if not isinstance(solver_slug, str) or not solver_slug:
            solver_slug = _slugify(
                meta_info.get("acronym") or meta_info.get("full_name") or solver_name
            )

        mapping[solver_name] = f"solvers/{cat_slug}/{solver_slug}/"

    return mapping


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    results_dir = repo_root / "results"
    if not results_dir.exists():
        raise RuntimeError(f"Missing results directory: {results_dir}")

    run_paths = _list_runs(results_dir)
    if not run_paths:
        raise RuntimeError(f"No benchmark results found in {results_dir}")

    default_run_id = _get_default_run_id(run_paths)
    manifest: dict[str, Any] = {"default_run": default_run_id, "runs": []}
    solvers_in_runs: set[str] = set()

    for path in run_paths:
        run_id = path.stem
        data = _read_json_sanitized(path)
        _compute_all_dataset_rows(data)
        _ensure_dataset_configs(data)
        for row in data.get("results") or []:
            if isinstance(row, dict) and isinstance(row.get("solver_name"), str):
                solvers_in_runs.add(row["solver_name"])

        info = _extract_run_info(run_id=run_id, data=data, source_path=path)
        manifest["runs"].append(
            {
                "id": info.run_id,
                "filename": f"{info.run_id}.json",
                "timestamp": info.timestamp,
                "n_samples": info.n_samples,
                "datasets": info.datasets,
                "metrics": info.metrics,
                "name": info.name,
                "description": info.description,
            }
        )

        with mkdocs_gen_files.open(f"assets/benchmarks/{run_id}.json", "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)

    with mkdocs_gen_files.open("assets/benchmarks/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    solver_pages = _build_solver_pages_map(only_solvers=solvers_in_runs)
    with mkdocs_gen_files.open("assets/benchmarks/solver_pages.json", "w") as f:
        json.dump(solver_pages, f, indent=2, sort_keys=True)


main()
