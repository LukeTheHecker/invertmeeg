"""Compare leaderboard.json vs leaderboard-test.json, flagging >5% changes."""
import json
import sys

def load(path):
    with open(path) as f:
        return json.load(f)

old = load("results/release/leaderboard.json")
new = load("results/release/leaderboard-test.json")

# Build lookup: (solver, dataset) -> {metric: {stat: val}}
def build_metrics(lb):
    out = {}
    for entry in lb["results"]:
        key = (entry["solver_name"], entry["dataset_name"])
        out[key] = entry["metrics"]
    return out

old_m = build_metrics(old)
new_m = build_metrics(new)

# Metrics where lower is better
LOWER_BETTER = {"mean_localization_error", "emd", "spatial_dispersion"}
HIGHER_BETTER = {"correlation", "average_precision"}

# Also compare ranks
old_ranks = old["ranks"]
new_ranks = new["ranks"]
old_global = old["global_ranks"]
new_global = new["global_ranks"]

# Collect all solvers present in both
all_solvers = sorted(set(s for s, _ in old_m) | set(s for s, _ in new_m))
all_datasets = sorted(set(d for _, d in old_m) | set(d for _, d in new_m))

THRESHOLD = 0.05  # 5% relative change is negligible

changes = []  # (solver, dataset, metric, stat, old_val, new_val, pct_change, direction)

for solver in all_solvers:
    for dataset in all_datasets:
        key = (solver, dataset)
        if key not in old_m or key not in new_m:
            continue
        for metric_name in old_m[key]:
            if metric_name not in new_m[key]:
                continue
            old_stats = old_m[key][metric_name]
            new_stats = new_m[key][metric_name]
            # Compare mean
            stat = "mean"
            ov = old_stats[stat]
            nv = new_stats[stat]
            if abs(ov) < 1e-9 and abs(nv) < 1e-9:
                continue
            denom = abs(ov) if abs(ov) > 1e-9 else abs(nv)
            pct = (nv - ov) / denom

            if abs(pct) <= THRESHOLD:
                continue

            if metric_name in LOWER_BETTER:
                direction = "improved" if nv < ov else "regressed"
            elif metric_name in HIGHER_BETTER:
                direction = "improved" if nv > ov else "regressed"
            else:
                direction = "changed"

            changes.append((solver, dataset, metric_name, ov, nv, pct, direction))

# Also compare global ranks
rank_changes = []
for solver in sorted(set(old_global) | set(new_global)):
    if solver in old_global and solver in new_global:
        ov = old_global[solver]
        nv = new_global[solver]
        diff = nv - ov  # lower rank = better, so negative diff = improved
        if abs(diff) > 1.0:  # only report rank changes > 1
            direction = "improved" if diff < 0 else "regressed"
            rank_changes.append((solver, ov, nv, diff, direction))
    elif solver not in old_global:
        rank_changes.append((solver, None, new_global[solver], None, "NEW"))
    else:
        rank_changes.append((solver, old_global[solver], None, None, "REMOVED"))

# Print summary
print("=" * 100)
print("GLOBAL RANK CHANGES (|diff| > 1.0)")
print("=" * 100)
print(f"{'Solver':<25} {'Old Rank':>10} {'New Rank':>10} {'Change':>10} {'Direction':<10}")
print("-" * 100)
for solver, ov, nv, diff, direction in sorted(rank_changes, key=lambda x: (x[4] != "improved", abs(x[3] or 0)), reverse=True):
    ov_s = f"{ov:.1f}" if ov is not None else "N/A"
    nv_s = f"{nv:.1f}" if nv is not None else "N/A"
    diff_s = f"{diff:+.1f}" if diff is not None else "N/A"
    print(f"{solver:<25} {ov_s:>10} {nv_s:>10} {diff_s:>10} {direction:<10}")

# Summarize per-solver across all datasets/metrics
print()
print("=" * 100)
print("PER-SOLVER METRIC CHANGES (>5% relative, mean values)")
print("=" * 100)

# Group by solver
from collections import defaultdict
solver_changes = defaultdict(list)
for solver, dataset, metric, ov, nv, pct, direction in changes:
    solver_changes[solver].append((dataset, metric, ov, nv, pct, direction))

# Print improved solvers first, then regressed
improved_solvers = set()
regressed_solvers = set()
mixed_solvers = set()
for solver, items in solver_changes.items():
    dirs = set(d for _, _, _, _, _, d in items)
    if dirs == {"improved"}:
        improved_solvers.add(solver)
    elif dirs == {"regressed"}:
        regressed_solvers.add(solver)
    else:
        mixed_solvers.add(solver)

metric_short = {
    "mean_localization_error": "MLE",
    "emd": "EMD",
    "spatial_dispersion": "SD",
    "correlation": "Corr",
    "average_precision": "AP",
}

def print_solver_section(title, solver_set):
    if not solver_set:
        return
    print(f"\n--- {title} ---")
    print(f"{'Solver':<25} {'Dataset':<18} {'Metric':<6} {'Old':>10} {'New':>10} {'Change':>8} {'Dir':<10}")
    print("-" * 90)
    for solver in sorted(solver_set):
        items = solver_changes[solver]
        for dataset, metric, ov, nv, pct, direction in sorted(items, key=lambda x: (x[0], x[1])):
            mn = metric_short.get(metric, metric[:6])
            print(f"{solver:<25} {dataset:<18} {mn:<6} {ov:>10.3f} {nv:>10.3f} {pct:>+7.0%} {direction:<10}")

print_solver_section("IMPROVED ONLY", improved_solvers)
print_solver_section("REGRESSED ONLY", regressed_solvers)
print_solver_section("MIXED (improved + regressed)", mixed_solvers)

# Summary count
print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)
n_improved = len([c for c in changes if c[6] == "improved"])
n_regressed = len([c for c in changes if c[6] == "regressed"])
print(f"Total metric changes >5%: {len(changes)}")
print(f"  Improved:  {n_improved}")
print(f"  Regressed: {n_regressed}")
print(f"Solvers with only improvements: {len(improved_solvers)}")
print(f"Solvers with only regressions:  {len(regressed_solvers)}")
print(f"Solvers with mixed changes:     {len(mixed_solvers)}")

# NEW/REMOVED solvers
old_solvers = set(s for s, _ in old_m)
new_solvers = set(s for s, _ in new_m)
added = new_solvers - old_solvers
removed = old_solvers - new_solvers
if added:
    print(f"\nNEW solvers in test: {sorted(added)}")
if removed:
    print(f"\nREMOVED solvers from test: {sorted(removed)}")
