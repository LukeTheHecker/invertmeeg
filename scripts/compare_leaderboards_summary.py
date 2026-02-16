"""Condensed comparison: global rank + per-dataset MLE/Corr changes."""
import json
from collections import defaultdict

def load(path):
    with open(path) as f:
        return json.load(f)

old = load("results/release/leaderboard.json")
new = load("results/release/leaderboard-test.json")

# Build lookup
def build_metrics(lb):
    out = {}
    for entry in lb["results"]:
        key = (entry["solver_name"], entry["dataset_name"])
        out[key] = entry["metrics"]
    return out

old_m = build_metrics(old)
new_m = build_metrics(new)
old_global = old["global_ranks"]
new_global = new["global_ranks"]

# All solvers in both
all_solvers = sorted(set(old_global) & set(new_global))
datasets = ["focal_source", "extended_source", "multi_source", "noisy"]

# For each solver: global rank change, and average MLE change across datasets
rows = []
for solver in all_solvers:
    old_r = old_global[solver]
    new_r = new_global[solver]
    rank_diff = new_r - old_r

    # Collect MLE changes per dataset
    mle_changes = {}
    for ds in datasets:
        key = (solver, ds)
        if key in old_m and key in new_m:
            om = old_m[key]["mean_localization_error"]["mean"]
            nm = new_m[key]["mean_localization_error"]["mean"]
            if abs(om) > 1e-9:
                mle_changes[ds] = (nm - om) / abs(om)
            else:
                mle_changes[ds] = 0

    avg_mle_change = sum(mle_changes.values()) / len(mle_changes) if mle_changes else 0
    rows.append((solver, old_r, new_r, rank_diff, mle_changes, avg_mle_change))

# Sort by rank change (most improved first)
rows.sort(key=lambda x: x[3])

# Only show solvers with |rank_diff| > 1.0 or any |MLE change| > 10%
RANK_THRESH = 1.0
MLE_THRESH = 0.10

filtered = [r for r in rows if abs(r[3]) > RANK_THRESH or any(abs(v) > MLE_THRESH for v in r[4].values())]

print(f"{'Solver':<22} {'OldRk':>6} {'NewRk':>6} {'RkChg':>6} │ {'Focal':>7} {'Extend':>7} {'Multi':>7} {'Noisy':>7} │ {'Verdict'}")
print("─" * 22 + " " + "─" * 6 + " " + "─" * 6 + " " + "─" * 6 + " │ " + "─" * 7 + " " + "─" * 7 + " " + "─" * 7 + " " + "─" * 7 + " │ " + "─" * 20)

for solver, old_r, new_r, rank_diff, mle_changes, avg_mle_change in filtered:
    def fmt_pct(ds):
        if ds in mle_changes:
            v = mle_changes[ds]
            if abs(v) < 0.05:
                return "    ·  "
            return f"{v:+6.0%} "
        return "   N/A "

    # Determine verdict
    if rank_diff < -5:
        verdict = "IMPROVED"
    elif rank_diff > 5:
        verdict = "REGRESSED"
    elif rank_diff < -1:
        verdict = "improved"
    elif rank_diff > 1:
        verdict = "regressed"
    else:
        # Check MLE changes
        imp = sum(1 for v in mle_changes.values() if v < -MLE_THRESH)
        reg = sum(1 for v in mle_changes.values() if v > MLE_THRESH)
        if imp > reg:
            verdict = "mixed (net better)"
        elif reg > imp:
            verdict = "mixed (net worse)"
        else:
            verdict = "mixed"

    print(f"{solver:<22} {old_r:>6.1f} {new_r:>6.1f} {rank_diff:>+6.1f} │ {fmt_pct('focal_source')}{fmt_pct('extended_source')}{fmt_pct('multi_source')}{fmt_pct('noisy')}│ {verdict}")

# Show removed/new
old_set = set(old_global)
new_set = set(new_global)
removed = old_set - new_set
added = new_set - old_set
if removed:
    print(f"\nREMOVED: {sorted(removed)}")
if added:
    print(f"\nADDED: {sorted(added)}")

# Systematics: many solvers regress on noisy
print("\n\n--- SYSTEMATIC PATTERN CHECK ---")
noisy_reg = 0
noisy_imp = 0
other_reg = 0
other_imp = 0
for solver in all_solvers:
    for ds in datasets:
        key = (solver, ds)
        if key in old_m and key in new_m:
            om = old_m[key]["mean_localization_error"]["mean"]
            nm = new_m[key]["mean_localization_error"]["mean"]
            if abs(om) > 1e-9:
                pct = (nm - om) / abs(om)
                if ds == "noisy":
                    if pct > 0.05: noisy_reg += 1
                    elif pct < -0.05: noisy_imp += 1
                else:
                    if pct > 0.05: other_reg += 1
                    elif pct < -0.05: other_imp += 1

print(f"Noisy dataset MLE:  {noisy_imp} improved, {noisy_reg} regressed")
print(f"Other datasets MLE: {other_imp} improved, {other_reg} regressed")
