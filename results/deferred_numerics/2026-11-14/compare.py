"""Keep all paired metrics visible; never hide per-scenario regressions."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("baseline", type=Path)
parser.add_argument("candidate", type=Path)
parser.add_argument("--out", type=Path)
args = parser.parse_args()


def read_result(path):
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    return json.loads(data)


a = read_result(args.baseline)
b = read_result(args.candidate)
lookup = {(r["solver_name"], r["dataset_name"]): r for r in a["results"]}
rows = []
for y in b["results"]:
    x = lookup[y["solver_name"], y["dataset_name"]]
    row = {"solver": y["solver_name"], "scenario": y["dataset_name"], "metrics": {}}
    for metric, v in x["metrics"].items():
        w = y["metrics"][metric]
        row["metrics"][metric] = {
            "baseline": v,
            "candidate": w,
            "mean_delta": w["mean"] - v["mean"],
        }
    sx = np.array(
        [[r[k] for k in ["mle", "emd", "ap", "correlation"]] for r in x["samples"]],
        float,
    )
    sy = np.array(
        [[r[k] for k in ["mle", "emd", "ap", "correlation"]] for r in y["samples"]],
        float,
    )
    assert sx.shape == sy.shape
    row["all_samples_identical"] = bool(np.array_equal(sx, sy, equal_nan=True))
    row["nonfinite_baseline"] = int(np.count_nonzero(~np.isfinite(sx)))
    row["nonfinite_candidate"] = int(np.count_nonzero(~np.isfinite(sy)))
    row["paired_standard_error"] = {
        k: float(np.std(sy[:, i] - sx[:, i], ddof=1) / np.sqrt(len(sx)))
        for i, k in enumerate(["mle", "emd", "ap", "correlation"])
    }
    rows.append(row)
    print(
        row["solver"],
        row["scenario"],
        {k: round(v["mean_delta"], 8) for k, v in row["metrics"].items()},
        "identical",
        row["all_samples_identical"],
        "nonfinite",
        row["nonfinite_baseline"],
        row["nonfinite_candidate"],
    )
if args.out:
    args.out.write_text(
        json.dumps(
            {
                "baseline": str(args.baseline),
                "candidate": str(args.candidate),
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
