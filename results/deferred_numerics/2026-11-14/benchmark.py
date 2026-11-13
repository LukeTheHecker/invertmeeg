"""Paired public-API benchmark; no hidden depth policy or matrix-only shortcut."""

# Imports follow source selection so baseline and candidate stay isolated.
# ruff: noqa: E402

import argparse
import hashlib
import json
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--source", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument(
    "--assets",
    type=Path,
    default=Path(
        "/Users/lukas/projects/invert/invert-package/results/math_audit/2026-11-07"
    ),
)
p.add_argument("--samples", type=int, default=30)
p.add_argument("--seed", type=int, default=20261114)
p.add_argument("--ssp", action="store_true")
p.add_argument("--no-noise-cov", action="store_true")
p.add_argument("--params", type=Path)
p.add_argument(
    "--solvers",
    default="mne,dspm,sloreta,eloreta,lcmv,lcmv-mvpure,esmv,ir-esmv,ssp-esmv,ssp-ir-esmv,esmv-mvpure",
)
a = p.parse_args()
sys.path.insert(0, str(a.source.resolve()))
import mne
import numpy as np

from invert import Solver
from invert.benchmark.datasets import BENCHMARK_DATASETS
from invert.benchmark.runner import (
    SimulationConfig,
    SimulationGenerator,
    _split_solver_params,
    evaluate_all,
    pos_from_forward,
)

mne.set_log_level("ERROR")
fwd = mne.read_forward_solution(a.assets / "hcp_ico2_fixed-fwd.fif", verbose=False)
fwd = mne.convert_forward_solution(
    fwd, surf_ori=True, force_fixed=True, use_cps=True, verbose=False
)
info = mne.io.read_info(a.assets / "hcp_ico2-info.fif", verbose=False)
P = np.eye(len(info["ch_names"]))
if a.ssp:
    ref = mne.EvokedArray(np.zeros((len(info["ch_names"]), 3)), info, verbose=False)
    ref.set_eeg_reference("average", projection=True, verbose=False)
    info = ref.info
    P -= np.ones_like(P) / len(P)
    # Simulate and fit in the same physical EEG reference for both versions.
    fwd["sol"]["data"] = P @ fwd["sol"]["data"]
    fwd["_orig_sol"] = P @ fwd["_orig_sol"]
    with fwd["info"]._unlock():
        fwd["info"]["projs"] = info["projs"]
params = json.loads(a.params.read_text()) if a.params else {}
pos = pos_from_forward(fwd)
adj = mne.spatial_src_adjacency(fwd["src"], verbose=False)
results = []
metric_keys = {
    "mean_localization_error": "mle",
    "emd": "emd",
    "spatial_dispersion": "sd",
    "average_precision": "ap",
    "correlation": "correlation",
}
for ds in ["focal_source", "multi_source", "extended_source", "noisy"]:
    d = BENCHMARK_DATASETS[ds]
    cfg = SimulationConfig(
        batch_size=a.samples,
        n_sources=d.n_sources,
        n_orders=d.n_orders,
        snr_range=d.snr_range,
        n_timepoints=d.n_timepoints,
        source_spatial_model=d.source_spatial_model,
        patch_rank=d.patch_rank,
        random_seed=a.seed,
        estimate_noise_cov=True,
        return_noise_cov=True,
    )
    X, Y, meta = next(SimulationGenerator(fwd, config=cfg).generate())
    for name in a.solvers.split(","):
        init, make, apply = _split_solver_params(params.get(name, {}))
        samples = []
        for i in range(a.samples):
            ev = mne.EvokedArray(P @ X[i], info.copy(), verbose=False)
            cov = (
                None
                if a.no_noise_cov
                else mne.Covariance(
                    P @ meta.iloc[i]["noise_cov_est_scaled"] @ P.T,
                    info["ch_names"],
                    [],
                    info["projs"],
                    1,
                )
            )
            solver = Solver(name, **init)
            solver.make_inverse_operator(fwd, ev, noise_cov=cov, alpha="auto", **make)
            pred = solver.apply_inverse_operator(ev, **apply).data
            metrics = evaluate_all(Y[i], pred, adj, adj, pos, pos)
            samples.append(
                {
                    "mle": float(metrics["Mean_Localization_Error"]),
                    "emd": float(metrics["EMD"]),
                    "sd": float(metrics["sd"]),
                    "ap": float(metrics["average_precision"]),
                    "correlation": float(metrics["correlation"]),
                }
            )
        results.append(
            {
                "solver_name": name,
                "dataset_name": ds,
                "samples": samples,
                "metrics": {
                    key: {
                        "mean": float(np.mean([r[short] for r in samples])),
                        "std": float(np.std([r[short] for r in samples])),
                    }
                    for key, short in metric_keys.items()
                },
            }
        )
        print(ds, name, flush=True)
output = {
    "source": str(a.source),
    "ssp": a.ssp,
    "noise_covariance": not a.no_noise_cov,
    "samples": a.samples,
    "seed": a.seed,
    "params": params,
    "numpy": np.__version__,
    "mne": mne.__version__,
    "forward": "HCP100307 ico2 324 fixed sources 32 EEG",
    "datasets": {
        k: BENCHMARK_DATASETS[k].model_dump()
        for k in ["focal_source", "multi_source", "extended_source", "noisy"]
    },
    "source_hashes": {
        str(f.relative_to(a.source)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in (a.source / "invert/solvers").rglob("*.py")
    },
    "results": results,
}
a.out.write_text(json.dumps(output, indent=2) + "\n")
