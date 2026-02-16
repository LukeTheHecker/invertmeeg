"""Quick BCS evaluation."""
import json
from pathlib import Path
from invert.benchmark import BenchmarkRunner
from invert.forward import create_forward_model, get_info

SOLVERS = ["BCS", "MNE", "Champagne"]
N_SAMPLES = 10

info = get_info(kind="biosemi32")
fwd = create_forward_model(sampling="ico2", info=info)

runner = BenchmarkRunner(
    fwd, info, solvers=SOLVERS, n_samples=N_SAMPLES, n_jobs=-1, random_seed=42,
)
runner.run()

out = Path("results/eval_bcs.json")
out.parent.mkdir(parents=True, exist_ok=True)
runner.save(out, compact=False, name="BCS eval")

# Print summary
data = json.loads(out.read_text())
print("\n=== BCS Evaluation ===")
for ds in ["focal_source", "multi_source", "extended_source", "noisy"]:
    metrics = data.get("metrics", {}).get(ds, {})
    print(f"\n--- {ds} ---")
    for solver in SOLVERS:
        m = metrics.get(solver, {})
        mle = m.get("MLE", "?")
        emd = m.get("EMD", "?")
        corr = m.get("Correlation", "?")
        print(f"  {solver:<15} MLE={mle:>6.1f}  EMD={emd:>6.1f}  Corr={corr:.3f}")
