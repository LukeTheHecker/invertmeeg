from invert.forward import create_forward_model, get_info
from invert.benchmark import BenchmarkRunner, visualize_results
from pathlib import Path

if __name__ == '__main__':
    info = get_info(kind="biosemi32")
    fwd = create_forward_model(sampling="ico2", info=info)

    runner = BenchmarkRunner(
        fwd, info, n_samples=100,
        solvers=["SubspaceSBL", "SSM", "NLChampagne", "FlexChampagne"],
        n_jobs=-1,
    )
    runner.run()
    out_path = Path("results/benchmark_results-claude.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    runner.save(out_path)
