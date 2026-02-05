from invert.forward import create_forward_model, get_info
from invert.benchmark import BenchmarkRunner, visualize_results
from pathlib import Path

if __name__ == '__main__':
    info = get_info(kind="biosemi32")
    fwd = create_forward_model(sampling="ico2", info=info)

    # runner = BenchmarkRunner(fwd, info, n_samples=25)
    # categories: beamformer, empirical_bayes, sparse_bayesian, music, matching_pursuit, other, baseline
    runner = BenchmarkRunner(fwd, info, n_samples=10, categories=["music"], n_jobs=-1)
    runner.run()
    out_path = Path("results/benchmark_results_music.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    runner.save(out_path)

    # visualize_results(out_path, save_path="figures/")
