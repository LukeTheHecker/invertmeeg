from invert.forward import create_forward_model, get_info
from invert.benchmark import BenchmarkRunner, visualize_results
from pathlib import Path

if __name__ == '__main__':
    info = get_info(kind="biosemi32")
    fwd = create_forward_model(sampling="ico2", info=info)

    # runner = BenchmarkRunner(fwd, info, n_samples=25)
    # categories: beamformer, empirical_bayes, sparse_bayesian, music, matching_pursuit, other, baseline
    # runner = BenchmarkRunner(fwd, info, n_samples=25, categories=["beamformer", "bayesian", "minimum_norm", "loreta", "music", "matching_pursuit", "other"], n_jobs=-1)
    runner = BenchmarkRunner(
        fwd, info, n_samples=50,
        categories=["beamformer", "bayesian", "minimum_norm", "loreta", "music", "matching_pursuit", "other"],
        solvers=["CovCNN", "CovCNN-KL", "CovCNN-KL-FLEXOMP"],
        n_jobs=-1,
    )    
    runner.run()
    out_path = Path("results/benchmark_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    runner.save(out_path, compact=True)
