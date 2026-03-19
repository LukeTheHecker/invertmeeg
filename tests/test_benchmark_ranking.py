from invert.benchmark.runner import AggregateStats, BenchmarkResult, BenchmarkRunner


def _stats(value: float) -> AggregateStats:
    return AggregateStats(mean=value, std=0.0, median=value)


def _result(
    solver_name: str,
    *,
    dataset_name: str,
    mle: float,
    emd: float,
    spatial_dispersion: float,
    average_precision: float,
    correlation: float,
) -> BenchmarkResult:
    return BenchmarkResult(
        solver_name=solver_name,
        dataset_name=dataset_name,
        category="other",
        metrics={
            "mean_localization_error": _stats(mle),
            "emd": _stats(emd),
            "spatial_dispersion": _stats(spatial_dispersion),
            "average_precision": _stats(average_precision),
            "correlation": _stats(correlation),
        },
        samples=[],
    )


def test_spatial_dispersion_is_excluded_from_best_solver_and_ranks():
    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner._results = [
        _result(
            "solver-accurate",
            dataset_name="toy",
            mle=1.0,
            emd=1.0,
            spatial_dispersion=4.0,
            average_precision=0.9,
            correlation=0.9,
        ),
        _result(
            "solver-sharp",
            dataset_name="toy",
            mle=10.0,
            emd=10.0,
            spatial_dispersion=1.0,
            average_precision=0.1,
            correlation=0.1,
        ),
    ]

    best_solvers = runner._compute_best_solvers()
    per_dataset_ranks, global_ranks = runner._compute_average_ranks()

    assert "spatial_dispersion" not in best_solvers["toy"]
    assert best_solvers["toy"]["mean_localization_error"]["solver"] == "solver-accurate"
    assert best_solvers["toy"]["emd"]["solver"] == "solver-accurate"
    assert best_solvers["toy"]["average_precision"]["solver"] == "solver-accurate"
    assert best_solvers["toy"]["correlation"]["solver"] == "solver-accurate"

    assert per_dataset_ranks["toy"]["solver-accurate"] == 1.0
    assert per_dataset_ranks["toy"]["solver-sharp"] == 2.0
    assert global_ranks["solver-accurate"] == 1.0
    assert global_ranks["solver-sharp"] == 2.0
