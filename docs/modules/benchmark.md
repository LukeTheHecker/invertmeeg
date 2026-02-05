# Benchmarking

The `benchmark` module provides a framework for systematically comparing solver performance across different datasets and conditions.

## Overview

The benchmarking framework supports:

- **Multiple datasets**: Test solvers on various simulated and real datasets
- **Solver categories**: Compare solvers within and across categories
- **Parallel execution**: Efficient batch processing of many solver-dataset combinations
- **Result visualization**: Built-in plotting for benchmark results

## Quick Start

```python
from invert.benchmark import BenchmarkRunner, create_datasets

# Create benchmark datasets
datasets = create_datasets(forward, n_samples=100)

# Run benchmarks
runner = BenchmarkRunner(
    solvers=["MNE", "dSPM", "LCMV", "Champagne"],
    datasets=datasets,
)
results = runner.run()

# Visualize results
from invert.benchmark import visualize_results
visualize_results(results)
```

## API Reference

### BenchmarkRunner

::: invert.benchmark.BenchmarkRunner
    options:
      show_root_heading: true
      show_source: true
      members_order: source

### BenchmarkResult

::: invert.benchmark.BenchmarkResult
    options:
      show_root_heading: true
      show_source: true

### Dataset Configuration

::: invert.benchmark.DatasetConfig
    options:
      show_root_heading: true
      show_source: true

::: invert.benchmark.create_datasets
    options:
      show_root_heading: true
      show_source: true

### Visualization

::: invert.benchmark.visualize_results
    options:
      show_root_heading: true
      show_source: true

### Solver Resolution

::: invert.benchmark.resolve_solvers
    options:
      show_root_heading: true
      show_source: true
