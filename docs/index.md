# invertmeeg

A high-level M/EEG Python library for EEG inverse solutions.

This package provides **82 inverse solvers** for M/EEG source imaging, integrating
with the [mne-python](https://mne.tools) framework. It covers minimum norm
methods, beamformers, Bayesian approaches, sparse recovery, subspace methods,
and deep learning models in a unified API.

## Highlights

- **82 inverse solvers** accessible through a single `Solver("solver_id")` interface
- Automatic regularization (GCV, L-curve, product methods)
- Returns standard `mne.SourceEstimate` objects
- Simulation utilities for benchmarking

## Quick links

- [Getting Started](getting-started.md) — install, extras, and a minimal end-to-end example
- [Solvers](solvers/index.md) — browse solver families and per-solver API docs
- [Benchmarks](benchmarks.md) — interactive benchmark dashboard (static, GitHub Pages friendly)
- [API Reference](api.md) — full Python API via mkdocstrings

## Quick Example

```python
from invert import Solver

# fwd = ...   (mne.Forward object)
# evoked = ... (mne.Evoked object)

solver = Solver("MNE")
solver.make_inverse_operator(fwd)
stc = solver.apply_inverse_operator(evoked)
stc.plot()
```
