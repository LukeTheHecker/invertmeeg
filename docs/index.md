# invertmeeg

A high-level Python library for M/EEG inverse solutions.

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

## Try it

[Localize a simulated EEG source](getting-started.md#localize-a-simulated-eeg-source)
with a 32-channel montage and measure the peak localization error in millimetres.
The example runs offline after installation.
