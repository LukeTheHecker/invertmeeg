# For Companies

`invertmeeg` is a Python library for M/EEG inverse solutions with a unified API, a large solver catalog, and a built-in benchmarking stack. It integrates with the `mne-python` ecosystem and returns standard `mne.SourceEstimate` objects.

This page is for teams who want to **use or ship** inverse methods in commercial software without taking on the obligations of GPLv3.

## Typical users

- **Neurotech & neurofeedback teams** working with low-channel EEG and tight runtime constraints
- **M/EEG software vendors** who want a broad set of inverse methods behind one stable interface
- **R&D groups** evaluating methods, building new ones, or standardizing internal pipelines

## What you get

- A **single entry point** (`Solver("solver_id")`) across many solver families (minimum norm, beamformers, Bayesian, sparse recovery, subspace, optional deep learning).
- **Simulation, evaluation, and benchmarking** utilities to compare methods under controlled conditions.
- **Repeatable artifacts** (JSON benchmark outputs) that can be turned into dashboards, reports, and internal decision docs.

## Why teams adopt invertmeeg

- **Faster integration:** one API and consistent outputs instead of many disconnected implementations.
- **Method selection you can justify:** benchmark solvers on scenarios that match your product (channels, SNR, focal vs. extended sources).
- **Room for productization:** the core stays open for science, while commercial licensing supports closed-source distribution.

## Licensing (commercial use)

`invertmeeg` is **dual-licensed**:

- **GPLv3 (open source):** for research, education, and open-source distribution under GPLv3.
- **Commercial license (closed source):** for proprietary/closed-source products or internal distribution that can’t meet GPLv3 obligations.

To discuss a commercial license, email: `lukas.hecker.job@gmail.com`.

## Intended use / compliance

`invertmeeg` is provided as a research and engineering tool. If you are building software in a regulated medical context, you are responsible for validating performance, managing risk, and meeting any applicable regulatory and quality requirements.

## Next step

If you share (1) your channel setup, (2) a representative forward model (or your constraints), and (3) your target use case, we can define a small, repeatable benchmark that answers: “which solver family is a good default for *our* product?”

