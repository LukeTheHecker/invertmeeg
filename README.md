# invertmeeg - A high-level M/EEG Python library for EEG inverse solutions

<p>
  <img src="docs/assets/images/logo-pixelbrain.png" alt="invertmeeg logo" width="120" />
</p>

[![PyPI version](https://badge.fury.io/py/invertmeeg.svg)](https://pypi.org/project/invertmeeg/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

This package provides **82 inverse solvers** for M/EEG source imaging, integrating
with the [mne-python](https://mne.tools) framework. It covers minimum norm
methods, beamformers, Bayesian approaches, sparse recovery, subspace methods,
and deep learning models in a unified API.

Read the [documentation here.](https://lukethehecker.github.io/invertmeeg/)

## Installation

The recommended way to install `invertmeeg` is using [uv](https://github.com/astral-sh/uv):

```bash
uv add invertmeeg
```

Alternatively, you can use `pip`:

```bash
pip install invertmeeg
```

### Extras
- **Neural Networks**: `uv add "invertmeeg[ann]"` (or `pip install "invertmeeg[ann]"`)
- **Visualization**: `uv add "invertmeeg[viz]"` (or `pip install "invertmeeg[viz]"`)

## Development

If you are contributing to `invertmeeg`, we use `uv` and `make` to manage the development environment and code quality.

### Setup
```bash
make install
```
This will sync all dependencies (including dev tools) and install the pre-commit hooks.

### Manual uv setup, install all packages except ann
```bash
uv sync --extra viz --extra docs --group dev
```

### Useful Commands
- `make lint`: Run Ruff and Mypy checks.
- `make format`: Auto-format code with Ruff.
- `make test`: Run the test suite with pytest.
- `make check`: Run all linting and tests.
- `make clean`: Remove temporary cache files (__pycache__, .mypy_cache, etc.).

### Benchmark Dashboard

Host dashboard locally:
```bash
uv run python3 -m http.server 8001
```

Visit site:
```
http://localhost:8001/dashboard.html
```

## Quick Start

```python
from invert import Solver

# fwd = ...   (mne.Forward object)
# evoked = ... (mne.Evoked object)

solver = Solver("MNE")
solver.make_inverse_operator(fwd)
stc = solver.apply_inverse_operator(evoked)
stc.plot()
```

## Features

- **82 inverse solvers** accessible through a single `Solver("solver_id")` interface
- Automatic regularization (GCV, L-curve, product methods)
- Returns standard `mne.SourceEstimate` objects
- Simulation utilities for benchmarking

## Solver Categories

| Category | Count | Examples |
|----------|-------|---------|
| Minimum Norm | 7 | MNE, wMNE, dSPM, FISTA, L1L2 |
| LORETA | 3 | LORETA, sLORETA, eLORETA |
| Other Minimum-Norm-like | 4 | LAURA, Backus-Gilbert, S-MAP |
| Bayesian | 13 | Champagne variants, Gamma-MAP, Source-MAP, VB-SBL |
| Beamformers | 12 | LCMV, DICS, MVAB, SAM, ReciPSIICOS, EBB |
| Dipole Fitting | 2 | ECD, SESAME |
| Structured Sparsity | 1 | Total Variation |
| Neural Networks | 4 | FC/ESInet, CovCNN, LSTM, CNN |
| Matching Pursuit | 11 | OMP, COSAMP, SOMP, BCS, Subspace Pursuit |
| MUSIC/Subspace | 8 | MUSIC, RAP-MUSIC, TRAP-MUSIC, FLEX-MUSIC |
| Basis Functions | 1 | GBF |
| Other | 1 | EPIFOCUS |

## Full Algorithm List

### Minimum Norm

| Full Solver Name | Abbreviation |
|------------------|--------------|
| Minimum Norm Estimate | "mne" |
| Minimum Norm Estimate with Graph Fourier Transform | "gft-mne" |
| Weighted Minimum Norm Estimate | "wmne" |
| Dynamic Statistical Parametric Mapping | "dspm" |
| Minimum Current Estimate | "l1", "fista", "mce" |
| Minimum L1 Norm GPT | "gpt", "l1-gpt" |
| Minimum L1L2 Norm | "l1l2" |

### LORETA

| Full Solver Name | Abbreviation |
|------------------|--------------|
| LORETA | "lor" |
| sLORETA | "slor" |
| eLORETA | "elor" |

### Other Minimum-Norm-like Algorithms

| Full Solver Name | Abbreviation |
|------------------|--------------|
| LAURA | "laura", "laur" |
| LAURA (Improved) | "laura-improved", "laur2", "lauraimproved" |
| Backus-Gilbert | "b-g", "bg" |
| S-MAP | "smap" |

### Bayesian

| Full Solver Name | Abbreviation |
|------------------|--------------|
| Champagne | "champ" |
| Low SNR Champagne | "lsc", "lowsnr-champagne" |
| MacKay Champagne | "mcc", "mackay-champagne" |
| Convexity Champagne | "coc", "convexity-champagne" |
| Noise Learning Champagne | "nl-champagne" |
| Expectation Maximization Champagne | "emc" |
| Majorization Maximization Champagne | "mmc" |
| Full-Structure Noise | "fun" |
| Heteroscedastic Champagne | "hsc" |
| Gamma-MAP | "gamma-map" |
| Source-MAP | "source-map" |
| Gamma-MAP-MSP | "gamma-map-msp" |
| Source-MAP-MSP | "source-map-msp" |
| Variational Bayes SBL | "vb-sbl" |

### Beamformers

| Full Solver Name | Abbreviation |
|------------------|--------------|
| Minimum Variance Adaptive Beamformer | "mvab" |
| Linearly Constrained Minimum Variance | "lcmv" |
| Dynamic Imaging of Coherent Sources | "dics" |
| Standardized Minimum Variance | "smv" |
| Weight-Normalized Minimum Variance | "wnmv" |
| Higher-Order Covariance Minimum Variance | "hocmv" |
| Eigenspace Scalar Minimum Variance | "esmv" |
| Multiple Constraint Minimum Variance | "mcmv" |
| Higher-Order Covariance Multiple Constraint Minimum Variance | "hocmcmv" |
| Reciprocal PSIICOS | "recipsiicos" |
| Synthetic Aperture Magnetometry | "sam" |
| Empirical Bayesian Beamformer | "ebb" |

### Dipole Fitting

| Full Solver Name | Abbreviation |
|------------------|--------------|
| Equivalent Current Dipole | "ecd" |
| SESAME | "sesame" |

### Structured Sparsity

| Full Solver Name | Abbreviation |
|------------------|--------------|
| Total Variation | "tv" |

### Artificial Neural Networks

| Full Solver Name | Abbreviation |
|------------------|--------------|
| Fully-Connected Network | "fc", "esinet" |
| Covariance CNN | "covcnn", "covnet" |
| Long Short-Term Memory | "lstm" |
| Convolutional Neural Network | "cnn" |

### Matching Pursuit / Compressive Sensing

| Full Solver Name | Abbreviation |
|------------------|--------------|
| Orthogonal Matching Pursuit | "omp" |
| Compressive Sampling Matching Pursuit | "cosamp" |
| Simultaneous Orthogonal Matching Pursuit | "somp" |
| Random Embedding Matching Pursuit | "rembo" |
| Subspace Pursuit | "sp" |
| Structured Subspace Pursuit | "ssp" |
| Subspace Matching Pursuit | "smp" |
| Structured Subspace Matching Pursuit | "ssmp" |
| Subspace-based Subspace Matching Pursuit | "subsmp" |
| Iterative Subspace-based Subspace Matching Pursuit | "isubsmp" |
| Bayesian Compressive Sensing | "bcs" |

### MUSIC/RAP/Subspace

| Full Solver Name | Abbreviation |
|------------------|--------------|
| Multiple Signal Classification | "music" |
| Recursively Applied and Projected MUSIC | "rap-music", "rap" |
| Truncated Recursively Applied and Projected MUSIC | "trap-music", "trap" |
| Flexible RAP-MUSIC | "flex-music", "flex" |
| Flexible Signal Subspace Matching | "flex-ssm" |
| Signal Subspace Matching | "ssm" |
| Flexible Alternating Projections | "flex-ap" |
| Alternating Projections | "ap" |

### Basis Functions

| Full Solver Name | Abbreviation |
|------------------|--------------|
| Geometrically Informed Basis Functions | "gbf" |

### Other

| Full Solver Name | Abbreviation |
|------------------|--------------|
| EPIFOCUS | "epifocus" |

## Licensing

This project is **dual-licensed**:

- **Non-commercial use:** CC BY-NC 4.0 (free with attribution).
- **Commercial use:** requires a separate commercial license/permission (contact below).

### For Researchers & Educators (Non-Commercial)
We love science. This library is free to use for research, education, thesis
projects, and tinkering as long as your use is **non-commercial**.

* **Attribution required:** please credit `invertmeeg` (and ideally cite this
  repository) in any publications, reports, or released artifacts that use it.
* **Non-commercial only:** if your usage is connected to a product/service, or
  you monetize it in any way, you need a commercial license.

### For Industry & Commercial Entities
If you are a company, a startup, or an individual intending to use this library
in a product/service (including internal tooling) or any other **commercial**
context, CC BY-NC 4.0 does **not** grant you that right. Please contact me to
obtain commercial permission/a commercial license.

**Contact:** `lukas.hecker.job@gmail.com`

## Citation

If you use this package and publish results, please cite as:

```
@Misc{invertmeeg2022,
  author =   {{Lukas Hecker}},
  title =    {{invertmeeg}: A high-level M/EEG Python library for EEG inverse solutions.},
  howpublished = {\url{https://github.com/LukeTheHecker/invertmeeg}},
  year = {since 2022}
}
```

Send feedback or feature requests to [lukas.hecker.job@gmail.com](mailto:lukas.hecker.job@gmail.com).
