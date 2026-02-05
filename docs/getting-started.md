# Getting Started

## Installation

### pip (recommended)

```bash
pip install invertmeeg
```

### uv

If you use [uv](https://github.com/astral-sh/uv):

```bash
uv add invertmeeg
```

### conda

```bash
conda create -n invertmeeg python=3.11
conda activate invertmeeg
pip install invertmeeg
```

### Extras

- **Neural Networks**: `pip install "invertmeeg[ann]"`
- **Visualization**: `pip install "invertmeeg[viz]"`
- **Documentation**: `pip install "invertmeeg[docs]"`

## Quick Start

```python
from invert import Solver

# Assume you have an mne.Forward and mne.Evoked object
# fwd = ...
# evoked = ...

solver = Solver("MNE")
solver.make_inverse_operator(fwd)
stc = solver.apply_inverse_operator(evoked)
stc.plot()
```

The `Solver` factory takes a `solver_id` string and returns the corresponding
solver instance. Browse the [Solvers](solvers/index.md) catalog to find the full
name and click-to-copy solver id for each method.
