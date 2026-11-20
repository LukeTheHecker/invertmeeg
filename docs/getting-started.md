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
conda create -n invertmeeg python=3.14
conda activate invertmeeg
pip install invertmeeg
```

### Extras

- **Neural Networks**: `pip install "invertmeeg[ann]"`
- **Visualization**: `pip install "invertmeeg[viz]"`
- **Documentation**: `pip install "invertmeeg[docs]"`

## Localize a simulated EEG source

With Python 3.14 and the base package installed, run the example below. It builds
a spherical head model and a coarse 20 mm grid in memory, adds seeded noise to
a single dipole, and estimates its location at the known pulse peak. The final
line gives the straight-line distance between the simulated source and the
estimated peak. On the seeded example the error is 0.0 mm, meaning the estimated peak lands
on the simulated grid point. The simulation and reconstruction share a head model.

[Download the Python example](downloads/eeg_quickstart.py)

```python
--8<-- "docs/downloads/eeg_quickstart.py"
```

To compare methods for your own recordings, see [Choosing a solver](benchmarks.md#choosing-a-solver).
