# Simulation

The `simulate` module provides tools for generating synthetic M/EEG data with known ground truth source activations. This is essential for validating and comparing inverse methods under controlled conditions.

## Overview

Simulation in invertmeeg supports:

- **Configurable source patterns**: Single dipoles, extended patches, or multiple simultaneous sources
- **Noise models**: White noise, colored noise, and realistic sensor noise
- **Covariance structures**: Correlated sources for testing robustness
- **Batch generation**: Efficient generation of many samples for training or benchmarking

## Quick Start

```python
from invert.simulate import SimulationConfig, SimulationGenerator

# Create a simulation configuration
config = SimulationConfig(
    n_sources=2,
    snr=5.0,
    source_extent=10.0,  # mm
)

# Generate simulations
generator = SimulationGenerator(forward, config)
evoked, stc_true = generator.generate()
```

## API Reference

::: invert.simulate
    options:
      show_root_heading: false
      members_order: source

### SimulationConfig

::: invert.simulate.SimulationConfig
    options:
      show_root_heading: true
      members_order: source

### SimulationGenerator

::: invert.simulate.SimulationGenerator
    options:
      show_root_heading: true
      members_order: source

### Noise Functions

::: invert.simulate.add_white_noise
    options:
      show_root_heading: true

::: invert.simulate.powerlaw_noise
    options:
      show_root_heading: true

### Covariance Utilities

::: invert.simulate.compute_covariance
    options:
      show_root_heading: true

::: invert.simulate.gen_correlated_sources
    options:
      show_root_heading: true
