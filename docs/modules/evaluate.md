# Evaluation

The `evaluate` module provides metrics and tools for assessing the quality of inverse solutions by comparing estimated source activity against ground truth.

## Overview

Evaluation metrics in invertmeeg include:

- **Localization error**: Distance between true and estimated source locations
- **Spatial dispersion**: Spread of the estimated activity around the true location
- **Amplitude accuracy**: Correlation and error metrics for source amplitudes
- **Resolution metrics**: Point spread and cross-talk functions

## Quick Start

```python
from invert.evaluate import Evaluation

# Create an evaluation object
evaluation = Evaluation(stc_true, stc_estimated, forward)

# Compute metrics
metrics = evaluation.compute_all()
print(f"Localization error: {metrics['localization_error']:.1f} mm")
print(f"Spatial dispersion: {metrics['spatial_dispersion']:.1f} mm")
```

## API Reference

### Evaluation Class

::: invert.evaluate.Evaluation
    options:
      show_root_heading: true
      show_source: true
      members_order: source
