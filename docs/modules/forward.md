# Forward Modeling

The `forward` module provides utilities for working with forward models (leadfield matrices) and source spaces.

## Overview

Forward modeling utilities include:

- **Leadfield manipulation**: Normalization, depth weighting, and subspace selection
- **Source space operations**: Working with MNE source spaces and vertices
- **Channel selection**: Matching channels between forward models and data

## Quick Start

```python
import mne
from invert.forward import prepare_forward

# Load a forward model
forward = mne.read_forward_solution('forward-sol.fif')

# Prepare for inverse modeling
forward_prepared = prepare_forward(forward, depth=0.8)
```

## API Reference

::: invert.forward
    options:
      show_root_heading: false
      show_source: true
      members_order: source
