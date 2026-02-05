# Visualization

The `viz` module provides plotting utilities for displaying source estimates and comparing inverse solutions.

## Overview

Visualization options include:

- **Glass brain plots**: 2D projections showing source activity overlaid on a transparent brain
- **3D glass brain**: Interactive 3D visualization of source estimates
- **Surface plots**: Activity mapped onto the cortical surface

## Quick Start

```python
from invert.viz import plot_glass_brain, plot_surface

# Plot source estimate on glass brain
fig = plot_glass_brain(stc, forward)

# Plot on cortical surface
fig = plot_surface(stc, subject='fsaverage', subjects_dir=subjects_dir)
```

## API Reference

### Glass Brain Plots

::: invert.viz.plot_glass_brain
    options:
      show_root_heading: true
      show_source: true

### 3D Glass Brain

::: invert.viz.plot_3d_glass_brain
    options:
      show_root_heading: true
      show_source: true

### Surface Plots

::: invert.viz.plot_surface
    options:
      show_root_heading: true
      show_source: true
