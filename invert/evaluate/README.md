# Evaluation System

This module provides a comprehensive evaluation system for comparing inverse solution algorithms across different source configurations.

## Quick Start

```python
from invert.evaluate.evaluation import Evaluation
from invert.models.priors import PriorEnum
import mne

# Load your forward model
forward = mne.read_forward_solution('path/to/forward.fif')

# Create evaluation instance
evaluation = Evaluation(
    forward=forward,
    solvers=['MNE', 'LORETA', 'Champagne'],  # or solver instances
    priors=[PriorEnum.DIPOLE, PriorEnum.PATCH, PriorEnum.BROAD],
    n_samples=100,
    random_seed=42
)

# Run evaluation
results = evaluation.evaluate()

# Access results
print(evaluation.summary)      # Summary statistics
print(evaluation.results)      # Detailed results
```

## Features

### 🧠 **Multiple Solver Support**
- Compare any combination of inverse solvers
- Automatic solver instantiation from names
- Support for custom solver instances

### 📊 **Comprehensive Metrics**
- **Mean Localization Error (MLE)**: Spatial accuracy of source estimates
- **Earth Mover's Distance (EMD)**: Distribution similarity between true and estimated sources
- **Spatial Dispersion**: Measure of source spreading/blurring
- **Average Precision**: Source detection accuracy

### 🎯 **Prior-Based Testing**
- **Dipole**: Focal point sources (sparse, localized activity)
- **Patch**: Moderately distributed sources 
- **Broad**: Widespread, diffuse activity
- **No Prior**: General case without assumptions

### 📈 **Rich Output**
- Detailed performance statistics
- Automatic best performer identification
- Built-in visualization tools
- Export-ready results format

## Available Solvers

The evaluation system supports all solvers that inherit from `BaseSolver`:

- **MNE**: Minimum Norm Estimate
- **LORETA**: Low Resolution Electromagnetic Tomography
- **sLORETA**: Standardized LORETA
- **eLORETA**: Exact LORETA
- **Champagne**: Sparse Bayesian learning
- **MUSIC**: Multiple Signal Classification
- **LCMV**: Linearly Constrained Minimum Variance beamformer
- **S-MAP**: Standardized Minimum Average Power
- **APSE**: Adaptive Patch Source Estimation
- And more...

## Priors and Source Patterns

### Dipole Sources
- **Best for**: Epileptic spikes, focal activation
- **Characteristics**: 1-5 point sources, highly localized
- **Simulation**: Low spatial order (0), moderate SNR

### Patch Sources  
- **Best for**: Cortical patches, moderate spread
- **Characteristics**: 1-10 sources with spatial extent
- **Simulation**: Medium spatial order (1-2), varied amplitudes

### Broad Sources
- **Best for**: Widespread networks, distributed processing
- **Characteristics**: Multiple extended sources
- **Simulation**: High spatial order (3-6), complex patterns

### No Prior
- **Best for**: General purpose, unknown source patterns
- **Characteristics**: Mixed scenarios, anything possible
- **Simulation**: Full range of parameters

## Usage Examples

### Basic Comparison
```python
# Compare two solvers on dipole sources
evaluation = Evaluation(
    forward=forward,
    solvers=['MNE', 'LORETA'],
    priors=[PriorEnum.DIPOLE],
    n_samples=50
)
results = evaluation.evaluate()
```

### Comprehensive Analysis
```python
# Full evaluation across all scenarios
evaluation = Evaluation(
    forward=forward,
    solvers=['MNE', 'LORETA', 'Champagne', 'MUSIC'],
    priors=list(PriorEnum),  # All available priors
    n_samples=100
)
results = evaluation.evaluate()

# Visualize results
evaluation.plot_results('mean_localization_error')
evaluation.plot_results('emd')
```

### Find Best Solver
```python
# Identify optimal solver for specific scenarios
best_for_dipoles = evaluation.get_best_solver('dipole', 'mean_localization_error')
best_for_patches = evaluation.get_best_solver('patch', 'emd')

print(f"Best for dipole sources: {best_for_dipoles}")
print(f"Best for patch sources: {best_for_patches}")
```

### Custom Solver Configuration
```python
from invert.solvers import SolverMNE, SolverLORETA

# Use custom solver instances with specific parameters
custom_mne = SolverMNE(regularisation_method="GCV", n_reg_params=10)
custom_loreta = SolverLORETA(regularisation_method="L", prep_leadfield=True)

evaluation = Evaluation(
    forward=forward,
    solvers=[custom_mne, custom_loreta],
    priors=[PriorEnum.DIPOLE, PriorEnum.PATCH]
)
```

## Output Format

### Summary Statistics
The `evaluation.summary` DataFrame contains:
- **solver**: Solver name
- **prior**: Source pattern type
- **mean_localization_error_mean/std**: MLE statistics (mm)
- **emd_mean/std**: EMD statistics
- **spatial_dispersion_mean/std**: Blurring metrics
- **average_precision_mean/std**: Detection accuracy
- **fit_time_mean/std**: Algorithm fitting time
- **apply_time_mean/std**: Source estimation time

### Detailed Results
The `evaluation.results` DataFrame contains sample-level results:
- All summary metrics per sample
- Simulation parameters (SNR, n_sources, etc.)
- Performance timing
- Sample indices for reproducibility

## Performance Considerations

- **Sample Size**: Start with n_samples=10-20 for quick testing, use 100+ for robust results
- **Solver Selection**: Some solvers are much slower than others
- **Memory Usage**: Large forward models and many samples can use significant RAM
- **Parallel Processing**: Currently single-threaded; consider running multiple evaluations in parallel

## Error Handling

The evaluation system is robust to solver failures:
- Failed samples are marked with NaN values
- Partial results are still usable
- Detailed error reporting with verbose mode
- Graceful degradation for missing dependencies

## Integration with Other Tools

### Export Results
```python
# Save results for further analysis
evaluation.summary.to_csv('solver_comparison.csv')
evaluation.results.to_pickle('detailed_results.pkl')
```

### Visualization
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Custom plotting
plt.figure(figsize=(10, 6))
sns.boxplot(data=evaluation.results, x='prior', y='mean_localization_error', hue='solver')
plt.title('Localization Error by Solver and Prior')
plt.show()
```

### Statistical Analysis
```python
from scipy import stats

# Statistical comparison
mne_results = evaluation.results[evaluation.results['solver'] == 'MNE']['mean_localization_error']
loreta_results = evaluation.results[evaluation.results['solver'] == 'LORETA']['mean_localization_error']

t_stat, p_value = stats.ttest_ind(mne_results, loreta_results)
print(f"MNE vs LORETA: t={t_stat:.3f}, p={p_value:.3f}")
```

## Contributing

To add support for new solvers:
1. Ensure solver inherits from `BaseSolver`
2. Implement `make_inverse_operator()` method
3. Add solver name mapping in `_create_solver_from_name()`
4. Test with evaluation system

To add new priors:
1. Add to `PriorEnum` in `priors.py`
2. Define appropriate simulation parameters
3. Test across multiple solvers
