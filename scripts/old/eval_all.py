"""
Evaluation script for inverse solution methods.

This script evaluates multiple inverse solution algorithms on simulated EEG/MEG data
and generates comparison plots and JSON results.

Main metrics:
- EMD (Earth Mover's Distance): Main focus metric
- Mean Localization Error: Spatial accuracy of source localization
- Spatial Dispersion: Measure of solution blurring
- Average Precision: Detection performance metric

Usage:
    python scripts/eval_all.py   # Run from project root
    # OR
    cd scripts && python eval_all.py

Outputs:
    - evaluation_results.json: Raw evaluation metrics for all solvers
    - evaluation_results_barplot.png: Bar plots comparing solver performance
"""

import sys
import os
# Add the project root to Python path so we can import from 'invert' module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
import mne

from invert.forward import get_info, create_forward_model
from invert.util import pos_from_forward
from invert.evaluate import eval_mean_localization_error
from invert import config
from invert import Solver

pp = dict(surface='inflated', hemi='both', verbose=0, cortex='low_contrast')

sampling = "ico3"
info = get_info(kind='biosemi64')
fwd = create_forward_model(info=info, sampling=sampling)
# fwd["sol"]["data"] /= np.linalg.norm(fwd["sol"]["data"], axis=0) 
pos = pos_from_forward(fwd)
leadfield = fwd["sol"]["data"]
n_chans, n_dipoles = leadfield.shape

source_model = fwd['src']
vertices = [source_model[0]['vertno'], source_model[1]['vertno']]
adjacency = mne.spatial_src_adjacency(fwd["src"], verbose=0)
distance_matrix = cdist(pos, pos)
fwd

from invert.simulate import generator
sim_params = dict(
    use_cov=False,
    return_mask=False,
    batch_repetitions=1,
    batch_size=20,
    n_sources=(2, 2),
    n_orders=(2, 2),
    snr_range=(5, 5),
    correlation_mode=None,
    amplitude_range=(1, 1),
    n_timecourses=200,
    n_timepoints=20,
    scale_data=False,
    add_forward_error=False,
    forward_error=0.1,
    inter_source_correlation=0.0,
    return_info=True,
    diffusion_parameter=0.1,
    beta_range = (1, 1),  # Determines the frequency spectrum of each simulted time course (1/f**beta)
    # correlation_mode="cholesky",
    # noise_color_coeff=0.5,
    normalize_leadfield=True,
    
    random_seed=None)

gen = generator(fwd, **sim_params)
x_test, y_test, sim_info = gen.__next__()

# solver_names = config.all_solvers
solver_names = ["MNE", "eLORETA", "Champagne", "LCMV", "APSE"]

from invert.evaluate import evaluate_all
import json

# Store inverse solutions for all samples
print(f"Computing inverse solutions for {x_test.shape[0]} samples...")
inverse_solutions = {}
from time import time

for solver_name in solver_names:
    print(f"Processing {solver_name}...")
    solver = Solver(solver_name)
    
    # Store solutions for all samples
    solver_solutions = []
    
    for i, x_sample in enumerate(x_test):
        print(f"  Sample {i+1}/{x_test.shape[0]}")
        evoked = mne.EvokedArray(x_sample.T, info)
        solver.make_inverse_operator(fwd, evoked, alpha="auto")
        stc_hat = solver.apply_inverse_operator(evoked)
        solver_solutions.append(stc_hat)
    
    inverse_solutions[solver_name] = solver_solutions

# Evaluate all solvers across all samples
print("Evaluating solvers...")
evaluation_results = {}

for solver_name in solver_names:
    print(f"Evaluating {solver_name}...")
    # try:
    stc_solutions = inverse_solutions[solver_name]
    
    # Store metrics for each sample
    sample_metrics = []
    
    for i, stc_hat in enumerate(stc_solutions):
        print(f"  Evaluating sample {i+1}/{len(stc_solutions)}")
        
        # Get the data arrays - assuming stc_hat has a .data attribute
        if hasattr(stc_hat, 'data'):
            y_pred = stc_hat.data
        else:
            y_pred = stc_hat
        
        # Use the corresponding ground truth sample
        y_true = y_test[i].T
        # convert from sparse to dense
        y_true = y_true
        
        print(f"    True data shape: {y_true.shape}")
        print(f"    Predicted data shape: {y_pred.shape}")

        print(f"    type(y_true): {type(y_true)}")
        print(f"    type(y_pred): {type(y_pred)}")
        print(f"    type(adjacency): {type(adjacency)}")
        print(f"    type(pos): {type(pos)}")
        
        
        # Evaluate using the evaluate_all function
        metrics = evaluate_all(
            y_true=y_true, 
            y_pred=y_pred, 
            adjacency_true=adjacency, 
            adjacency_pred=adjacency,  # Same source space
            pos_true=pos, 
            pos_pred=pos,  # Same source space
            mode="dle", 
            threshold=0.1
        )
        
        sample_metrics.append(metrics)
        print(f"    Sample {i+1} EMD: {metrics['EMD']:.4f}")
    
    # Store all sample metrics (without aggregation as requested)
    evaluation_results[solver_name] = sample_metrics
    
    # Calculate mean EMD for summary
    mean_emd = np.mean([m['EMD'] for m in sample_metrics])
    print(f"  {solver_name} Mean EMD: {mean_emd:.4f}")
        
    # except Exception as e:
    #     print(f"  Error evaluating {solver_name}: {str(e)}")
    #     # Create dummy metrics for all samples to keep the script running
    #     num_samples = len(x_test)
    #     evaluation_results[solver_name] = [{
    #         'Mean_Localization_Error': np.nan,
    #         'EMD': np.nan,
    #         'sd': np.nan,
    #         'average_precision': np.nan
    #     }] * num_samples

# Save results to JSON (all samples, no aggregation)
output_file = "evaluation_results.json"
print(f"\nSaving results to {output_file}")
with open(output_file, 'w') as f:
    # Convert numpy types to native Python types for JSON serialization
    json_results = {}
    for solver, sample_metrics_list in evaluation_results.items():
        json_results[solver] = []
        for sample_metrics in sample_metrics_list:
            sample_json = {}
            for metric, value in sample_metrics.items():
                if isinstance(value, np.ndarray):
                    sample_json[metric] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    sample_json[metric] = float(value)
                else:
                    sample_json[metric] = value
            json_results[solver].append(sample_json)
    
    json.dump(json_results, f, indent=2)

# Create bar plots for each metric (aggregated across samples)
print("\nCreating plots...")
metrics_to_plot = ['Mean_Localization_Error', 'EMD', 'sd', 'average_precision']

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, metric in enumerate(metrics_to_plot):
    ax = axes[i]
    
    # Extract and aggregate values for this metric across all samples
    solvers = list(evaluation_results.keys())
    values = []
    errors = []  # Standard errors for error bars
    
    for solver in solvers:
        sample_metrics = evaluation_results[solver]
        metric_values = [sample[metric] for sample in sample_metrics]
        
        # Filter out NaN values
        valid_values = [v for v in metric_values if not np.isnan(v)]
        
        if valid_values:
            mean_value = np.mean(valid_values)
            std_error = np.std(valid_values) / np.sqrt(len(valid_values))
            values.append(mean_value)
            errors.append(std_error)
        else:
            values.append(np.nan)
            errors.append(0)
    
    # Handle NaN values for plotting
    plot_values = [0 if np.isnan(v) else v for v in values]
    plot_errors = [0 if np.isnan(v) else e for v, e in zip(values, errors)]
    colors = ['red' if np.isnan(v) else 'steelblue' for v in values]
    
    # Create bar plot with error bars
    bars = ax.bar(solvers, plot_values, yerr=plot_errors, color=colors, 
                  capsize=5, alpha=0.8)
    ax.set_title(f'{metric} (Mean ± SE)')
    ax.set_ylabel('Value')
    
    # Rotate x-axis labels for better readability
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value, error in zip(bars, values, errors):
        height = bar.get_height()
        if np.isnan(value):
            label = 'NaN'
        else:
            label = f'{value:.3f}±{error:.3f}'
        ax.text(bar.get_x() + bar.get_width()/2., height + error,
                label, ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

# Save the plot
plot_filename = "evaluation_results_barplot.png"
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"Plot saved as {plot_filename}")

print("\nEvaluation completed!")
print(f"\nResults summary (averaged across {len(x_test)} samples):")
print("=" * 60)
for solver_name in solver_names:
    sample_metrics = evaluation_results[solver_name]
    
    # Calculate means across samples
    emd_values = [sample['EMD'] for sample in sample_metrics]
    mle_values = [sample['Mean_Localization_Error'] for sample in sample_metrics]
    
    # Filter out NaN values
    valid_emd = [v for v in emd_values if not np.isnan(v)]
    valid_mle = [v for v in mle_values if not np.isnan(v)]
    
    if valid_emd and valid_mle:
        mean_emd = np.mean(valid_emd)
        mean_mle = np.mean(valid_mle)
        std_emd = np.std(valid_emd)
        std_mle = np.std(valid_mle)
        print(f"{solver_name:<12}: EMD={mean_emd:.4f}±{std_emd:.4f}, MLE={mean_mle:.4f}±{std_mle:.4f}")
    else:
        print(f"{solver_name:<12}: EMD=NaN, MLE=NaN (evaluation failed)")

print(f"\nFiles created:")
print(f"  - {output_file} (contains all {len(x_test)} samples without aggregation)")
print(f"  - {plot_filename} (shows mean ± standard error)")
print("\nNote: Lower EMD and MLE values indicate better performance.")

