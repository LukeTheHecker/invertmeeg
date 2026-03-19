"""Cortical folding estimation from EEG source reconstruction on a sphere."""

from .experiment import create_sphere_forward, create_sphere_forward_template, run_accumulation, compute_geometry_metrics
from .estimator import CorticalFoldingEstimator
from .riemannian_transport import compute_transport_matrix, riemannian_leadfield_transport
