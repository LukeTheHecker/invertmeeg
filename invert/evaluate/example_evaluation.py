"""
Example usage of the Evaluation class for comparing inverse solvers.

This script demonstrates how to use the comprehensive evaluation system
to compare multiple inverse solution algorithms across different source configurations.
"""

import logging

import mne
import numpy as np
from mne.datasets import sample

from invert.evaluate.evaluation import Evaluation
from invert.models.priors import PriorEnum

logger = logging.getLogger(__name__)


def create_sample_forward_model():
    """Create a simple forward model for testing."""
    # Use MNE's sample dataset
    data_path = sample.data_path()
    data_path / "subjects"

    # Load forward model
    fname_fwd = data_path / "MEG" / "sample" / "sample_audvis-meg-eeg-oct-6-fwd.fif"

    if fname_fwd.exists():
        forward = mne.read_forward_solution(fname_fwd, verbose=False)
        # Convert to surface orientation and reduce source space for faster testing
        forward = mne.convert_forward_solution(
            forward, surf_ori=True, force_fixed=True, verbose=False
        )

        # Optionally subsample for faster testing
        # forward = mne.pick_channels_forward(forward, include=['EEG001', 'EEG002', ...])

        return forward
    else:
        # Create a simple synthetic forward model if sample data not available
        logger.info("Sample data not found, creating synthetic forward model...")
        return create_synthetic_forward_model()


def create_synthetic_forward_model():
    """Create a synthetic forward model for testing when sample data is not available."""
    n_channels = 64
    n_sources = 1000

    # Create random leadfield
    leadfield = np.random.randn(n_channels, n_sources)

    # Create info
    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)], sfreq=1000, ch_types="eeg"
    )

    # Create minimal source space
    pos = np.random.randn(n_sources, 3)

    # Create forward solution dictionary
    forward = {
        "sol": {"data": leadfield},
        "info": info,
        "src": [{"rr": pos, "use_tris": None, "inuse": np.ones(n_sources, dtype=bool)}],
    }

    return forward


def run_basic_evaluation_example():
    """Run a basic evaluation example."""
    logger.info("Creating Evaluation Example")
    logger.info("=" * 50)

    # Create forward model
    try:
        forward = create_sample_forward_model()
        logger.info("Forward model loaded successfully")
    except Exception as e:
        logger.warning(f"Using synthetic forward model: {e}")
        forward = create_synthetic_forward_model()

    # Define solvers to compare - use string names (easier)
    solvers = ["MNE"]  # Start with just one solver for testing

    # Define priors to test (using subset for faster testing)
    priors = [PriorEnum.DIPOLE]  # Start with just one prior for testing

    # Create evaluation instance
    evaluation = Evaluation(
        forward=forward,
        solvers=solvers,
        priors=priors,
        n_samples=5,  # Very small number for quick testing
        random_seed=42,
        verbose=2,
    )

    # Run evaluation
    logger.info("Starting evaluation...")
    evaluation.evaluate()

    # Display results
    if not evaluation.results.empty:
        logger.info("Results Summary:")
        logger.info(evaluation.summary.to_string(index=False))

        # Check for NaN values
        n_valid_mle = evaluation.results["mean_localization_error"].notna().sum()
        n_valid_emd = evaluation.results["emd"].notna().sum()
        logger.info(
            f"Valid results: MLE={n_valid_mle}/{len(evaluation.results)}, EMD={n_valid_emd}/{len(evaluation.results)}"
        )
    else:
        logger.warning("No results generated")

    return evaluation


def run_comprehensive_evaluation_example():
    """Run a more comprehensive evaluation with all priors."""
    logger.info("Comprehensive Evaluation Example")
    logger.info("=" * 50)

    # Create forward model
    try:
        forward = create_sample_forward_model()
    except Exception:
        forward = create_synthetic_forward_model()

    # Use all available solvers (if they exist)
    solvers = ["MNE", "LORETA"]  # Add more as available

    # Use all priors
    priors = list(PriorEnum)

    # Create evaluation instance
    evaluation = Evaluation(
        forward=forward,
        solvers=solvers,
        priors=priors,
        n_samples=20,  # Moderate number for comprehensive testing
        random_seed=42,
        verbose=1,
    )

    # Run evaluation
    evaluation.evaluate()

    # Create visualization (if matplotlib available)
    try:
        evaluation.plot_results("mean_localization_error")
        evaluation.plot_results("emd")
    except Exception as e:
        logger.warning(f"Could not create plots: {e}")

    return evaluation


if __name__ == "__main__":
    # Run basic example
    basic_eval = run_basic_evaluation_example()

    # Run comprehensive example (commented out for speed)
    # comprehensive_eval = run_comprehensive_evaluation_example()

    logger.info("Evaluation examples completed!")
    logger.info(
        "To use in your own code:\n"
        "from invert.evaluate.evaluation import Evaluation\n"
        "from invert.models.priors import PriorEnum\n\n"
        "# Create evaluation\n"
        "eval = Evaluation(forward, solvers=['MNE', 'LORETA'], n_samples=100)\n\n"
        "# Run evaluation\n"
        "results = eval.evaluate()\n\n"
        "# Access results\n"
        "print(eval.summary)  # Summary statistics\n"
        "print(eval.results)  # Detailed results\n\n"
        "# Get best solver\n"
        "best = eval.get_best_solver('dipole', 'mean_localization_error')"
    )
