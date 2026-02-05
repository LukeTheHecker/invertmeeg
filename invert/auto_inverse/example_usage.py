"""
Example usage of the AutoInverse class.

This script demonstrates how to use the AutoInverse functionality to automatically
select the best inverse solver based on simulated data with different prior assumptions.
"""

import logging

import mne
import numpy as np

from invert.auto_inverse.auto_inverse import AutoInverse
from invert.models.priors import PriorEnum

logger = logging.getLogger(__name__)


def create_example_forward():
    """Create a simple forward model for testing purposes."""
    # This is just an example - in practice you would load your own forward model
    mne.create_info(
        ch_names=[f"EEG_{i:03d}" for i in range(64)], sfreq=1000, ch_types="eeg"
    )

    # Create a simple source space (this is simplified for demo)
    # In practice, you would use mne.setup_source_space() and mne.make_forward_solution()
    logger.info("Note: This example requires a real forward model to work properly.")
    logger.info("Please replace this with your actual forward model.")
    return None


def example_usage():
    """Demonstrate usage of AutoInverse with different priors."""

    # Load your forward model (replace with actual forward model)
    forward = create_example_forward()
    if forward is None:
        logger.info("Skipping example - no forward model available")
        return

    # Create some dummy data (in practice, this would be your real EEG/MEG data)
    info = mne.create_info(
        ch_names=[f"EEG_{i:03d}" for i in range(64)], sfreq=1000, ch_types="eeg"
    )
    data = np.random.randn(64, 1000)
    evoked = mne.EvokedArray(data, info)

    # Test different priors - now using string parameters!
    priors_to_test = ["dipole", "patch", "broad", "focal", "widespread"]

    for prior_str in priors_to_test:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Testing with '{prior_str}' prior")
        logger.info(f"{'=' * 60}")

        # Create AutoInverse instance with string parameter
        auto_inverse = AutoInverse(
            prior=prior_str,  # Now accepts strings!
            n_samples=50,  # Use fewer samples for faster testing
            snr="auto",
        )

        # Fit and get recommendations
        auto_inverse.fit(evoked, forward)

        # Print results
        auto_inverse.print_summary()

        # Get recommended solver
        recommended = auto_inverse.get_recommended_solver()
        logger.info(f"Recommended solver for {prior_str} sources: {recommended}")

        logger.info("You can now use this solver for your real data:")
        logger.info("from invert import Solver")
        logger.info(f"solver = Solver('{recommended}')")
        logger.info("result = solver.solve(your_data, forward)")

    # Demonstrate error handling for invalid priors
    logger.info(f"\n{'=' * 60}")
    logger.info("Demonstrating error handling for invalid priors:")
    logger.info(f"{'=' * 60}")

    try:
        auto_inverse = AutoInverse(prior="invalid_prior")
    except ValueError as e:
        logger.info(f"Caught expected error: {e}")

    # Show all valid prior options
    logger.info("Valid prior options:")
    valid_options = PriorEnum.get_valid_strings()
    for i, option in enumerate(valid_options, 1):
        logger.info(f"{i:2d}. {option}")


if __name__ == "__main__":
    example_usage()
