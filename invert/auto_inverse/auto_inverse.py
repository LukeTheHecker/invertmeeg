import logging
from typing import Optional, Union

import mne
import numpy as np
import pandas as pd
from mne import Epochs, EpochsArray, Evoked, EvokedArray, Forward
from scipy.spatial.distance import cdist

from .. import config
from ..evaluate.evaluate import eval_emd, eval_mean_localization_error
from ..invert import Solver
from ..models.priors import PriorEnum
from ..simulate import SimulationGenerator
from ..util import pos_from_forward

logger = logging.getLogger(__name__)

EVOKED_TYPES = [EvokedArray, Evoked]
EPOCHS_TYPES = [Epochs, EpochsArray]


class Report:
    """Class to store auto inverse results and generate recommendations."""

    def __init__(self, prior_type: str, results_df: pd.DataFrame):
        self.prior_type = prior_type
        self.results_df = results_df
        self.best_mle_solver = None
        self.best_emd_solver = None
        self.recommended_solver = None
        self._analyze_results()

    def _analyze_results(self):
        """Analyze results and determine best solvers."""
        # Find best solvers based on median performance
        median_results = (
            self.results_df.groupby("solver")
            .agg({"mle": "median", "emd": "median"})
            .reset_index()
        )

        self.best_mle_solver = median_results.loc[
            median_results["mle"].idxmin(), "solver"
        ]
        self.best_emd_solver = median_results.loc[
            median_results["emd"].idxmin(), "solver"
        ]

        # Choose recommendation based on prior type
        if self.prior_type.upper() == "DIPOLE":
            self.recommended_solver = self.best_mle_solver
            self.recommendation_metric = "MLE"
        else:
            self.recommended_solver = self.best_emd_solver
            self.recommendation_metric = "EMD"

    def generate_summary(self) -> str:
        """Generate a summary text of the results."""
        summary = f"Auto-Inverse Analysis Results for {self.prior_type} Prior:\n"
        summary += f"{'=' * 50}\n\n"

        summary += f"Best solver by MLE: {self.best_mle_solver}\n"
        summary += f"Best solver by EMD: {self.best_emd_solver}\n\n"

        summary += f"Recommended solver: {self.recommended_solver}\n"
        summary += f"Recommendation based on: {self.recommendation_metric} "
        summary += f"(optimal for {self.prior_type.lower()} sources)\n\n"

        # Add performance summary
        median_results = (
            self.results_df.groupby("solver")
            .agg({"mle": ["median", "std"], "emd": ["median", "std"]})
            .round(4)
        )

        # Sort by the recommended metric (ascending order for best performance first)
        if self.recommendation_metric == "MLE":
            # Sort by MLE median (lower is better)
            median_results = median_results.sort_values(("mle", "median"))
        else:
            # Sort by EMD median (lower is better)
            median_results = median_results.sort_values(("emd", "median"))

        summary += "Performance Summary (median ± std):\n"
        summary += f"Sorted by {self.recommendation_metric} (best first):\n"
        summary += median_results.to_string()

        return summary

    def get_recommended_solver(self) -> Optional[str]:
        """Get the recommended solver name."""
        return self.recommended_solver


class AutoInverse:
    def __init__(
        self,
        prior: Union[str, PriorEnum] = "patch",
        snr="auto",
        alpha="auto",
        n_samples=100,
        n_timepoints=20,
        verbose=True,
    ):
        """
        Initialize AutoInverse with flexible prior parameter handling.

        Parameters
        ----------
        prior : Union[str, PriorEnum], default="noprior"
            Prior type for source reconstruction. Can be:
            - String: "dipole", "patch", "broad", "noprior" (case-insensitive)
            - PriorEnum member: PriorEnum.DIPOLE, PriorEnum.PATCH, etc.
            - Aliases: "focal", "sparse", "localized" for dipole;
                      "moderate", "distributed" for patch;
                      "widespread", "extended", "diffuse" for broad;
                      "none", "uninformed", "default" for noprior
        snr : Union[str, float, tuple], default="auto"
            Signal-to-noise ratio for simulation. Can be:
            - "auto": Use default SNR range from prior settings
            - float: Fixed SNR value
            - tuple: (min_snr, max_snr) range
        alpha : str, default="auto"
            Regularization parameter (currently unused, kept for interface consistency)
        n_samples : int, default=100
            Number of samples to generate for evaluation
        verbose : bool, default=True
            Whether to print debugging information during evaluation

        Examples
        --------
        >>> # Using string parameters
        >>> auto_inv = AutoInverse(prior="dipole")
        >>> auto_inv = AutoInverse(prior="focal")  # alias for dipole
        >>> auto_inv = AutoInverse(prior="PATCH")  # case-insensitive

        >>> # Using enum members (backward compatibility)
        >>> auto_inv = AutoInverse(prior=PriorEnum.DIPOLE)

        >>> # With custom SNR
        >>> auto_inv = AutoInverse(prior="broad", snr=5.0)
        """
        # Validate and convert prior parameter
        self.prior = PriorEnum.validate_prior(prior)
        self.snr = snr
        self.alpha = alpha
        self.n_samples = n_samples
        self.n_timepoints = n_timepoints
        self.verbose = verbose
        self.report: Optional[Report] = None
        self.sim_params = None
        self.solvers_to_test = None
        self.simulation_config = None

    def fit(
        self,
        data: Union[EvokedArray, Evoked],
        forward: Forward,
    ) -> Optional[Report]:
        """
        Fit the auto inverse model to determine the best solver.

        Parameters
        ----------
        data : Union[EVOKED_TYPES, EPOCHS_TYPES]
            The EEG/MEG data (not used for simulation, but kept for interface consistency)
        forward : Forward
            The forward model to use for simulation and inversion

        Returns
        -------
        report : Report
            Report object containing results and recommendations
        """
        if data.info is not None:
            self.info = data.info
        # Step 1: Determine prior type and get simulation settings
        self._determine_prior_settings()

        # Step 2: Get list of solvers to test based on prior
        self._get_solvers_for_prior()

        # Step 3: Simulate data based on priors
        simulated_data = self._simulate_data(forward)

        # Step 4: Calculate inverse solutions with all selected solvers
        self.results = self._compute_inverse_solutions(simulated_data, forward)

        # Step 5: Create and return report
        self.report = Report(self.prior.value.name, self.results)
        return self.report

    def _determine_prior_settings(self):
        """Extract simulation settings from PriorEnum based on selected prior type."""
        self.sim_params = self.prior.value.sim_params.copy()

        # Handle auto SNR - use the range from sim_params
        if self.snr == "auto":
            self.sim_params["snr_range"] = self.sim_params["snr_range"]
        else:
            # If specific SNR provided, use it
            if isinstance(self.snr, (int, float)):
                self.sim_params["snr_range"] = (self.snr, self.snr)
            else:
                self.sim_params["snr_range"] = self.snr

    def _get_solvers_for_prior(self):
        """Get list of solvers from config based on the determined prior type."""
        # Use enum member name (DIPOLE, PATCH, BROAD, NOPRIOR) to index config
        prior_name = self.prior.name

        if prior_name in config.SOLVERS_AUTOINVERSE:
            self.solvers_to_test = config.SOLVERS_AUTOINVERSE[prior_name]
        else:
            # Fallback to a general set of solvers
            self.solvers_to_test = config.SOLVERS_AUTOINVERSE["NOPRIOR"]

        logger.info(
            f"Testing {len(self.solvers_to_test)} solvers for {prior_name} prior: {self.solvers_to_test}"
        )

    def _simulate_data(self, forward: Forward) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate data based on the prior settings.

        Parameters
        ----------
        forward : Forward
            The forward model

        Returns
        -------
        x_sim : np.ndarray
            Simulated EEG/MEG data [batch_size, n_channels, n_timepoints]
        y_sim : np.ndarray
            True source activity [batch_size, n_dipoles, n_timepoints]
        """
        logger.info(
            f"Simulating {self.n_samples} samples with {self.prior.value.name} prior..."
        )

        # Create generator with prior-specific parameters
        assert self.simulation_config is not None  # type: ignore[attr-defined]
        sim_gen = SimulationGenerator(
            fwd=forward,
            batch_size=self.n_samples,
            batch_repetitions=1,
            n_timepoints=self.n_timepoints,
            verbose=0,
            **self.sim_params,  # type: ignore[arg-type]
        )

        # Get one batch of simulated data
        x_sim, y_sim, info = next(sim_gen.generate())

        logger.info(f"Generated data shapes: X={x_sim.shape}, Y={y_sim.shape}")
        return x_sim, y_sim

    def _compute_inverse_solutions(
        self, simulated_data: tuple[np.ndarray, np.ndarray], forward: Forward
    ) -> pd.DataFrame:
        """
        Compute inverse solutions for all solvers and evaluate metrics.

        Parameters
        ----------
        simulated_data : Tuple[np.ndarray, np.ndarray]
            Simulated EEG/MEG data and true sources
        forward : Forward
            The forward model

        Returns
        -------
        results_df : pd.DataFrame
            DataFrame containing results for each solver and sample
        """
        x_sim, y_sim = simulated_data

        # Get positions for EMD and MLE calculation
        pos = pos_from_forward(forward)
        adjacency = mne.spatial_src_adjacency(forward["src"], verbose=0)
        distance_matrix = cdist(pos, pos)

        # Get info from forward model for creating EvokedArray
        info = self.info

        results: list = []

        assert self.solvers_to_test is not None
        logger.info(
            f"Computing inverse solutions for {len(self.solvers_to_test)} solvers..."
        )

        for solver_name in self.solvers_to_test:
            logger.info(f"Testing solver: {solver_name}")

            # try:
            # Create solver instance
            solver = Solver(solver_name)

            # Compute inverse for each sample
            for i in range(self.n_samples):
                try:
                    # Get current sample (already in correct shape)
                    # x_sample: [n_channels, n_timepoints]
                    # y_true: [n_dipoles, n_timepoints]
                    x_sample = x_sim[i]
                    y_true = y_sim[i]

                    # Debug: Print shapes
                    if i == 0 and self.verbose:
                        logger.debug(
                            f"  Sample 0 - x_sample shape: {x_sample.shape}, y_true shape: {y_true.shape}"
                        )

                    # Create EvokedArray from the data (already in channels x time format)
                    evoked = mne.EvokedArray(x_sample, info, tmin=0)

                    # Use the correct solver pattern from eval_all.py
                    solver.make_inverse_operator(forward, evoked, alpha="auto")
                    stc_hat = solver.apply_inverse_operator(evoked)

                    # Extract data from the source time course
                    if hasattr(stc_hat, "data"):
                        y_pred = stc_hat.data
                    else:
                        y_pred = stc_hat

                    # Debug: Print y_pred shape and check for NaN/inf
                    if i == 0 and self.verbose:
                        logger.debug(f"  Sample 0 - y_pred shape: {y_pred.shape}")
                        logger.debug(
                            f"  Sample 0 - y_pred has NaN: {np.any(np.isnan(y_pred))}, has inf: {np.any(np.isinf(y_pred))}"
                        )
                        logger.debug(
                            f"  Sample 0 - y_pred range: [{np.nanmin(np.abs(y_pred))}, {np.nanmax(np.abs(y_pred))}]"
                        )
                        logger.debug(
                            f"  Sample 0 - y_true has NaN: {np.any(np.isnan(y_true))}, has inf: {np.any(np.isinf(y_true))}"
                        )
                        logger.debug(
                            f"  Sample 0 - y_true range: [{np.nanmin(np.abs(y_true))}, {np.nanmax(np.abs(y_true))}]"
                        )

                    # Ensure y_pred and y_true are 1D for evaluation
                    if y_pred.ndim > 1:
                        y_pred = np.abs(y_pred).mean(axis=-1)
                    if y_true.ndim > 1:
                        y_true = np.abs(y_true).mean(axis=-1)

                    # Check if arrays are valid for metrics
                    if np.all(y_pred == 0) or np.all(y_true == 0):
                        if i == 0 and self.verbose:
                            logger.warning("  y_pred or y_true is all zeros")
                        mle = np.nan
                        emd = np.nan
                    elif np.any(np.isnan(y_pred)) or np.any(np.isnan(y_true)):
                        if i == 0 and self.verbose:
                            logger.warning("  y_pred or y_true contains NaN")
                        mle = np.nan
                        emd = np.nan
                    else:
                        mle = eval_mean_localization_error(
                            y_true[:, np.newaxis],
                            y_pred[:, np.newaxis],
                            adjacency,
                            adjacency,
                            pos,
                            pos,
                            distance_matrix,
                        )

                        emd = eval_emd(distance_matrix, y_true, y_pred)

                    results.append(
                        {"solver": solver_name, "sample": i, "mle": mle, "emd": emd}
                    )

                except Exception as e:
                    logger.error(f"  Error with sample {i}: {e}")
                    import traceback

                    if self.verbose:
                        traceback.print_exc()
                    results.append(
                        {
                            "solver": solver_name,
                            "sample": i,
                            "mle": np.nan,
                            "emd": np.nan,
                        }
                    )

            # except Exception as e:
            #     print(f"  Error creating solver {solver_name}: {e}")
            #     # Add NaN results for all samples for this solver
            #     for i in range(self.n_samples):
            #         results.append({
            #             'solver': solver_name,
            #             'sample': i,
            #             'mle': np.nan,
            #             'emd': np.nan
            #         })

        results_df = pd.DataFrame(results)
        return results_df

    def get_recommended_solver(self) -> str:
        """Get the recommended solver name."""
        if self.report is None:
            raise ValueError("Must call fit() first before getting recommendations")
        return self.report.get_recommended_solver()  # type: ignore[return-value]

    def print_summary(self):
        """Print the summary report."""
        if self.report is None:
            raise ValueError("Must call fit() first before printing summary")
        logger.info(self.report.generate_summary())
