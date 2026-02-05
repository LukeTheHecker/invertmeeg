"""
Comprehensive evaluation system for comparing inverse solution algorithms.

This module provides the Evaluation class for systematically comparing
multiple inverse solvers across different source configurations and datasets.
"""

import logging
import time
from typing import TYPE_CHECKING, Any, Optional, Union

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns

from ..invert import Solver
from ..models.priors import PriorEnum
from ..simulate import SimulationGenerator
from ..util.util import pos_from_forward
from .evaluate import eval_emd, eval_mean_localization_error

if TYPE_CHECKING:
    from ..solvers.base import BaseSolver

logger = logging.getLogger(__name__)


class Evaluation:
    """
    Comprehensive evaluation system for comparing inverse solution algorithms.

    This class enables systematic comparison of multiple inverse solvers across
    various source configurations defined by prior knowledge. It simulates
    realistic EEG/MEG data based on different source patterns and evaluates
    solver performance using established metrics.

    Parameters
    ----------
    forward : mne.Forward
        The forward solution containing source space and leadfield matrix
    solvers : List[BaseSolver] or List[str]
        List of solver instances or solver names to evaluate
    priors : List[PriorEnum] or List[str], optional
        List of priors to test. If None, tests all available priors
    n_samples : int, optional
        Number of samples to simulate per prior. Default is 100
    random_seed : int, optional
        Random seed for reproducible results. Default is 42
    verbose : int, optional
        Verbosity level (0=silent, 1=progress, 2=detailed). Default is 1

    Attributes
    ----------
    results : pd.DataFrame
        Detailed results dataframe with all metrics
    summary : pd.DataFrame
        Summary statistics by solver and prior
    """

    def __init__(
        self,
        forward: mne.Forward,
        info: mne.Info,
        solvers: list[Union["BaseSolver", str]],
        priors: Optional[list[Union[PriorEnum, str]]] = None,
        n_samples: int = 100,
        n_timepoints: Optional[int] = None,
        random_seed: int = 42,
        alpha: Union[float, str] = "auto",
        verbose: int = 1,
    ):
        self.verbose = verbose
        self.alpha = alpha
        self.forward = forward
        self.info = info
        self.solvers = self._validate_solvers(solvers)
        self.priors = self._validate_priors(priors)
        self.n_samples = n_samples
        self.n_timepoints = n_timepoints
        self.random_seed = random_seed

        # Track which solvers need fitting (those given as strings)
        self.solvers_need_fitting = []
        for _i, solver_input in enumerate(solvers):
            self.solvers_need_fitting.append(isinstance(solver_input, str))

        # Extract forward model info
        self.leadfield = forward["sol"]["data"]
        self.pos = pos_from_forward(forward)
        self.adjacency = mne.spatial_src_adjacency(forward["src"], verbose=0)

        # Pre-compute distance matrix (used in metrics)
        from scipy.spatial.distance import cdist

        self.distance_matrix = cdist(self.pos, self.pos)

        # Results storage
        self.results = None
        self.summary = None
        self.detailed_results: list[dict[str, Any]] = []

    def _validate_solvers(self, solvers):
        """Validate and prepare solver instances."""
        validated_solvers = []

        for solver in solvers:
            if isinstance(solver, str):
                # Create solver instance from string name
                solver_instance = Solver(solver)  # , verbose=self.verbose)
                validated_solvers.append(solver_instance)
            else:
                # Assume it's already a solver instance
                validated_solvers.append(solver)

        return validated_solvers

    def _create_solver_from_name(self, solver_name: str):
        """Create solver instance from string name."""
        # Import solvers dynamically to avoid circular imports
        from .. import solvers as solver_module

        # Map common solver names to classes
        solver_map = {
            "MNE": solver_module.SolverMNE,  # type: ignore[attr-defined]
            "LORETA": solver_module.SolverLORETA,  # type: ignore[attr-defined]
            "sLORETA": solver_module.SolverStandardizedLORETA,  # type: ignore[attr-defined]
            "eLORETA": solver_module.SolverExactLORETA,  # type: ignore[attr-defined]
            "Champagne": solver_module.SolverChampagne,  # type: ignore[attr-defined]
            "MUSIC": solver_module.SolverMUSIC,  # type: ignore[attr-defined]
            "LCMV": solver_module.SolverLCMVBeamformer,  # type: ignore[attr-defined]
            "MVAB": solver_module.SolverMVABeamformer,  # type: ignore[attr-defined]
            "S-MAP": solver_module.SolverSMAP,  # type: ignore[attr-defined]
            "APSE": solver_module.SolverAPSE,  # type: ignore[attr-defined]
        }

        if solver_name in solver_map:
            return solver_map[solver_name]()
        else:
            raise ValueError(f"Unknown solver: {solver_name}")

    def _validate_priors(self, priors):
        """Validate and prepare prior configurations."""
        if priors is None:
            # Use all available priors
            return list(PriorEnum)

        validated_priors = []
        for prior in priors:
            if isinstance(prior, str):
                validated_priors.append(PriorEnum.from_string(prior))
            elif isinstance(prior, PriorEnum):
                validated_priors.append(prior)
            else:
                raise ValueError(f"Invalid prior type: {type(prior)}")

        return validated_priors

    def evaluate(self) -> dict[str, Any]:
        """
        Run comprehensive evaluation comparing all solvers across all priors.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing results summary, detailed results, and performance metrics
        """
        if self.verbose >= 1:
            logger.info("Starting comprehensive inverse solver evaluation...")
            logger.info(
                f"   Solvers: {[s.name if hasattr(s, 'name') else str(type(s).__name__) for s in self.solvers]}"
            )
            logger.info(f"   Priors: {[p.value.name for p in self.priors]}")
            logger.info(f"   Samples per prior: {self.n_samples}")

        # Reset results
        self.detailed_results = []

        # Evaluate each prior
        for prior in self.priors:
            if self.verbose >= 1:
                logger.info(f"Evaluating {prior.value.name} sources...")

            prior_results = self._evaluate_prior(prior)
            self.detailed_results.extend(prior_results)

        # Process and summarize results
        self._process_results()

        # Print summary
        if self.verbose >= 1:
            self._print_summary()

        return {
            "summary": self.summary,
            "detailed_results": self.results,
            "evaluation_info": {
                "n_samples": self.n_samples,
                "n_solvers": len(self.solvers),
                "n_priors": len(self.priors),
                "random_seed": self.random_seed,
            },
        }

    def _evaluate_prior(self, prior: PriorEnum) -> list[dict]:
        """Evaluate all solvers for a specific prior configuration."""
        prior_results = []

        # Generate simulation data for this prior
        sim_data = self._generate_simulation_data(prior)

        # Test each solver
        for solver_idx, solver in enumerate(self.solvers):
            solver_name = (
                solver.name if hasattr(solver, "name") else type(solver).__name__
            )
            needs_fitting = self.solvers_need_fitting[solver_idx]

            if self.verbose >= 2:
                logger.info(f"  Testing {solver_name}...")
            elif self.verbose >= 1:
                logger.info(f"  {solver_name}...")

            solver_results = self._evaluate_solver_on_data(
                solver, prior, sim_data, needs_fitting
            )
            prior_results.extend(solver_results)

            if self.verbose >= 1:
                avg_mle = np.nanmean(
                    [r["mean_localization_error"] for r in solver_results]
                )
                avg_emd = np.nanmean([r["emd"] for r in solver_results])
                avg_temporal_corr = np.nanmean(
                    [r["temporal_corr"] for r in solver_results]
                )
                logger.info(
                    f"MLE: {avg_mle:.2f}mm, EMD: {avg_emd:.2f}, Temporal Corr: {avg_temporal_corr:.2f}"
                )

        return prior_results

    def _generate_simulation_data(self, prior: PriorEnum) -> list[dict]:
        """Generate simulation data for a specific prior."""
        params = prior.value.sim_params

        # Create generator with prior-specific parameters
        sim_gen = SimulationGenerator(
            fwd=self.forward,
            batch_size=self.n_samples,
            batch_repetitions=1,
            n_sources=params["n_sources"],
            n_orders=params["n_orders"],
            amplitude_range=params["amplitude_range"],
            n_timepoints=params["n_timepoints"]
            if self.n_timepoints is None
            else self.n_timepoints,
            snr_range=params["snr_range"],
            random_seed=self.random_seed,
            normalize_leadfield=False,
            verbose=0,
        )

        # Get one batch
        x_batch, y_batch, info_batch = next(sim_gen.generate())

        if self.verbose >= 2:
            logger.debug(
                f"    Generated data shapes: X={x_batch.shape}, Y={y_batch.shape}"
            )

        # Convert to list of samples
        sim_data = []
        for i in range(self.n_samples):
            # x_batch[i] shape: [n_channels, n_timepoints]
            # y_batch[i] shape: [n_dipoles, n_timepoints]
            # Already in correct format!
            sim_data.append(
                {
                    "eeg_data": x_batch[i],
                    "source_data": y_batch[i],
                    "sim_info": info_batch.iloc[i].to_dict(),
                }
            )

        return sim_data

    def _evaluate_solver_on_data(
        self, solver, prior: PriorEnum, sim_data: list[dict], needs_fitting: bool = True
    ) -> list[dict]:
        """Evaluate a single solver on simulation data."""
        solver_name = (
            solver.name if hasattr(solver, "name") else str(solver.solver_name)
        )
        solver_results = []

        # Get info
        info = self.info

        try:
            # Apply to all samples, fitting per sample if required
            for i, sample in enumerate(sim_data):
                try:
                    # Prepare data - already in correct format now
                    x_sample = sample["eeg_data"]  # (n_channels, n_timepoints)
                    y_true = sample["source_data"]  # (n_sources, n_timepoints)

                    # Create EvokedArray from the data (already in channels x time format)
                    evoked = mne.EvokedArray(x_sample, info, tmin=0)

                    # # Compute common average reference
                    # evoked.set_eeg_reference("average", projection=True, verbose=0).apply_proj(verbose=0)

                    # Fit the solver for this specific sample if needed
                    if needs_fitting:
                        fit_start = time.time()
                        solver.make_inverse_operator(
                            self.forward, evoked, alpha=self.alpha
                        )
                        fit_time = time.time() - fit_start
                    else:
                        fit_time = 0.0

                    # Apply solver following auto_inverse.py pattern
                    start_time = time.time()
                    stc_hat = solver.apply_inverse_operator(evoked)
                    apply_time = time.time() - start_time

                    # Extract data from the source time course
                    if hasattr(stc_hat, "data"):
                        y_pred = stc_hat.data
                    else:
                        y_pred = stc_hat

                    # Calculate metrics on individual timepoints instead of temporally averaged sources
                    if y_pred.ndim > 1 and y_true.ndim > 1:
                        # Sample 10 equally spaced timepoints
                        n_timepoints = y_pred.shape[-1]
                        n_samples = 10
                        timepoint_indices = np.linspace(
                            0, n_timepoints - 1, n_samples, dtype=int
                        )

                        # Compute metrics for each timepoint
                        mle_values = []
                        emd_values = []

                        for t_idx in timepoint_indices:
                            y_pred_t = np.abs(y_pred[:, t_idx])
                            y_true_t = np.abs(y_true[:, t_idx])

                            # Calculate metrics for this timepoint
                            timepoint_metrics = self._calculate_metrics(
                                y_true_t, y_pred_t
                            )
                            mle_values.append(
                                timepoint_metrics["mean_localization_error"]
                            )
                            emd_values.append(timepoint_metrics["emd"])

                        # Calculate temporal correlation out of the loop
                        # Because all time points need to be evaluated
                        temporal_corr = self.eval_temporal_correlation(y_true, y_pred)

                        # Average the metrics across timepoints
                        metrics = {
                            "mean_localization_error": np.nanmean(mle_values),
                            "emd": np.nanmean(emd_values),
                            "temporal_corr": temporal_corr,
                            "spatial_dispersion": np.nan,  # Not calculated
                            "average_precision": np.nan,  # Not calculated
                        }
                    else:
                        logger.debug(f"1D data: {y_pred.shape}, {y_true.shape}")
                        # Fallback to original behavior for 1D data
                        if y_pred.ndim > 1:
                            y_pred = np.abs(y_pred).mean(axis=-1)
                        if y_true.ndim > 1:
                            y_true_1d = np.abs(y_true).mean(axis=-1)
                        else:
                            y_true_1d = y_true

                        # Calculate metrics following auto_inverse.py pattern
                        metrics = self._calculate_metrics(y_true_1d, y_pred)  # type: ignore[assignment]

                    # Store results
                    result = {
                        "solver": solver_name,
                        "prior": prior.value.name,
                        "prior_description": prior.value.description,
                        "sample_idx": i,
                        "fit_time": fit_time,
                        "apply_time": apply_time,
                        **metrics,
                        **sample["sim_info"],
                    }
                    solver_results.append(result)

                except Exception as e:
                    if self.verbose >= 2:
                        logger.warning(f"Sample {i} failed for {solver_name}: {e}")
                    # Add failed result with proper sim_info handling
                    failed_result = {
                        "solver": solver_name,
                        "prior": prior.value.name,
                        "prior_description": prior.value.description,
                        "sample_idx": i,
                        "fit_time": np.nan,
                        "apply_time": np.nan,
                        "mean_localization_error": np.nan,
                        "emd": np.nan,
                        "temporal_corr": np.nan,
                        "spatial_dispersion": np.nan,
                        "average_precision": np.nan,
                    }
                    # Add sim_info if available
                    if "sim_info" in sample:
                        failed_result.update(sample["sim_info"])
                    solver_results.append(failed_result)

        except Exception as e:
            if self.verbose >= 2:
                logger.error(f"Solver {solver_name} failed completely: {e}")

            # Add failed results for all samples
            for i, sample in enumerate(sim_data):
                failed_result = {
                    "solver": solver_name,
                    "prior": prior.value.name,
                    "prior_description": prior.value.description,
                    "sample_idx": i,
                    "fit_time": np.nan,
                    "apply_time": np.nan,
                    "mean_localization_error": np.nan,
                    "emd": np.nan,
                    "temporal_corr": np.nan,
                    "spatial_dispersion": np.nan,
                    "average_precision": np.nan,
                }
                # Add sim_info if available
                if "sim_info" in sample:
                    failed_result.update(sample["sim_info"])
                solver_results.append(failed_result)

        return solver_results

    def eval_temporal_correlation(
        self, X_true: np.ndarray, X_est: np.ndarray, mode: str = "true"
    ) -> float:
        """
        Calculate temporal correlation between true and estimated source time courses.

        This metric measures how well temporal dynamics are preserved, allowing for
        spatial displacement. For each true source, we find the best matching
        estimated source based on temporal correlation.

        Parameters
        ----------
        X_true : np.ndarray, shape (n_dipoles, n_timepoints)
            True source time courses
        X_est : np.ndarray, shape (n_dipoles, n_timepoints)
            Estimated source time courses
        mode : str, optional
            - "true": For each true source, find best match in estimated (default)
            - "est": For each estimated source, find best match in true
            - "match": Use Hungarian algorithm to find optimal one-to-one matching
            - "bidirectional": Average of "true" and "est" modes

        Returns
        -------
        float
            Average temporal correlation score (0 to 1, higher is better)
            Returns np.nan if calculation fails
        """
        # Validate inputs
        if X_true.ndim != 2 or X_est.ndim != 2:
            logger.warning("Input arrays must be 2D (n_dipoles, n_timepoints)")
            return np.nan

        if X_true.shape[1] != X_est.shape[1]:
            logger.warning("Time dimensions must match")
            return np.nan

        n_true, n_time = X_true.shape
        X_est.shape[0]

        # Handle edge cases
        if n_time < 2:
            logger.warning("Need at least 2 timepoints for correlation")
            return np.nan

        # Identify active sources (those with non-zero activity)
        # Use threshold of 1% of max activity
        threshold = 0.01
        active_true = np.abs(X_true).max(axis=1) > threshold * np.abs(X_true).max()
        active_est = np.abs(X_est).max(axis=1) > threshold * np.abs(X_est).max()

        if not np.any(active_true) or not np.any(active_est):
            logger.warning("No active sources found")
            return np.nan

        # Extract active sources
        X_true_active = X_true[active_true]
        X_est_active = X_est[active_est]

        # Compute correlation matrix between all pairs
        # Shape: (n_true_active, n_est_active)
        corr_matrix = np.zeros((X_true_active.shape[0], X_est_active.shape[0]))

        for i in range(X_true_active.shape[0]):
            for j in range(X_est_active.shape[0]):
                # Pearson correlation of time courses
                true_tc = X_true_active[i]
                est_tc = X_est_active[j]

                # Normalize
                true_tc_norm = (true_tc - true_tc.mean()) / (true_tc.std() + 1e-10)
                est_tc_norm = (est_tc - est_tc.mean()) / (est_tc.std() + 1e-10)

                # Compute correlation (use absolute value to handle sign flips)
                corr = np.abs(np.corrcoef(true_tc_norm, est_tc_norm)[0, 1])
                corr_matrix[i, j] = corr

        # Handle NaN values in correlation matrix
        if np.any(np.isnan(corr_matrix)):
            logger.warning("NaN values in correlation matrix")
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Calculate metric based on mode
        if mode == "true":
            # For each true source, find best matching estimated source
            best_corr = np.max(corr_matrix, axis=1)
            temporal_corr = np.mean(best_corr)

        elif mode == "est":
            # For each estimated source, find best matching true source
            best_corr = np.max(corr_matrix, axis=0)
            temporal_corr = np.mean(best_corr)

        elif mode == "bidirectional":
            # Average of both directions
            best_corr_true = np.max(corr_matrix, axis=1).mean()
            best_corr_est = np.max(corr_matrix, axis=0).mean()
            temporal_corr = (best_corr_true + best_corr_est) / 2

        elif mode == "match":
            # Use Hungarian algorithm for optimal one-to-one matching
            # Maximize correlation = minimize negative correlation
            from scipy.optimize import linear_sum_assignment

            row_ind, col_ind = linear_sum_assignment(-corr_matrix)
            temporal_corr = np.mean(corr_matrix[row_ind, col_ind])

        else:
            raise ValueError(f"Invalid mode: {mode}")

        return temporal_corr

    def _calculate_metrics(
        self, source_true: np.ndarray, source_pred: np.ndarray
    ) -> dict[str, float]:
        """Calculate evaluation metrics between true and predicted sources."""
        try:
            if self.verbose >= 2:
                logger.debug(
                    f"Metric calc - True shape: {source_true.shape}, Pred shape: {source_pred.shape}"
                )
                logger.debug(
                    f"Metric calc - True range: [{source_true.min():.6f}, {source_true.max():.6f}]"
                )
                logger.debug(
                    f"Metric calc - Pred range: [{source_pred.min():.6f}, {source_pred.max():.6f}]"
                )

            # Use pre-computed distance matrix
            # Calculate MLE following auto_inverse.py pattern
            mle = eval_mean_localization_error(
                source_true[:, np.newaxis],
                source_pred[:, np.newaxis],
                self.adjacency,
                self.adjacency,
                self.pos,
                self.pos,
                self.distance_matrix,
                mode="match",
            )

            # Calculate EMD following auto_inverse.py pattern
            emd = eval_emd(self.distance_matrix, source_true, source_pred)

            if self.verbose >= 2:
                logger.debug(f"Computed MLE: {mle}, EMD: {emd}")

            return {
                "mean_localization_error": mle,
                "emd": emd,
                "temporal_corr": np.nan,
                "spatial_dispersion": np.nan,  # Not calculated in auto_inverse.py
                "average_precision": np.nan,  # Not calculated in auto_inverse.py
            }

        except Exception as e:
            if self.verbose >= 2:
                logger.warning(f"Metric calculation failed: {e}", exc_info=True)
            return {
                "mean_localization_error": np.nan,
                "emd": np.nan,
                "temporal_corr": np.nan,
                "spatial_dispersion": np.nan,
                "average_precision": np.nan,
            }

    def _process_results(self):
        """Process raw results into structured dataframes."""
        # Create detailed results dataframe
        self.results = pd.DataFrame(self.detailed_results)

        # Create summary statistics
        if not self.results.empty:
            # Group by solver and prior
            groupby_cols = ["solver", "prior", "prior_description"]

            summary_stats = (
                self.results.groupby(groupby_cols)
                .agg(
                    {
                        "mean_localization_error": ["mean", "std", "median", "count"],
                        "emd": ["mean", "std", "median"],
                        "temporal_corr": ["mean", "std", "median"],
                        "spatial_dispersion": ["mean", "std", "median"],
                        "average_precision": ["mean", "std", "median"],
                        "fit_time": ["mean", "std"],
                        "apply_time": ["mean", "std"],
                    }
                )
                .round(4)
            )

            # Flatten column names
            summary_stats.columns = [
                "_".join(col).strip() for col in summary_stats.columns
            ]
            summary_stats = summary_stats.reset_index()

            self.summary = summary_stats
        else:
            self.summary = pd.DataFrame()

    def _print_summary(self):
        """Print a nice summary of the evaluation results."""
        logger.info("=" * 80)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 80)

        if self.summary.empty:
            logger.warning("No successful results to display")
            return

        logger.info("MEAN LOCALIZATION ERROR (mm) - Lower is better")
        logger.info("-" * 60)

        # Create pivot table for MLE
        mle_pivot = self.summary.pivot(
            index="solver", columns="prior", values="mean_localization_error_mean"
        )

        # Print formatted table
        logger.info("\n" + mle_pivot.to_string(float_format="{:.2f}".format))

        logger.info("EARTH MOVER'S DISTANCE - Lower is better")
        logger.info("-" * 60)

        # Create pivot table for EMD
        emd_pivot = self.summary.pivot(
            index="solver", columns="prior", values="emd_mean"
        )

        logger.info("\n" + emd_pivot.to_string(float_format="{:.4f}".format))

        logger.info("TEMPORAL CORRELATION - Higher is better")
        logger.info("-" * 60)
        # Create pivot table for temporal correlation
        temporal_corr_pivot = self.summary.pivot(
            index="solver", columns="prior", values="temporal_corr_mean"
        )

        logger.info("\n" + temporal_corr_pivot.to_string(float_format="{:.4f}".format))

        # Print best performers
        logger.info("BEST PERFORMERS")
        logger.info("-" * 60)

        for prior in self.priors:
            prior_data = self.summary[self.summary["prior"] == prior.value.name]
            if not prior_data.empty:
                best_mle = prior_data.loc[
                    prior_data["mean_localization_error_mean"].idxmin()
                ]
                best_emd = prior_data.loc[prior_data["emd_mean"].idxmin()]
                best_temporal_corr = prior_data.loc[
                    prior_data["temporal_corr_mean"].idxmax()
                ]

                logger.info(
                    f"\n{prior.value.name.upper()} sources ({prior.value.description}):"
                )
                logger.info(
                    f"  Best MLE: {best_mle['solver']} ({best_mle['mean_localization_error_mean']:.2f}mm)"
                )
                logger.info(
                    f"  Best EMD: {best_emd['solver']} ({best_emd['emd_mean']:.4f})"
                )
                logger.info(
                    f"  Best Temporal Correlation: {best_temporal_corr['solver']} ({best_temporal_corr['temporal_corr_mean']:.4f})"
                )

        logger.info("PERFORMANCE TIMING")
        logger.info("-" * 60)

        timing_summary = (
            self.results.groupby("solver")
            .agg({"fit_time": "mean", "apply_time": "mean"})
            .round(3)
        )

        logger.info("\n" + timing_summary.to_string())

        logger.info("=" * 80)
        logger.info(
            "Evaluation complete! Use .summary and .results for detailed analysis."
        )
        logger.info("=" * 80)

    def plot_results(
        self,
        metric: str = "mean_localization_error",
        percentile: Optional[int] = None,
        save_path: Optional[str] = None,
    ):
        """
        Create visualization of evaluation results.

        Parameters
        ----------
        metric : str
            Metric to plot ('mean_localization_error', 'emd', 'spatial_dispersion', 'average_precision')
        percentile : int, optional
            If specified, plot the given percentile (e.g., 10 for 10th percentile to show worst-case).
            If None (default), plots the median with 95% CI error bars.
        save_path : str, optional
            Path to save the plot
        """
        if self.results is None or self.results.empty:
            logger.warning("No results to plot. Run evaluate() first.")
            return

        plt.figure(figsize=(12, 8))

        # Determine ranking direction (lower is better for most metrics except average_precision)
        lower_is_better = metric not in ["average_precision", "temporal_corr"]

        # Compute overall solver order based on aggregated performance
        perf = (
            self.results[["solver", metric]]
            .dropna(subset=[metric])
            .groupby("solver", as_index=False)
            # .median()
            .mean()
        )
        perf = perf.sort_values(by=metric, ascending=lower_is_better)
        solver_order = perf["solver"].tolist()

        # Professional, publication-friendly style and palette
        sns.set_theme(style="whitegrid", context="talk")
        palette = sns.color_palette("colorblind", n_colors=max(len(solver_order), 3))
        solver_to_color = {
            s: palette[i % len(palette)] for i, s in enumerate(solver_order)
        }

        # Plot with percentile or default behavior
        if percentile is not None:
            # Compute percentile aggregation
            def percentile_func(x):
                return np.nanpercentile(x, percentile)

            aggregated = (
                self.results[["solver", "prior", metric]]
                .dropna(subset=[metric])
                .groupby(["solver", "prior"], as_index=False)
                .agg({metric: percentile_func})
            )

            sns.barplot(
                data=aggregated,
                x="prior",
                y=metric,
                hue="solver",
                hue_order=solver_order,
                palette=solver_to_color,
                errorbar=None,
            )
            title_suffix = f" ({percentile}th Percentile)"
        else:
            # Default: median with 95% CI
            sns.barplot(
                data=self.results,
                x="prior",
                y=metric,
                hue="solver",
                hue_order=solver_order,
                palette=solver_to_color,
                errorbar=("ci", 95),
            )
            title_suffix = " (Median with 95% CI)"

        # Labels and title
        metric_title = metric.replace("_", " ").title()
        ylabel_map = {
            "mean_localization_error": "Mean Localization Error (mm)",
            "temporal_corr": "Temporal Correlation",
            "emd": "Earth Mover's Distance",
            "spatial_dispersion": "Spatial Dispersion",
            "average_precision": "Average Precision",
        }
        plt.xlabel("Prior")
        plt.ylabel(ylabel_map.get(metric, metric_title))
        plt.title(f"{metric_title} by Solver and Prior{title_suffix}")
        plt.xticks(rotation=45, ha="right")

        # Legend formatting
        plt.legend(
            title="Solver", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False
        )

        sns.despine()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)

        plt.show()

    def get_best_solver(
        self, prior: str, metric: str = "mean_localization_error"
    ) -> str:
        """
        Get the best performing solver for a specific prior and metric.

        Parameters
        ----------
        prior : str
            Prior name to analyze
        metric : str
            Metric to optimize ('mean_localization_error', 'emd', etc.)

        Returns
        -------
        str
            Name of the best performing solver
        """
        if self.summary is None or self.summary.empty:
            raise ValueError("No results available. Run evaluate() first.")

        prior_data = self.summary[self.summary["prior"] == prior]
        if prior_data.empty:
            raise ValueError(f"No results found for prior: {prior}")

        metric_col = f"{metric}_mean"
        if metric_col not in prior_data.columns:
            raise ValueError(f"Metric {metric} not found in results")

        best_idx = prior_data[metric_col].idxmin()
        return prior_data.loc[best_idx, "solver"]
