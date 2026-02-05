from __future__ import annotations

import importlib.util as _ilu
import json
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import mne
import numpy as np
from pydantic import BaseModel
from tqdm import tqdm

_spec = _ilu.spec_from_file_location(
    "invert.evaluate.evaluate",
    str(Path(__file__).resolve().parents[1] / "evaluate" / "evaluate.py"),
)
assert _spec is not None
assert _spec.loader is not None
_eval_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_eval_mod)
evaluate_all = _eval_mod.evaluate_all
from invert.simulate import SimulationConfig, SimulationGenerator  # noqa: E402
from invert.solvers.base import BaseSolver  # noqa: E402
from invert.util.util import pos_from_forward  # noqa: E402

from .datasets import BENCHMARK_DATASETS, DatasetConfig  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Solver registry: maps short name -> (module_path, class_name)
# Lazy-imported to avoid loading all solvers at module level.
# ---------------------------------------------------------------------------

_SOLVER_REGISTRY: dict[str, tuple[str, str]] = {
    # Minimum-norm family
    "MNE": ("invert.solvers.minimum_norm.mne", "SolverMNE"),
    "GFTMNE": ("invert.solvers.minimum_norm.gft_mne", "SolverGFTMNE"),
    "WMNE": ("invert.solvers.minimum_norm.wmne", "SolverWMNE"),
    "dSPM": ("invert.solvers.minimum_norm.dspm", "SolverDSPM"),
    "MCE": ("invert.solvers.minimum_norm.minimum_l1_norm", "SolverMinimumL1Norm"),
    "GFTMCE": (
        "invert.solvers.minimum_norm.gft_minimum_l1_norm",
        "SolverGFTMinimumL1Norm",
    ),
    "MCE-GPT": (
        "invert.solvers.minimum_norm.minimum_l1_norm_gpt",
        "SolverMinimumL1NormGPT",
    ),
    "MCE-L1L2": (
        "invert.solvers.minimum_norm.minimum_l1_l2_norm",
        "SolverMinimumL1L2Norm",
    ),
    "SSLOFO": ("invert.solvers.minimum_norm.sslofo", "SolverSSLOFO"),
    "SMAP": ("invert.solvers.minimum_norm.smap", "SolverSMAP"),
    "EPIFOCUS": ("invert.solvers.minimum_norm.epifocus", "SolverEPIFOCUS"),
    "BasisFunctions": (
        "invert.solvers.minimum_norm.basis_functions",
        "SolverBasisFunctions",
    ),
    # "BackusGilbert": ("invert.solvers.minimum_norm.backus_gilbert", "SolverBackusGilbert"),
    "LAURA": ("invert.solvers.minimum_norm.laura", "SolverLAURA"),
    "TV": ("invert.solvers.minimum_norm.total_variation", "SolverTotalVariation"),
    # LORETA family
    "LORETA": ("invert.solvers.minimum_norm.loreta", "SolverLORETA"),
    "sLORETA": ("invert.solvers.minimum_norm.sloreta", "SolverSLORETA"),
    "eLORETA": ("invert.solvers.minimum_norm.eloreta", "SolverELORETA"),
    "SelfRegELORETA": (
        "invert.solvers.minimum_norm.self_regularized",
        "SolverSelfRegularizedELORETA",
    ),
    # Beamformers
    "MVAB": ("invert.solvers.beamformers.mvab", "SolverMVAB"),
    "LCMV": ("invert.solvers.beamformers.lcmv", "SolverLCMV"),
    "SMV": ("invert.solvers.beamformers.smv", "SolverSMV"),
    "WNMV": ("invert.solvers.beamformers.wnmv", "SolverWNMV"),
    "HOCMV": ("invert.solvers.beamformers.hocmv", "SolverHOCMV"),
    "MCMV": ("invert.solvers.beamformers.mcmv", "SolverMCMV"),
    "UNIG": ("invert.solvers.beamformers.unit_noise_gain", "SolverUnitNoiseGain"),
    "HOCMCMV": ("invert.solvers.beamformers.hocmcmv", "SolverHOCMCMV"),
    "SAM": ("invert.solvers.beamformers.sam", "SolverSAM"),
    # "EBB": ("invert.solvers.beamformers.ebb", "SolverEBB"),
    "ESMV": ("invert.solvers.beamformers.esmv", "SolverESMV"),
    "ESMV2": ("invert.solvers.beamformers.esmv2", "SolverESMV2"),
    "ESMV3": ("invert.solvers.beamformers.esmv3", "SolverESMV3"),
    "DeblurFlexESMV": (
        "invert.solvers.beamformers.deblur_flex_esmv",
        "SolverDeblurFlexESMV",
    ),
    "FlexESMV": ("invert.solvers.beamformers.flex_esmv", "SolverFlexESMV"),
    "SafeFlexESMV": ("invert.solvers.beamformers.safe_flex_esmv", "SolverSafeFlexESMV"),
    "SharpFlexESMV": (
        "invert.solvers.beamformers.sharp_flex_esmv",
        "SolverSharpFlexESMV",
    ),
    "SharpFlexESMV2": (
        "invert.solvers.beamformers.sharp_flex_esmv2",
        "SolverSharpFlexESMV2",
    ),
    "AdaptFlexESMV": (
        "invert.solvers.beamformers.adapt_flex_esmv",
        "SolverAdaptFlexESMV",
    ),
    "SSP-ESMV": ("invert.solvers.beamformers.ssp_esmv", "SolverSSPESMV"),
    "IR-ESMV": ("invert.solvers.beamformers.iresmv", "SolverIRESMV"),
    "SSP-IR-ESMV": ("invert.solvers.beamformers.ssp_iresmv", "SolverSSPIRESMV"),
    # "ReciPSIICOS-Plain": (
    #     "invert.solvers.beamformers.recipsiicos_plain",
    #     "SolverReciPSIICOSPlain",
    # ),
    # "ReciPSIICOS-Whitened": (
    #     "invert.solvers.beamformers.recipsiicos_whitened",
    #     "SolverReciPSIICOSWhitened",
    # ),
    # Empirical Bayes
    "Champagne": ("invert.solvers.bayesian.champagne", "SolverChampagne"),
    "NLChampagne": ("invert.solvers.bayesian.nl_champagne", "SolverNLChampagne"),
    "FlexChampagne": ("invert.solvers.bayesian.flex_champagne", "SolverFlexChampagne"),
    "FlexNLChampagne": (
        "invert.solvers.bayesian.flex_nl_champagne",
        "SolverFlexNLChampagne",
    ),
    "OmniChampagne": ("invert.solvers.bayesian.omni_champagne", "SolverOmniChampagne"),
    # Sparse Bayesian
    "MSP": ("invert.solvers.bayesian.msp", "SolverMSP"),
    "BCS": ("invert.solvers.bayesian.bcs", "SolverBCS"),
    "GammaMAP": ("invert.solvers.bayesian.gamma_map", "SolverGammaMAP"),
    "SourceMAP": ("invert.solvers.bayesian.source_map", "SolverSourceMAP"),
    "GammaMAPMSP": ("invert.solvers.bayesian.gamma_map_msp", "SolverGammaMAPMSP"),
    "SourceMAPMSP": ("invert.solvers.bayesian.source_map_msp", "SolverSourceMAPMSP"),
    # "CMEM": ("invert.solvers.bayesian.cmem", "SolverCMEM"),
    "SubspaceSBL": ("invert.solvers.bayesian.subspace_sbl", "SolverSubspaceSBL"),
    "SubspaceSBLPlus": (
        "invert.solvers.bayesian.subspace_sbl",
        "SolverSubspaceSBLPlus",
    ),
    "VBSBL": ("invert.solvers.bayesian.vb_sbl", "SolverVBSBL"),
    # MUSIC / subspace
    "MUSIC": ("invert.solvers.music", "SolverMUSIC"),
    # "ExSoMUSIC": ("invert.solvers.music", "SolverExSoMUSIC"),
    "FLEX-MUSIC": ("invert.solvers.music", "SolverFLEXMUSIC"),
    "FLEX-MUSIC-2": ("invert.solvers.music", "SolverFLEXMUSIC_2"),
    "AltProj": ("invert.solvers.music", "SolverAlternatingProjections"),
    "AdaptiveAltProj": ("invert.solvers.music", "SolverAdaptiveAlternatingProjections"),
    "SSM": ("invert.solvers.music", "SolverSignalSubspaceMatching"),
    "GenIterative": ("invert.solvers.music", "SolverGeneralizedIterative"),
    # Matching pursuit
    "OMP": ("invert.solvers.matching_pursuit", "SolverOMP"),
    "SOMP": ("invert.solvers.matching_pursuit", "SolverSOMP"),
    "COSAMP": ("invert.solvers.matching_pursuit", "SolverCOSAMP"),
    "REMBO": ("invert.solvers.matching_pursuit", "SolverREMBO"),
    "SP": ("invert.solvers.matching_pursuit", "SolverSP"),
    "SSP": ("invert.solvers.matching_pursuit", "SolverSSP"),
    # Smooth matching pursuit
    "SMP": ("invert.solvers.matching_pursuit", "SolverSMP"),
    "SSMP": ("invert.solvers.matching_pursuit", "SolverSSMP"),
    "SubSMP": ("invert.solvers.matching_pursuit", "SolverSubSMP"),
    "ISubSMP": ("invert.solvers.matching_pursuit", "SolverISubSMP"),
    # Other
    "APSE": ("invert.solvers.hybrids.apse", "SolverAPSE"),
    "Chimera": ("invert.solvers.hybrids.chimera", "SolverChimera"),
    "ECD": ("invert.solvers.dipoles", "SolverECD"),
    "SESAME": ("invert.solvers.dipoles", "SolverSESAME"),
    # Baseline
    "RandomNoise": ("invert.solvers.random_noise", "SolverRandomNoise"),
    # Neural Networks (optional torch)
    "FC": ("invert.solvers.neural_networks.fc", "SolverFC"),
    "CovCNN": ("invert.solvers.neural_networks.covcnn", "SolverCovCNN"),
    "CovCNN-Centers": (
        "invert.solvers.neural_networks.covcnn_centers",
        "SolverCovCNNCenters",
    ),
    "CovCNN-Mask": ("invert.solvers.neural_networks.covcnn_mask", "SolverCovCNNMask"),
    "CovCNN-KL": ("invert.solvers.neural_networks.covcnn_kl", "SolverCovCNNKL"),
    "CovCNN-KL-FLEXOMP": (
        "invert.solvers.neural_networks.covcnn_kl_flexomp",
        "SolverCovCNNKLFlexOMP",
    ),
    "CovCNN-KL-Diff": (
        "invert.solvers.neural_networks.covcnn_kl_diff",
        "SolverCovCNNKLDiff",
    ),
    "CovCNN-KL-Adapt": (
        "invert.solvers.neural_networks.covcnn_kl_adapt",
        "SolverCovCNNKLAdapt",
    ),
    "CovCNN-StructKL-Diff": (
        "invert.solvers.neural_networks.covcnn_structkl_diff",
        "SolverCovCNNStructKLDiff",
    ),
    "CovCNN-BasisDiagKL-Diff": (
        "invert.solvers.neural_networks.covcnn_basisdiag_kl_diff",
        "SolverCovCNNBasisDiagKLDiff",
    ),
    "LSTM": ("invert.solvers.neural_networks.lstm", "SolverLSTM"),
    "CNN": ("invert.solvers.neural_networks.cnn", "SolverCNN"),
}


def _expects_simulation_config(solver_cls: type[BaseSolver]) -> bool:
    """Return True if solver.make_inverse_operator expects a SimulationConfig.

    Neural-network solvers in this codebase train from simulated data and expose
    a signature like: make_inverse_operator(forward, simulation_config, ...).
    """
    import inspect

    try:
        sig = inspect.signature(solver_cls.make_inverse_operator)
    except (TypeError, ValueError):
        return False

    params = list(sig.parameters.values())
    # params[0] is "self"; params[1] is typically "forward"
    for p in params[2:]:
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        return p.name == "simulation_config"
    return False


def _default_nn_batch_size(forward: mne.Forward) -> int:
    """Default ANN training batch size: 2x number of dipoles in the source model."""
    try:
        n_dipoles = int(forward["sol"]["data"].shape[1])
    except Exception:
        return 1
    return max(1, 2 * n_dipoles)


def _generate_solver_categories() -> dict[str, list[str]]:
    """Automatically generate solver categories based on directory structure.

    Categories are determined by the solver's module path:
    - beamformers/ -> beamformer
    - bayesian/ -> bayesian
    - minimum_norm/ -> loreta (if name contains "LORETA") or minimum_norm
    - music/ -> music
    - matching_pursuit/ -> matching_pursuit
    - hybrids/ -> other
    - neural_networks/ -> neural_networks
    - random_noise -> other
    - _old/ -> excluded

    Special cases:
    - RandomNoise -> other (not baseline)
    """
    categories: dict[str, list[str]] = {}

    # Solver names that should be categorized as "other" regardless of directory
    other_solvers = {
        "RandomNoise",  # Should be other, not baseline
    }

    for solver_name, (module_path, _) in _SOLVER_REGISTRY.items():
        # Skip solvers from _old directory
        if "_old" in module_path:
            continue

        # Handle special cases
        if solver_name in other_solvers:
            categories.setdefault("other", []).append(solver_name)
            continue

        # Extract directory from module path
        # e.g., "invert.solvers.beamformers.esmv" -> "beamformers"
        parts = module_path.split(".")
        if len(parts) < 3 or parts[0] != "invert" or parts[1] != "solvers":
            # Fallback: put in other
            categories.setdefault("other", []).append(solver_name)
            continue

        directory = parts[2]

        # Map directory to category
        if directory == "beamformers":
            category = "beamformer"
        elif directory == "bayesian":
            category = "bayesian"
        elif directory == "minimum_norm":
            # Split based on solver name
            if "LORETA" in solver_name:
                category = "loreta"
            else:
                category = "minimum_norm"
        elif directory == "music":
            category = "music"
        elif directory == "matching_pursuit":
            category = "matching_pursuit"
        elif directory == "hybrids":
            category = "other"
        elif directory == "neural_networks":
            category = "neural_networks"
        elif (
            directory == "random_noise" or module_path == "invert.solvers.random_noise"
        ):
            category = "other"
        else:
            # Unknown directory, put in other
            category = "other"

        categories.setdefault(category, []).append(solver_name)

    # Sort solver lists within each category for consistency
    for category in categories:
        categories[category].sort()

    return categories


SOLVER_CATEGORIES = _generate_solver_categories()


def get_solver_class(name: str) -> type[BaseSolver]:
    """Lazy-import and return a solver class by its short name."""
    if name not in _SOLVER_REGISTRY:
        raise ValueError(
            f"Unknown solver {name!r}. Available: {sorted(_SOLVER_REGISTRY)}"
        )
    module_path, class_name = _SOLVER_REGISTRY[name]
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def get_solver_category(solver_name: str) -> str:
    """Get the category for a solver name.

    Parameters
    ----------
    solver_name : str
        The solver's short name.

    Returns
    -------
    str
        The category name, or "other" if not found in any category.
    """
    for category, solvers in SOLVER_CATEGORIES.items():
        if solver_name in solvers:
            return category
    return "other"


def resolve_solvers(
    solvers: list[str] | None = None,
    categories: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[str]:
    """Resolve a list of solver names from explicit names and/or categories.

    Parameters
    ----------
    solvers : list of str, optional
        Explicit solver short names (e.g. ``["MNE", "LCMV"]``).
    categories : list of str, optional
        Category names to include (e.g. ``["beamformer", "loreta"]``).
        Use ``"all"`` to include every registered (non-neural-net) solver.
    exclude : list of str, optional
        Solver names to exclude from the result.

    Returns
    -------
    list of str
        Deduplicated, order-preserved list of solver names.
    """
    result: list[str] = []
    seen: set[str] = set()
    exclude_set = set(exclude) if exclude else set()

    if categories:
        for cat in categories:
            if cat == "all":
                names = list(_SOLVER_REGISTRY.keys())
            elif cat in SOLVER_CATEGORIES:
                names = SOLVER_CATEGORIES[cat]
            else:
                raise ValueError(
                    f"Unknown category {cat!r}. "
                    f"Available: {sorted(SOLVER_CATEGORIES)} or 'all'"
                )
            for n in names:
                if n not in seen and n not in exclude_set:
                    seen.add(n)
                    result.append(n)

    if solvers:
        for n in solvers:
            if n not in _SOLVER_REGISTRY:
                raise ValueError(
                    f"Unknown solver {n!r}. Available: {sorted(_SOLVER_REGISTRY)}"
                )
            if n not in seen and n not in exclude_set:
                seen.add(n)
                result.append(n)

    return result


# Default solvers when nothing is specified
_DEFAULT_SOLVERS = ["MNE", "eLORETA", "LCMV", "Champagne", "RandomNoise"]


# ---------------------------------------------------------------------------
# Worker functions for parallelization (must be at module level for pickling)
# ---------------------------------------------------------------------------


def _apply_inverse_worker(
    idx: int,
    inv_op_matrix: np.ndarray,
    x_sample: np.ndarray,
    y_sample: np.ndarray,
    adjacency,
    pos: np.ndarray,
) -> tuple[int, dict]:
    """Worker for applying precomputed inverse operator.

    Returns (sample_idx, metrics_dict) for ordering results.
    """
    y_pred = inv_op_matrix @ x_sample
    metrics = evaluate_all(y_sample, y_pred, adjacency, adjacency, pos, pos)
    return idx, metrics


def _compute_and_apply_worker(
    idx: int,
    solver_module: str,
    solver_class: str,
    forward: mne.Forward,
    info: mne.Info,
    x_sample: np.ndarray,
    y_sample: np.ndarray,
    adjacency,
    pos: np.ndarray,
    require_data: bool,
) -> tuple[int, dict]:
    """Worker for computing fresh inverse operator per sample.

    Returns (sample_idx, metrics_dict) for ordering results.
    """
    import importlib

    mod = importlib.import_module(solver_module)
    solver_cls = getattr(mod, solver_class)
    solver = solver_cls()

    evoked = mne.EvokedArray(x_sample, info, tmin=0.0, verbose=0)
    if require_data:
        solver.make_inverse_operator(forward, evoked, alpha="auto")
    else:
        solver.make_inverse_operator(forward, alpha="auto")
    stc = solver.apply_inverse_operator(evoked)
    y_pred = stc.data
    metrics = evaluate_all(y_sample, y_pred, adjacency, adjacency, pos, pos)
    return idx, metrics


class SampleMetrics(BaseModel):
    mle: float
    emd: float
    sd: float
    ap: float
    correlation: float


class AggregateStats(BaseModel):
    mean: float
    std: float
    median: float
    # Optional for compact dashboard exports.
    worst_10_pct: float | None = None  # 10th/90th percentile of worst predictions


class BenchmarkResult(BaseModel):
    solver_name: str
    dataset_name: str
    category: str | None = None
    metrics: dict[str, AggregateStats]
    samples: list[SampleMetrics]


class BenchmarkRunner:
    def __init__(
        self,
        forward: mne.Forward,
        info: mne.Info,
        solvers: list[str] | None = None,
        categories: list[str] | None = None,
        exclude_solvers: list[str] | None = None,
        datasets: dict[str, DatasetConfig] | None = None,
        n_samples: int = 50,
        n_jobs: int | None = None,
        random_seed: int | None = None,
        solver_params: dict[str, dict[str, Any]] | None = None,
    ):
        self.forward = forward
        self.info = info
        if solvers is None and categories is None:
            self.solvers = [
                s for s in _DEFAULT_SOLVERS if s not in set(exclude_solvers or [])
            ]
        else:
            self.solvers = resolve_solvers(
                solvers=solvers, categories=categories, exclude=exclude_solvers
            )
        self.datasets = datasets or dict(BENCHMARK_DATASETS)
        self.n_samples = n_samples
        self.random_seed = random_seed
        self.solver_params = solver_params or {}

        # Auto-detect n_jobs if not specified
        if n_jobs is None:
            cpu_count = os.cpu_count()
            self.n_jobs = max(1, cpu_count - 1) if cpu_count is not None else 1
        elif n_jobs == -1:
            self.n_jobs = os.cpu_count() or 1
        else:
            self.n_jobs = max(1, n_jobs)

        self._results: list[BenchmarkResult] = []

    def run(self) -> list[BenchmarkResult]:
        pos = pos_from_forward(self.forward)
        adjacency = mne.spatial_src_adjacency(self.forward["src"], verbose=0)
        results = []

        # Total number of (dataset, solver) combinations for overall progress
        total_combinations = len(self.datasets) * len(self.solvers)

        with tqdm(
            total=total_combinations, desc="Overall Progress", position=0
        ) as pbar_overall:
            for ds_name, ds_config in self.datasets.items():
                logger.info("Dataset: %s", ds_name)

                # Generate all samples for this dataset once
                sim_config = SimulationConfig(
                    batch_size=self.n_samples,
                    n_sources=ds_config.n_sources,
                    n_orders=ds_config.n_orders,
                    snr_range=ds_config.snr_range,
                    n_timepoints=ds_config.n_timepoints,
                    random_seed=self.random_seed,
                )
                gen = SimulationGenerator(self.forward, config=sim_config)
                x_batch, y_batch, _ = next(gen.generate())

                for solver_name in self.solvers:
                    logger.info("  Solver: %s", solver_name)
                    solver_cls = get_solver_class(solver_name)
                    solver = solver_cls()

                    # Best-effort determinism for fair comparisons when a seed is provided.
                    if self.random_seed is not None:
                        seed = int(self.random_seed)
                        random.seed(seed)
                        np.random.seed(seed)
                        try:  # torch is optional
                            import torch  # type: ignore[import-not-found]

                            torch.manual_seed(seed)
                            if torch.cuda.is_available():
                                torch.cuda.manual_seed_all(seed)
                        except Exception:
                            pass

                    # Neural-network solvers train from SimulationConfig and then
                    # apply the trained model to each sample.
                    if _expects_simulation_config(solver_cls):
                        train_sim_config = sim_config.model_copy(
                            update={"batch_size": _default_nn_batch_size(self.forward)}
                        )
                        logger.info(
                            "ANN training batch_size=%d (default=2*n_dipoles) for %s",
                            int(train_sim_config.batch_size),
                            solver_name,
                        )
                        params = dict(self.solver_params.get(solver_name, {}))
                        alpha = params.pop("alpha", "auto")
                        solver.make_inverse_operator(
                            self.forward, train_sim_config, alpha=alpha, **params
                        )
                        sample_metrics: list[SampleMetrics] = []
                        for i in tqdm(
                            range(self.n_samples),
                            desc=f"{ds_name}/{solver_name}",
                            position=1,
                            leave=False,
                        ):
                            evoked = mne.EvokedArray(
                                x_batch[i], self.info, tmin=0.0, verbose=0
                            )
                            stc = solver.apply_inverse_operator(evoked)
                            y_pred = stc.data
                            metrics = evaluate_all(
                                y_batch[i], y_pred, adjacency, adjacency, pos, pos
                            )
                            sample_metrics.append(self._metrics_from_dict(metrics))

                    # Parallelize based on require_recompute
                    elif not solver.require_recompute:
                        # Compute inverse operator once, then parallelize application
                        if solver.require_data:
                            evoked = mne.EvokedArray(
                                x_batch[0], self.info, tmin=0.0, verbose=0
                            )
                            solver.make_inverse_operator(
                                self.forward, evoked, alpha="auto"
                            )
                        else:
                            solver.make_inverse_operator(self.forward, alpha="auto")

                        # Check if solver has inverse_operators attribute
                        # Some solvers (e.g., SolverRandomNoise) don't create inverse operators
                        if not hasattr(solver, "inverse_operators"):
                            # Fall back to direct application for each sample
                            sample_metrics = []
                            for i in range(len(x_batch)):
                                evoked = mne.EvokedArray(
                                    x_batch[i], self.info, tmin=0.0, verbose=0
                                )
                                stc = solver.apply_inverse_operator(evoked)
                                y_pred = stc.data
                                metrics = evaluate_all(
                                    y_batch[i], y_pred, adjacency, adjacency, pos, pos
                                )
                                sample_metrics.append(self._metrics_from_dict(metrics))
                        else:
                            # Select optimal regularization via L-curve/GCV
                            if len(solver.inverse_operators) > 1:  # type: ignore[attr-defined]
                                _, optimal_idx = solver.regularise_gcv(x_batch[0])  # type: ignore[attr-defined]
                            else:
                                optimal_idx = 0
                            inv_op = solver.inverse_operators[optimal_idx]  # type: ignore[attr-defined]

                            # Extract the inverse operator matrix (numpy array)
                            inv_op_matrix = inv_op.data[0]

                            # Parallel application
                            sample_metrics = self._run_parallel_apply(
                                inv_op_matrix,
                                x_batch,
                                y_batch,
                                adjacency,
                                pos,
                                ds_name,
                                solver_name,
                            )
                    else:
                        # Parallelize full computation (require_recompute=True)
                        module_path, class_name = _SOLVER_REGISTRY[solver_name]

                        sample_metrics = self._run_parallel_compute(
                            module_path,
                            class_name,
                            self.forward,
                            self.info,
                            x_batch,
                            y_batch,
                            adjacency,
                            pos,
                            solver.require_data,
                            ds_name,
                            solver_name,
                        )

                    result = self._aggregate(solver_name, ds_name, sample_metrics)
                    results.append(result)
                    pbar_overall.update(1)
                    pbar_overall.set_postfix(
                        {"dataset": ds_name, "solver": solver_name}
                    )

        self._results = results
        return results

    def _run_parallel_apply(
        self,
        inv_op_matrix: np.ndarray,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        adjacency,
        pos: np.ndarray,
        ds_name: str,
        solver_name: str,
    ) -> list[SampleMetrics]:
        """Parallelize inverse operator application (require_recompute=False)."""
        if self.n_jobs == 1:
            sample_metrics = []
            for i in tqdm(
                range(self.n_samples),
                desc=f"{ds_name}/{solver_name}",
                position=1,
                leave=False,
            ):
                _, metrics = _apply_inverse_worker(
                    i, inv_op_matrix, x_batch[i], y_batch[i], adjacency, pos
                )
                sample_metrics.append(self._metrics_from_dict(metrics))
            return sample_metrics

        sample_metrics_dict = {}
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {
                executor.submit(
                    _apply_inverse_worker,
                    i,
                    inv_op_matrix,
                    x_batch[i],
                    y_batch[i],
                    adjacency,
                    pos,
                ): i
                for i in range(self.n_samples)
            }

            with tqdm(
                total=self.n_samples,
                desc=f"{ds_name}/{solver_name}",
                position=1,
                leave=False,
            ) as pbar:
                for future in as_completed(futures):
                    try:
                        idx, metrics = future.result()
                        sample_metrics_dict[idx] = self._metrics_from_dict(metrics)
                    except Exception as e:
                        logger.error(f"Sample {futures[future]} failed: {e}")
                        idx = futures[future]
                        sample_metrics_dict[idx] = SampleMetrics(
                            mle=float("nan"),
                            emd=float("nan"),
                            sd=float("nan"),
                            ap=float("nan"),
                            correlation=float("nan"),
                        )
                    pbar.update(1)

        return [sample_metrics_dict[i] for i in range(self.n_samples)]

    def _run_parallel_compute(
        self,
        solver_module: str,
        solver_class: str,
        forward: mne.Forward,
        info: mne.Info,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        adjacency,
        pos: np.ndarray,
        require_data: bool,
        ds_name: str,
        solver_name: str,
    ) -> list[SampleMetrics]:
        """Parallelize full computation (require_recompute=True)."""
        if self.n_jobs == 1:
            sample_metrics = []
            for i in tqdm(
                range(self.n_samples),
                desc=f"{ds_name}/{solver_name}",
                position=1,
                leave=False,
            ):
                _, metrics = _compute_and_apply_worker(
                    i,
                    solver_module,
                    solver_class,
                    forward,
                    info,
                    x_batch[i],
                    y_batch[i],
                    adjacency,
                    pos,
                    require_data,
                )
                sample_metrics.append(self._metrics_from_dict(metrics))
            return sample_metrics

        sample_metrics_dict = {}
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {
                executor.submit(
                    _compute_and_apply_worker,
                    i,
                    solver_module,
                    solver_class,
                    forward,
                    info,
                    x_batch[i],
                    y_batch[i],
                    adjacency,
                    pos,
                    require_data,
                ): i
                for i in range(self.n_samples)
            }

            with tqdm(
                total=self.n_samples,
                desc=f"{ds_name}/{solver_name}",
                position=1,
                leave=False,
            ) as pbar:
                for future in as_completed(futures):
                    try:
                        idx, metrics = future.result()
                        sample_metrics_dict[idx] = self._metrics_from_dict(metrics)
                    except Exception as e:
                        logger.error(f"Sample {futures[future]} failed: {e}")
                        idx = futures[future]
                        sample_metrics_dict[idx] = SampleMetrics(
                            mle=float("nan"),
                            emd=float("nan"),
                            sd=float("nan"),
                            ap=float("nan"),
                            correlation=float("nan"),
                        )
                    pbar.update(1)

        return [sample_metrics_dict[i] for i in range(self.n_samples)]

    @staticmethod
    def _metrics_from_dict(m: dict) -> SampleMetrics:
        return SampleMetrics(
            mle=float(m["Mean_Localization_Error"]),
            emd=float(m["EMD"]),
            sd=float(m["sd"]),
            ap=float(m["average_precision"]),
            correlation=float(m["correlation"]),
        )

    @staticmethod
    def _aggregate(
        solver_name: str,
        dataset_name: str,
        samples: list[SampleMetrics],
    ) -> BenchmarkResult:
        # Metrics where higher is better (worst = 10th percentile)
        # Others are lower is better (worst = 90th percentile)
        higher_is_better = {"average_precision", "correlation"}

        arrays = {
            "mean_localization_error": np.array([s.mle for s in samples]),
            "emd": np.array([s.emd for s in samples]),
            "spatial_dispersion": np.array([s.sd for s in samples]),
            "average_precision": np.array([s.ap for s in samples]),
            "correlation": np.array([s.correlation for s in samples]),
        }
        metrics = {}
        for key, arr in arrays.items():
            # For higher-is-better metrics, worst 10% = 10th percentile (lowest)
            # For lower-is-better metrics, worst 10% = 90th percentile (highest)
            worst_pct = 10 if key in higher_is_better else 90
            metrics[key] = AggregateStats(
                mean=float(np.nanmean(arr)),
                std=float(np.nanstd(arr)),
                median=float(np.nanmedian(arr)),
                worst_10_pct=float(np.nanpercentile(arr, worst_pct)),
            )
        return BenchmarkResult(
            solver_name=solver_name,
            dataset_name=dataset_name,
            category=get_solver_category(solver_name),
            metrics=metrics,
            samples=samples,
        )

    def _compute_best_solvers(self) -> dict[str, Any]:
        """Compute best solver per dataset for each metric.

        Returns
        -------
        dict
            Structure: {dataset_name: {metric_name: {"solver": solver_name, "value": metric_value}}}
        """
        # Metrics where lower is better
        lower_is_better = {"mean_localization_error", "emd", "spatial_dispersion"}
        # Metrics where higher is better
        higher_is_better = {"average_precision", "correlation"}

        datasets = sorted(set(r.dataset_name for r in self._results))
        all_metrics: set[str] = set()
        for r in self._results:
            all_metrics.update(r.metrics.keys())

        best_solvers: dict[str, Any] = {}
        for dataset in datasets:
            best_solvers[dataset] = {}
            for metric in all_metrics:
                # Find all results for this dataset
                dataset_results = [
                    r for r in self._results if r.dataset_name == dataset
                ]

                if not dataset_results:
                    continue

                # Get metric values for all solvers on this dataset
                # Filter out NaN values
                solver_values = {}
                for r in dataset_results:
                    if metric in r.metrics:
                        value = r.metrics[metric].mean
                        if not np.isnan(value):
                            solver_values[r.solver_name] = value

                if not solver_values:
                    continue

                # Determine best based on metric type
                if metric in lower_is_better:
                    best_solver = min(solver_values.items(), key=lambda x: x[1])
                elif metric in higher_is_better:
                    best_solver = max(solver_values.items(), key=lambda x: x[1])
                else:
                    # Default to lower is better if unknown
                    best_solver = min(solver_values.items(), key=lambda x: x[1])

                best_solvers[dataset][metric] = {
                    "solver": best_solver[0],
                    "value": round(best_solver[1], 4),
                }

        return best_solvers

    def _compute_average_ranks(
        self,
    ) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
        """Compute per-dataset and global average ranks for each solver.

        For each (dataset, metric) pair, solvers are ranked 1..N (best=1).
        Per-dataset rank: average rank across metrics for that dataset.
        Global rank: average of the per-dataset ranks.

        Returns
        -------
        tuple
            (per_dataset_ranks, global_ranks) where:
            - per_dataset_ranks: {dataset_name: {solver_name: avg_rank}}
            - global_ranks: {solver_name: avg_rank}
        """
        lower_is_better = {"mean_localization_error", "emd", "spatial_dispersion"}

        datasets = sorted(set(r.dataset_name for r in self._results))
        all_metrics: set[str] = set()
        for r in self._results:
            all_metrics.update(r.metrics.keys())

        # Collect ranks per dataset: dataset -> solver_name -> list of ranks
        dataset_solver_ranks: dict[str, dict[str, list[float]]] = {}

        for dataset in datasets:
            dataset_results = [r for r in self._results if r.dataset_name == dataset]
            dataset_solver_ranks[dataset] = {}
            for metric in all_metrics:
                # Gather (solver, value) pairs, skip NaN
                solver_values = []
                for r in dataset_results:
                    if metric in r.metrics:
                        val = r.metrics[metric].mean
                        if not np.isnan(val):
                            solver_values.append((r.solver_name, val))

                if not solver_values:
                    continue

                # Sort: ascending for lower-is-better, descending for higher-is-better
                reverse = metric not in lower_is_better
                solver_values.sort(key=lambda x: x[1], reverse=reverse)

                # Dense ranking: tied values get the same rank
                rank = 1
                for i, (solver_name, val) in enumerate(solver_values):
                    if i > 0 and val != solver_values[i - 1][1]:
                        rank = i + 1
                    dataset_solver_ranks[dataset].setdefault(solver_name, []).append(
                        rank
                    )

        # Per-dataset ranks: average across metrics for each dataset
        per_dataset_ranks: dict[str, dict[str, float]] = {}
        for dataset in datasets:
            per_dataset_ranks[dataset] = {
                name: round(float(np.mean(ranks)), 2)
                for name, ranks in sorted(
                    dataset_solver_ranks[dataset].items(),
                    key=lambda x: np.mean(x[1]),
                )
            }

        # Global ranks: average of per-dataset ranks
        solver_dataset_ranks: dict[str, list[float]] = {}
        for _dataset, solver_ranks in per_dataset_ranks.items():
            for solver_name, avg_rank in solver_ranks.items():
                solver_dataset_ranks.setdefault(solver_name, []).append(avg_rank)

        global_ranks = {
            name: round(float(np.mean(ranks)), 2)
            for name, ranks in sorted(
                solver_dataset_ranks.items(), key=lambda x: np.mean(x[1])
            )
        }

        return per_dataset_ranks, global_ranks

    def save(
        self,
        path: str | Path,
        *,
        compact: bool = False,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        path = Path(path)

        per_dataset_ranks, global_ranks = self._compute_average_ranks()

        datasets_payload = {
            key: cfg.model_dump() if isinstance(cfg, BaseModel) else dict(cfg)  # type: ignore[arg-type]
            for key, cfg in self.datasets.items()
        }

        m_electrodes: int | None = None
        n_leadfield_columns: int | None = None
        n_sources_space: int | None = None
        n_orient: int | None = None
        try:
            lf = self.forward["sol"]["data"]
            m_electrodes = int(lf.shape[0])
            n_leadfield_columns = int(lf.shape[1])
        except Exception:
            m_electrodes = None
            n_leadfield_columns = None

        try:
            raw_nsource = self.forward.get("nsource")  # type: ignore[call-arg]
            if raw_nsource is not None:
                n_sources_space = int(raw_nsource)
        except Exception:
            n_sources_space = None

        if n_sources_space is None:
            try:
                src = self.forward.get("src")  # type: ignore[call-arg]
                if isinstance(src, (list, tuple)):
                    n_sources_space = int(
                        sum(
                            len(s.get("vertno", [])) for s in src if isinstance(s, dict)
                        )
                    )
            except Exception:
                n_sources_space = None

        if (
            n_sources_space is not None
            and n_leadfield_columns is not None
            and n_sources_space > 0
            and n_leadfield_columns % n_sources_space == 0
        ):
            n_orient = int(n_leadfield_columns // n_sources_space)

        if compact:
            # Minimal payload for the MkDocs dashboard: aggregated metrics only.
            output = {
                "ranks": per_dataset_ranks,
                "global_ranks": global_ranks,
                "metadata": {
                    "name": name,
                    "description": description,
                    "timestamp": datetime.now().isoformat(),
                    "n_samples": self.n_samples,
                    "random_seed": self.random_seed,
                    "solvers": self.solvers,
                    "m": m_electrodes,
                    "n": n_sources_space,
                    "m_electrodes": m_electrodes,
                    "n_sources": n_sources_space,
                    "n_leadfield_columns": n_leadfield_columns,
                    "n_orient": n_orient,
                },
                "datasets": datasets_payload,
                "results": [
                    {
                        "solver_name": r.solver_name,
                        "dataset_name": r.dataset_name,
                        "category": r.category,
                        "metrics": {
                            metric: {
                                "mean": float(stats.mean),
                                "std": float(stats.std),
                                "median": float(stats.median),
                                "worst_10_pct": (
                                    float(stats.worst_10_pct)
                                    if stats.worst_10_pct is not None
                                    else None
                                ),
                            }
                            for metric, stats in r.metrics.items()
                        },
                        "samples": [],
                    }
                    for r in self._results
                ],
            }
            path.write_text(json.dumps(output, indent=2))
            logger.info("Compact results saved to %s", path)
            return

        # Full payload (includes per-sample metrics)
        summary = {}
        for r in self._results:
            key = f"{r.solver_name} | {r.dataset_name}"
            summary[key] = {m: round(s.mean, 4) for m, s in r.metrics.items()}

        best_solvers = self._compute_best_solvers()
        output = {
            "summary": summary,
            "best_solvers": best_solvers,
            "ranks": per_dataset_ranks,
            "global_ranks": global_ranks,
            "metadata": {
                "name": name,
                "description": description,
                "timestamp": datetime.now().isoformat(),
                "n_samples": self.n_samples,
                "random_seed": self.random_seed,
                "solvers": self.solvers,
                "m": m_electrodes,
                "n": n_sources_space,
                "m_electrodes": m_electrodes,
                "n_sources": n_sources_space,
                "n_leadfield_columns": n_leadfield_columns,
                "n_orient": n_orient,
            },
            "datasets": datasets_payload,
            "results": [r.model_dump() for r in self._results],
        }
        path.write_text(json.dumps(output, indent=2))
        logger.info("Results saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> list[BenchmarkResult]:
        path = Path(path)
        data = json.loads(path.read_text())
        results = []
        for r in data["results"]:
            # Populate category if missing (for backward compatibility)
            if "category" not in r or r["category"] is None:
                r["category"] = get_solver_category(r["solver_name"])
            # Add correlation if missing (backward compatibility)
            for sample in r.get("samples", []):
                if "correlation" not in sample:
                    sample["correlation"] = float("nan")
            results.append(BenchmarkResult(**r))
        return results

    @classmethod
    def update_summary_statistics(cls, path: str | Path) -> None:
        """Update summary statistics (including best_solvers) for an existing results file.

        This is useful when you want to regenerate the summary from existing results
        without re-running the benchmark.

        Parameters
        ----------
        path : str or Path
            Path to the benchmark results JSON file.
        """
        path = Path(path)
        data = json.loads(path.read_text())
        results = []
        for r in data["results"]:
            # Populate category if missing (for backward compatibility)
            if "category" not in r or r["category"] is None:
                r["category"] = get_solver_category(r["solver_name"])
            # Add correlation if missing (backward compatibility)
            for sample in r.get("samples", []):
                if "correlation" not in sample:
                    sample["correlation"] = float("nan")
            results.append(BenchmarkResult(**r))

        # Create a temporary runner instance to use the _compute_best_solvers method
        # We need to set _results manually
        temp_runner = cls.__new__(cls)
        temp_runner._results = results

        # Recompute summary
        summary = {}
        for r in results:
            key = f"{r.solver_name} | {r.dataset_name}"
            summary[key] = {m: round(s.mean, 4) for m, s in r.metrics.items()}

        # Compute best solvers and ranks
        best_solvers = temp_runner._compute_best_solvers()
        per_dataset_ranks, global_ranks = temp_runner._compute_average_ranks()

        # Update the data structure
        data["summary"] = summary
        data["best_solvers"] = best_solvers
        data["ranks"] = per_dataset_ranks
        data["global_ranks"] = global_ranks

        # Write back
        path.write_text(json.dumps(data, indent=2))
        logger.info("Updated summary statistics in %s", path)
