"""Solver factory for M/EEG inverse solutions."""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING, Any

from . import config

if TYPE_CHECKING:
    from .solvers.base import BaseSolver

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Solver registry
# ---------------------------------------------------------------------------
# Each key maps to either a solver class or a partial() that pre-fills kwargs.
# The registry is populated lazily on first access so that heavyweight imports
# (especially tensorflow-based solvers) are deferred.
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Any] | None = None


def _build_registry() -> dict[str, Any]:
    """Build the name -> constructor mapping.

    Imports are done here (not at module level) so that the cost is paid once,
    only when ``Solver()`` is first called.
    """
    from invert import solvers

    # Helper: register a class under one or more lowercase aliases.
    reg: dict[str, Any] = {}

    def _add(aliases: str | list[str], factory):
        if isinstance(aliases, str):
            aliases = [aliases]
        for alias in aliases:
            reg[alias.lower()] = factory

    # -- Minimum Norm -------------------------------------------------------
    _add("mne", solvers.SolverMNE)
    _add("gft-mne", solvers.SolverGFTMNE)
    _add("wmne", solvers.SolverWMNE)
    _add("dspm", solvers.SolverDSPM)
    _add(
        ["l1", "fista", "mce", "minimum current estimate"], solvers.SolverMinimumL1Norm
    )
    _add(["gpt", "l1-gpt", "l1gpt"], solvers.SolverMinimumL1NormGPT)
    _add("l1l2", solvers.SolverMinimumL1L2Norm)
    _add(["gft-l1", "gft-minimum-l1"], solvers.SolverGFTMinimumL1Norm)
    _add(
        ["self-regularized-eloreta", "sr-eloreta", "sreloreta"],
        solvers.SolverSelfRegularizedELORETA,
    )

    # -- LORETA -------------------------------------------------------------
    _add(["loreta", "lor"], solvers.SolverLORETA)
    _add(["sloreta", "slor"], solvers.SolverSLORETA)
    _add(["eloreta", "elor"], solvers.SolverELORETA)
    _add("sslofo", solvers.SolverSSLOFO)

    # -- Other minimum-norm-like --------------------------------------------
    _add(["laura", "laur"], solvers.SolverLAURA)
    _add(["backus-gilbert", "b-g", "bg"], solvers.SolverBackusGilbert)
    _add(["s-map", "smap"], solvers.SolverSMAP)

    # -- Bayesian / Champagne -----------------------------------------------
    _add(["champagne", "champ"], solvers.SolverChampagne)
    _add(
        ["emchampagne", "em-champagne", "emc"],
        partial(solvers.SolverChampagne, update_rule="EM"),
    )
    _add(
        ["convexitychampagne", "convexity-champagne", "coc", "mm-champagne"],
        partial(solvers.SolverChampagne, update_rule="Convexity"),
    )
    _add(
        ["mackaychampagne", "mackay-champagne", "mcc"],
        partial(solvers.SolverChampagne, update_rule="MacKay"),
    )
    _add(
        ["temchampagne", "tem-champagne", "temc", "t-em-champagne"],
        partial(solvers.SolverChampagne, update_rule="TEM"),
    )
    _add(
        ["aremchampagne", "arem-champagne", "aremc", "ar-em-champagne"],
        partial(solvers.SolverChampagne, update_rule="AR-EM"),
    )
    _add(
        ["lowsnrchampagne", "low-snr-champagne", "lowsnr-champagne", "lsc"],
        partial(solvers.SolverChampagne, update_rule="LowSNR"),
    )
    _add(
        ["adaptivechampagne", "adaptive-champagne", "ac"],
        partial(solvers.SolverChampagne, update_rule="Adaptive"),
    )
    _add(["nlchampagne", "nl-champagne", "nlc"], solvers.SolverNLChampagne)
    _add(
        ["omnichampagne", "omni-champagne", "omni champagne", "oc"],
        solvers.SolverOmniChampagne,
    )
    _add(
        ["flexchampagne", "flex-champagne", "fc-champ"],
        solvers.SolverFlexChampagne,
    )
    _add(
        ["flexnlchampagne", "flex-nl-champagne", "fnlc"],
        solvers.SolverFlexNLChampagne,
    )

    # -- Other Bayesian -----------------------------------------------------
    _add(["gamma-map", "gmap"], solvers.SolverGammaMAP)
    _add("source-map", solvers.SolverSourceMAP)
    _add("gamma-map-msp", solvers.SolverGammaMAPMSP)
    _add("source-map-msp", solvers.SolverSourceMAPMSP)
    _add(["multiple-sparse-priors", "msp"], solvers.SolverMSP)
    _add("cmem", solvers.SolverCMEM)
    _add(["subspace-sbl", "ssm-nlc", "subspacesbl"], solvers.SolverSubspaceSBL)
    _add(
        ["subspace-sbl-plus", "ssm-nlc-plus", "subspacesblplus"],
        solvers.SolverSubspaceSBLPlus,
    )
    _add(["vb-sbl", "vbsbl", "variational-bayes-sbl"], solvers.SolverVBSBL)

    # -- Beamformers --------------------------------------------------------
    _add("mvab", solvers.SolverMVAB)
    _add("lcmv", solvers.SolverLCMV)
    _add(["dics", "dics-beamformer"], solvers.SolverDICS)
    _add("smv", solvers.SolverSMV)
    _add(["wnmv", "wmnv"], solvers.SolverWNMV)
    _add("hocmv", solvers.SolverHOCMV)
    _add("esmv", solvers.SolverESMV)
    _add("esmv2", solvers.SolverESMV2)
    _add("esmv3", solvers.SolverESMV3)
    _add("mcmv", solvers.SolverMCMV)
    _add("hocmcmv", solvers.SolverHOCMCMV)
    _add(["recipsiicos-plain", "recipsiicos"], solvers.SolverReciPSIICOSPlain)
    _add("recipsiicos-whitened", solvers.SolverReciPSIICOSWhitened)
    _add("sam", solvers.SolverSAM)
    _add(["ebb", "empirical-bayesian-beamformer"], solvers.SolverEBB)
    _add(["adapt-flex-esmv", "adaptflexesmv"], solvers.SolverAdaptFlexESMV)
    _add(["flex-esmv", "flexesmv"], solvers.SolverFlexESMV)
    _add(["flex-esmv2", "flexesmv2"], solvers.SolverFlexESMV2)
    _add(["deblur-flex-esmv", "deblurflexesmv"], solvers.SolverDeblurFlexESMV)
    _add(["safe-flex-esmv", "safeflexesmv"], solvers.SolverSafeFlexESMV)
    _add(["sharp-flex-esmv", "sharpflexesmv"], solvers.SolverSharpFlexESMV)
    _add(["sharp-flex-esmv2", "sharpflexesmv2"], solvers.SolverSharpFlexESMV2)
    _add(["ssp-esmv", "sspesmv"], solvers.SolverSSPESMV)
    _add(["ir-esmv", "iresmv"], solvers.SolverIRESMV)
    _add(["ssp-ir-esmv", "sspiresmv"], solvers.SolverSSPIRESMV)
    _add(["unit-noise-gain", "ung"], solvers.SolverUnitNoiseGain)

    # -- Dipole fitting ----------------------------------------------------
    _add(["ecd", "equivalent-current-dipole"], solvers.SolverECD)
    _add(["sesame"], solvers.SolverSESAME)

    # -- Structured sparsity / edge-preserving -----------------------------
    _add(["tv", "total-variation", "graph-tv"], solvers.SolverTotalVariation)

    # -- Artificial Neural Networks (optional torch) ------------------------
    try:
        _add(["fully-connected", "fc", "fullyconnected", "esinet"], solvers.SolverFC)  # type: ignore[attr-defined]
        _add(["covcnn", "cov-cnn", "covnet"], solvers.SolverCovCNN)  # type: ignore[attr-defined]
        _add("lstm", solvers.SolverLSTM)  # type: ignore[attr-defined]
        _add("cnn", solvers.SolverCNN)  # type: ignore[attr-defined]
    except AttributeError:
        logger.debug("ANN solvers not available (torch not installed)")

    # -- Matching Pursuit / Compressive Sensing -----------------------------
    _add("omp", solvers.SolverOMP)
    _add("cosamp", solvers.SolverCOSAMP)
    _add("somp", solvers.SolverSOMP)
    _add("rembo", solvers.SolverREMBO)
    _add("sp", solvers.SolverSP)
    _add("ssp", solvers.SolverSSP)
    _add("smp", solvers.SolverSMP)
    _add("ssmp", solvers.SolverSSMP)
    _add("subsmp", solvers.SolverSubSMP)
    _add("isubsmp", solvers.SolverISubSMP)
    _add("bcs", solvers.SolverBCS)

    # -- MUSIC / Subspace ---------------------------------------------------
    _add("music", solvers.SolverMUSIC)
    _add(["rap-music", "rap"], partial(solvers.SolverFLEXMUSIC, name="RAP-MUSIC"))
    _add(["trap-music", "trap"], partial(solvers.SolverFLEXMUSIC, name="TRAP-MUSIC"))
    _add(["flex-rap-music", "flex-music", "flex"], solvers.SolverFLEXMUSIC)
    _add(
        "flex-ssm",
        partial(
            solvers.SolverSignalSubspaceMatching,
            name="Flexible Signal Subspace Matching",
        ),
    )
    _add(
        "ssm",
        partial(solvers.SolverSignalSubspaceMatching, name="Signal Subspace Matching"),
    )
    _add(
        "flex-ap",
        partial(
            solvers.SolverAlternatingProjections,
            name="Flexible Alternating Projections",
        ),
    )
    _add(
        "ap",
        partial(solvers.SolverAlternatingProjections, name="Alternating Projections"),
    )

    _add(
        ["adaptive-ap", "adaptive-alternating-projections", "aap"],
        solvers.SolverAdaptiveAlternatingProjections,
    )
    _add(["flex-music-2", "flexmusic2"], solvers.SolverFLEXMUSIC_2)
    _add(
        ["generalized-iterative", "gi"],
        solvers.SolverGeneralizedIterative,
    )

    # -- Basis Functions ----------------------------------------------------
    _add("gbf", solvers.SolverBasisFunctions)

    # -- Other --------------------------------------------------------------
    _add("epifocus", solvers.SolverEPIFOCUS)
    _add("apse", solvers.hybrids.SolverAPSE)
    _add(["random-noise", "random", "noise-baseline"], solvers.SolverRandomNoise)

    return reg


def _get_registry() -> dict[str, Any]:
    """Return the solver registry, building it on first call."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


def list_solvers() -> list[str]:
    """Return a sorted list of all registered solver aliases."""
    return sorted(_get_registry().keys())


def Solver(solver: str = "mne", **kwargs) -> BaseSolver:
    """Create a solver instance by name.

    Parameters
    ----------
    solver : str
        Name or alias of the solver (case-insensitive).
        Use ``list_solvers()`` to see all available names.
    **kwargs
        Forwarded to the solver constructor.

    Returns
    -------
    BaseSolver
        The solver instance.

    Raises
    ------
    ValueError
        If the solver name is not recognised.
    """
    registry = _get_registry()
    key = solver.lower()

    if key not in registry:
        raise ValueError(
            f"'{solver}' is not a recognised solver. "
            f"Available solvers: {config.all_solvers}"
        )

    return registry[key](**kwargs)
