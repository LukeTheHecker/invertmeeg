"""Central registry for solver discovery and lazy construction.

This module is intentionally lightweight (stdlib-only) so it can be imported
without pulling in every solver implementation (some have heavyweight optional
dependencies).
"""

from __future__ import annotations

import importlib
import importlib.util
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


def normalize_solver_name(name: str) -> str:
    """Normalize a user-provided solver name/alias to a canonical lookup key."""
    value = name.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value


@dataclass(frozen=True, slots=True)
class SolverSpec:
    """Lazy solver specification (import path + naming metadata)."""

    solver_id: str  # The canonical solver id (which is used to get the solver via Solver(solver_id))
    module_path: str  # The import path of the solver class
    class_name: str  # The name of the solver class
    default_kwargs: Mapping[str, Any] = field(default_factory=dict)  # Default keyword arguments for the solver
    aliases: tuple[str, ...] = ()  # Aliases for the solver
    display_name: str | None = None  # The display name of the solver
    requires: tuple[str, ...] = ()  # Optional requirements for the solver
    internal: bool = False  # Whether the solver is internal

    def normalized_id(self) -> str:
        return normalize_solver_name(self.solver_id)


# ---------------------------------------------------------------------------
# Specs (single source of truth for solver ids and aliases)
# ---------------------------------------------------------------------------
_SOLVER_SPECS: tuple[SolverSpec, ...] = (
    # -- Minimum Norm -------------------------------------------------------
    SolverSpec(
        solver_id="mne",
        module_path="invert.solvers.minimum_norm.mne",
        class_name="SolverMNE",
    ),
    SolverSpec(
        solver_id="gft-mne",
        module_path="invert.solvers.minimum_norm.gft_mne",
        class_name="SolverGFTMNE",
    ),
    SolverSpec(
        solver_id="wmne",
        module_path="invert.solvers.minimum_norm.wmne",
        class_name="SolverWMNE",
    ),
    SolverSpec(
        solver_id="dspm",
        module_path="invert.solvers.minimum_norm.dspm",
        class_name="SolverDSPM",
    ),
    SolverSpec(
        solver_id="mce",
        module_path="invert.solvers.minimum_norm.minimum_l1_norm",
        class_name="SolverMinimumL1Norm",
        aliases=("l1", "fista", "minimum current estimate"),
        display_name="MCE",
    ),
    SolverSpec(
        solver_id="gpt",
        module_path="invert.solvers.minimum_norm.minimum_l1_norm_gpt",
        class_name="SolverMinimumL1NormGPT",
        aliases=("l1-gpt", "l1gpt", "mce-gpt"),
        display_name="GPT",
        internal=True,
    ),
    SolverSpec(
        solver_id="l1l2",
        module_path="invert.solvers.minimum_norm.minimum_l1_l2_norm",
        class_name="SolverMinimumL1L2Norm",
        aliases=("mce-l1l2",),
    ),
    SolverSpec(
        solver_id="gft-l1",
        module_path="invert.solvers.minimum_norm.gft_minimum_l1_norm",
        class_name="SolverGFTMinimumL1Norm",
        aliases=("gft-minimum-l1", "gftmce"),
    ),
    SolverSpec(
        solver_id="self-regularized-eloreta",
        module_path="invert.solvers.minimum_norm.self_regularized",
        class_name="SolverSelfRegularizedELORETA",
        aliases=("sr-eloreta", "sreloreta"),
    ),
    # -- LORETA -------------------------------------------------------------
    SolverSpec(
        solver_id="loreta",
        module_path="invert.solvers.minimum_norm.loreta",
        class_name="SolverLORETA",
        aliases=("lor",),
    ),
    SolverSpec(
        solver_id="sloreta",
        module_path="invert.solvers.minimum_norm.sloreta",
        class_name="SolverSLORETA",
        aliases=("slor",),
    ),
    SolverSpec(
        solver_id="eloreta",
        module_path="invert.solvers.minimum_norm.eloreta",
        class_name="SolverELORETA",
        aliases=("elor",),
    ),
    SolverSpec(
        solver_id="sslofo",
        module_path="invert.solvers.minimum_norm.sslofo",
        class_name="SolverSSLOFO",
    ),
    # -- Other minimum-norm-like -------------------------------------------
    SolverSpec(
        solver_id="laura",
        module_path="invert.solvers.minimum_norm.laura",
        class_name="SolverLAURA",
        aliases=("laur",),
    ),
    SolverSpec(
        solver_id="backus-gilbert",
        module_path="invert.solvers.minimum_norm.backus_gilbert",
        class_name="SolverBackusGilbert",
        aliases=("b-g", "bg"),
    ),
    SolverSpec(
        solver_id="s-map",
        module_path="invert.solvers.minimum_norm.smap",
        class_name="SolverSMAP",
        aliases=("smap",),
    ),
    # -- Structured sparsity / edge-preserving -----------------------------
    SolverSpec(
        solver_id="tv",
        module_path="invert.solvers.minimum_norm.total_variation",
        class_name="SolverTotalVariation",
        aliases=("total-variation", "graph-tv"),
    ),
    # -- Bayesian / Champagne ----------------------------------------------
    SolverSpec(
        solver_id="champagne-mackay",
        module_path="invert.solvers.bayesian.champagne",
        class_name="SolverChampagne",
        default_kwargs={"update_rule": "MacKay"},
        aliases=("champagne", "mackaychampagne", "mackay-champagne", "mcc"),
        display_name="MacKay-Champagne",
    ),
    SolverSpec(
        solver_id="champagne-em",
        module_path="invert.solvers.bayesian.champagne",
        class_name="SolverChampagne",
        default_kwargs={"update_rule": "EM"},
        aliases=("emchampagne", "em-champagne", "emc"),
        display_name="EM-Champagne",
    ),
    SolverSpec(
        solver_id="champagne-convexity",
        module_path="invert.solvers.bayesian.champagne",
        class_name="SolverChampagne",
        default_kwargs={"update_rule": "Convexity"},
        aliases=(
            "convexitychampagne",
            "convexity-champagne",
            "coc",
            "mm-champagne",
        ),
        display_name="Convexity-Champagne",
    ),
    SolverSpec(
        solver_id="champagne-tem",
        module_path="invert.solvers.bayesian.champagne",
        class_name="SolverChampagne",
        default_kwargs={"update_rule": "TEM"},
        aliases=("temchampagne", "tem-champagne", "temc", "t-em-champagne"),
        display_name="TEM-Champagne",
    ),
    SolverSpec(
        solver_id="champagne-ar-em",
        module_path="invert.solvers.bayesian.champagne",
        class_name="SolverChampagne",
        default_kwargs={"update_rule": "AR-EM"},
        aliases=("aremchampagne", "arem-champagne", "aremc", "ar-em-champagne"),
        display_name="AR-EM-Champagne",
    ),
    SolverSpec(
        solver_id="champagne-low-snr",
        module_path="invert.solvers.bayesian.champagne",
        class_name="SolverChampagne",
        default_kwargs={"update_rule": "LowSNR"},
        aliases=("lowsnrchampagne", "low-snr-champagne", "lowsnr-champagne", "lsc"),
        display_name="Low-SNR-Champagne",
    ),
    SolverSpec(
        solver_id="champagne-adaptive",
        module_path="invert.solvers.bayesian.champagne",
        class_name="SolverChampagne",
        default_kwargs={"update_rule": "Adaptive"},
        aliases=("adaptivechampagne", "adaptive-champagne", "ac"),
        display_name="Adaptive-Champagne",
    ),
    SolverSpec(
        solver_id="nl-champagne",
        module_path="invert.solvers.bayesian.nl_champagne",
        class_name="SolverNLChampagne",
        aliases=("nlchampagne", "nl-champagne", "nlc"),
    ),
    SolverSpec(
        solver_id="omni-champagne",
        module_path="invert.solvers.bayesian.omni_champagne",
        class_name="SolverOmniChampagne",
        aliases=("omnichampagne", "omni-champagne", "omni champagne", "oc"),
    ),
    SolverSpec(
        solver_id="flex-champagne",
        module_path="invert.solvers.bayesian.flex_champagne",
        class_name="SolverFlexChampagne",
        aliases=("flexchampagne", "flex-champagne", "fc-champ"),
    ),
    SolverSpec(
        solver_id="flex-nl-champagne",
        module_path="invert.solvers.bayesian.flex_nl_champagne",
        class_name="SolverFlexNLChampagne",
        aliases=("flexnlchampagne", "flex-nl-champagne", "fnlc"),
    ),
    # -- Other Bayesian -----------------------------------------------------
    SolverSpec(
        solver_id="gamma-map",
        module_path="invert.solvers.bayesian.gamma_map",
        class_name="SolverGammaMAP",
        aliases=("gmap",),
    ),
    SolverSpec(
        solver_id="source-map",
        module_path="invert.solvers.bayesian.source_map",
        class_name="SolverSourceMAP",
    ),
    SolverSpec(
        solver_id="gamma-map-msp",
        module_path="invert.solvers.bayesian.gamma_map_msp",
        class_name="SolverGammaMAPMSP",
    ),
    SolverSpec(
        solver_id="source-map-msp",
        module_path="invert.solvers.bayesian.source_map_msp",
        class_name="SolverSourceMAPMSP",
    ),
    SolverSpec(
        solver_id="msp",
        module_path="invert.solvers.bayesian.msp",
        class_name="SolverMSP",
        aliases=("multiple-sparse-priors",),
    ),
    SolverSpec(
        solver_id="cmem",
        module_path="invert.solvers.bayesian.cmem",
        class_name="SolverCMEM",
    ),
    SolverSpec(
        solver_id="subspace-sbl",
        module_path="invert.solvers.bayesian.subspace_sbl",
        class_name="SolverSubspaceSBL",
        aliases=("ssm-nlc", "subspacesbl"),
    ),
    SolverSpec(
        solver_id="subspace-sbl-plus",
        module_path="invert.solvers.bayesian.subspace_sbl",
        class_name="SolverSubspaceSBLPlus",
        aliases=("ssm-nlc-plus", "subspacesblplus"),
    ),
    SolverSpec(
        solver_id="vb-sbl",
        module_path="invert.solvers.bayesian.vb_sbl",
        class_name="SolverVBSBL",
        aliases=("vbsbl", "variational-bayes-sbl"),
    ),
    # -- Beamformers --------------------------------------------------------
    SolverSpec(
        solver_id="mvab",
        module_path="invert.solvers.beamformers.mvab",
        class_name="SolverMVAB",
    ),
    SolverSpec(
        solver_id="lcmv",
        module_path="invert.solvers.beamformers.lcmv",
        class_name="SolverLCMV",
    ),
    SolverSpec(
        solver_id="dics",
        module_path="invert.solvers.beamformers.dics",
        class_name="SolverDICS",
        aliases=("dics-beamformer",),
    ),
    SolverSpec(
        solver_id="smv",
        module_path="invert.solvers.beamformers.smv",
        class_name="SolverSMV",
    ),
    SolverSpec(
        solver_id="wnmv",
        module_path="invert.solvers.beamformers.wnmv",
        class_name="SolverWNMV",
        aliases=("wmnv",),
    ),
    SolverSpec(
        solver_id="hocmv",
        module_path="invert.solvers.beamformers.hocmv",
        class_name="SolverHOCMV",
    ),
    SolverSpec(
        solver_id="esmv",
        module_path="invert.solvers.beamformers.esmv",
        class_name="SolverESMV",
    ),
    SolverSpec(
        solver_id="esmv2",
        module_path="invert.solvers.beamformers.esmv2",
        class_name="SolverESMV2",
    ),
    SolverSpec(
        solver_id="esmv3",
        module_path="invert.solvers.beamformers.esmv3",
        class_name="SolverESMV3",
    ),
    SolverSpec(
        solver_id="mcmv",
        module_path="invert.solvers.beamformers.mcmv",
        class_name="SolverMCMV",
    ),
    SolverSpec(
        solver_id="hocmcmv",
        module_path="invert.solvers.beamformers.hocmcmv",
        class_name="SolverHOCMCMV",
    ),
    SolverSpec(
        solver_id="recipsiicos",
        module_path="invert.solvers.beamformers.recipsiicos_plain",
        class_name="SolverReciPSIICOSPlain",
        aliases=("recipsiicos-plain",),
        display_name="ReciPSIICOS",
    ),
    SolverSpec(
        solver_id="recipsiicos-whitened",
        module_path="invert.solvers.beamformers.recipsiicos_whitened",
        class_name="SolverReciPSIICOSWhitened",
    ),
    SolverSpec(
        solver_id="sam",
        module_path="invert.solvers.beamformers.sam",
        class_name="SolverSAM",
    ),
    SolverSpec(
        solver_id="ebb",
        module_path="invert.solvers.beamformers.ebb",
        class_name="SolverEBB",
        aliases=("empirical-bayesian-beamformer",),
    ),
    SolverSpec(
        solver_id="adapt-flex-esmv",
        module_path="invert.solvers.beamformers.adapt_flex_esmv",
        class_name="SolverAdaptFlexESMV",
        aliases=("adaptflexesmv",),
    ),
    SolverSpec(
        solver_id="flex-esmv",
        module_path="invert.solvers.beamformers.flex_esmv",
        class_name="SolverFlexESMV",
        aliases=("flexesmv",),
    ),
    SolverSpec(
        solver_id="flex-esmv2",
        module_path="invert.solvers.beamformers.flex_esmv2",
        class_name="SolverFlexESMV2",
        aliases=("flexesmv2",),
    ),
    SolverSpec(
        solver_id="deblur-flex-esmv",
        module_path="invert.solvers.beamformers.deblur_flex_esmv",
        class_name="SolverDeblurFlexESMV",
        aliases=("deblurflexesmv",),
    ),
    SolverSpec(
        solver_id="safe-flex-esmv",
        module_path="invert.solvers.beamformers.safe_flex_esmv",
        class_name="SolverSafeFlexESMV",
        aliases=("safeflexesmv",),
    ),
    SolverSpec(
        solver_id="sharp-flex-esmv",
        module_path="invert.solvers.beamformers.sharp_flex_esmv",
        class_name="SolverSharpFlexESMV",
        aliases=("sharpflexesmv",),
    ),
    SolverSpec(
        solver_id="sharp-flex-esmv2",
        module_path="invert.solvers.beamformers.sharp_flex_esmv2",
        class_name="SolverSharpFlexESMV2",
        aliases=("sharpflexesmv2",),
    ),
    SolverSpec(
        solver_id="ssp-esmv",
        module_path="invert.solvers.beamformers.ssp_esmv",
        class_name="SolverSSPESMV",
        aliases=("sspesmv",),
    ),
    SolverSpec(
        solver_id="ir-esmv",
        module_path="invert.solvers.beamformers.iresmv",
        class_name="SolverIRESMV",
        aliases=("iresmv",),
    ),
    SolverSpec(
        solver_id="ssp-ir-esmv",
        module_path="invert.solvers.beamformers.ssp_iresmv",
        class_name="SolverSSPIRESMV",
        aliases=("sspiresmv",),
    ),
    SolverSpec(
        solver_id="unit-noise-gain",
        module_path="invert.solvers.beamformers.unit_noise_gain",
        class_name="SolverUnitNoiseGain",
        aliases=("ung", "unig"),
    ),
    # -- Dipole fitting -----------------------------------------------------
    SolverSpec(
        solver_id="ecd",
        module_path="invert.solvers.dipoles.ecd",
        class_name="SolverECD",
        aliases=("equivalent-current-dipole",),
    ),
    SolverSpec(
        solver_id="sesame",
        module_path="invert.solvers.dipoles.sesame",
        class_name="SolverSESAME",
    ),
    # -- Artificial Neural Networks (optional torch) ------------------------
    SolverSpec(
        solver_id="fc",
        module_path="invert.solvers.neural_networks.fc",
        class_name="SolverFC",
        aliases=("fully-connected", "fullyconnected", "esinet"),
        requires=("torch",),
    ),
    SolverSpec(
        solver_id="covcnn",
        module_path="invert.solvers.neural_networks.covcnn",
        class_name="SolverCovCNN",
        aliases=("cov-cnn", "covnet"),
        requires=("torch",),
    ),
    SolverSpec(
        solver_id="lstm",
        module_path="invert.solvers.neural_networks.lstm",
        class_name="SolverLSTM",
        requires=("torch",),
    ),
    SolverSpec(
        solver_id="cnn",
        module_path="invert.solvers.neural_networks.cnn",
        class_name="SolverCNN",
        requires=("torch",),
    ),
    # -- Matching Pursuit / Compressive Sensing -----------------------------
    SolverSpec(
        solver_id="omp",
        module_path="invert.solvers.matching_pursuit.omp",
        class_name="SolverOMP",
    ),
    SolverSpec(
        solver_id="cosamp",
        module_path="invert.solvers.matching_pursuit.cosamp",
        class_name="SolverCOSAMP",
    ),
    SolverSpec(
        solver_id="somp",
        module_path="invert.solvers.matching_pursuit.somp",
        class_name="SolverSOMP",
    ),
    SolverSpec(
        solver_id="rembo",
        module_path="invert.solvers.matching_pursuit.rembo",
        class_name="SolverREMBO",
    ),
    SolverSpec(
        solver_id="sp",
        module_path="invert.solvers.matching_pursuit.sp",
        class_name="SolverSP",
    ),
    SolverSpec(
        solver_id="ssp",
        module_path="invert.solvers.matching_pursuit.ssp",
        class_name="SolverSSP",
    ),
    SolverSpec(
        solver_id="smp",
        module_path="invert.solvers.matching_pursuit.smp",
        class_name="SolverSMP",
    ),
    SolverSpec(
        solver_id="ssmp",
        module_path="invert.solvers.matching_pursuit.ssmp",
        class_name="SolverSSMP",
    ),
    SolverSpec(
        solver_id="subsmp",
        module_path="invert.solvers.matching_pursuit.subsmp",
        class_name="SolverSubSMP",
    ),
    SolverSpec(
        solver_id="isubsmp",
        module_path="invert.solvers.matching_pursuit.isubsmp",
        class_name="SolverISubSMP",
    ),
    SolverSpec(
        solver_id="bcs",
        module_path="invert.solvers.bayesian.bcs",
        class_name="SolverBCS",
    ),
    # -- MUSIC / Subspace ---------------------------------------------------
    SolverSpec(
        solver_id="music",
        module_path="invert.solvers.music.music",
        class_name="SolverMUSIC",
    ),
    SolverSpec(
        solver_id="rap-music",
        module_path="invert.solvers.music.flex_music",
        class_name="SolverFLEXMUSIC",
        default_kwargs={"name": "RAP-MUSIC"},
        aliases=("rap",),
        display_name="RAP-MUSIC",
    ),
    SolverSpec(
        solver_id="trap-music",
        module_path="invert.solvers.music.flex_music",
        class_name="SolverFLEXMUSIC",
        default_kwargs={"name": "TRAP-MUSIC"},
        aliases=("trap",),
        display_name="TRAP-MUSIC",
    ),
    SolverSpec(
        solver_id="flex-music",
        module_path="invert.solvers.music.flex_music",
        class_name="SolverFLEXMUSIC",
        aliases=("flex-rap-music", "flex"),
        display_name="FLEX-MUSIC",
    ),
    SolverSpec(
        solver_id="flex-ssm",
        module_path="invert.solvers.music.signal_subspace_matching",
        class_name="SolverSignalSubspaceMatching",
        default_kwargs={"name": "Flexible Signal Subspace Matching"},
        display_name="FLEX-SSM",
    ),
    SolverSpec(
        solver_id="ssm",
        module_path="invert.solvers.music.signal_subspace_matching",
        class_name="SolverSignalSubspaceMatching",
        default_kwargs={"name": "Signal Subspace Matching"},
        display_name="SSM",
    ),
    SolverSpec(
        solver_id="flex-ap",
        module_path="invert.solvers.music.alternating_projections",
        class_name="SolverAlternatingProjections",
        default_kwargs={"name": "Flexible Alternating Projections"},
        aliases=("altproj",),
        display_name="FLEX-AP",
    ),
    SolverSpec(
        solver_id="ap",
        module_path="invert.solvers.music.alternating_projections",
        class_name="SolverAlternatingProjections",
        default_kwargs={"name": "Alternating Projections"},
        display_name="AP",
    ),
    SolverSpec(
        solver_id="adaptive-ap",
        module_path="invert.solvers.music.adaptive_alternating_projections",
        class_name="SolverAdaptiveAlternatingProjections",
        aliases=("adaptive-alternating-projections", "aap", "adaptivealtproj"),
    ),
    SolverSpec(
        solver_id="flex-music-2",
        module_path="invert.solvers.music.flex_music_2",
        class_name="SolverFLEXMUSIC_2",
        aliases=("flexmusic2",),
    ),
    SolverSpec(
        solver_id="generalized-iterative",
        module_path="invert.solvers.music.generalized_iterative",
        class_name="SolverGeneralizedIterative",
        aliases=("gi", "geniterative"),
    ),
    # -- Basis Functions ----------------------------------------------------
    SolverSpec(
        solver_id="gbf",
        module_path="invert.solvers.minimum_norm.basis_functions",
        class_name="SolverBasisFunctions",
        aliases=("basisfunctions", "basis-functions"),
        display_name="GBF",
    ),
    # -- State-space --------------------------------------------------------
    SolverSpec(
        solver_id="kalman",
        module_path="invert.solvers._old.kalman",
        class_name="SolverKalman",
        aliases=("kf", "stkf"),
    ),
    # -- Other --------------------------------------------------------------
    SolverSpec(
        solver_id="epifocus",
        module_path="invert.solvers.minimum_norm.epifocus",
        class_name="SolverEPIFOCUS",
    ),
    SolverSpec(
        solver_id="apse",
        module_path="invert.solvers.hybrids.apse",
        class_name="SolverAPSE",
    ),
    SolverSpec(
        solver_id="chimera",
        module_path="invert.solvers.hybrids.chimera",
        class_name="SolverChimera",
    ),
    SolverSpec(
        solver_id="random-noise",
        module_path="invert.solvers.random_noise",
        class_name="SolverRandomNoise",
        aliases=("random", "noise-baseline"),
    ),
)


def _build_solver_indexes(
    specs: tuple[SolverSpec, ...],
) -> tuple[dict[str, SolverSpec], dict[str, str]]:
    by_id: dict[str, SolverSpec] = {}
    alias_to_id: dict[str, str] = {}

    for spec in specs:
        sid = spec.normalized_id()
        if sid in by_id:
            raise ValueError(f"Duplicate solver id: {sid!r}")
        by_id[sid] = spec

    for spec in specs:
        sid = spec.normalized_id()
        candidates = [
            spec.solver_id,
            sid,
            *(spec.aliases or ()),
        ]
        if spec.display_name:
            candidates.append(spec.display_name)
        for alias in candidates:
            key = normalize_solver_name(str(alias))
            if not key:
                continue
            keys = {key, key.replace("-", "")}
            for k in keys:
                if not k:
                    continue
                existing = alias_to_id.get(k)
                if existing is not None and existing != sid:
                    raise ValueError(
                        f"Alias conflict for {k!r}: {existing!r} vs {sid!r}"
                    )
                alias_to_id[k] = sid

    return by_id, alias_to_id


_SPECS_BY_ID, _ALIAS_TO_ID = _build_solver_indexes(_SOLVER_SPECS)


def iter_solver_specs(
    *,
    include_internal: bool = True,
    include_unavailable: bool = False,
) -> list[SolverSpec]:
    """Return solver specs, optionally filtering internal/unavailable ones."""
    out: list[SolverSpec] = []
    for spec in _SOLVER_SPECS:
        if not include_internal and spec.internal:
            continue
        if not include_unavailable and not is_solver_available(spec):
            continue
        out.append(spec)
    return out


def list_solver_ids(
    *,
    include_internal: bool = True,
    include_unavailable: bool = False,
) -> list[str]:
    """List canonical solver ids."""
    return sorted(
        spec.normalized_id()
        for spec in iter_solver_specs(
            include_internal=include_internal, include_unavailable=include_unavailable
        )
    )


def list_solver_aliases(
    solver: str,
    *,
    include_id: bool = True,
    include_display_name: bool = True,
) -> list[str]:
    """List all registered aliases for a canonical solver id."""
    spec = get_solver_spec(solver)
    aliases: list[str] = []
    if include_id:
        aliases.append(spec.normalized_id())
    aliases.extend(spec.aliases)
    if include_display_name and spec.display_name:
        aliases.append(spec.display_name)
    # Normalize + dedupe while keeping order
    out: list[str] = []
    seen: set[str] = set()
    for a in aliases:
        if a in seen:
            continue
        seen.add(a)
        out.append(a)
    return out


def get_solver_spec(name_or_alias: str) -> SolverSpec:
    """Resolve a solver by canonical id or alias."""
    key = normalize_solver_name(name_or_alias)
    if not key or key not in _ALIAS_TO_ID:
        raise ValueError(
            f"Unknown solver {name_or_alias!r}. Available: {list_solver_ids()}"
        )
    return _SPECS_BY_ID[_ALIAS_TO_ID[key]]


def is_solver_available(spec: SolverSpec) -> bool:
    """Return True if all declared optional requirements are installed."""
    for requirement in spec.requires:
        if importlib.util.find_spec(requirement) is None:
            return False
    return True


def get_solver_class(name_or_alias: str):
    """Lazy-import and return a solver class."""
    spec = get_solver_spec(name_or_alias)
    if not is_solver_available(spec):
        missing = [r for r in spec.requires if importlib.util.find_spec(r) is None]
        raise ImportError(
            f"Solver {spec.normalized_id()!r} is unavailable; missing: {missing}"
        )
    mod = importlib.import_module(spec.module_path)
    return getattr(mod, spec.class_name)


def create_solver(name_or_alias: str = "mne", **kwargs: Any):
    """Create a solver instance by canonical id or alias."""
    spec = get_solver_spec(name_or_alias)
    solver_cls = get_solver_class(spec.solver_id)
    init_kwargs = dict(spec.default_kwargs)
    init_kwargs.update(kwargs)
    return solver_cls(**init_kwargs)
