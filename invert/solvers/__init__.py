from .base import *
from .bayesian import *
from .beamformers import *
from .dipoles import *
from . import hybrids
from .hybrids import *
from .matching_pursuit import *
from .minimum_norm import *
from .music import *
from .random_noise import *
from ._old.kalman import SolverKalman

# ANN solvers — require torch (optional dependency)
try:
    from .neural_networks import *
except ImportError:
    pass

__all__ = [
    "BaseSolver",
    "InverseOperator",
    # Minimum Norm
    "SolverMNE",
    "SolverGFTMNE",
    "SolverWMNE",
    "SolverDSPM",
    "SolverMinimumL1Norm",
    "SolverMinimumL1NormGPT",
    "SolverMinimumL1L2Norm",
    "SolverGFTMinimumL1Norm",
    "SolverSelfRegularizedELORETA",
    "SolverTotalVariation",
    # LORETA
    "SolverLORETA",
    "SolverSLORETA",
    "SolverELORETA",
    "SolverSSLOFO",
    # Other min-norm-like
    "SolverLAURA",
    "SolverBackusGilbert",
    "SolverSMAP",
    # Bayesian
    "SolverChampagne",
    "SolverNLChampagne",
    "SolverFlexChampagne",
    "SolverFlexNLChampagne",
    "SolverOmniChampagne",
    "SolverGammaMAP",
    "SolverSourceMAP",
    "SolverGammaMAPMSP",
    "SolverSourceMAPMSP",
    "SolverMSP",
    "SolverCMEM",
    "SolverSubspaceSBL",
    "SolverSubspaceSBLPlus",
    "SolverVBSBL",
    # Beamformers
    "SolverMVAB",
    "SolverLCMV",
    "SolverDICS",
    "SolverSMV",
    "SolverWNMV",
    "SolverHOCMV",
    "SolverESMV",
    "SolverESMV2",
    "SolverESMV3",
    "SolverMCMV",
    "SolverHOCMCMV",
    "SolverReciPSIICOSPlain",
    "SolverReciPSIICOSWhitened",
    "SolverSAM",
    "SolverEBB",
    "SolverAdaptFlexESMV",
    "SolverFlexESMV",
    "SolverFlexESMV2",
    "SolverDeblurFlexESMV",
    "SolverSafeFlexESMV",
    "SolverSharpFlexESMV",
    "SolverSharpFlexESMV2",
    "SolverSSPESMV",
    "SolverIRESMV",
    "SolverSSPIRESMV",
    "SolverUnitNoiseGain",
    # Dipoles
    "SolverECD",
    "SolverSESAME",
    # Matching Pursuit
    "SolverOMP",
    "SolverCOSAMP",
    "SolverSOMP",
    "SolverREMBO",
    "SolverSP",
    "SolverSSP",
    "SolverSMP",
    "SolverSSMP",
    "SolverSubSMP",
    "SolverISubSMP",
    "SolverBCS",
    # MUSIC / Subspace
    "SolverMUSIC",
    "SolverFLEXMUSIC",
    "SolverSignalSubspaceMatching",
    "SolverAlternatingProjections",
    "SolverAdaptiveAlternatingProjections",
    "SolverFLEXMUSIC_2",
    "SolverGeneralizedIterative",
    "SolverExSoMUSIC",
    # Basis Functions
    "SolverBasisFunctions",
    # State-space
    "SolverKalman",
    # Other
    "SolverEPIFOCUS",
    # Hybrids
    "SolverAPSE",
    "SolverChimera",
    # ANN (optional)
    "SolverCNN",
    "SolverCovCNN",
    "SolverCovCNNCenters",
    "SolverCovCNNMask",
    "SolverCovCNNKL",
    "SolverCovCNNKLDiff",
    "SolverCovCNNKLAdapt",
    "SolverCovCNNStructKLDiff",
    "SolverCovCNNBasisDiagKLDiff",
    "SolverFC",
    "SolverLSTM",
    # Baseline
    "SolverRandomNoise",
]
