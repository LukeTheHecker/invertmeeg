from . import hybrids
from .base import *
from .bayesian import *
from .beamformers import *
from .dipoles import *
from .hybrids import *
from .matching_pursuit import *
from .minimum_norm import *
from .music import *
from .random_noise import *

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
    "SolverDSPMMNE",
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
    "SolverLCMVMVPURE",
    "SolverDICS",
    "SolverSMV",
    "SolverWNMV",
    "SolverHOCMV",
    "SolverESMV",
    "SolverESMVMVPURE",
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
    "SolverFlexESMVMVPURE",
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
    "SolverREMBO",
    "SolverSP",
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
