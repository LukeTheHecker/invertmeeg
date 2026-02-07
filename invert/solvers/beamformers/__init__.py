"""
Beamformer solvers for M/EEG source reconstruction.
"""

from .adapt_flex_esmv import SolverAdaptFlexESMV
from .deblur_flex_esmv import SolverDeblurFlexESMV
from .dics import SolverDICS
from .ebb import SolverEBB
from .esmv import SolverESMV
from .esmv2 import SolverESMV2
from .esmv3 import SolverESMV3
from .flex_esmv import SolverFlexESMV
from .flex_esmv2 import SolverFlexESMV2
from .hocmcmv import SolverHOCMCMV
from .hocmv import SolverHOCMV
from .iresmv import SolverIRESMV
from .lcmv import SolverLCMV
from .lcmv_mvpure import SolverLCMVMVPURE
from .mcmv import SolverMCMV
from .mvab import SolverMVAB
from .recipsiicos_plain import SolverReciPSIICOSPlain
from .recipsiicos_whitened import SolverReciPSIICOSWhitened
from .safe_flex_esmv import SolverSafeFlexESMV
from .sam import SolverSAM
from .sharp_flex_esmv import SolverSharpFlexESMV
from .sharp_flex_esmv2 import SolverSharpFlexESMV2
from .smv import SolverSMV
from .ssp_esmv import SolverSSPESMV
from .ssp_iresmv import SolverSSPIRESMV
from .unit_noise_gain import SolverUnitNoiseGain
from .wnmv import SolverWNMV

__all__ = [
    "SolverMVAB",
    "SolverLCMV",
    "SolverLCMVMVPURE",
    "SolverDICS",
    "SolverSMV",
    "SolverWNMV",
    "SolverHOCMV",
    "SolverESMV",
    "SolverESMV2",
    "SolverESMV3",
    "SolverAdaptFlexESMV",
    "SolverFlexESMV",
    "SolverFlexESMV2",
    "SolverDeblurFlexESMV",
    "SolverSafeFlexESMV",
    "SolverSharpFlexESMV",
    "SolverSharpFlexESMV2",
    "SolverMCMV",
    "SolverUnitNoiseGain",
    "SolverHOCMCMV",
    "SolverSAM",
    "SolverEBB",
    "SolverReciPSIICOSPlain",
    "SolverReciPSIICOSWhitened",
    "SolverSSPESMV",
    "SolverIRESMV",
    "SolverSSPIRESMV",
]
