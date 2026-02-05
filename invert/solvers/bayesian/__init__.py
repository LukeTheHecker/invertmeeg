"""Bayesian solvers for M/EEG source reconstruction."""

from .bcs import SolverBCS
from .champagne import SolverChampagne
from .cmem import SolverCMEM
from .flex_champagne import SolverFlexChampagne
from .flex_nl_champagne import SolverFlexNLChampagne
from .gamma_map import SolverGammaMAP
from .gamma_map_msp import SolverGammaMAPMSP
from .msp import SolverMSP
from .nl_champagne import SolverNLChampagne
from .omni_champagne import SolverOmniChampagne
from .source_map import SolverSourceMAP
from .source_map_msp import SolverSourceMAPMSP
from .subspace_sbl import SolverSubspaceSBL, SolverSubspaceSBLPlus
from .vb_sbl import SolverVBSBL

__all__ = [
    "SolverChampagne",
    "SolverNLChampagne",
    "SolverFlexChampagne",
    "SolverFlexNLChampagne",
    "SolverOmniChampagne",
    "SolverBCS",
    "SolverGammaMAP",
    "SolverSourceMAP",
    "SolverGammaMAPMSP",
    "SolverSourceMAPMSP",
    "SolverMSP",
    "SolverCMEM",
    "SolverSubspaceSBL",
    "SolverSubspaceSBLPlus",
    "SolverVBSBL",
]
