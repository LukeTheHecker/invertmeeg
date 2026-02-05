"""
Matching pursuit solvers for M/EEG source reconstruction.
"""

from .cosamp import SolverCOSAMP
from .isubsmp import SolverISubSMP
from .omp import SolverOMP
from .rembo import SolverREMBO
from .smp import SolverSMP
from .somp import SolverSOMP
from .sp import SolverSP
from .ssmp import SolverSSMP
from .ssp import SolverSSP
from .subsmp import SolverSubSMP

__all__ = [
    "SolverOMP",
    "SolverSOMP",
    "SolverCOSAMP",
    "SolverREMBO",
    "SolverSP",
    "SolverSSP",
    "SolverSMP",
    "SolverSSMP",
    "SolverSubSMP",
    "SolverISubSMP",
]
