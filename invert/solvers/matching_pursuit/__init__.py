"""
Matching pursuit solvers for M/EEG source reconstruction.
"""

from .cosamp import SolverCOSAMP
from .isubsmp import SolverISubSMP
from .omp import SolverOMP
from .rembo import SolverREMBO
from .sp import SolverSP
from .subsmp import SolverSubSMP

__all__ = [
    "SolverOMP",
    "SolverCOSAMP",
    "SolverREMBO",
    "SolverSP",
    "SolverSubSMP",
    "SolverISubSMP",
]
