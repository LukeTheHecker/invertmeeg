"""
MUSIC / Subspace solvers for M/EEG source reconstruction.
"""

from .adaptive_alternating_projections import SolverAdaptiveAlternatingProjections
from .alternating_projections import SolverAlternatingProjections
from .exhaustive_subspace_ml import SolverExhaustiveSubspaceML
from .exso_music import SolverExSoMUSIC
from .flex_greedy_ml import SolverFlexGreedyML
from .flex_music import SolverFLEXMUSIC
from .flex_music_2 import SolverFLEXMUSIC_2
from .generalized_iterative import SolverGeneralizedIterative
from .greedy_ml import SolverGreedyML
from .music import SolverMUSIC
from .signal_subspace_matching import SolverSignalSubspaceMatching

__all__ = [
    "SolverMUSIC",
    "SolverExSoMUSIC",
    "SolverFLEXMUSIC",
    "SolverSignalSubspaceMatching",
    "SolverAlternatingProjections",
    "SolverAdaptiveAlternatingProjections",
    "SolverFLEXMUSIC_2",
    "SolverGeneralizedIterative",
    "SolverGreedyML",
    "SolverExhaustiveSubspaceML",
    "SolverFlexGreedyML",
]
