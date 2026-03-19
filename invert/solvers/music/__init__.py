"""
MUSIC / Subspace solvers for M/EEG source reconstruction.
"""

from .alternating_projections import SolverAlternatingProjections
from .exhaustive_subspace_ml import SolverExhaustiveSubspaceML
from .exso_music import SolverExSoMUSIC
from .flex_greedy_ml_bgproj import SolverFlexGreedyMLBgProj
from .flex_greedy_ml_bgwhiten import SolverFlexGreedyMLBgWhiten
from .flex_greedy_ml import SolverFlexGreedyML
from .flex_greedy_ml_diffuse import SolverFlexGreedyMLDiffuse
from .flex_greedy_ml_support import SolverFlexGreedyMLSupport
from .flex_greedy_ml_temporal_diff import SolverFlexGreedyMLTemporalDiff
from .flex_greedy_ml_temporal_diff2 import SolverFlexGreedyMLTemporalDiff2
from .flex_music import SolverFLEXMUSIC
from .generalized_iterative import SolverGeneralizedIterative
from .greedy_ml import SolverGreedyML
from .greedy_ml_support import SolverGreedyMLSupport
from .music import SolverMUSIC
from .signal_subspace_matching import SolverSignalSubspaceMatching

__all__ = [
    "SolverMUSIC",
    "SolverExSoMUSIC",
    "SolverFLEXMUSIC",
    "SolverSignalSubspaceMatching",
    "SolverAlternatingProjections",
    "SolverGeneralizedIterative",
    "SolverGreedyML",
    "SolverGreedyMLSupport",
    "SolverExhaustiveSubspaceML",
    "SolverFlexGreedyML",
    "SolverFlexGreedyMLSupport",
    "SolverFlexGreedyMLBgProj",
    "SolverFlexGreedyMLBgWhiten",
    "SolverFlexGreedyMLTemporalDiff",
    "SolverFlexGreedyMLTemporalDiff2",
    "SolverFlexGreedyMLDiffuse",
]
