"""
Neural network solvers for M/EEG source reconstruction.
"""

_available = {}

for _name, _mod in [
    ("SolverCNN", ".cnn"),
    ("SolverCovCNN", ".covcnn"),
    ("SolverCovCNNCenters", ".covcnn_centers"),
    ("SolverCovCNNMask", ".covcnn_mask"),
    ("SolverCovCNNKL", ".covcnn_kl"),
    ("SolverCovCNNKLFlexOMP", ".covcnn_kl_flexomp"),
    ("SolverCovCNNKLDiff", ".covcnn_kl_diff"),
    ("SolverCovCNNKLAdapt", ".covcnn_kl_adapt"),
    ("SolverCovCNNStructKLDiff", ".covcnn_structkl_diff"),
    ("SolverCovCNNBasisDiagKLDiff", ".covcnn_basisdiag_kl_diff"),
    ("SolverFC", ".fc"),
    ("SolverLSTM", ".lstm"),
]:
    try:
        _module = __import__(__name__ + _mod, fromlist=[_name])
        _available[_name] = getattr(_module, _name)
    except Exception:
        continue

globals().update(_available)

__all__ = [
    *_available.keys(),
]
