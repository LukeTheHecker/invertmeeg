from importlib.metadata import PackageNotFoundError, version

from .invert import Solver, list_solvers

try:
    __version__ = version("invertmeeg")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["Solver", "list_solvers", "__version__"]
