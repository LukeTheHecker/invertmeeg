"""Base class for beamformer solvers.

Beamformers construct spatial filters from covariance structure rather than
fitting the forward model to the data in a residual-minimization sense.
Residual-based regularization selection (GCV, L-curve) is therefore not
applicable; instead, covariance-based strategies (OAS shrinkage, condition
number targeting, trace loading) are used during ``make_inverse_operator()``.
"""

from ..base import BaseSolver


class BaseBeamformer(BaseSolver):
    """Base class for all beamformer solvers.

    Inherits from :class:`BaseSolver` and sets beamformer-specific defaults:

    - ``_is_beamformer = True`` — used by the base class to skip residual-based
      regularization selection and to enable free-orientation support.
    - ``regularisation_method`` defaults to ``"L"`` (L-curve), though for
      beamformers the base class bypasses this entirely and uses index 0.
    """

    _is_beamformer = True

    def __init__(self, *, reduce_rank=True, rank="auto", **kwargs):
        kwargs.setdefault("regularisation_method", "L")
        super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)
