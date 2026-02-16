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

    Regularization strategy
    -----------------------
    Beamformers optimize spatial filter power (maximize signal-to-noise of a
    virtual sensor), not data-fit residuals. Residual-based regularization
    criteria (GCV, L-curve) are therefore not applicable.

    Instead, regularization is applied to the data covariance during
    ``make_inverse_operator()`` via covariance-based strategies:

    - **OAS shrinkage** — Oracle Approximating Shrinkage toward a scaled
      identity, estimated analytically from the data.
    - **Condition number targeting** — Diagonal loading to reach a target
      condition number (e.g. 100), via ``condition_number_loaded_covariance``.
    - **Trace loading** — ``C + beta * trace(C)/n * I``, controlled by a
      scalar ``beta``.

    The benchmark runner respects this by always using index 0 for beamformers
    rather than running GCV/L-curve selection across multiple inverse operators.
    """

    _is_beamformer = True

    def __init__(self, *, reduce_rank=True, rank="auto", **kwargs):
        kwargs.setdefault("regularisation_method", "L")
        super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)
