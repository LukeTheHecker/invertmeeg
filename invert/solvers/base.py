from __future__ import annotations

import logging
import os
import pickle as pkl
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import mne
import numpy as np
import numpy.typing as npt
from mne.io.constants import FIFF

from ..util import find_corner
from .orientation import (
    ensure_surface_free_surf_ori,
    estimate_orientation_pca,
    reduce_free_to_scalar,
)

logger = logging.getLogger(__name__)


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value or "solver"


@dataclass
class SolverMeta:
    """Metadata describing a solver for documentation and catalog generation."""

    full_name: str
    category: str
    description: str
    acronym: str | None = None
    slug: str | None = None
    references: list[str] = field(default_factory=list)
    internal: bool = False

    def __post_init__(self) -> None:
        if self.acronym is None and self.slug is None:
            raise ValueError("SolverMeta requires at least one of 'acronym' or 'slug'.")
        if self.acronym is None:
            self.acronym = str(self.slug).upper()
        if self.slug is None:
            self.slug = _slugify(str(self.acronym))


@dataclass
class WhitenedForward:
    """Preprocessed forward model with optional whitening, priors, and normalization."""

    # Always populated
    G_white: np.ndarray  # (n_eff, n_sources) — whitened+projected leadfield
    sensor_transform: (
        np.ndarray
    )  # (n_eff, n_chans)   — W @ P (or just P if no noise_cov)
    projector: np.ndarray  # (n_chans, n_chans)  — SSP projector
    whitener_mode: (
        str  # "projected" | "identity_projector" | "identity_projector+jitter" | "none"
    )
    n_eff: int  # effective sensor rank (= rows of sensor_transform)

    # Populated when source_cov or depth is given
    prior_diag: np.ndarray | None = None  # (n_sources,)
    R_sqrt: np.ndarray | None = None  # sqrt(prior_diag), trace-scaled if requested

    # Populated when trace_normalize=True
    A: np.ndarray | None = None  # effective operator for regularized inversion
    trace_scale: float = 1.0  # scale factor applied by trace normalization

    # Populated when precompute_svd=True
    svd: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None


@dataclass
class ChannelAlignmentReport:
    """Channel alignment result across forward/data/noise covariance."""

    context: str
    kept_ch_names: list[str]
    dropped_from_forward: list[str] = field(default_factory=list)
    dropped_from_data: list[str] = field(default_factory=list)
    dropped_from_noise_cov: list[str] = field(default_factory=list)

    def has_drops(self) -> bool:
        return bool(
            self.dropped_from_forward
            or self.dropped_from_data
            or self.dropped_from_noise_cov
        )


class InverseOperator:
    """Container for precomputed inverse operators.

    Parameters
    ----------
    inverse_operator : numpy.ndarray or list of numpy.ndarray
        The inverse operator matrix (or matrices).  Each matrix should have
        shape ``(n_sources, n_channels)`` so that applying it to sensor data
        ``y`` of shape ``(n_channels, n_times)`` produces a source estimate
        ``(n_sources, n_times)``.

        For free-orientation solvers, ``n_sources = n_locations * n_orient``
        with rows ordered as ``[x1, y1, z1, x2, y2, z2, ...]``.
    solver_name : str
        Name of the solver that produced this operator.
    expected_shape : tuple[int, int] | None
        If given ``(n_sources, n_channels)``, validate each matrix matches.
    """

    def __init__(
        self,
        inverse_operator: Any,
        solver_name: str,
        expected_shape: tuple[int, int] | None = None,
    ) -> None:
        self.solver_name = solver_name
        self.expected_shape = expected_shape
        self.data = inverse_operator
        self.handle_inverse_operator()

    def has_multiple_operators(self) -> bool:
        """Return True if this object wraps more than one operator."""
        if isinstance(self.data, list) and len(self.data) > 1:
            return True
        return False

    def handle_inverse_operator(self) -> None:
        if not isinstance(self.data, list):
            self.data = [self.data]
        self.type = type(self.data[0])

        if self.expected_shape is not None:
            for i, mat in enumerate(self.data):
                arr = np.asarray(mat)
                if arr.ndim != 2:
                    raise ValueError(
                        f"InverseOperator[{i}] from {self.solver_name!r}: "
                        f"expected 2D array, got {arr.ndim}D with shape {arr.shape}"
                    )
                if arr.shape != self.expected_shape:
                    raise ValueError(
                        f"InverseOperator[{i}] from {self.solver_name!r}: "
                        f"expected shape {self.expected_shape}, got {arr.shape}"
                    )

    def apply(self, M: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Apply the precomputed inverse operator to data matrix *M*.

        Parameters
        ----------
        M : numpy.ndarray
            The M/EEG data matrix ``(n_channels, n_timepoints)``.

        Returns
        -------
        numpy.ndarray
            The source estimate matrix ``(n_sources, n_timepoints)``.
        """
        return self.data[0] @ M


class BaseSolver:
    """
    Parameters
    ----------
    regularisation_method : str
        Can be either
            "GCV"       -> generalized cross validation
            "MGCV"      -> modified GCV (GCV with a gamma correction factor)
            "RGCV"      -> robust GCV (penalises leverage concentration)
            "R1GCV"     -> strong robust GCV (penalises solution variance)
            "L"         -> L-Curve method using triangle method
            "Product"   -> Minimal product method
    n_reg_params : int
        The number of regularisation parameters to try. The higher, the
        more accurate the regularisation and the slower the computations.
    prep_leadfield : bool
        If True -> Apply common average referencing and normalisation of the leadfield columns.
    reduce_rank : bool
        Whether to reduce the rank of the M/EEG data
    rank : str/int
        Can be either int -> select only the <rank> largest eigenvectors of the data
        "auto" -> automatically select the optimal rank using the L-curve method and
                  an eigenvalue drop-off criterion
    plot_reg : bool
        Plot the regularization parameters.
    gcv_gamma : float
        Gamma factor for GCV. Default 1.0 (plain GCV).
    mgcv_gamma : float
        Gamma factor used when `regularisation_method="MGCV"`. Values slightly
        above 1.0 typically bias toward more regularization.
    rgcv_gamma : float
        Mixing parameter γ ∈ (0, 1) for RGCV. ``V_RGCV = [γ + (1-γ)·μ₂] · V_GCV``
        where μ₂ = tr(A²)/n penalises leverage concentration. Default 0.5.
    r1gcv_gamma : float
        Mixing parameter γ ∈ (0, 1) for R1GCV. ``V_R1GCV = [γ + (1-γ)·μ₁₂] · V_GCV``
        where μ₁₂ = -dμ₁/dα penalises solution variance amplification. Default 0.5.

    """

    meta: SolverMeta | None = None
    # Opt-in: solvers that support true 3-component (vector) source models.
    SUPPORTS_VECTOR_ORIENTATION: bool = False
    # Default regularization grid exponents for r_values = logspace(min_exp, max_exp, n_reg_params).
    # Solver subclasses can override this to widen/shift the search grid without affecting others.
    # Note: the upper end was widened to reduce frequent "edge-of-grid" GCV picks in practice.
    R_VALUE_EXPONENTS: tuple[float, float] = (-10.0, 2.0)

    def __init__(
        self,
        regularisation_method: str = "GCV",
        n_reg_params: int = 10,
        prep_leadfield: bool = True,
        use_depth_weighting: bool = False,
        use_last_alpha: bool = False,
        rank: str | int = "enhanced",
        depth_weighting: float = 0.5,
        reduce_rank: bool = False,
        plot_reg: bool = False,
        common_average_reference: bool = False,
        gcv_gamma: float = 1.0,
        mgcv_gamma: float = 1.05,
        rgcv_gamma: float = 0.5,
        r1gcv_gamma: float = 0.5,
        orientation: str = "auto",
        orientation_pca_reg: float = 0.01,
        orientation_pca_deterministic_sign: bool = True,
        verbose: int = 0,
        **kwargs: Any,
    ) -> None:
        self.verbose = verbose

        min_exp, max_exp = getattr(self, "R_VALUE_EXPONENTS", (-10.0, 1.0))
        self.r_values = np.logspace(float(min_exp), float(max_exp), n_reg_params)
        # self.r_values = np.logspace(7, 9, n_reg_params)

        self.n_reg_params = n_reg_params
        self.regularisation_method = regularisation_method
        self._is_beamformer = getattr(self.__class__, "_is_beamformer", False)
        self.prep_leadfield = prep_leadfield
        self.use_depth_weighting = use_depth_weighting
        self.use_last_alpha = use_last_alpha
        self.last_reg_idx: int | None = None
        self.rank = rank
        self.depth_weighting = depth_weighting
        self.depth_weights = None
        self.reduce_rank = reduce_rank
        self.plot_reg = plot_reg
        self.made_inverse_operator = False
        self.common_average_reference = common_average_reference
        self.gcv_gamma = gcv_gamma
        self.mgcv_gamma = mgcv_gamma
        self.rgcv_gamma = rgcv_gamma
        self.r1gcv_gamma = r1gcv_gamma
        orientation = str(orientation).lower()
        if orientation not in {"auto", "fixed", "pca", "free"}:
            raise ValueError(
                "orientation must be one of {'auto','fixed','pca','free'}, "
                f"got {orientation!r}"
            )
        self.orientation = orientation
        self.orientation_pca_reg = float(orientation_pca_reg)
        if not np.isfinite(self.orientation_pca_reg) or self.orientation_pca_reg < 0:
            raise ValueError(
                f"orientation_pca_reg must be a finite non-negative float, got {orientation_pca_reg!r}"
            )
        self.orientation_pca_deterministic_sign = bool(
            orientation_pca_deterministic_sign
        )
        self.require_recompute = True
        self.require_data = True
        self._free_orientation = False
        self._n_orient = 1
        self._orientation_data_obj: Any | None = None
        self._orientation_q: np.ndarray | None = None
        # Attributes populated during operator construction.
        self.forward: Any = None
        self.leadfield: np.ndarray = np.empty((0, 0), dtype=float)
        self.inverse_operators: list[InverseOperator] = []
        self._leadfield_orig: np.ndarray | None = None
        self._sensor_transform: np.ndarray | None = None
        self._operator_ch_names: tuple[str, ...] = tuple()
        self._last_data_ch_names: tuple[str, ...] | None = None
        self._last_input_data_ch_names: tuple[str, ...] | None = None

    def make_inverse_operator(
        self,
        forward: mne.Forward,
        *args,
        alpha: str | float = "auto",
        reference: npt.NDArray | None = None,
        **kwargs: Any,
    ) -> None:
        """Base function to create the inverse operator based on the forward
            model.

        Parameters
        ----------
        forward : mne.Forward
            The mne Forward model object.
        alpha : ["auto", float]
            Dimensionless regularization knob ``r``.

            - If ``"auto"``: create a grid of values using ``self.r_values``.
            - If a float: interpreted as a single ``r`` value.

            The solver then converts ``r`` to an **effective**, dimensionful
            regularization parameter by multiplying with the largest eigenvalue
            of a reference matrix (default: ``L @ L.T``).


        Return
        ------
        None

        """
        # Optional: store the data object used for orientation estimation (PCA mode).
        # Many solvers accept an mne_obj in make_inverse_operator(forward, mne_obj, ...);
        # when provided positionally, it will appear here as args[0].
        self._orientation_data_obj = None
        self._orientation_q = None
        if args:
            candidate = args[0]
            if isinstance(
                candidate,
                (
                    mne.Evoked,
                    mne.EvokedArray,
                    mne.Epochs,
                    mne.EpochsArray,
                    mne.io.BaseRaw,
                ),
            ):
                self._orientation_data_obj = candidate

        self.forward = deepcopy(forward)
        self.prepare_forward()
        # Reset whitening state — will be set by prepare_whitened_forward()
        self._leadfield_orig = None
        self._sensor_transform = None
        self._operator_ch_names = tuple(self.forward.ch_names)
        self._last_data_ch_names = None
        self._last_input_data_ch_names = None
        self.alpha = alpha
        self.alphas = self.get_alphas(reference=reference)
        self.made_inverse_operator = True

    def validate_inverse_operators(self) -> None:
        """Validate that inverse_operators is well-formed after construction.

        Call this at the end of a subclass ``make_inverse_operator()`` to catch
        shape mismatches early.  Failures raise ``ValueError`` with a clear
        message identifying the solver and the shape discrepancy.
        """
        ops = getattr(self, "inverse_operators", None)
        if ops is None or len(ops) == 0:
            return  # some solvers (e.g. NN) populate operators lazily

        n_chans = int(self.leadfield.shape[0])
        n_sources = int(self.leadfield.shape[1])

        # Expected rows: for scalar orientation n_sources, for free n_sources already
        # reflects the orientation (leadfield has shape (n_chans, n_sources * n_orient)
        # when free, but after prepare_forward n_sources in leadfield already includes
        # the orient dimension).
        expected_rows = n_sources
        solver_name = getattr(self, "name", self.__class__.__name__)

        for i, op in enumerate(ops):
            for j, mat in enumerate(op.data):
                arr = np.asarray(mat)
                if arr.ndim != 2:
                    raise ValueError(
                        f"{solver_name}: inverse_operators[{i}].data[{j}] is "
                        f"{arr.ndim}D (shape {arr.shape}), expected 2D "
                        f"(n_sources={expected_rows}, n_channels={n_chans})"
                    )
                if arr.shape[1] != n_chans:
                    raise ValueError(
                        f"{solver_name}: inverse_operators[{i}].data[{j}] has "
                        f"{arr.shape[1]} columns, expected {n_chans} (n_channels)"
                    )
                if arr.shape[0] != expected_rows:
                    logger.debug(
                        "%s: inverse_operators[%d].data[%d] has %d rows, "
                        "expected %d (n_sources). This may be intentional for "
                        "solvers that collapse orientation internally.",
                        solver_name,
                        i,
                        j,
                        arr.shape[0],
                        expected_rows,
                    )

    def store_obj_information(self, mne_obj):
        if hasattr(mne_obj, "tmin"):
            self.tmin = mne_obj.tmin
        else:
            self.tmin = 0

        self.obj_info = mne_obj.info

    def apply_inverse_operator(self, mne_obj) -> mne.SourceEstimate:
        """Apply the inverse operator

        Parameters
        ----------
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.

        Return
        ------
        stc : mne.SourceEstimate
            The mne SourceEstimate object.

        """

        data = self.unpack_data_obj(mne_obj)
        self.validate_operator_data_compatibility(data)
        source_mat, _idx = self._compute_source_mat(data)
        stc = self.source_to_object(source_mat)
        return stc

    def _compute_source_mat(
        self, data: npt.NDArray[np.floating]
    ) -> tuple[npt.NDArray[np.floating], int]:
        """Return (source_mat, idx) for the current regularisation selection policy."""
        if self.use_last_alpha and self.last_reg_idx is not None:
            idx = int(self.last_reg_idx)
            return self.inverse_operators[idx].apply(data), idx

        # Beamformer note: residual-based criteria like GCV/L-curve are designed for
        # regularized linear inverse operators (G @ x ≈ y). For classic covariance
        # beamformers (LCMV, SMV, SAM, ESMV, ...), those criteria are generally not
        # meaningful. Default to the first (typically smallest) regularization in
        # the precomputed grid, or the single provided alpha.
        if self._is_beamformer and getattr(self, "inverse_operators", None):
            idx = 0
            self.last_reg_idx = idx
            return self.inverse_operators[idx].apply(data), idx

        method = self.regularisation_method.lower()
        if method == "l":
            source_mat, idx = self.regularise_lcurve(data, plot=self.plot_reg)
        elif method in {"gcv", "mgcv", "rgcv", "r1gcv", "composite"}:
            if method == "gcv":
                gamma = self.gcv_gamma
            elif method == "mgcv":
                gamma = self.mgcv_gamma
            elif method in {"rgcv", "composite"}:
                gamma = self.rgcv_gamma
            else:  # r1gcv
                gamma = self.r1gcv_gamma
            source_mat, idx = self.regularise_gcv(
                data, plot=self.plot_reg, gamma=gamma, method=method
            )
        elif method == "product":
            source_mat, idx = self.regularise_product(data, plot=self.plot_reg)
        else:
            raise AttributeError(
                f"{self.regularisation_method} is no valid regularisation method."
            )

        self.last_reg_idx = int(idx)
        return source_mat, int(idx)

    def apply_inverse_operator_vector(self, mne_obj):
        """Apply the inverse operator and return a vector (3-component) STC.

        This is an opt-in API: the default ``apply_inverse_operator`` remains scalar
        and will collapse free-orientation outputs by vector norm.
        """

        if (
            not getattr(self, "_free_orientation", False)
            or int(getattr(self, "_n_orient", 1)) != 3
        ):
            raise ValueError(
                "apply_inverse_operator_vector() requires a 3-component source model. "
                "Build the operator with orientation='free' (vector-capable solvers only), "
                "or use a volume/discrete free-orientation forward with orientation='auto'."
            )

        data = self.unpack_data_obj(mne_obj)
        self.validate_operator_data_compatibility(data)
        source_mat, _idx = self._compute_source_mat(data)
        return self.source_to_object_vector(source_mat)

    def prep_data(self, mne_obj):
        if (
            not mne_obj.proj
            and "eeg" in mne_obj.get_channel_types()
            and self.common_average_reference
        ):
            mne_obj.set_eeg_reference("average", projection=True, verbose=0).apply_proj(
                verbose=0
            )

        return mne_obj

    def validate_operator_data_compatibility(self, data: np.ndarray) -> None:
        """Ensure operator input dimension matches apply-time data channels."""
        inverse_ops = getattr(self, "inverse_operators", None)
        expected_chans: int | None = None
        if inverse_ops:
            try:
                op_mat = np.asarray(inverse_ops[0].data[0], dtype=float)
            except Exception:
                op_mat = None
            if op_mat is not None and op_mat.ndim == 2:
                expected_chans = int(op_mat.shape[1])

        op_ch_names = list(getattr(self, "_operator_ch_names", ()) or [])
        if expected_chans is None and op_ch_names:
            expected_chans = len(op_ch_names)
        if expected_chans is None:
            return

        got_chans = int(np.asarray(data).shape[0])
        data_ch_names = list(
            getattr(self, "_last_data_ch_names", ())
            or getattr(self, "_last_input_data_ch_names", ())
            or []
        )
        missing_in_data: list[str] = []
        extra_in_data: list[str] = []
        if op_ch_names and data_ch_names:
            data_set = set(data_ch_names)
            op_set = set(op_ch_names)
            missing_in_data = [ch for ch in op_ch_names if ch not in data_set]
            extra_in_data = [ch for ch in data_ch_names if ch not in op_set]

        if expected_chans == got_chans and not (missing_in_data or extra_in_data):
            return

        msg = (
            "Data channels do not match the inverse operator input dimension "
            f"(operator expects {expected_chans}, data has {got_chans}). "
            "Recompute the inverse operator on the current data channel set."
        )
        if missing_in_data or extra_in_data:
            msg = (
                f"{msg} Missing in data: {len(missing_in_data)}"
                f"{self._format_channel_list(missing_in_data, max_items=None)}. "
                f"Extra in data: {len(extra_in_data)}"
                f"{self._format_channel_list(extra_in_data, max_items=None)}."
            )
        raise ValueError(msg)

    @staticmethod
    def _validate_unique_channel_names(ch_names: list[str], *, label: str) -> None:
        if len(ch_names) != len(set(ch_names)):
            msg = f"{label} contains duplicate channel names."
            raise ValueError(msg)

    @staticmethod
    def _format_channel_list(ch_names: list[str], *, max_items: int | None = 8) -> str:
        if not ch_names:
            return ""
        if max_items is None:
            head = ", ".join(ch_names)
            return f" [{head}]"
        head = ", ".join(ch_names[:max_items])
        if len(ch_names) > max_items:
            head = f"{head}, ..."
        return f" [{head}]"

    def log_channel_alignment(self, report: ChannelAlignmentReport) -> None:
        if not report.has_drops():
            return
        logger.info(
            "%s channel alignment: kept=%d, dropped_from_forward=%d%s, dropped_from_data=%d%s, dropped_from_noise_cov=%d%s",
            report.context,
            len(report.kept_ch_names),
            len(report.dropped_from_forward),
            self._format_channel_list(report.dropped_from_forward, max_items=None),
            len(report.dropped_from_data),
            self._format_channel_list(report.dropped_from_data, max_items=None),
            len(report.dropped_from_noise_cov),
            self._format_channel_list(report.dropped_from_noise_cov, max_items=None),
        )

    def log_regularisation_edge_choice(self, *, optimum_idx: int, method: str) -> None:
        """Warn when alpha selection falls on the boundary of the search grid."""
        n_alphas = len(getattr(self, "alphas", []) or [])
        inverse_ops = getattr(self, "inverse_operators", None)
        n_tested = len(inverse_ops) if inverse_ops else n_alphas

        # Only meaningful if the solver evaluated more than one candidate.
        if n_tested <= 1:
            return
        if optimum_idx not in (0, n_tested - 1):
            return

        edge = "lowest" if optimum_idx == 0 else "highest"
        alpha = float(self.alphas[optimum_idx]) if optimum_idx < n_alphas else np.nan
        solver_label = getattr(self, "name", self.__class__.__name__)
        logger.warning(
            "%s selected the %s regularization parameter for %s in the search range "
            "(idx=%d of %d, alpha=%.6e). Consider widening or shifting `r_values`.",
            method,
            edge,
            solver_label,
            optimum_idx,
            n_tested - 1,
            alpha,
        )

    def align_channel_sets(
        self,
        *,
        forward_ch_names: list[str],
        data_ch_names: list[str] | None = None,
        noise_cov_ch_names: list[str] | None = None,
        context: str = "BaseSolver",
    ) -> ChannelAlignmentReport:
        """Align channel names using forward order as the canonical order."""
        forward_ch_names = list(forward_ch_names)
        self._validate_unique_channel_names(forward_ch_names, label="forward")

        kept_ch_names = list(forward_ch_names)

        if data_ch_names is not None:
            data_ch_names = list(data_ch_names)
            self._validate_unique_channel_names(data_ch_names, label="data")
            data_set = set(data_ch_names)
            kept_ch_names = [ch for ch in kept_ch_names if ch in data_set]

        if noise_cov_ch_names is not None:
            noise_cov_ch_names = list(noise_cov_ch_names)
            self._validate_unique_channel_names(noise_cov_ch_names, label="noise_cov")
            cov_set = set(noise_cov_ch_names)
            kept_ch_names = [ch for ch in kept_ch_names if ch in cov_set]

        kept_set = set(kept_ch_names)
        dropped_from_forward = [ch for ch in forward_ch_names if ch not in kept_set]
        dropped_from_data: list[str] = []
        dropped_from_noise_cov: list[str] = []

        if data_ch_names is not None:
            dropped_from_data = [ch for ch in data_ch_names if ch not in kept_set]

        if noise_cov_ch_names is not None:
            dropped_from_noise_cov = [
                ch for ch in noise_cov_ch_names if ch not in kept_set
            ]

        return ChannelAlignmentReport(
            context=context,
            kept_ch_names=kept_ch_names,
            dropped_from_forward=dropped_from_forward,
            dropped_from_data=dropped_from_data,
            dropped_from_noise_cov=dropped_from_noise_cov,
        )

    @staticmethod
    def coerce_noise_cov(
        noise_cov: mne.Covariance,
    ) -> tuple[np.ndarray, list[str]]:
        """Return covariance matrix and channel names from an MNE covariance."""
        if not isinstance(noise_cov, mne.Covariance):
            msg = f"noise_cov must be an mne.Covariance or None. Got {type(noise_cov)}."
            raise TypeError(msg)

        cov_mat = np.asarray(noise_cov["data"], dtype=float)
        cov_names = list(noise_cov.get("names", noise_cov.get("ch_names", [])))
        if cov_mat.ndim != 2 or cov_mat.shape[0] != cov_mat.shape[1]:
            msg = f"noise_cov must be a square 2D array, got shape {cov_mat.shape}"
            raise ValueError(msg)
        if len(cov_names) != cov_mat.shape[0]:
            msg = (
                "noise_cov channel-name length does not match covariance shape: "
                f"{len(cov_names)} names vs {cov_mat.shape}"
            )
            raise ValueError(msg)
        return cov_mat, cov_names

    @staticmethod
    def reorder_covariance_to_channels(
        noise_cov: np.ndarray,
        noise_cov_ch_names: list[str],
        target_ch_names: list[str],
    ) -> np.ndarray:
        index_map = {ch: idx for idx, ch in enumerate(noise_cov_ch_names)}
        picks = [index_map[ch] for ch in target_ch_names]
        return np.asarray(noise_cov[np.ix_(picks, picks)], dtype=float)

    @staticmethod
    def make_identity_noise_cov(
        ch_names: list[str],
        *,
        nfree: int = 1,
    ) -> mne.Covariance:
        n_chans = len(ch_names)
        return mne.Covariance(
            data=np.eye(n_chans, dtype=float),
            names=list(ch_names),
            bads=[],
            projs=[],
            nfree=int(max(nfree, 1)),
        )

    def unpack_data_obj(self, mne_obj, pick_types=None):
        """Unpacks the mne data object and returns the data.

        Parameters
        ----------
        mne_obj : [mne.Evoked, mne.EvokedArray, mne.Epochs, mne.EpochsArray, mne.Raw]

        Return
        ------
        data : numpy.ndarray
            The M/EEG data matrix.

        """

        type_list = [
            mne.Evoked,
            mne.EvokedArray,
            mne.Epochs,
            mne.EpochsArray,
            mne.io.Raw,
            mne.io.RawArray,
        ]  # mne.io.brainvision.brainvision.RawBrainVision]
        if pick_types is None:
            pick_types = ["meg", "eeg", "fnirs"]
        assert not isinstance(pick_types, dict), (
            "pick_types must be of type str or list(str), but is of type dict()"
        )

        # Prepare Data
        mne_obj = self.prep_data(mne_obj)
        channels_before_pick = list(mne_obj.ch_names)
        mne_obj_meeg = mne_obj.copy().pick(pick_types)
        channels_after_pick = list(mne_obj_meeg.ch_names)
        self._last_input_data_ch_names = tuple(channels_after_pick)
        dropped_by_pick_type = [
            ch for ch in channels_before_pick if ch not in set(channels_after_pick)
        ]
        if dropped_by_pick_type:
            logger.info(
                "unpack_data_obj type filter: dropped=%d%s",
                len(dropped_by_pick_type),
                self._format_channel_list(dropped_by_pick_type),
            )

        canonical_ch_names = list(
            getattr(self, "_operator_ch_names", ()) or self.forward.ch_names
        )
        alignment = self.align_channel_sets(
            forward_ch_names=canonical_ch_names,
            data_ch_names=channels_after_pick,
            context="unpack_data_obj",
        )
        self.log_channel_alignment(alignment)
        missing_operator_channels = list(alignment.dropped_from_forward)
        extra_data_channels = list(alignment.dropped_from_data)
        if (
            getattr(self, "made_inverse_operator", False)
            and getattr(self, "_operator_ch_names", ())
            and missing_operator_channels
        ):
            msg = (
                "Data is missing channels required by the inverse operator "
                f"({len(missing_operator_channels)} missing)"
                f"{self._format_channel_list(missing_operator_channels, max_items=None)}. "
                "Recompute the inverse operator on the current data channel set."
            )
            raise ValueError(msg)
        if (
            getattr(self, "made_inverse_operator", False)
            and getattr(self, "_operator_ch_names", ())
            and extra_data_channels
        ):
            msg = (
                "Data channels do not match the inverse operator channel set. "
                f"Extra in data: {len(extra_data_channels)}"
                f"{self._format_channel_list(extra_data_channels, max_items=None)}. "
                "Recompute the inverse operator on the current data channel set."
            )
            raise ValueError(msg)
        picks = alignment.kept_ch_names

        # Select only data channels in mne_obj
        mne_obj_meeg.pick(picks)

        # Store original forward model for later
        self.forward_original = deepcopy(self.forward)

        # Select only available data channels in forward
        self.forward = self.forward.pick_channels(picks, ordered=True)

        # Test if ch_names in forward model and mne_obj_meeg are equal
        if self.forward.ch_names != mne_obj_meeg.ch_names:
            msg = "channels available in mne object are not equal to those present in the forward model."
            raise ValueError(msg)
        if len(self.forward.ch_names) <= 1:
            raise ValueError("forward model contains only a single channel")

        # check if the object is an evoked object
        if isinstance(mne_obj, (mne.Evoked, mne.EvokedArray)):
            # handle evoked object
            data = mne_obj_meeg.data

        # check if the object is a raw object
        elif isinstance(mne_obj, (mne.Epochs, mne.EpochsArray)):
            data = mne_obj_meeg.average().data

        # check if the object is a raw object
        elif isinstance(
            mne_obj,
            (
                mne.io.Raw,
                mne.io.RawArray,
                mne.io.brainvision.brainvision.RawBrainVision,
            ),
        ):
            # handle raw object
            data = mne_obj_meeg._data

        # handle other cases
        else:
            msg = f"mne_obj is of type {type(mne_obj)} but needs to be one of the following types: {type_list}"
            raise AttributeError(msg)

        self.store_obj_information(mne_obj)

        if self.reduce_rank:
            data = self.select_signal_subspace(data, rank=self.rank)

        self._last_data_ch_names = tuple(mne_obj_meeg.ch_names)

        # Restore the original forward model so it matches what the
        # inverse operators were built with.  Do NOT call prepare_forward()
        # here — it would overwrite self.leadfield, clobbering any whitened
        # or otherwise transformed leadfield set up by make_inverse_operator.
        self.forward = self.forward_original

        return data

    def unpack_covariance_samples(
        self,
        mne_obj,
        *,
        pick_types: list[str] | None = None,
        require_forward_channel_match: bool = True,
        context: str = "BaseSolver",
    ) -> tuple[np.ndarray, list[str]]:
        """Return a (n_channels, n_samples) matrix for covariance estimation.

        Compared to :meth:`unpack_data_obj`, this method preserves sample count
        for epochs by concatenating all epoch time samples instead of averaging.
        Channel handling follows forward-model ordering.
        """
        if pick_types is None:
            pick_types = ["meg", "eeg", "fnirs"]

        mne_obj = self.prep_data(mne_obj)
        obj = mne_obj.copy().pick(pick_types)

        forward_ch_names = list(self.forward.ch_names)
        obj_set = set(obj.ch_names)
        if require_forward_channel_match:
            missing = [ch for ch in forward_ch_names if ch not in obj_set]
            if missing:
                raise ValueError(
                    f"{context} covariance estimation requires data channels to "
                    "match the forward model. Missing in data: "
                    f"{missing[:10]}{'...' if len(missing) > 10 else ''}."
                )
            kept = forward_ch_names
        else:
            kept = [ch for ch in forward_ch_names if ch in obj_set]

        if len(kept) <= 1:
            raise ValueError(
                f"{context} covariance estimation requires at least 2 shared channels."
            )

        obj.pick(kept)
        if getattr(obj, "ch_names", None) != kept and hasattr(obj, "reorder_channels"):
            obj.reorder_channels(kept)

        if isinstance(obj, (mne.Epochs, mne.EpochsArray)):
            X = obj.get_data()  # (n_epochs, n_ch, n_times)
            Y = X.transpose(1, 0, 2).reshape(int(X.shape[1]), -1)
        elif isinstance(obj, (mne.Evoked, mne.EvokedArray)):
            Y = np.asarray(obj.data, dtype=float)
        elif isinstance(obj, mne.io.BaseRaw):
            Y = np.asarray(obj.get_data(), dtype=float)
        else:
            Y = np.asarray(self.unpack_data_obj(mne_obj), dtype=float)

        if self.reduce_rank:
            Y = self.select_signal_subspace(np.asarray(Y, dtype=float), rank=self.rank)
        return np.asarray(Y, dtype=float), kept

    def get_alphas(self, reference=None):
        """Create list of regularization parameters (alphas) based on the
        largest eigenvalue of the leadfield or some reference matrix.

        Parameters
        ----------
        reference : [None, numpy.ndarray]
            If None: use ``L @ L.T`` to calculate the scaling, else use the
            provided reference matrix (e.g., data covariance / CSD).

        Return
        ------
        alphas : list
            List of **effective** regularization parameters (dimensionful),
            obtained as ``alphas = r * max_eig(reference)``.

        """
        if reference is None:
            reference = self.leadfield @ self.leadfield.T

        _, eigs, _ = np.linalg.svd(reference, full_matrices=False)
        self.max_eig = eigs.max()

        if self.alpha == "auto":
            alphas = list(self.max_eig * self.r_values)
        else:
            alphas = [
                self.alpha * self.max_eig,
            ]
        # Side effect: keep a consistent place for downstream code/tests to read.
        self.alphas = alphas
        return alphas

    @staticmethod
    def data_covariance(
        Y: npt.NDArray[np.floating], *, center: bool = True, ddof: int = 1
    ):
        """Compute normalized sensor-space covariance.

        Parameters
        ----------
        Y : np.ndarray
            Data array of shape (n_channels, n_times) or (n_channels,).
        center : bool
            If True, subtract the per-channel temporal mean before computing the covariance.
        ddof : int
            Delta degrees of freedom used in the normalization: divide by ``max(n_times - ddof, 1)``.

        Returns
        -------
        C : np.ndarray
            Covariance matrix of shape (n_channels, n_channels).
        """
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]
        if center:
            Y = Y - Y.mean(axis=1, keepdims=True)
        n_times = int(Y.shape[1])
        denom = max(n_times - int(ddof), 1)
        return (Y @ Y.T) / float(denom)

    @staticmethod
    def coerce_diag_source_prior(
        source_cov: np.ndarray | None, n_sources: int
    ) -> np.ndarray:
        """Coerce source covariance input to a diagonal prior vector."""
        if n_sources <= 0:
            raise ValueError(f"n_sources must be > 0, got {n_sources}")

        if source_cov is None:
            return np.ones(n_sources, dtype=float)

        source_cov = np.asarray(source_cov, dtype=float)
        if source_cov.ndim == 0:
            return np.full(n_sources, float(source_cov), dtype=float)

        if source_cov.ndim == 1:
            if source_cov.shape[0] != n_sources:
                msg = (
                    f"source_cov has length {source_cov.shape[0]}, expected {n_sources}"
                )
                raise ValueError(msg)
            return source_cov

        if source_cov.ndim == 2:
            expected = (n_sources, n_sources)
            if source_cov.shape != expected:
                msg = f"source_cov has shape {source_cov.shape}, expected {expected}"
                raise ValueError(msg)
            off_diag = source_cov - np.diag(np.diag(source_cov))
            if not np.allclose(off_diag, 0.0):
                msg = (
                    "Full (non-diagonal) source_cov is not supported. "
                    "Pass a 1D diagonal prior instead."
                )
                raise ValueError(msg)
            return np.diag(source_cov)

        raise ValueError(f"Invalid source_cov with ndim={source_cov.ndim}")

    def compute_sensor_projector(
        self,
        forward_or_info: Any | None = None,
        n_chans: int | None = None,
        fallback_identity: bool = True,
    ) -> np.ndarray:
        """Compute SSP projector in sensor space (identity if unavailable)."""
        if forward_or_info is None:
            forward_or_info = getattr(self, "forward", None)

        if n_chans is None:
            if self.leadfield.size:
                n_chans = int(self.leadfield.shape[0])
            elif forward_or_info is not None:
                try:
                    n_chans = int(forward_or_info["sol"]["data"].shape[0])
                except Exception:
                    n_chans = None

        if n_chans is None or n_chans <= 0:
            raise ValueError("n_chans must be provided or inferable from solver state.")

        info = None
        if isinstance(forward_or_info, dict):
            if "projs" in forward_or_info:
                info = forward_or_info
            else:
                info = forward_or_info.get("info")
        elif forward_or_info is not None:
            info = getattr(forward_or_info, "info", None)

        projs = []
        bads = []
        ch_names = []
        if info is not None:
            projs = list(info.get("projs", []) or [])
            bads = list(info.get("bads", []) or [])
            ch_names = list(info.get("ch_names", []) or [])

        if not ch_names and forward_or_info is not None:
            try:
                ch_names = list(forward_or_info.ch_names)  # type: ignore[union-attr]
            except Exception:
                ch_names = []

        if not projs:
            return np.eye(n_chans, dtype=float)

        if len(ch_names) != n_chans:
            ch_names = [str(i) for i in range(n_chans)]

        try:
            projector, _nproj, _ = mne.make_projector(
                projs, ch_names, bads=bads, verbose=0
            )
            projector = np.asarray(projector, dtype=float)
            if projector.shape != (n_chans, n_chans):
                msg = (
                    f"Projector has shape {projector.shape}, expected "
                    f"{(n_chans, n_chans)}"
                )
                raise ValueError(msg)
            return projector
        except Exception as err:
            if not fallback_identity:
                raise
            logger.warning("Failed to build SSP projector (%s); using I.", err)
            return np.eye(n_chans, dtype=float)

    @staticmethod
    def compute_sensor_whitener(
        noise_cov: np.ndarray,
        projector: np.ndarray | None = None,
        *,
        rank_tol: float = 1e-12,
        eps: float = 1e-15,
    ) -> np.ndarray:
        """Compute PCA-space sensor whitener with rank truncation."""
        noise_cov = np.asarray(noise_cov, dtype=float)
        if noise_cov.ndim != 2 or noise_cov.shape[0] != noise_cov.shape[1]:
            msg = f"noise_cov must be a square 2D array, got shape {noise_cov.shape}"
            raise ValueError(msg)
        n_chans = int(noise_cov.shape[0])

        if projector is None:
            projector = np.eye(n_chans, dtype=float)
        projector = np.asarray(projector, dtype=float)
        if projector.shape != (n_chans, n_chans):
            msg = (
                f"projector has shape {projector.shape}, expected {(n_chans, n_chans)}"
            )
            raise ValueError(msg)

        Cn = 0.5 * (noise_cov + noise_cov.T)
        Cn_proj = projector @ Cn @ projector.T
        Cn_proj = 0.5 * (Cn_proj + Cn_proj.T)

        eigvals, eigvecs = np.linalg.eigh(Cn_proj)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        max_ev = float(np.max(eigvals)) if eigvals.size else 0.0
        if max_ev <= 0:
            return np.zeros((0, n_chans), dtype=float)

        # Important: eigenvalues can be extremely small in native MEG units
        # (e.g., ~1e-28..1e-24). Using an *absolute* floor here would incorrectly
        # discard the full rank and trigger "jitter" fallbacks. Use a relative
        # threshold instead.
        rel_tol = max(float(rank_tol), float(eps))
        mask = eigvals > (rel_tol * max_ev)
        if not np.any(mask):
            return np.zeros((0, n_chans), dtype=float)

        return np.asarray((eigvecs[:, mask] / np.sqrt(eigvals[mask])).T, dtype=float)

    @classmethod
    def compute_sensor_whitener_robust(
        cls,
        noise_cov: np.ndarray,
        projector: np.ndarray | None = None,
        *,
        rank_tol: float = 1e-12,
        eps: float = 1e-15,
    ) -> tuple[np.ndarray, str]:
        """Compute a sensor whitener with fallback strategies."""
        noise_cov = np.asarray(noise_cov, dtype=float)
        if noise_cov.ndim != 2 or noise_cov.shape[0] != noise_cov.shape[1]:
            msg = f"noise_cov must be a square 2D array, got shape {noise_cov.shape}"
            raise ValueError(msg)

        W = cls.compute_sensor_whitener(
            noise_cov,
            projector=projector,
            rank_tol=rank_tol,
            eps=eps,
        )
        if W.shape[0] > 0:
            return W, "projected"

        n_chans = int(noise_cov.shape[0])
        W = cls.compute_sensor_whitener(
            noise_cov,
            projector=np.eye(n_chans, dtype=float),
            rank_tol=rank_tol,
            eps=eps,
        )
        if W.shape[0] > 0:
            return W, "identity_projector"

        Cn = 0.5 * (noise_cov + noise_cov.T)
        diag_abs = np.abs(np.diag(Cn))
        scale = float(np.max(diag_abs)) if diag_abs.size else 1.0
        jitter = max(rank_tol * max(scale, 1.0), eps)
        Cn_reg = Cn + jitter * np.eye(n_chans, dtype=float)
        W = cls.compute_sensor_whitener(
            Cn_reg,
            projector=np.eye(n_chans, dtype=float),
            rank_tol=rank_tol,
            eps=eps,
        )
        if W.shape[0] > 0:
            return W, "identity_projector+jitter"

        return W, "failed"

    @staticmethod
    def compute_depth_prior_whitened(
        G_white: np.ndarray,
        *,
        depth: float = 0.8,
        depth_limit: float = 10.0,
        eps: float = 1e-15,
    ) -> np.ndarray:
        """Compute fixed-orientation depth prior from whitened leadfield."""
        G_white = np.asarray(G_white, dtype=float)
        if G_white.ndim != 2:
            raise ValueError(f"G_white must be 2D, got ndim={G_white.ndim}")

        sens = np.sum(G_white * G_white, axis=0)
        sens = np.maximum(sens, eps)
        weights = 1.0 / sens

        if depth_limit > 0:
            w_min = float(np.min(weights))
            w_max = w_min * float(depth_limit) ** 2
            weights = np.minimum(weights, w_max)

        if depth != 0:
            weights = weights ** float(depth)
        return weights

    @staticmethod
    def trace_normalize_operator(
        A: np.ndarray, target_rank: int, *, eps: float = 1e-15
    ) -> tuple[np.ndarray, float]:
        """Scale A so trace(A A^T) equals target_rank."""
        A = np.asarray(A, dtype=float)
        if A.ndim != 2:
            raise ValueError(f"A must be 2D, got ndim={A.ndim}")
        if target_rank <= 0:
            raise ValueError(f"target_rank must be > 0, got {target_rank}")

        trace_aat = float(np.sum(A * A))
        scale = np.sqrt(float(target_rank) / max(trace_aat, eps))
        return A * scale, float(scale)

    @staticmethod
    def solve_tikhonov_svd(
        A: np.ndarray,
        lambda2: float,
        *,
        left_scale: np.ndarray | None = None,
        svd: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        eps: float = 1e-15,
    ) -> np.ndarray:
        """Solve Tikhonov system via SVD with optional row scaling."""
        if lambda2 < 0:
            raise ValueError(f"lambda2 must be >= 0, got {lambda2}")

        A = np.asarray(A, dtype=float)
        if A.ndim != 2:
            raise ValueError(f"A must be 2D, got ndim={A.ndim}")
        if A.shape[0] == 0 or A.shape[1] == 0:
            return np.zeros((A.shape[1], A.shape[0]), dtype=float)

        if svd is None:
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
        else:
            U, s, Vt = svd
            U = np.asarray(U, dtype=float)
            s = np.asarray(s, dtype=float)
            Vt = np.asarray(Vt, dtype=float)

        denom = np.maximum(s * s + float(lambda2), eps)
        gamma = s / denom
        kernel = (Vt.T * gamma[np.newaxis, :]) @ U.T

        if left_scale is None:
            return kernel

        left_scale = np.asarray(left_scale, dtype=float)
        if left_scale.ndim != 1 or left_scale.shape[0] != kernel.shape[0]:
            msg = f"left_scale has shape {left_scale.shape}, expected {(kernel.shape[0],)}"
            raise ValueError(msg)
        return kernel * left_scale[:, np.newaxis]

    @staticmethod
    def noise_normalize_rows(
        K_white: np.ndarray,
        *,
        K_full: np.ndarray | None = None,
        eps: float = 1e-15,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalize rows by noise std derived from whitened operator."""
        K_white = np.asarray(K_white, dtype=float)
        if K_white.ndim != 2:
            raise ValueError(f"K_white must be 2D, got ndim={K_white.ndim}")

        if K_full is None:
            K_full = K_white
        K_full = np.asarray(K_full, dtype=float)
        if K_full.ndim != 2:
            raise ValueError(f"K_full must be 2D, got ndim={K_full.ndim}")
        if K_full.shape[0] != K_white.shape[0]:
            msg = (
                "K_full and K_white must have the same number of rows, got "
                f"{K_full.shape[0]} and {K_white.shape[0]}"
            )
            raise ValueError(msg)

        noise_var = np.sum(K_white * K_white, axis=1)
        noise_std = np.sqrt(np.maximum(noise_var, eps))
        return K_full / noise_std[:, np.newaxis], noise_std

    def prepare_whitened_forward(
        self,
        noise_cov: mne.Covariance | None = None,
        *,
        apply_projector_when_no_cov: bool = True,
        source_cov: np.ndarray | None = None,
        depth: float | None = None,
        depth_limit: float = 10.0,
        trace_normalize: bool = False,
        precompute_svd: bool = False,
        rank_tol: float = 1e-12,
        eps: float = 1e-15,
    ) -> WhitenedForward:
        """Pipeline: SSP projection, whitening, source prior, trace norm, SVD.

        Parameters
        ----------
        noise_cov : mne.Covariance | None
            Noise covariance. Channels are aligned by name. If *None*, no
            whitening is applied.
        apply_projector_when_no_cov : bool
            If ``True`` and ``noise_cov`` is ``None``, still apply the SSP
            projector from the forward model. If ``False``, use identity in
            that case.
        source_cov : array or None
            Diagonal source prior (1D or 2D diagonal).
        depth : float or None
            Depth-weighting exponent. *None* disables depth weighting.
        depth_limit : float
            Maximum depth-weight ratio (squared).
        trace_normalize : bool
            Scale the effective operator so trace(A A^T) == n_eff.
        precompute_svd : bool
            Store economy SVD of the effective operator.
        rank_tol : float
            Relative eigenvalue threshold for whitener rank truncation.
        eps : float
            Numerical floor.

        Returns
        -------
        WhitenedForward
        """
        # Save original leadfield (before any whitening) for regularization trace.
        # If already saved (repeated call without intervening make_inverse_operator),
        # keep the original to prevent double-whitening.
        if self._leadfield_orig is None:
            self._leadfield_orig = np.array(self.leadfield, dtype=float)
        G = np.asarray(self._leadfield_orig, dtype=float)
        forward_ch_names = list(self.forward.ch_names)
        n_sources = int(G.shape[1])
        noise_cov_mat: np.ndarray | None = None

        if noise_cov is not None:
            noise_cov_mat, noise_cov_ch_names = self.coerce_noise_cov(noise_cov)
            alignment = self.align_channel_sets(
                forward_ch_names=forward_ch_names,
                noise_cov_ch_names=noise_cov_ch_names,
                context="prepare_whitened_forward",
            )
            self.log_channel_alignment(alignment)
            if len(alignment.kept_ch_names) <= 1:
                msg = (
                    "forward/noise_cov channel intersection has <= 1 channel. "
                    "Check channel naming and montage consistency."
                )
                raise ValueError(msg)

            if len(alignment.kept_ch_names) != len(forward_ch_names):
                fwd_idx_map = {ch: idx for idx, ch in enumerate(forward_ch_names)}
                keep_idx = [fwd_idx_map[ch] for ch in alignment.kept_ch_names]
                G = G[keep_idx, :]
                self._leadfield_orig = G.copy()
                self.forward = self.forward.pick_channels(
                    alignment.kept_ch_names, ordered=True
                )
                forward_ch_names = alignment.kept_ch_names
                n_sources = int(G.shape[1])

            noise_cov_mat = self.reorder_covariance_to_channels(
                noise_cov_mat, noise_cov_ch_names, forward_ch_names
            )
        n_chans = int(G.shape[0])

        # --- SSP projector ---
        if noise_cov_mat is None and not apply_projector_when_no_cov:
            P = np.eye(n_chans, dtype=float)
        else:
            P = self.compute_sensor_projector(
                forward_or_info=self.forward,
                n_chans=n_chans,
            )

        # --- Sensor whitening ---
        if noise_cov_mat is not None:
            noise_cov_mat = np.asarray(noise_cov_mat, dtype=float)
            if noise_cov_mat.shape != (n_chans, n_chans):
                msg = (
                    f"noise_cov has shape {noise_cov_mat.shape}, expected {(n_chans, n_chans)}. "
                    "Pass noise_cov with channel names to allow automatic alignment."
                )
                raise ValueError(msg)
            noise_cov_mat = 0.5 * (noise_cov_mat + noise_cov_mat.T)

            W, whitener_mode = self.compute_sensor_whitener_robust(
                noise_cov_mat,
                projector=P,
                rank_tol=rank_tol,
                eps=eps,
            )
            if W.shape[0] == 0:
                raise ValueError(
                    "Whitening rank is zero after projected/identity/jitter "
                    "fallbacks. Check noise covariance channel alignment and "
                    "positivity."
                )
            sensor_transform = W @ P  # (n_eff, n_chans)
            G_white = sensor_transform @ G  # (n_eff, n_sources)
            n_eff = int(W.shape[0])
        else:
            sensor_transform = P  # (n_chans, n_chans)
            G_white = P @ G  # (n_chans, n_sources)
            n_eff = n_chans
            whitener_mode = "none"

        # --- Source prior ---
        prior_diag = None
        R_sqrt = None
        if source_cov is not None or depth is not None:
            prior_diag = self.coerce_diag_source_prior(source_cov, n_sources)
            if depth is not None:
                prior_diag = prior_diag * self.compute_depth_prior_whitened(
                    G_white,
                    depth=depth,
                    depth_limit=depth_limit,
                    eps=eps,
                )
            prior_diag = np.maximum(prior_diag, eps)
            R_sqrt = np.sqrt(prior_diag)

        # --- Trace normalization ---
        A = None
        trace_scale = 1.0
        if trace_normalize:
            A = (
                G_white * R_sqrt[np.newaxis, :]
                if R_sqrt is not None
                else G_white.copy()
            )
            A, trace_scale = self.trace_normalize_operator(
                A, target_rank=n_eff, eps=eps
            )
            if R_sqrt is not None:
                R_sqrt = R_sqrt * trace_scale

        # --- SVD ---
        svd = None
        if precompute_svd:
            svd_target = A if A is not None else G_white
            svd = np.linalg.svd(svd_target, full_matrices=False)

        # Install whitened state so regularisation methods work in whitened space
        self.leadfield = G_white
        self._sensor_transform = sensor_transform
        self._operator_ch_names = tuple(self.forward.ch_names)

        return WhitenedForward(
            G_white=G_white,
            sensor_transform=sensor_transform,
            projector=P,
            whitener_mode=whitener_mode,
            n_eff=n_eff,
            prior_diag=prior_diag,
            R_sqrt=R_sqrt,
            A=A,
            trace_scale=trace_scale,
            svd=svd,
        )

    def regularise_lcurve(self, M, plot=False):
        """Find optimally regularized inverse solution using the L-Curve method [1].

        Parameters
        ----------
        M : numpy.ndarray
            The M/EEG data matrix (n_channels, n_timepoints)

        Return
        ------
        source_mat : numpy.ndarray
            The inverse solution  (dipoles x time points)
        optimum_idx : int
            The index of the selected (optimal) regularization parameter

        References
        ----------
        [1] Grech, R., Cassar, T., Muscat, J., Camilleri, K. P., Fabri, S. G.,
        Zervakis, M., ... & Vanrumste, B. (2008). Review on solving the inverse
        problem in EEG source analysis. Journal of neuroengineering and
        rehabilitation, 5(1), 1-33.

        """

        if not self.inverse_operators:
            msg = (
                "No inverse operators available for regularisation. "
                "Call make_inverse_operator() and ensure it populates "
                "self.inverse_operators before apply_inverse_operator()."
            )
            raise ValueError(msg)

        L_resid = self.leadfield  # G_white if whitened, else original
        _st = getattr(self, "_sensor_transform", None)

        source_mats = [
            inverse_operator.apply(M) for inverse_operator in self.inverse_operators
        ]

        l2_norms = np.asarray(
            [np.linalg.norm(source_mat) for source_mat in source_mats]
        )

        M_eff = _st @ M if _st is not None else M
        residual_norms = np.asarray(
            [np.linalg.norm(L_resid @ source_mat - M_eff) for source_mat in source_mats]
        )

        # L-curve corner finding is conventionally done in log-log space:
        # x-axis: residual norm ||Gx - M||, y-axis: solution norm ||x||.
        optimum_idx = self.find_corner(residual_norms, l2_norms)

        source_mat = source_mats[optimum_idx]
        if plot:
            plt.figure()
            plt.loglog(residual_norms, l2_norms, "ok-")
            plt.loglog(
                residual_norms[optimum_idx],
                l2_norms[optimum_idx],
                "r*",
                markersize=10,
            )
            plt.xlabel(r"Residual norm $\|Gx - M\|_2$")
            plt.ylabel(r"Solution norm $\|x\|_2$")
            plt.grid(True, which="both")
            alpha = self.alphas[optimum_idx]
            plt.title(f"L-Curve: {alpha}")

        self.log_regularisation_edge_choice(
            optimum_idx=optimum_idx,
            method="L-curve",
        )
        return source_mat, optimum_idx

    @staticmethod
    def get_curvature(x, y):
        x_t = np.gradient(x)
        y_t = np.gradient(y)
        vel = np.array([[x_t[i], y_t[i]] for i in range(x_t.size)])
        speed = np.sqrt(x_t * x_t + y_t * y_t)
        np.array([1 / speed] * 2).transpose() * vel

        np.gradient(speed)
        xx_t = np.gradient(x_t)
        yy_t = np.gradient(y_t)

        curvature_val = np.abs(xx_t * y_t - x_t * yy_t) / (x_t * x_t + y_t * y_t) ** 1.5

        return curvature_val

    def regularise_gcv(
        self,
        M,
        plot: bool = False,
        gamma: float | None = None,
        method: str = "gcv",
    ):
        """Find optimally regularized inverse solution using GCV or a variant.

        Supported methods
        -----------------
        ``"gcv"``
            Classical Generalized Cross-Validation [1].
            ``V(α) = ||r||² / (n − tr(A))²``

        ``"mgcv"``
            Modified GCV – uses ``γ > 1`` in the denominator to bias toward
            more regularization when the forward model is rank-deficient.
            ``V(α) = ||r||² / (n − γ·tr(A))²``

        ``"rgcv"``
            Robust GCV [2] – penalises leverage concentration via ``μ₂``.
            ``V_RGCV(α) = [γ + (1−γ)·μ₂(α)] · V_GCV(α)``
            where ``μ₂ = tr(A²)/n``.

        ``"r1gcv"``
            Strong robust GCV [2] – penalises solution variance amplification.
            ``V_R1GCV(α) = [γ + (1−γ)·μ₁₂(α)] · V_GCV(α)``
            where ``μ₁₂ = [tr(A) − tr(A²)] / (n·α)``, which is ``−dμ₁/dα``
            for Tikhonov-type influence matrices.

        ``"composite"``
            Geometric mean of GCV and RGCV scores.
            ``V_comp(α) = sqrt(V_GCV · V_RGCV) = sqrt(γ+(1−γ)μ₂) · V_GCV``
            A milder version of RGCV that balances spatial precision (favoured
            by GCV) and noise robustness (favoured by RGCV).

        Parameters
        ----------
        M : numpy.ndarray
            The M/EEG data matrix (n_channels, n_timepoints).
        gamma : float | None
            Method-specific parameter.  For ``"gcv"``/``"mgcv"`` this is the
            denominator correction factor.  For ``"rgcv"``/``"r1gcv"`` this is
            the mixing parameter γ ∈ (0, 1).  If *None*, falls back to the
            instance attribute for the chosen method.
        method : str
            One of ``"gcv"``, ``"mgcv"``, ``"rgcv"``, ``"r1gcv"``.

        Returns
        -------
        source_mat : numpy.ndarray
            The inverse solution (n_dipoles, n_timepoints).
        optimum_idx : int
            Index of the selected regularization parameter.

        References
        ----------
        [1] Grech et al. (2008). Review on solving the inverse problem in EEG
            source analysis.  J Neuroeng Rehab, 5(1), 1–33.
        [2] Lukas, M. A. (2006/2010). Robust GCV choice of the regularization
            parameter for correlated data; Comparing parameter choice methods
            for regularization of ill-posed problems.
        """
        if not self.inverse_operators:
            msg = (
                "No inverse operators available for regularisation. "
                "Call make_inverse_operator() and ensure it populates "
                "self.inverse_operators before apply_inverse_operator()."
            )
            raise ValueError(msg)

        method = method.lower()
        if gamma is None:
            gamma = self._resolve_gcv_gamma(method)

        gcv_values = self.compute_gcv_scores(M, method=method, gamma=gamma)

        # Find optimal regularization parameter
        valid_indices = np.isfinite(gcv_values)
        if not np.any(valid_indices):
            optimum_idx = len(gcv_values) // 2
        else:
            valid_gcv = gcv_values[valid_indices]
            valid_idx_positions = np.where(valid_indices)[0]
            min_pos = np.argmin(valid_gcv)
            optimum_idx = valid_idx_positions[min_pos]

        if plot and len(self.alphas) == len(gcv_values):
            plt.figure()
            plt.semilogx(
                self.alphas, gcv_values, "o-", label=f"{method.upper()} values"
            )
            plt.semilogx(
                self.alphas[optimum_idx],
                gcv_values[optimum_idx],
                "r*",
                markersize=10,
                label=f"Optimal α = {self.alphas[optimum_idx]:.2e}",
            )
            plt.xlabel("Regularization parameter α")
            plt.ylabel(f"{method.upper()} value")
            plt.title(
                f"{method.upper()} (γ={gamma:g}): Optimal α = {self.alphas[optimum_idx]:.2e}"
            )
            plt.legend()
            plt.grid(True)

        self.log_regularisation_edge_choice(
            optimum_idx=optimum_idx,
            method=method.upper(),
        )
        source_mat = self.inverse_operators[optimum_idx].apply(M)  # type: ignore[attr-defined]
        return source_mat, optimum_idx

    def _resolve_gcv_gamma(self, method: str) -> float:
        """Return the default gamma parameter for the given GCV method."""
        if method == "mgcv":
            return self.mgcv_gamma
        elif method in {"rgcv", "composite"}:
            return self.rgcv_gamma
        elif method == "r1gcv":
            return self.r1gcv_gamma
        return self.gcv_gamma

    def compute_gcv_scores(
        self,
        M,
        method: str = "gcv",
        gamma: float | None = None,
    ) -> np.ndarray:
        """Compute GCV scores for each inverse operator without selecting one.

        This is the scoring component of ``regularise_gcv``, exposed separately
        so that the benchmark runner can aggregate scores across multiple data
        samples before choosing the optimal regularization level.

        Parameters
        ----------
        M : numpy.ndarray
            Data matrix (n_channels, n_timepoints).
        method : str
            GCV variant (``"gcv"``, ``"mgcv"``, ``"rgcv"``, ``"r1gcv"``,
            ``"composite"``).
        gamma : float | None
            Method-specific parameter.  If *None*, uses instance default.

        Returns
        -------
        numpy.ndarray
            Array of GCV scores, one per inverse operator.
        """
        method = method.lower()
        if gamma is None:
            gamma = self._resolve_gcv_gamma(method)

        # Trace: use ORIGINAL leadfield (correct by cyclic property of trace)
        _lo = getattr(self, "_leadfield_orig", None)
        L_orig = _lo if _lo is not None else self.leadfield
        # Residual: use whitened space if whitening was applied
        L_resid = self.leadfield  # G_white if whitened, else original
        n_eff = L_resid.shape[0]
        _st = getattr(self, "_sensor_transform", None)

        need_trace_A2 = method in {"rgcv", "r1gcv", "composite"}

        gcv_values = []
        for _i, inverse_operator in enumerate(self.inverse_operators):  # type: ignore[attr-defined]
            # Apply inverse operator to get source estimate
            x = inverse_operator.apply(M)

            # Trace with original G (correct by cyclic property of trace)
            W = inverse_operator.data[0]  # Get the actual matrix
            trace_H = float(np.sum(L_orig * W.T))

            # Residual in whitened space
            M_hat = L_resid @ x
            M_eff = _st @ M if _st is not None else M
            residual_ss = float(np.linalg.norm(np.asarray(M_eff - M_hat)) ** 2)

            # ------ plain / modified GCV score ------
            if method in {"gcv", "mgcv"}:
                effective_dof = n_eff - gamma * trace_H
                if effective_dof <= 0 or abs(effective_dof) < 1e-10:
                    gcv_values.append(np.inf)
                else:
                    gcv_values.append(residual_ss / (effective_dof**2))
                continue

            # ------ base GCV (γ=1) for RGCV / R1GCV / composite ------
            effective_dof = n_eff - trace_H
            if effective_dof <= 0 or abs(effective_dof) < 1e-10:
                gcv_values.append(np.inf)
                continue
            V_gcv = residual_ss / (effective_dof**2)

            if need_trace_A2:
                # A = L_orig @ W  (n_chans × n_chans — small matrix, e.g. 32×32)
                A = L_orig @ W
                trace_A2 = float(np.sum(A * A))  # ||A||_F² = tr(A²) since A symmetric

            if method == "rgcv":
                mu2 = trace_A2 / n_eff
                weight = gamma + (1 - gamma) * mu2
            elif method == "r1gcv":
                # μ₁₂ = [tr(A) − tr(A²)] / (n·α)
                alpha = self.alphas[_i]
                if alpha <= 0 or abs(alpha) < 1e-30:
                    gcv_values.append(np.inf)
                    continue
                mu12 = (trace_H - trace_A2) / (n_eff * alpha)
                weight = gamma + (1 - gamma) * mu12
            else:  # composite — geometric mean of GCV and RGCV
                mu2 = trace_A2 / n_eff
                rgcv_weight = gamma + (1 - gamma) * mu2
                weight = np.sqrt(max(rgcv_weight, 0.0))

            if weight <= 0:
                gcv_values.append(np.inf)
            else:
                gcv_values.append(weight * V_gcv)

        return np.array(gcv_values)

    def regularise_product(self, M, plot=False):
        """Find optimally regularized inverse solution using the product method [1].

        Parameters
        ----------
        M : numpy.ndarray
            The M/EEG data matrix (n_channels, n_timepoints)

        Return
        ------
        source_mat : numpy.ndarray
            The inverse solution  (dipoles x time points)
        optimum_idx : int
            The index of the selected (optimal) regularization parameter

        References
        ----------
        [1] Grech, R., Cassar, T., Muscat, J., Camilleri, K. P., Fabri, S. G.,
        Zervakis, M., ... & Vanrumste, B. (2008). Review on solving the inverse
        problem in EEG source analysis. Journal of neuroengineering and
        rehabilitation, 5(1), 1-33.

        """

        if not self.inverse_operators:
            msg = (
                "No inverse operators available for regularisation. "
                "Call make_inverse_operator() and ensure it populates "
                "self.inverse_operators before apply_inverse_operator()."
            )
            raise ValueError(msg)

        L_resid = self.leadfield  # G_white if whitened, else original
        _st = getattr(self, "_sensor_transform", None)

        product_values = []

        for inverse_operator in self.inverse_operators:
            x = inverse_operator.apply(M)

            M_hat = L_resid @ x
            M_eff = _st @ M if _st is not None else M
            residual_norm = np.linalg.norm(M_hat - M_eff)
            semi_norm = np.linalg.norm(x)
            product_value = semi_norm * residual_norm
            product_values.append(product_value)

        optimum_idx = np.argmin(product_values)

        if plot:
            plt.figure()
            plt.plot(self.alphas, product_values)
            plt.plot(self.alphas[optimum_idx], product_values[optimum_idx], "r*")
            alpha = self.alphas[optimum_idx]
            plt.title(f"Product: {alpha}")

        self.log_regularisation_edge_choice(
            optimum_idx=optimum_idx,
            method="Product",
        )
        source_mat = self.inverse_operators[optimum_idx].apply(M)
        return source_mat, optimum_idx

    @staticmethod
    def delete_from_list(a, idc):
        """Delete elements of list at idc."""

        idc = np.sort(idc)[::-1]
        for idx in idc:
            a.pop(idx)
        return a

    def find_corner(self, residual_norms, solution_norms, eps: float = 1e-30):
        """Find the corner of an L-curve in a numerically stable way.

        Uses the maximum-area triangle heuristic on the *log-log* curve of
        residual norm vs solution norm. Working in log space makes the method
        invariant to global scaling and matches the conventional L-curve
        definition used for Tikhonov-style regularization.

        Parameters
        ----------
        residual_norms : array-like
            Residual norms (e.g. ``||Gx - M||``) across the regularization grid.
        solution_norms : array-like
            Solution norms (e.g. ``||x||``) across the regularization grid.
        eps : float
            Floor applied before taking logs.

        Return
        ------
        idx : int
            Index at which the L-Curve has its corner.
        """

        x = np.asarray(residual_norms, dtype=float)
        y = np.asarray(solution_norms, dtype=float)

        if x.size < 3:
            return int(max(0, x.size - 1))

        # Log-log coordinates (L-curve convention).
        x = np.log10(np.maximum(x, eps))
        y = np.log10(np.maximum(y, eps))

        # Normalize both axes to keep the triangle metric well-scaled.
        x_span = float(np.max(x) - np.min(x))
        y_span = float(np.max(y) - np.min(y))
        if x_span > 0:
            x = (x - np.min(x)) / x_span
        else:
            x = x * 0.0
        if y_span > 0:
            y = (y - np.min(y)) / y_span
        else:
            y = y * 0.0

        A = np.array([x[0], y[0]])
        C = np.array([x[-1], y[-1]])
        areas = []
        for j in range(1, len(y) - 1):
            B = np.array([x[j], y[j]])
            AB = self.euclidean_distance(A, B)
            AC = self.euclidean_distance(A, C)
            CB = self.euclidean_distance(C, B)
            area = abs(self.calc_area_tri(AB, AC, CB))
            areas.append(area)
        if len(areas) > 0:
            idx = int(np.argmax(areas) + 1)
        else:
            idx = 0

        return idx

    @staticmethod
    def get_comps_L(D):
        """Estimate number of components using L-curve method.

        Parameters
        ----------
        D : numpy.ndarray
            Array of eigenvalues or singular values

        Return
        ------
        n_comp_L : int
            Number of components based on L-curve criterion
        """
        iters = np.arange(len(D))
        n_comp_L = find_corner(deepcopy(iters), deepcopy(D))
        return n_comp_L

    @staticmethod
    def get_comps_drop(D):
        """Estimate number of components using eigenvalue drop-off method.

        Parameters
        ----------
        D : numpy.ndarray
            Array of eigenvalues or singular values

        Return
        ------
        n_comp_drop : int
            Number of components based on eigenvalue drop-off criterion
        """
        D_ = D / D.max()
        n_comp_drop = np.where(abs(np.diff(D_)) < 0.001)[0]

        if len(n_comp_drop) > 0:
            n_comp_drop = n_comp_drop[0] + 1
        else:
            n_comp_drop = 1
        return n_comp_drop

    def estimate_n_sources(self, data, method="auto"):
        """Estimate the number of active sources from data.

        Parameters
        ----------
        data : numpy.ndarray
            The M/EEG data matrix (n_channels, n_timepoints)
        method : str
            Method for estimation. Options:
            - "auto": Combine multiple criteria (default)
            - "L": L-curve method
            - "drop": Eigenvalue drop-off
            - "mean": Eigenvalues above mean
            - "enhanced": Enhanced method

        Return
        ------
        n_sources : int
            Estimated number of active sources
        """
        # Compute data covariance
        C = data @ data.T

        # SVD decomposition
        _, D, _ = np.linalg.svd(C, full_matrices=False)

        if method == "L":
            # Based on L-Curve Criterion
            n_comp = self.get_comps_L(D)
        elif method == "drop":
            # Based on eigenvalue drop-off
            n_comp = self.get_comps_drop(D)
        elif method == "mean":
            # Based on mean eigenvalue
            n_comp = np.where(D < D.mean())[0]
            n_comp = n_comp[0] if len(n_comp) > 0 else len(D)
        elif method == "enhanced":
            n_comp = self.robust_estimate_n_sources(data)
        elif method == "auto":
            # Combine multiple criteria
            n_comp_L = self.get_comps_L(D)
            n_comp_drop = self.get_comps_drop(D)
            mean_idx = np.where(D < D.mean())[0]
            n_comp_mean = mean_idx[0] if len(mean_idx) > 0 else len(D)

            # Combine the criteria with slight bias
            n_comp = np.ceil(
                1.1 * np.mean([n_comp_L, n_comp_drop, n_comp_mean])
            ).astype(int)
        else:
            raise ValueError(
                f"Unknown method: {method}. Must be 'auto', 'L', 'drop', or 'mean'"
            )

        return n_comp

    # More robust approach for correlated sources:
    def robust_estimate_n_sources(self, data):
        C = data @ data.T
        _, S, _ = np.linalg.svd(C, full_matrices=False)
        S_norm = S / S.max()

        # 1. Stricter drop-off (better for correlated sources)
        diff_S = np.abs(np.diff(S_norm))
        n_drop = np.where(diff_S < 0.02)[0]  # Higher threshold
        n_drop = n_drop[0] + 1 if len(n_drop) > 0 else len(S)

        # 2. Cumulative energy (90% instead of 95%)
        cumsum_S = np.cumsum(S) / np.sum(S)
        n_energy = np.where(cumsum_S > 0.90)[0][0] + 1

        # 3. L-curve (limit to reasonable range)
        n_lcurve = self.get_comps_L(S[: min(15, len(S))])

        # Conservative combination (use 33rd percentile)
        estimates = [n_drop, n_lcurve, n_energy]
        n_sources = int(np.percentile(estimates, 33))

        return np.clip(n_sources, 1, len(S) // 2)

    def select_signal_subspace(
        self, data_matrix: np.ndarray, rank: str | int = "auto"
    ) -> np.ndarray:
        """Select the signal subspace of the data matrix.

        Parameters
        ----------
        data_matrix : np.ndarray
            The data matrix to select the signal subspace from.
        rank : str or int
            The rank to select the signal subspace from.

        Return
        ------
        data_matrix_approx : np.ndarray
            The low-rank approximation of the data matrix.
        """
        # Compute the SVD of the data matrix
        U, S, V = np.linalg.svd(data_matrix, full_matrices=False)

        if isinstance(rank, str):
            rank = self.estimate_n_sources(data_matrix, method=rank)
        elif isinstance(rank, int):
            rank = rank
        else:
            raise ValueError(
                f"Unknown rank: {rank}. Must be 'auto', 'L', 'drop', 'mean', 'enhanced', or some positive integer"
            )
        # Select the top `rank` singular values and corresponding singular vectors
        U_subset = U[:, :rank]  # type: ignore[misc, index]
        S_subset = S[:rank]  # type: ignore[misc]
        V_subset = V[:rank, :]  # type: ignore[misc, index]

        # Reconstruct a low-rank approximation of the data matrix using the selected singular values and vectors
        data_matrix_approx = U_subset @ np.diag(S_subset) @ V_subset

        return data_matrix_approx

    @staticmethod
    def filter_norms(r_vals, l2_norms):
        """Filter l2_norms where they are not monotonically decreasing.

        Parameters
        ----------
        r_vals : [list, numpy.ndarray]
            List or array of r-values
        l2_norms : [list, numpy.ndarray]
            List or array of l2_norms

        Return
        ------
        bad_idc : list
            List where indices are increasing

        """
        diffs = np.diff(l2_norms)
        bad_idc = []
        all_idc = np.arange(len(l2_norms))
        while np.any(diffs > 0):
            pop_idx = np.where(diffs > 0)[0][0] + 1
            r_vals = np.delete(r_vals, pop_idx)
            l2_norms = np.delete(l2_norms, pop_idx)
            diffs = np.diff(l2_norms)

            bad_idc.append(all_idc[pop_idx])
            all_idc = np.delete(all_idc, pop_idx)
        return bad_idc

    def prepare_forward(self):
        """Prepare leadfield for calculating inverse solutions by applying
        common average referencing and unit norm scaling.

        Parameters
        ----------


        Return
        ------
        """
        orientation = str(getattr(self, "orientation", "auto")).lower()
        if orientation not in {"auto", "fixed", "pca", "free"}:
            raise ValueError(
                "orientation must be one of {'auto','fixed','pca','free'}, "
                f"got {orientation!r}"
            )

        self._orientation_q = None
        solver_label = getattr(self, "name", self.__class__.__name__)

        src_types = [str(s.get("type", "")) for s in self.forward["src"]]
        has_volume_like = any(t in ("vol", "discrete") for t in src_types)
        is_surface = len(src_types) == 2 and all(t == "surf" for t in src_types)

        leadfield: np.ndarray | None = None

        if self.forward["source_ori"] == FIFF.FIFFV_MNE_FREE_ORI:
            if orientation in {"auto", "fixed"}:
                if has_volume_like:
                    if orientation == "fixed":
                        raise ValueError(
                            "orientation='fixed' is only supported for surface source spaces. "
                            "Use orientation='pca' (data-driven) or orientation='auto'."
                        )
                    # Volume/discrete source space: free orientation not meaningful
                    # for most solvers.  Check whether this solver can handle it.
                    supports_vector = bool(
                        getattr(self, "SUPPORTS_VECTOR_ORIENTATION", False)
                    )
                    if supports_vector:
                        # Solver handles free orientation natively.
                        self._free_orientation = True
                        self._n_orient = 3
                    elif self._orientation_data_obj is not None:
                        # Auto PCA reduction: project 3-component leadfield to
                        # scalar using data-driven orientation estimates.
                        logger.info(
                            "%s orientation=auto + volume source: "
                            "auto PCA reduction (free→scalar).",
                            solver_label,
                        )
                        Y = self.unpack_data_obj(self._orientation_data_obj)
                        kept_ch_names = list(
                            getattr(self, "_last_data_ch_names", ()) or []
                        )
                        if kept_ch_names:
                            self.forward = self.forward.pick_channels(
                                kept_ch_names, ordered=True
                            )
                        G_free = np.asarray(
                            self.forward["sol"]["data"], dtype=float
                        )
                        q = estimate_orientation_pca(
                            G_free,
                            Y,
                            reg=float(self.orientation_pca_reg),
                            deterministic_sign=bool(
                                self.orientation_pca_deterministic_sign
                            ),
                        )
                        self._orientation_q = q
                        leadfield = reduce_free_to_scalar(G_free, q)
                        self._free_orientation = False
                        self._n_orient = 1
                    else:
                        # No data available — keep free and hope for the best.
                        import warnings

                        warnings.warn(
                            f"{solver_label}: volume source space with "
                            "orientation='auto' but no data for PCA reduction. "
                            "Keeping free orientation; solver may fail.",
                            stacklevel=2,
                        )
                        self._free_orientation = True
                        self._n_orient = 3
                else:
                    # Surface source space: convert to fixed (cortical normal).
                    self.forward = mne.convert_forward_solution(
                        self.forward,
                        surf_ori=True,
                        force_fixed=True,
                        use_cps=True,
                        verbose=0,
                    )
                    self._free_orientation = False
                    self._n_orient = 1

            elif orientation == "pca":
                if self._orientation_data_obj is None:
                    raise ValueError(
                        "orientation='pca' requires passing an MNE data object to "
                        "make_inverse_operator(forward, mne_obj, ...)."
                    )
                logger.info(
                    "%s orientation: using PCA reduction (free->scalar).", solver_label
                )
                # Extract data using the same preprocessing/picking pipeline.
                Y = self.unpack_data_obj(self._orientation_data_obj)
                kept_ch_names = list(getattr(self, "_last_data_ch_names", ()) or [])
                if kept_ch_names:
                    # Pick forward channels to match the PCA data channels.
                    self.forward = self.forward.pick_channels(
                        kept_ch_names, ordered=True
                    )

                G_free = np.asarray(self.forward["sol"]["data"], dtype=float)
                q = estimate_orientation_pca(
                    G_free,
                    Y,
                    reg=float(self.orientation_pca_reg),
                    deterministic_sign=bool(self.orientation_pca_deterministic_sign),
                )
                self._orientation_q = q
                leadfield = reduce_free_to_scalar(G_free, q)
                self._free_orientation = False
                self._n_orient = 1

            else:  # orientation == "free"
                supports_vector = bool(
                    getattr(self, "SUPPORTS_VECTOR_ORIENTATION", False)
                )
                if not supports_vector:
                    # Surface forwards can be converted to a fixed (cortical-normal)
                    # orientation; use that as a pragmatic fallback so solvers that
                    # require scalar leadfields (some beamformers) can still run.
                    # Volume/discrete forwards have no such canonical fixed-orientation.
                    if not (is_surface and bool(getattr(self, "_is_beamformer", False))):
                        solver_label = getattr(self, "name", self.__class__.__name__)
                        supported = ["auto", "fixed", "pca"]
                        raise ValueError(
                            f"{solver_label} does not support orientation='free'. "
                            f"Supported: {supported}."
                        )
                    logger.info(
                        "%s orientation='free' requested but solver does not support "
                        "vector leadfields; using surface fixed orientation instead.",
                        solver_label,
                    )
                    self.forward = mne.convert_forward_solution(
                        self.forward,
                        surf_ori=True,
                        force_fixed=True,
                        use_cps=True,
                        verbose=0,
                    )
                    self._free_orientation = False
                    self._n_orient = 1
                    # leadfield will be pulled from forward["sol"]["data"] below.
                else:
                    if is_surface:
                        self.forward = ensure_surface_free_surf_ori(self.forward)
                    self._free_orientation = True
                    self._n_orient = 3
                    logger.info(
                        "%s orientation: using true free orientation (3-component).",
                        solver_label,
                    )
        else:
            if orientation == "free":
                raise ValueError(
                    "orientation='free' requires a forward model with source_ori=FREE."
                )
            self._free_orientation = False
            self._n_orient = 1

        if leadfield is None:
            leadfield = np.asarray(self.forward["sol"]["data"], dtype=float)

        # Always copy: several solvers modify leadfield in-place.
        self.leadfield = np.array(leadfield, dtype=float, copy=True)

        if self.common_average_reference:
            n = self.leadfield.shape[0]
            H = np.eye(n) - np.ones((n, n)) / n
            self.leadfield = H @ self.leadfield

        # Depth weighting is now opt-in per solver instead of globally applied.
        if self.prep_leadfield and self.use_depth_weighting:
            self.leadfield, self.depth_weights = self.depth_weight_fixed(
                self.leadfield, degree=self.depth_weighting
            )
        else:
            self.depth_weights = None

        # Final orientation summary at INFO level.
        eff = "free" if getattr(self, "_free_orientation", False) else "scalar"
        logger.info(
            "%s orientation: requested=%s effective=%s n_orient=%d src_types=%s",
            solver_label,
            orientation,
            eff,
            int(getattr(self, "_n_orient", 1)),
            src_types,
        )

    @staticmethod
    def depth_weight_fixed(L, degree=0.8, ref="median", eps=1e-12):
        """
        Depth-weight a fixed-orientation leadfield L (m,n) with strength in [0,1].
        Returns (L_dw, w) where L_dw = L * w and w are per-column weights.
        """
        norms = np.linalg.norm(L, axis=0) ** degree
        return L / norms, norms

    def apply_depth_weighting_to_leadfield(
        self, leadfield: np.ndarray | None = None, degree: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a depth-weighted copy of the leadfield and the weights."""
        if leadfield is None:
            leadfield = self.leadfield
        if degree is None:
            degree = self.depth_weighting
        return self.depth_weight_fixed(leadfield, degree=degree)

    @staticmethod
    def euclidean_distance(A, B):
        """Euclidean Distance between two points (A -> B)."""
        return np.sqrt(np.sum((A - B) ** 2))

    @staticmethod
    def calc_area_tri(AB, AC, CB):
        """Calculates area of a triangle given the length of each side."""
        s = (AB + AC + CB) / 2
        area = (s * (s - AB) * (s - AC) * (s - CB)) ** 0.5
        return area

    def robust_inverse(self, matrix, cond_threshold=1e12, regularization=1e-12):
        """Robustly invert a matrix, handling singular/ill-conditioned cases.

        Parameters
        ----------
        matrix : numpy.ndarray
            The matrix to invert
        cond_threshold : float
            Condition number threshold above which regularization is applied
        regularization : float
            Regularization parameter for ill-conditioned matrices

        Return
        ------
        matrix_inv : numpy.ndarray
            The inverted matrix
        """
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            return np.zeros_like(matrix.T)

        # Check condition number for numerical stability
        cond_num = np.linalg.cond(matrix)

        if cond_num > cond_threshold:
            # Use regularized inversion for ill-conditioned matrices
            reg_matrix = regularization * np.eye(matrix.shape[0])
            matrix_inv = np.linalg.inv(matrix + reg_matrix)
        else:
            # Use standard inversion for well-conditioned matrices
            try:
                matrix_inv = np.linalg.inv(matrix)
            except np.linalg.LinAlgError:
                # Fall back to regularized inversion if standard inversion fails
                reg_matrix = regularization * np.eye(matrix.shape[0])
                matrix_inv = np.linalg.inv(matrix + reg_matrix)

        return matrix_inv

    def robust_inverse_solution(
        self, leadfield, data, cond_threshold=1e12, regularization=1e-12
    ):
        """Compute numerically stable pseudoinverse for leadfield matrices.

        Parameters
        ----------
        leadfield : numpy.ndarray
            The leadfield matrix to invert (n_channels, n_sources)
        data : numpy.ndarray
            The data to project (n_channels,) or (n_channels, n_timepoints)
        cond_threshold : float
            Condition number threshold above which regularization is applied
        regularization : float
            Regularization parameter for ill-conditioned matrices

        Return
        ------
        result : numpy.ndarray
            The result of the robust inverse operation
        """
        if leadfield.shape[1] == 0:
            # Handle empty leadfield case
            if data.ndim == 1:
                return np.zeros(0)
            else:
                return np.zeros((0, data.shape[1]))

        # Check condition number for numerical stability
        cond_num = np.linalg.cond(leadfield)

        if cond_num > cond_threshold:
            # Use least-squares for ill-conditioned matrices
            result = np.linalg.lstsq(leadfield, data, rcond=None)[0]
        else:
            # Use pseudoinverse for well-conditioned matrices
            result = np.linalg.pinv(leadfield) @ data

        return result

    def robust_residual(
        self, data, leadfield, cond_threshold=1e12, regularization=1e-12
    ):
        """Compute residual with numerically stable projection.

        Parameters
        ----------
        data : numpy.ndarray
            The data vector/matrix
        leadfield : numpy.ndarray
            The leadfield matrix for projection
        cond_threshold : float
            Condition number threshold above which regularization is applied
        regularization : float
            Regularization parameter for ill-conditioned matrices

        Return
        ------
        residual : numpy.ndarray
            The residual after projection
        """
        if leadfield.shape[1] == 0:
            return data.copy()

        # Check condition number for numerical stability
        cond_num = np.linalg.cond(leadfield)

        if cond_num > cond_threshold:
            # Use least-squares projection for ill-conditioned matrices
            proj_coeff = np.linalg.lstsq(leadfield, data, rcond=None)[0]
            return data - leadfield @ proj_coeff
        else:
            # Use pseudoinverse projection for well-conditioned matrices
            return data - leadfield @ np.linalg.pinv(leadfield) @ data

    def robust_normalize_leadfield(self, leadfield):
        """Robustly normalize leadfield columns to unit norm.

        Parameters
        ----------
        leadfield : numpy.ndarray
            The leadfield matrix to normalize (n_channels, n_sources)

        Return
        ------
        leadfield_normed : numpy.ndarray
            The column-normalized leadfield matrix
        """
        # Calculate column norms
        leadfield_norms = np.linalg.norm(leadfield, axis=0)

        # Avoid division by zero for columns with zero norm
        leadfield_norms[leadfield_norms == 0] = 1

        # Create normalized leadfield for atom selection
        leadfield_normed = leadfield / leadfield_norms

        return leadfield_normed

    def compute_matrix_power_robust(self, matrix, power):
        """Compute matrix power using eigenvalue decomposition for numerical stability."""
        try:
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            # Avoid negative eigenvalues that could cause issues
            eigenvals = np.maximum(eigenvals, 1e-12)
            eigenvals_power = eigenvals**power
            return eigenvecs @ np.diag(eigenvals_power) @ eigenvecs.T
        except np.linalg.LinAlgError:
            # Fallback to regularized approach
            reg = 1e-12 * np.eye(matrix.shape[0])
            return np.linalg.matrix_power(matrix + reg, power)

    def source_to_object(
        self, source_mat: npt.NDArray[np.floating]
    ) -> mne.SourceEstimate:
        """Converts the source_mat matrix to the mne.SourceEstimate object.

        Parameters
        ----------
        source_mat : numpy.ndarray
            Source matrix (dipoles, time points)-

        Return
        ------
        stc : mne.SourceEstimate

        """
        # Undo depth weighting only for solvers that explicitly use it.
        if (
            self.prep_leadfield
            and self.use_depth_weighting
            and hasattr(self, "depth_weights")
            and self.depth_weights is not None
        ):
            source_mat = (
                source_mat / self.depth_weights[:, np.newaxis]
                if source_mat.ndim > 1
                else source_mat / self.depth_weights
            )

        # Combine free-orientation components (3 per source -> 1 via vector norm).
        # Expected row ordering: [x1, y1, z1, x2, y2, z2, ...].
        if getattr(self, "_free_orientation", False):
            n_orient = getattr(self, "_n_orient", 3)
            if source_mat.shape[0] % n_orient != 0:
                raise ValueError(
                    f"Free-orientation source_mat has {source_mat.shape[0]} rows "
                    f"which is not divisible by n_orient={n_orient}. "
                    f"Expected rows = n_locations * {n_orient}."
                )
            n_src = source_mat.shape[0] // n_orient
            if source_mat.ndim == 2:
                source_mat = np.linalg.norm(
                    source_mat.reshape(n_src, n_orient, -1), axis=1
                )
            else:
                source_mat = np.linalg.norm(source_mat.reshape(n_src, n_orient), axis=1)

        # Convert source to an MNE STC object matching the source-space type.
        source_model = self.forward["src"]
        vertices = [np.asarray(s["vertno"]) for s in source_model]
        src_types = [str(s.get("type", "")) for s in source_model]
        tmin = self.tmin
        sfreq = self.obj_info["sfreq"]
        tstep = 1 / sfreq
        subject = self.obj_info["subject_info"]

        if isinstance(subject, dict) and "his_id" in subject:
            subject = subject["his_id"]
        # else assume fsaverage as subject id
        else:
            subject = "fsaverage"

        if len(vertices) == 2 and all(src_type == "surf" for src_type in src_types):
            stc = mne.SourceEstimate(
                source_mat,
                vertices,
                tmin=tmin,
                tstep=tstep,
                subject=subject,
                verbose=self.verbose,
            )
        elif len(vertices) == 1 and src_types[0] in {"vol", "discrete"}:
            stc = mne.VolSourceEstimate(
                source_mat,
                vertices,
                tmin=tmin,
                tstep=tstep,
                subject=subject,
                verbose=self.verbose,
            )
        else:
            stc = mne.MixedSourceEstimate(
                source_mat,
                vertices,
                tmin=tmin,
                tstep=tstep,
                subject=subject,
                verbose=self.verbose,
            )
        return stc

    def source_to_object_vector(
        self, source_mat: npt.NDArray[np.floating]
    ) -> (
        mne.VectorSourceEstimate
        | mne.VolVectorSourceEstimate
        | mne.MixedVectorSourceEstimate
    ):
        """Convert a (3*n_sources, n_times) matrix into a VectorSourceEstimate."""

        # Undo depth weighting only for solvers that explicitly use it.
        if (
            self.prep_leadfield
            and self.use_depth_weighting
            and hasattr(self, "depth_weights")
            and self.depth_weights is not None
        ):
            source_mat = source_mat / self.depth_weights[:, np.newaxis]

        source_mat = np.asarray(source_mat, dtype=float)
        if source_mat.ndim == 1:
            source_mat = source_mat[:, np.newaxis]

        n_orient = int(getattr(self, "_n_orient", 3))
        if n_orient != 3:
            raise ValueError(
                f"source_to_object_vector requires n_orient==3, got {n_orient}"
            )
        if source_mat.shape[0] % n_orient != 0:
            raise ValueError(
                f"source_mat has {source_mat.shape[0]} rows which is not divisible by n_orient={n_orient}"
            )

        n_src = int(source_mat.shape[0] // n_orient)
        vec = source_mat.reshape(n_src, n_orient, -1)

        source_model = self.forward["src"]
        vertices = [np.asarray(s["vertno"]) for s in source_model]
        src_types = [str(s.get("type", "")) for s in source_model]
        tmin = self.tmin
        sfreq = self.obj_info["sfreq"]
        tstep = 1 / sfreq
        subject = self.obj_info["subject_info"]

        if isinstance(subject, dict) and "his_id" in subject:
            subject = subject["his_id"]
        else:
            subject = "fsaverage"

        if len(vertices) == 2 and all(src_type == "surf" for src_type in src_types):
            return mne.VectorSourceEstimate(
                vec,
                vertices,
                tmin=tmin,
                tstep=tstep,
                subject=subject,
                verbose=self.verbose,
            )
        if len(vertices) == 1 and src_types[0] in {"vol", "discrete"}:
            return mne.VolVectorSourceEstimate(
                vec,
                vertices,
                tmin=tmin,
                tstep=tstep,
                subject=subject,
                verbose=self.verbose,
            )
        return mne.MixedVectorSourceEstimate(
            vec,
            vertices,
            tmin=tmin,
            tstep=tstep,
            subject=subject,
            verbose=self.verbose,
        )

    def _fix_class_identity(self):
        """Fix class identity issues that can occur with module reloading (e.g., Jupyter autoreload).

        This ensures that self.__class__ points to the currently imported class,
        preventing PicklingError: "it's not the same object as..."
        """
        import sys

        # Get the module and class name
        module_name = self.__class__.__module__
        class_name = self.__class__.__name__

        # Check if the module is already loaded
        if module_name in sys.modules:
            # Reload the module reference (don't actually reload, just get current reference)
            try:
                module = sys.modules[module_name]
                # Get the current class from the module
                current_class = getattr(module, class_name, None)
                if current_class is not None and current_class is not self.__class__:
                    # Update the instance's class to the current module's class
                    self.__class__ = current_class
            except (AttributeError, ImportError):
                # If we can't fix it, let pickle try anyway
                pass

    def save(self, path: str) -> BaseSolver:
        """Saves the solver object.

        Paramters
        ---------
        path : str
            The path to save the solver.
        Return
        ------
        self : BaseSolver
            Function returns itself.
        """

        name = self.name  # type: ignore[attr-defined]

        # get list of folders in path
        if os.path.exists(path):
            list_of_folders = os.listdir(path)
        else:
            # create path
            os.makedirs(path, exist_ok=True)
            list_of_folders = []

        model_ints = []
        for folder in list_of_folders:
            full_path = os.path.join(path, folder)
            if not os.path.isdir(full_path):
                continue
            if folder.startswith(name):
                new_integer = int(folder.split("_")[-1])
                model_ints.append(new_integer)
        if len(model_ints) == 0:
            model_name = f"{name}_0"
        else:
            model_name = f"{name}_{max(model_ints) + 1}"
        new_path = os.path.join(path, model_name)
        os.makedirs(new_path, exist_ok=True)

        # Fix class identity before pickling to prevent autoreload issues
        self._fix_class_identity()

        # Clear generator before pickling (can't pickle local functions)
        if hasattr(self, "generator"):
            self.generator = None

        if hasattr(self, "model"):
            # Save model inside the directory as model.keras
            model_path = os.path.join(new_path, "model.keras")
            self.model.save(model_path)  # type: ignore[attr-defined, has-type]

            # Save rest
            # Delete model since it is not serializable
            self.model = None
            if hasattr(self, "history"):
                self.history = None

            with open(os.path.join(new_path, "instance.pkl"), "wb") as f:
                pkl.dump(self, f, protocol=pkl.HIGHEST_PROTOCOL)

            # Attach model again now that everything is saved
            # Build custom_objects dict for loading
            custom_objects = {}
            if hasattr(self, "loss") and self.loss is not None:
                custom_objects["loss"] = self.loss

            # Try to import and register CustomConv2D if it exists
            try:
                from .neural_networks.utils import (  # type: ignore[attr-defined]
                    CustomConv2D,
                )

                custom_objects["CustomConv2D"] = CustomConv2D
            except (ImportError, AttributeError):
                pass

            try:
                import tensorflow as tf

                self.model = tf.keras.models.load_model(
                    model_path, custom_objects=custom_objects
                )
            except Exception as e:
                logger.warning(f"Load model with custom_objects failed: {e}")
                logger.info("Trying to load model without custom_objects...")
                try:
                    import tensorflow as tf

                    self.model = tf.keras.models.load_model(model_path)
                except Exception as e2:
                    logger.error(f"Load model without custom_objects also failed: {e2}")
                    logger.warning(
                        "Model will need to be loaded manually after unpickling."
                    )
        else:
            with open(os.path.join(new_path, "instance.pkl"), "wb") as f:
                pkl.dump(self, f)
        return self
