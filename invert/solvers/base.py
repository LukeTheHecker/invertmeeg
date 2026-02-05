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


class InverseOperator:
    """Container for precomputed inverse operators.

    Parameters
    ----------
    inverse_operator : numpy.ndarray or list of numpy.ndarray
        The inverse operator matrix (or matrices).
    solver_name : str
        Name of the solver that produced this operator.
    """

    def __init__(self, inverse_operator: Any, solver_name: str) -> None:
        self.solver_name = solver_name
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

    """

    meta: SolverMeta | None = None

    def __init__(
        self,
        regularisation_method: str = "GCV",
        n_reg_params: int = 10,
        prep_leadfield: bool = True,
        use_last_alpha: bool = False,
        rank: str | int = "enhanced",
        depth_weighting: float = 0.5,
        reduce_rank: bool = False,
        plot_reg: bool = False,
        common_average_reference: bool = False,
        gcv_gamma: float = 1.0,
        mgcv_gamma: float = 1.05,
        verbose: int = 0,
        **kwargs: Any,
    ) -> None:
        self.verbose = verbose

        self.r_values = np.logspace(-10, 1, n_reg_params)

        self.n_reg_params = n_reg_params
        self.regularisation_method = regularisation_method
        self.prep_leadfield = prep_leadfield
        self.use_last_alpha = use_last_alpha
        self.last_reg_idx = None
        self.rank = rank
        self.depth_weighting = depth_weighting
        self.reduce_rank = reduce_rank
        self.plot_reg = plot_reg
        self.made_inverse_operator = False
        self.common_average_reference = common_average_reference
        self.gcv_gamma = gcv_gamma
        self.mgcv_gamma = mgcv_gamma
        self.require_recompute = True
        self.require_data = True

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
        self.forward = deepcopy(forward)
        self.prepare_forward()
        self.alpha = alpha
        self.alphas = self.get_alphas(reference=reference)
        self.made_inverse_operator = True

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

        if self.use_last_alpha and self.last_reg_idx is not None:
            source_mat = self.inverse_operators[self.last_reg_idx].apply(data)

        else:
            if self.regularisation_method.lower() == "l":
                source_mat, idx = self.regularise_lcurve(data, plot=self.plot_reg)
                self.last_reg_idx = idx
            elif self.regularisation_method.lower() in {"gcv", "mgcv"}:
                gamma = (
                    self.gcv_gamma
                    if self.regularisation_method.lower() == "gcv"
                    else self.mgcv_gamma
                )
                source_mat, idx = self.regularise_gcv(
                    data, plot=self.plot_reg, gamma=gamma
                )
                self.last_reg_idx = idx
            elif self.regularisation_method.lower() == "product":
                source_mat, idx = self.regularise_product(data, plot=self.plot_reg)
                self.last_reg_idx = idx
            else:
                msg = f"{self.regularisation_method} is no valid regularisation method."
                raise AttributeError(msg)

        stc = self.source_to_object(source_mat)
        return stc

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
        mne_obj_meeg = mne_obj.copy().pick(pick_types)

        channels_in_fwd = self.forward.ch_names
        channels_in_mne_obj = mne_obj_meeg.ch_names
        picks = self.select_list_intersection(channels_in_fwd, channels_in_mne_obj)

        # Select only data channels in mne_obj
        mne_obj_meeg.pick(picks)

        # Store original forward model for later
        self.forward_original = deepcopy(self.forward)

        # Select only available data channels in forward
        self.forward = self.forward.pick_channels(picks)

        # Prepare the potentially new forward model
        self.prepare_forward()

        # Test if ch_names in forward model and mne_obj_meeg are equal
        assert self.forward.ch_names == mne_obj_meeg.ch_names, (
            "channels available in mne object are not equal to those present in the forward model."
        )
        assert len(self.forward.ch_names) > 1, (
            "forward model contains only a single channel"
        )

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

        # Restore the original forward model and leadfield so they match
        # what the inverse operators were built with.
        self.forward = self.forward_original
        self.prepare_forward()

        return data

    @staticmethod
    def select_list_intersection(list1, list2):
        new_list = []
        for element in list1:
            if element in list2:
                new_list.append(element)
        return new_list

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
    def data_covariance(Y: npt.NDArray[np.floating], *, center: bool = True, ddof: int = 1):
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

        leadfield = self.leadfield
        source_mats = [
            inverse_operator.apply(M) for inverse_operator in self.inverse_operators
        ]

        l2_norms = [np.linalg.norm(source_mat) for source_mat in source_mats]

        residual_norms = [
            np.linalg.norm(leadfield @ source_mat - M) for source_mat in source_mats
        ]

        optimum_idx = self.find_corner(l2_norms, residual_norms)

        source_mat = source_mats[optimum_idx]
        if plot:
            plt.figure()
            plt.plot(residual_norms, l2_norms, "ok")
            plt.plot(residual_norms[optimum_idx], l2_norms[optimum_idx], "r*")
            alpha = self.alphas[optimum_idx]
            plt.title(f"L-Curve: {alpha}")

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

    def regularise_gcv(self, M, plot: bool = False, gamma: float | None = None):
        """Find optimally regularized inverse solution using the generalized
        cross-validation method [1].

        The GCV criterion is defined as:
        GCV(α) = ||M - H_α M||² / (n - γ·trace(H_α))²

        Where H_α is the hat matrix (influence matrix) for regularization parameter α.
        For Tikhonov regularized minimum norm estimation:
        H_α = L @ (L^T @ L + α*I)^(-1) @ L^T

        Setting γ>1 (Modified GCV / "MGCV") biases selection toward slightly
        larger α, which can be helpful when the forward model is rank-deficient
        (e.g., due to common-average referencing) and plain GCV tends to
        under-regularize.

        Parameters
        ----------
        M : numpy.ndarray
            The M/EEG data matrix (n_channels, n_timepoints)
        gamma : float | None
            GCV correction factor γ. If None, uses `self.gcv_gamma`.

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
        n_chans = self.leadfield.shape[0]
        if gamma is None:
            gamma = self.gcv_gamma
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma}")

        gcv_values = []
        for _i, inverse_operator in enumerate(self.inverse_operators):  # type: ignore[attr-defined]
            # Apply inverse operator to get source estimate
            x = inverse_operator.apply(M)

            # Calculate reconstructed data
            M_hat = self.leadfield @ x

            W = inverse_operator.data[0]  # Get the actual matrix
            # trace(H) = trace(L @ W) = sum(L * W.T), avoid forming H explicitly
            trace_H = float(np.sum(self.leadfield * W.T))

            # Calculate residual sum of squares
            residual = M - M_hat
            # Default norm is Euclidean (1D) / Frobenius (2D), which is what we want here.
            residual_ss = float(np.linalg.norm(np.asarray(residual)) ** 2)

            # Calculate effective degrees of freedom
            effective_dof = n_chans - gamma * trace_H

            # Calculate GCV value
            # Handle case where effective_dof is near zero to avoid division issues
            if effective_dof <= 0 or abs(effective_dof) < 1e-10:
                gcv_value = np.inf
            else:
                gcv_value = residual_ss / (effective_dof**2)

            gcv_values.append(gcv_value)

        # Find optimal regularization parameter
        gcv_values = np.array(gcv_values)  # type: ignore[assignment]

        # Filter out invalid values (inf, nan)
        valid_indices = np.isfinite(gcv_values)
        if not np.any(valid_indices):
            # If all values are invalid, use the middle index
            optimum_idx = len(gcv_values) // 2
        else:
            # Find minimum among valid values
            valid_gcv = gcv_values[valid_indices]
            valid_idx_positions = np.where(valid_indices)[0]
            min_pos = np.argmin(valid_gcv)
            optimum_idx = valid_idx_positions[min_pos]

        if plot and len(self.alphas) == len(gcv_values):
            plt.figure()
            plt.semilogx(self.alphas, gcv_values, "o-", label="GCV values")
            plt.semilogx(
                self.alphas[optimum_idx],
                gcv_values[optimum_idx],
                "r*",
                markersize=10,
                label=f"Optimal α = {self.alphas[optimum_idx]:.2e}",
            )
            plt.xlabel("Regularization parameter α")
            plt.ylabel("GCV value")
            title = f"GCV: Optimal α = {self.alphas[optimum_idx]:.2e}"
            if gamma != 1.0:
                title = f"Modified GCV (γ={gamma:g}): Optimal α = {self.alphas[optimum_idx]:.2e}"
            plt.title(title)
            plt.legend()
            plt.grid(True)

        source_mat = self.inverse_operators[optimum_idx].apply(M)  # type: ignore[attr-defined]
        return source_mat, optimum_idx

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

        product_values = []

        for inverse_operator in self.inverse_operators:
            x = inverse_operator.apply(M)

            M_hat = self.leadfield @ x
            residual_norm = np.linalg.norm(M_hat - M)
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

        source_mat = self.inverse_operators[optimum_idx].apply(M)
        return source_mat, optimum_idx

    @staticmethod
    def delete_from_list(a, idc):
        """Delete elements of list at idc."""

        idc = np.sort(idc)[::-1]
        for idx in idc:
            a.pop(idx)
        return a

    def find_corner(self, r_vals, l2_norms):
        """Find the corner of the l-curve given by plotting regularization
        levels (r_vals) against norms of the inverse solutions (l2_norms).

        Parameters
        ----------
        r_vals : list
            Levels of regularization
        l2_norms : list
            L2 norms of the inverse solutions per level of regularization.

        Return
        ------
        idx : int
            Index at which the L-Curve has its corner.
        """

        # Normalize l2 norms
        l2_norms /= np.max(l2_norms)

        A = np.array([r_vals[0], l2_norms[0]])
        C = np.array([r_vals[-1], l2_norms[-1]])
        areas = []
        for j in range(1, len(l2_norms) - 1):
            B = np.array([r_vals[j], l2_norms[j]])
            AB = self.euclidean_distance(A, B)
            AC = self.euclidean_distance(A, C)
            CB = self.euclidean_distance(C, B)
            area = abs(self.calc_area_tri(AB, AC, CB))
            areas.append(area)
        if len(areas) > 0:
            idx = np.argmax(areas) + 1
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
        # Check whether forward model has free source orientation
        # if yes -> convert to fixed
        if self.forward["source_ori"] == FIFF.FIFFV_MNE_FREE_ORI:
            logger.warning(
                "Forward model has free source orientation. This is currently not possible, converting to fixed."
            )
            # convert to fixed
            self.forward = mne.convert_forward_solution(
                self.forward, force_fixed=True, verbose=0
            )

        self.leadfield = deepcopy(self.forward["sol"]["data"])

        if self.common_average_reference:
            n = self.leadfield.shape[0]
            H = np.eye(n) - np.ones((n, n)) / n
            self.leadfield = H @ self.leadfield

        if self.prep_leadfield:
            self.leadfield, self.depth_weights = self.depth_weight_fixed(
                self.leadfield, degree=self.depth_weighting
            )

    @staticmethod
    def depth_weight_fixed(L, degree=0.8, ref="median", eps=1e-12):
        """
        Depth-weight a fixed-orientation leadfield L (m,n) with strength in [0,1].
        Returns (L_dw, w) where L_dw = L * w and w are per-column weights.
        """
        norms = np.linalg.norm(L, axis=0) ** degree
        return L / norms, norms

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
            # Use regularized least squares for ill-conditioned matrices
            reg_matrix = regularization * np.eye(leadfield.shape[1])
            result = np.linalg.solve(
                leadfield.T @ leadfield + reg_matrix, leadfield.T @ data
            )
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
            # Use regularized projection for ill-conditioned matrices
            reg_matrix = regularization * np.eye(leadfield.shape[1])
            proj_coeff = np.linalg.solve(
                leadfield.T @ leadfield + reg_matrix, leadfield.T @ data
            )
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
        # Undo depth weighting to recover physical amplitudes
        if self.prep_leadfield and hasattr(self, "depth_weights"):
            source_mat = (
                source_mat / self.depth_weights[:, np.newaxis]
                if source_mat.ndim > 1
                else source_mat / self.depth_weights
            )

        # Convert source to mne.SourceEstimate object
        source_model = self.forward["src"]
        vertices = [source_model[0]["vertno"], source_model[1]["vertno"]]
        tmin = self.tmin
        sfreq = self.obj_info["sfreq"]
        tstep = 1 / sfreq
        subject = self.obj_info["subject_info"]

        if isinstance(subject, dict) and "his_id" in subject:
            subject = subject["his_id"]
        # else assume fsaverage as subject id
        else:
            subject = "fsaverage"

        stc = mne.SourceEstimate(
            source_mat,
            vertices,
            tmin=tmin,
            tstep=tstep,
            subject=subject,
            verbose=self.verbose,
        )
        return stc

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
                from .neural_networks.utils import CustomConv2D

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
