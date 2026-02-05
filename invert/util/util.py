import logging
import os
import pickle as pkl

import matplotlib.pyplot as plt
import mne
import numpy as np

logger = logging.getLogger(__name__)


def read_solver(path, name="instance", custom_objects=None):
    import tensorflow as tf  # noqa: F811 – lazy import, only needed for NN solvers

    instance_path = os.path.join(path, name + ".pkl")
    logger.info("loading: %s", instance_path)
    with open(instance_path, "rb") as f:
        solver = pkl.load(f)

    # Check if neural network model is in folder:
    folder_ls = os.listdir(path)

    # Check for Keras 3 format (.keras file)
    if "model.keras" in folder_ls:
        model_path = os.path.join(path, "model.keras")

        # Build custom_objects dict with default custom layers/losses
        if custom_objects is None:
            custom_objects = {}
        else:
            custom_objects = custom_objects.copy()  # Don't modify the input dict

        # Add CustomConv2D if not already present
        if "CustomConv2D" not in custom_objects:
            try:
                from ..solvers.esinet import CustomConv2D

                custom_objects["CustomConv2D"] = CustomConv2D
            except (ImportError, AttributeError):
                pass

        # Add loss function if the solver has one and it's not in custom_objects
        if (
            hasattr(solver, "loss")
            and solver.loss is not None
            and "loss" not in custom_objects
        ):
            custom_objects["loss"] = solver.loss

        try:
            model = tf.keras.models.load_model(
                model_path, custom_objects=custom_objects
            )
        except Exception as e:
            logger.warning(f"Load model with custom_objects failed: {e}")
            logger.info("Trying without custom_objects...")
            try:
                model = tf.keras.models.load_model(model_path)
            except Exception as e2:
                logger.error(f"Load model without custom_objects also failed: {e2}")
                logger.warning(
                    "Model could not be loaded. You may need to provide custom_objects manually."
                )
                model = None

        if model is not None:
            solver.model = model
    # Check for old SavedModel format (Keras 2)
    elif "keras_metadata.pb" in folder_ls:
        if custom_objects is None:
            custom_objects = {}
        try:
            from ..solvers.esinet import CustomConv2D

            custom_objects["CustomConv2D"] = CustomConv2D
        except (ImportError, AttributeError):
            pass
        model = tf.keras.models.load_model(path, custom_objects=custom_objects)
        solver.model = model

    return solver


def pos_from_forward(forward, verbose=0):
    """Get vertex/dipole positions from mne.Forward model

    Parameters
    ----------
    forward : instance of mne.Forward
        The forward model.

    Return
    ------
    pos : numpy.ndarray
        A 2D matrix containing the MNI coordinates of the vertices/ dipoles

    Note
    ----
    forward must contain some subject id in forward["src"][0]["subject_his_id"]
    in order to work.
    """
    # Get Subjects ID
    subject_his_id = forward["src"][0]["subject_his_id"]
    src = forward["src"]

    # Extract vertex positions from left and right source space
    pos_left = mne.vertex_to_mni(src[0]["vertno"], 0, subject_his_id, verbose=verbose)
    pos_right = mne.vertex_to_mni(src[1]["vertno"], 1, subject_his_id, verbose=verbose)

    # concatenate coordinates from both hemispheres
    pos = np.concatenate([pos_left, pos_right], axis=0)

    return pos


def thresholding(x, k):
    """Set all but the k largest magnitudes in x to zero (0).

    Parameters
    ----------
    x : numpy.ndarray
        Data vector
    k : int
        The k number of largest magnitudes to maintain.

    Return
    ------
    x_new : numpy.ndarray
        Array of same length as input array x.
    """
    if isinstance(x, list):
        x = np.array(x)

    # Handle edge cases
    if k <= 0:
        return np.zeros_like(x)
    if k >= len(x):
        return x.copy()

    # Get absolute values and sort indices
    abs_x = np.abs(x)
    sorted_indices = np.argsort(abs_x)

    # Handle ties by including all elements with the same magnitude as the k-th largest
    threshold_value = abs_x[sorted_indices[-k]]

    # Find all indices with magnitude >= threshold
    selected_indices = np.where(abs_x >= threshold_value)[0]

    # If we have more than k elements due to ties, select the first k based on original order
    if len(selected_indices) > k:
        # Sort by magnitude (descending) then by original index (ascending) for tie-breaking
        tie_breaking_order = np.lexsort(
            (np.arange(len(x))[selected_indices], -abs_x[selected_indices])
        )
        selected_indices = selected_indices[tie_breaking_order[:k]]

    x_new = np.zeros_like(x)
    x_new[selected_indices] = x[selected_indices]
    return x_new


def calc_residual_variance(M_hat, M):
    """Calculate the residual variance (relative squared error) in percent (%).

    Parameters
    ----------
    M_hat : numpy.ndarray
        Estimated M/EEG data
    M : numpy.ndarray
        True M/EEG data

    Return
    ------
    rv : float
        Residual variance in %.
            0  -> M_hat perfectly explains M
            100 -> M_hat explains none of M (equivalent to zero estimate)

    """
    return 100 * np.sum((M - M_hat) ** 2) / np.sum(M**2)


def euclidean_distance(A, B):
    """Euclidean Distance between two points."""
    return np.sqrt(np.sum((A - B) ** 2))


def calc_area_tri(AB, AC, CB):
    """Calculates area of a triangle given the length of each side."""

    s = (AB + AC + CB) / 2
    area = (s * (s - AB) * (s - AC) * (s - CB)) ** 0.5
    return area


def find_corner(source_power, residual, plot=False):
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
    if len(residual) < 3:
        return len(residual) - 1
    # Normalize l2 norms
    # source_power /= np.max(source_power)

    A = np.array([residual[0], source_power[0]])
    C = np.array([residual[-1], source_power[-1]])
    areas = []
    for j in range(1, len(source_power) - 1):
        B = np.array([residual[j], source_power[j]])
        AB = euclidean_distance(A, B)
        AC = euclidean_distance(A, C)
        CB = euclidean_distance(C, B)
        area = abs(calc_area_tri(AB, AC, CB))
        areas.append(area)
    if len(areas) > 0:
        idx = np.argmax(areas) + 1
    else:
        idx = 0
    if plot:
        plt.figure()
        plt.plot(source_power, residual, "*k")
        plt.plot(source_power[idx], residual[idx], "or")
    return idx


def best_index_residual(residuals, x_hats, plot=False):
    """Finds the idx that optimally regularises the inverse solution.
    Parameters
    ----------
    residuals : numpy.ndarray
        The residual variances of the inverse solutions to the data.

    Return
    ------
    corner_idx : int
        The index at which the trade-off between explaining
        data and source complexity is optimal.
    """
    iters = np.arange(len(residuals)).astype(float)
    # Remove indices where residual variance started to rise

    if np.any(np.diff(residuals) > 0):
        bad_idx = (np.where(np.diff(residuals) > 0)[0] + 1)[0]
    else:
        bad_idx = len(residuals)
    bad_idx = len(residuals)

    iters = iters[:bad_idx]
    x_hats = x_hats[:bad_idx]
    residuals = residuals[:bad_idx]

    # iters = iters[1:bad_idx]
    # x_hats = x_hats[1:bad_idx]
    # residuals = residuals[1:bad_idx]

    # L-Curve Corner
    # corner_idx = find_corner(iters, residuals)

    # Residual criterion
    span = abs(residuals[-1] - residuals[0])
    min_change = span * 0.1
    try:
        corner_idx = np.where(abs(np.diff(residuals)) < min_change)[0][0]
    except IndexError:
        corner_idx = np.argmin(residuals)

    if plot:
        plt.figure()
        plt.plot(iters, residuals, "*k")
        plt.plot(iters[corner_idx], residuals[corner_idx], "or")
        plt.ylabel("residual")
        plt.xlabel("iteration no.")

    return x_hats[corner_idx]
