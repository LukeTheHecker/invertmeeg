from copy import deepcopy

import mne
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.sparse.csgraph import laplacian
from scipy.stats import pearsonr


def rescale_sources(leadfield, source_pred, y_original):
    """
    Rescale the predicted sources to match the original data.

    Parameters:
    -----------
    leadfield : numpy.ndarray
        The leadfield matrix.
    source_pred : numpy.ndarray
        The predicted source activity.
    y_original : numpy.ndarray
        The original observed data.

    Returns:
    --------
    source_pred_scaled : numpy.ndarray
        The rescaled source predictions.
    """
    # Forward calculate the sensor data from the predicted sources
    y_pred = leadfield @ source_pred

    # Calculate the scaling factor
    scaling_factor = np.sum(y_original * y_pred) / np.sum(y_pred**2)

    # Apply the scaling factor to the predicted sources
    source_pred_scaled = source_pred * scaling_factor

    return source_pred_scaled


def solve_p_wrap(leadfield, y_est, x_true):
    """Wrapper for parallel (or, alternatively, serial) scaling of
    predicted sources.
    """

    y_est_scaled = deepcopy(y_est)

    for trial, _ in enumerate(x_true):
        for time in range(x_true[trial].shape[-1]):
            scaled = solve_p(leadfield, y_est[trial][:, time], x_true[trial][:, time])
            y_est_scaled[trial][:, time] = scaled

    return y_est_scaled


def solve_p(leadfield, y_est, x_true):
    """
    Parameters
    ---------
    y_est : numpy.ndarray
        The estimated source vector.
    x_true : numpy.ndarray
        The original input EEG vector.

    Return
    ------
    y_scaled : numpy.ndarray
        The scaled estimated source vector.

    """
    # Check if y_est is just zeros:
    if np.max(y_est) == 0:
        return y_est
    y_est = np.squeeze(np.array(y_est))
    x_true = np.squeeze(np.array(x_true))
    # Get EEG from predicted source using leadfield
    x_est = np.matmul(leadfield, y_est)

    # optimize forward solution
    tol = 1e-9
    options = dict(maxiter=1000, disp=False)

    # base scaling
    rms_est = np.mean(np.abs(x_est))
    rms_true = np.mean(np.abs(x_true))
    base_scaler = rms_true / rms_est

    opt = minimize_scalar(
        correlation_criterion,
        args=(leadfield, y_est * base_scaler, x_true),
        bounds=(0, 1),
        options=options,
        tol=tol,
    )

    scaler = opt.x
    y_scaled = y_est * scaler * base_scaler
    return y_scaled


def correlation_criterion(scaler, leadfield, y_est, x_true):
    """Perform forward projections of a source using the leadfield.
    This is the objective function which is minimized in Net::solve_p().

    Parameters
    ----------
    scaler : float
        scales the source y_est
    leadfield : numpy.ndarray
        The leadfield (or sometimes called gain matrix).
    y_est : numpy.ndarray
        Estimated/predicted source.
    x_true : numpy.ndarray
        True, unscaled EEG.
    """

    x_est = np.matmul(leadfield, y_est)
    error = np.abs(pearsonr(x_true - x_est, x_true)[0])
    return error


class Compressor:
    """Compression using Graph Fourier Transform"""

    def __init__(self):
        pass

    def fit(self, fwd, k=600):
        A = mne.spatial_src_adjacency(fwd["src"], verbose=0).toarray()
        L = laplacian(A)
        U, s, V = np.linalg.svd(L)

        self.U = U[:, -k:]
        self.s = s[-k:]
        self.V = V[:, -k:]
        return self

    def encode(self, X):
        """Encodes a true signal X
        Parameters
        ----------
        X : numpy.ndarray
            True signal

        Return
        ------
        X_comp : numpy.ndarray
            Compressed signal
        """
        X_comp = self.U.T @ X

        return X_comp

    def decode(self, X_comp):
        """Decodes a compressed signal X

        Parameters
        ----------
        X : numpy.ndarray
            Compressed signal

        Return
        ------
        X_unfold : numpy.ndarray
            Decoded signal
        """
        X_unfold = self.U @ X_comp

        return X_unfold
