import logging
from copy import deepcopy

import numpy as np
import ot
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import laplacian
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.metrics import auc, roc_curve

logger = logging.getLogger(__name__)


def evaluate_all(
    y_true,
    y_pred,
    adjacency_true,
    adjacency_pred,
    pos_true,
    pos_pred,
    mode: str = "dle",
    threshold: float = 0.1,
):
    """
    Comprehensive evaluation of source reconstruction.

    Parameters
    ----------
    y_true : np.ndarray
        True source activity
    y_pred : np.ndarray
        Predicted source activity
    adjacency_true : scipy.sparse matrix
        Adjacency matrix for true source space
    adjacency_pred : scipy.sparse matrix
        Adjacency matrix for predicted source space
    pos_true : np.ndarray
        Positions of true sources
    pos_pred : np.ndarray
        Positions of predicted sources (typically same as pos_true)
    mode : str
        MLE calculation mode
    threshold : float
        Threshold for maxima detection

    Returns
    -------
    dict
        Dictionary of evaluation metrics

    Notes
    -----
    In typical usage, pos_true and pos_pred are identical (same source space),
    but this function supports different source spaces for generality.
    """
    # Distance matrices for different metric calculations
    distance_matrix_true_pred = cdist(pos_true, pos_pred)  # For EMD and MLE
    distance_matrix_true = cdist(pos_true, pos_true)  # For spatial dispersion of true
    distance_matrix_pred = cdist(pos_pred, pos_pred)  # For spatial dispersion of pred

    y_true_collapsed = abs(y_true).mean(axis=-1)
    y_pred_collapsed = abs(y_pred).mean(axis=-1)

    logger.debug(
        "Shapes - y_true: %s, y_pred: %s, adj_true: %s, adj_pred: %s, dist_matrix: %s",
        y_true_collapsed.shape,
        y_pred_collapsed.shape,
        adjacency_true.shape,
        adjacency_pred.shape,
        distance_matrix_true_pred.shape,
    )

    # mse = [eval_mse(yy_true, yy_pred) for yy_true, yy_pred in zip(y_true.T, y_pred.T)]
    # nmse = [eval_nmse(yy_true, yy_pred) for yy_true, yy_pred in zip(y_true.T, y_pred.T)]
    # auc = [np.mean(eval_auc(yy_true, yy_pred, pos_1, epsilon=0.01, n_redraw=10)) for yy_true, yy_pred in zip(y_true.T, y_pred.T)]
    # corr = [pearsonr(yy_true, yy_pred)[0] for yy_true, yy_pred in zip(y_true.T, y_pred.T)]
    # sparsity_pred = eval_sparsity(y_pred)
    # sparsity_true = eval_sparsity(y_true)
    # active_true = eval_active(y_true)
    # active_pred = eval_active(y_pred)

    mle = eval_mean_localization_error(
        y_true_collapsed,
        y_pred_collapsed,
        adjacency_true,
        adjacency_pred,
        pos_true,
        pos_pred,
        distance_matrix_true_pred,
        mode=mode,
        threshold=threshold,
    )
    emd = eval_emd(distance_matrix_true_pred, y_true_collapsed, y_pred_collapsed)
    sd = get_spatial_dispersion(
        y_pred,
        y_true,
        distance_matrix_pred,
        distance_matrix_true,
        adjacency_pred.toarray(),
        adjacency_true.toarray(),
    )

    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    average_precision = compute_average_precision(precision, recall)

    correlation = np.nanmedian(corr(y_true, y_pred))

    d = dict(
        Mean_Localization_Error=np.nanmedian(mle),
        EMD=emd,
        sd=sd,
        average_precision=average_precision,
        correlation=correlation,
    )

    return d


def eval_active(y):
    if len(y.shape) > 1:
        return np.linalg.norm(y[:, 0], ord=0)
    else:
        return np.linalg.norm(y, ord=0)


def eval_sparsity(y):
    y_scaled = y / np.linalg.norm(y, axis=0)
    return np.linalg.norm(y_scaled, ord=1)


def eval_emd(M, values_1, values_2, threshold=0.25, p=1):
    """
    Compute Earth Mover's Distance between two distributions.

    Parameters
    ----------
    M : np.ndarray, shape (n1, n2)
        Distance/cost matrix between positions
    values_1 : np.ndarray, shape (n1,)
        First distribution (will be normalized to sum to 1)
    values_2 : np.ndarray, shape (n2,)
        Second distribution (will be normalized to sum to 1)

    Returns
    -------
    float
        EMD value, or np.nan if calculation fails
    """
    # Validate inputs
    if np.any(np.isnan(values_1)) or np.any(np.isnan(values_2)):
        logger.warning("NaN values in input distributions")
        return np.nan

    if np.any(np.isinf(values_1)) or np.any(np.isinf(values_2)):
        logger.warning("Inf values in input distributions")
        return np.nan

    # Set values smaller 0.25 of the max to 0
    values_1 = np.where(abs(values_1) < threshold * values_1.max(), 0, values_1)
    values_2 = np.where(abs(values_2) < threshold * values_2.max(), 0, values_2)

    sum_1 = np.sum(values_1)
    sum_2 = np.sum(values_2)

    # Check for zero or negative sums
    if sum_1 <= 0:
        logger.warning(f"First distribution sums to {sum_1} (must be positive)")
        return np.nan

    if sum_2 <= 0:
        logger.warning(f"Second distribution sums to {sum_2} (must be positive)")
        return np.nan

    # Normalize to sum to 1

    # ## Add a small constant to avoid division by zero
    # values_1 += 1e-4 * values_1.max()
    # values_2 += 1e-4 * values_2.max()

    values_1 = values_1 / sum_1
    values_2 = values_2 / sum_2

    # Validate dimensions match cost matrix
    if M.shape != (len(values_1), len(values_2)):
        raise ValueError(
            f"Cost matrix shape {M.shape} doesn't match distribution shapes "
            f"({len(values_1)}, {len(values_2)})"
        )

    try:
        emd_value = ot.emd2(values_1, values_2, M**p)
        # emd_value = sinkhorn_divergence(values_1, values_2, M**2, reg)
    except Exception as e:
        logger.error(f"Error calculating EMD: {e}")
        return np.nan

    # ot.emd2 can return a list in some cases; ensure scalar
    if isinstance(emd_value, (list, np.ndarray)):
        emd_value = float(np.asarray(emd_value).item())

    return emd_value


def emd(
    positions_1: np.ndarray,
    values_1: np.ndarray,
    positions_2: np.ndarray,
    values_2: np.ndarray,
) -> float:
    """
    Compute the Earth Movers Distance between two distributions on different grids.

    This is a convenience wrapper that computes the distance matrix and calls eval_emd.
    Use eval_emd directly if you already have a precomputed distance matrix.

    Parameters
    ----------
    positions_1 : np.ndarray, shape (n1, d)
        Positions for first distribution (d = spatial dimensions)
    values_1 : np.ndarray, shape (n1,)
        Values at positions_1
    positions_2 : np.ndarray, shape (n2, d)
        Positions for second distribution
    values_2 : np.ndarray, shape (n2,)
        Values at positions_2

    Returns
    -------
    float
        EMD value, or np.nan if calculation fails
    """

    # Calculate pairwise distance between positions
    M = ot.dist(positions_1, positions_2)

    # Use the validated eval_emd function
    return eval_emd(M, values_1, values_2)


def eval_mse(y_true, y_est):
    """Returns the mean squared error between predicted and true source."""
    return np.mean((y_true - y_est) ** 2)


def eval_nmse(y_true, y_est):
    """Returns the normalized mean squared error between predicted and true
    source."""

    y_true_normed = y_true / np.max(np.abs(y_true))
    y_est_normed = y_est / np.max(np.abs(y_est))
    return np.mean((y_true_normed - y_est_normed) ** 2)


def nmse(y_true, y_pred):
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        return np.nan
    error = np.zeros(y_true.shape[1])
    for i, (y_true_slice, y_pred_slice) in enumerate(zip(y_true.T, y_pred.T)):
        error[i] = np.mean(
            (
                (y_true_slice / abs(y_true_slice).max())
                - (y_pred_slice / abs(y_pred_slice).max())
            )
            ** 2
        )
    return error


def corr(y_true, y_pred):
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        return np.nan
    error = np.zeros(y_true.shape[1])
    for i, (y_true_slice, y_pred_slice) in enumerate(zip(y_true.T, y_pred.T)):
        error[i] = pearsonr(y_true_slice, y_pred_slice)[0]
    return error


def true_variance_explained(y_true, y_pred, leadfield):
    """Calculate the true variance explained using the predicted source and the noiseless EEG."""
    M = leadfield @ y_true
    M_hat = leadfield @ y_pred
    # Common average reference
    # M -= M.mean(axis=-1)
    # M_hat -= M_hat.mean(axis=-1)

    return calc_residual_variance(M_hat, M)


def calc_residual_variance(M_hat, M):
    return 100 * np.sum((M - M_hat) ** 2) / np.sum(M**2)


def coeff_det(M, M_hat):
    M_prep = deepcopy(M)
    M_hat_prep = deepcopy(M_hat)

    # M_prep -= M_prep.mean(axis=0)
    # M_hat_prep -= M_hat_prep.mean(axis=0)

    M_prep /= M_prep.std()
    M_hat_prep /= M_hat_prep.std()
    SS_res = np.sum((M_prep - M_hat_prep) ** 2)
    SS_tot = np.sum((M_prep - np.mean(M_prep)) ** 2)

    return 1 - (SS_res / SS_tot)


def compute_fwhm(current_density, distance_matrix, adj_matrix):
    # Identify nodes with density above 50% of the maximum
    threshold = 0.5 * current_density.max()
    above_threshold = current_density > threshold

    # Calculate the effective volume of these nodes
    node_count = np.sum(above_threshold)
    if node_count == 0:
        return 0

    # Use the adjacency matrix to sum up the "volume"
    fwhm = 0
    for i in range(len(current_density)):
        if above_threshold[i]:
            # Volume contribution from node i
            connected_nodes = np.where(adj_matrix[i] > 0)[0]
            local_volume = np.sum(distance_matrix[i, connected_nodes])
            fwhm += local_volume

    return fwhm


def get_spatial_dispersion(
    X_hat: np.ndarray,
    X: np.ndarray,
    distance_matrix_hat: np.ndarray,
    distance_matrix: np.ndarray,
    adjacency_matrix_hat,
    adjacency_matrix: np.ndarray,
) -> float:
    """This implementation of spatial dispersion is based on the FWHM of the
    current density. It returns the blurring, which is the ratio of the FWHM of the
    estimated current density to the FWHM of the true current density.
    """
    X_hat = abs(X_hat)
    X = abs(X)

    if X_hat.ndim == 2:
        X_hat = X_hat.mean(axis=1)
    if X.ndim == 2:
        X = X.mean(axis=1)

    fwhm_hat = compute_fwhm(X_hat, distance_matrix_hat, adjacency_matrix_hat)
    fwhm_true = compute_fwhm(X, distance_matrix, adjacency_matrix)

    if fwhm_true == 0:
        return (
            np.inf
        )  # Avoid division by zero; indicates no significant activity in true solution
    blurring = fwhm_hat / fwhm_true

    return blurring


def eval_mean_localization_error(
    y_true: np.ndarray,
    y_est: np.ndarray,
    adjacency_true: scipy.sparse.csr_matrix,
    adjacency_est: scipy.sparse.csr_matrix,
    pos_true: np.ndarray,
    pos_pred: np.ndarray,
    distance_matrix: np.ndarray,
    mode: str = "dle",
    threshold: float = 0.1,
    smooth_solution: bool = False,
    max_maxima: int = 5,
    max_iter: int = 100,
) -> float:
    """Calculate the Mean Localization Error (MLE) between a true and predicted source.
    Parameters
    ----------
    y_true : np.ndarray
        The ground truth values.
    y_est : np.ndarray
        The estimated values.
    adjacency_true : scipy.sparse.csr_matrix
        The adjacency matrix for the true graph.
    adjacency_est : scipy.sparse.csr_matrix
        The adjacency matrix for the estimated graph.
    distance_matrix : np.ndarray
        The euclidean distance between each dipole in y_true and each dipole in y_est.
    mode : str
        The mode to use for the MLE calculation. Options are "dle" (default), "truth" and "est" and "match".

    """
    if len(y_true.shape) == 2:
        y_true_collapsed = abs(y_true).mean(axis=-1)
    else:
        y_true_collapsed = abs(y_true)
    if len(y_est.shape) == 2:
        y_est_collapsed = abs(y_est).mean(axis=-1)
    else:
        y_est_collapsed = abs(y_est)

    maxima_idc_true = get_maxima(
        y_true_collapsed,
        adjacency_true,
        pos_true,
        threshold=threshold,
        get_smoothed=False,
        smooth_solution=False,
        max_maxima=max_maxima,
        max_iter=max_iter,
    )
    maxima_idc_est = get_maxima(
        y_est_collapsed,
        adjacency_est,
        pos_pred,
        threshold=threshold,
        get_smoothed=False,
        smooth_solution=smooth_solution,
        max_maxima=max_maxima,
        max_iter=max_iter,
    )
    # print(f"maxima_idc_true: {len(maxima_idc_true)}, maxima_idc_est: {len(maxima_idc_est)}")
    # print(len(maxima_idc_est), len(maxima_idc_true))
    if len(maxima_idc_est) == 0 or len(maxima_idc_true) == 0:
        logger.warning("No maxima found in true or estimated source")
        return np.nan

    # Get pairwise distance between true and estimated source locations.
    pairwise_dist = np.zeros((len(maxima_idc_true), len(maxima_idc_est)))

    for ii, idx_true in enumerate(maxima_idc_true):
        for jj, idx_est in enumerate(maxima_idc_est):
            pairwise_dist[ii, jj] = distance_matrix[idx_true, idx_est]
    if mode == "dle":
        mle = (pairwise_dist.min(axis=0).mean() + pairwise_dist.min(axis=1).mean()) / 2
    elif mode == "est":
        mle = pairwise_dist.min(axis=0).mean()
    elif mode == "true":
        mle = pairwise_dist.min(axis=1).mean()
    elif mode == "match":
        # Solve the assignment problem (i.e., find the matching with minimum total distance)
        true_indices, estimated_indices = linear_sum_assignment(pairwise_dist)
        # Calculate the sum of distances for the optimal assignment
        mle = pairwise_dist[true_indices, estimated_indices].mean()
    elif mode == "amir":
        mle = shortest_dists_amir(pairwise_dist)

    else:
        raise ValueError(f"Invalid mode '{mode}' for MLE calculation.")
    # explanation:
    # pairwise_dist.min(axis=0) returns the minimum distance for each estimated source

    return mle


def find_maxima(
    y: np.ndarray,
    adjacency: np.ndarray,
    threshold: float = 0.1,
    max_val_return: int = 9999999999,
) -> list:
    """
    Return indices of local maxima based on the adjacency matrix.

    Parameters
    ----------
    y : np.ndarray
        1D array containing the values of nodes. Each value corresponds to a
        node's "intensity" or "activity".
    adjacency : np.ndarray
        2D square matrix representing the adjacency of nodes. An entry of 1 at
        position (i,j) implies node i is adjacent to node j.

    Returns
    -------
    list_of_maxima : list
        List of indices where the nodes' values in 'y' are greater than all of
        their adjacent nodes.
    """
    # Apply thresholding in-place to avoid copying the array
    y_thresholded = np.where(abs(y) >= abs(y).max() * threshold, y, 0)
    list_of_maxima = []

    # Only iterate over non-zero thresholded elements
    nonzeros = np.nonzero(y_thresholded)[0]

    if isinstance(adjacency, scipy.sparse.csr_matrix):
        adjacency = adjacency.toarray()

    for i in nonzeros:
        # Get indices of neighbors
        neighbors = np.flatnonzero(adjacency[i])

        # remove itself
        neighbors = np.delete(neighbors, np.where(neighbors == i)[0])

        # Check if the value at index i is greater than or equal to all its neighbors
        if np.all(y_thresholded[i] > y_thresholded[neighbors]):
            list_of_maxima.append(i)
        # if len(list_of_maxima) == max_val_return:
        #     break
    # print("Maxima before")
    # print(y_thresholded[np.array(list_of_maxima)])
    if len(list_of_maxima) > max_val_return:
        # retain only the largest maxima
        list_of_maxima = sorted(
            list_of_maxima, key=lambda x: y_thresholded[x], reverse=True
        )[:max_val_return]
    # print("Maxima after")
    # print(y_thresholded[np.array(list_of_maxima)])

    # print(f"Found {len(list_of_maxima)} maxima.")
    return list_of_maxima


def check_maxima_larger_than(
    y: np.ndarray,
    adjacency: np.ndarray,
    threshold: float = 0.1,
    max_val_return: int = 9999999999,
) -> bool:
    """
    Return indices of local maxima based on the adjacency matrix.

    Parameters
    ----------
    y : np.ndarray
        1D array containing the values of nodes. Each value corresponds to a
        node's "intensity" or "activity".
    adjacency : np.ndarray
        2D square matrix representing the adjacency of nodes. An entry of 1 at
        position (i,j) implies node i is adjacent to node j.

    Returns
    -------
    True if there are more than max_val_return maxima, False otherwise.
    """
    # Apply thresholding in-place to avoid copying the array
    y_thresholded = np.where(abs(y) >= abs(y).max() * threshold, y, 0)
    list_of_maxima = []

    # Only iterate over non-zero thresholded elements
    nonzeros = np.nonzero(y_thresholded)[0]

    if isinstance(adjacency, scipy.sparse.csr_matrix):
        adjacency = adjacency.toarray()

    for i in nonzeros:
        # Get indices of neighbors
        neighbors = np.flatnonzero(adjacency[i])

        # Check if the value at index i is greater than or equal to all its neighbors
        if np.all(y_thresholded[i] >= y_thresholded[neighbors]):
            list_of_maxima.append(i)
            if len(list_of_maxima) == max_val_return:
                return True
    return False


def adaptive_smooth_solution(
    Y: np.ndarray,
    adjacency: np.ndarray,
    get_smoothed: bool = False,
    max_maxima: int = 5,
    threshold: float = 0.1,
    max_iter: int = 100,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    alpha = 0.1
    Y_smoothed = deepcopy(Y)
    adjacency_ = adjacency.toarray()  # type: ignore[union-attr, attr-defined]

    if max_iter > 0:
        nonzero_idc = np.where(Y_smoothed != 0)[0]
        L = laplacian(adjacency_, normed=False)

        Y_smoothed[abs(Y_smoothed) < abs(Y_smoothed).max() * threshold] = 0

    for n_iter in range(max_iter):
        if n_iter % 1 == 0:
            too_many_maxima = check_maxima_larger_than(
                Y_smoothed, adjacency_, threshold=threshold, max_val_return=max_maxima
            )
            if not too_many_maxima:
                break
        # Y_smoothed -= alpha * L @ Y_smoothed
        Y_smoothed[nonzero_idc] -= (
            alpha * L[nonzero_idc, :][:, nonzero_idc] @ Y_smoothed[nonzero_idc]
        )
        Y_smoothed /= np.abs(Y_smoothed).max()
    maxima_idc = find_maxima(
        Y_smoothed, adjacency_, threshold=threshold, max_val_return=max_maxima
    )

    if get_smoothed:
        return maxima_idc, Y_smoothed  # type: ignore[return-value]
    else:
        return maxima_idc  # type: ignore[return-value]


def get_maxima(
    Y: np.ndarray,
    adjacency: scipy.sparse.csr_matrix,
    pos: np.ndarray,
    threshold: float = 0.1,
    get_smoothed: bool = False,
    smooth_solution: bool = False,
    max_maxima: int = 5,
    max_iter: int = 100,
):
    maxima_idc_raw = find_maxima(
        Y, adjacency.toarray(), threshold=threshold, max_val_return=max_maxima
    )

    Y_smoothed = Y
    if smooth_solution:
        if get_smoothed:
            maxima_idc, Y_smoothed = adaptive_smooth_solution(
                Y,
                adjacency,
                get_smoothed=get_smoothed,
                threshold=threshold,
                max_maxima=max_maxima,
                max_iter=max_iter,
            )
        else:
            maxima_idc = adaptive_smooth_solution(
                Y,
                adjacency,
                get_smoothed=get_smoothed,
                threshold=threshold,
                max_maxima=max_maxima,
                max_iter=max_iter,
            )
        #  all maxima are in the original maxima list
        for i, idx in enumerate(maxima_idc):
            if idx not in maxima_idc_raw:
                match = np.argmin(cdist(pos[idx][np.newaxis], pos[maxima_idc_raw]))
                maxima_idc[i] = maxima_idc_raw[match]

    else:
        maxima_idc = maxima_idc_raw

    if get_smoothed:
        return maxima_idc, Y_smoothed
    else:
        return maxima_idc


def shortest_dists_amir(m_dists):
    """Code translated form amirs matlab code"""
    n_predictions, n_ground_truths = m_dists.shape
    n_matches = min(n_predictions, n_ground_truths)  # Choose number of iterations
    mean_dist = 0
    m_dists = m_dists.T  # Transpose for convenience
    # Pick smallest distance in matrix and exclude its row and col
    for _ in range(n_matches):
        d = np.min(m_dists)
        n, m = np.unravel_index(np.argmin(m_dists), m_dists.shape)
        m_dists = np.delete(np.delete(m_dists, n, axis=0), m, axis=1)  # Exclude
        mean_dist += d  # Add to cumulative error
    mean_dist /= n_matches  # Calculate mean error
    return mean_dist


def get_maxima_pos(mask, pos):
    """Returns the positions of the maxima within mask.
    Parameters
    ----------
    mask : numpy.ndarray
        The source mask
    pos : numpy.ndarray
        The dipole position matrix
    """
    return pos[np.where(mask == 1)[0]]


def eval_auc(y_true, y_est, pos, n_redraw=25, epsilon=0.25):
    """Returns the area under the curve metric between true and predicted
    source.

    Parameters
    ----------
    y_true : numpy.ndarray
        True source vector
    y_est : numpy.ndarray
        Estimated source vector
    pos : numpy.ndarray
        Dipole positions (points x dims)
    n_redraw : int
        Defines how often the negative samples are redrawn.
    epsilon : float
        Defines threshold on which sources are considered
        active.
    Return
    ------
    auc_close : float
        Area under the curve for dipoles close to source.
    auc_far : float
        Area under the curve for dipoles far from source.
    """
    if y_est.sum() == 0 or y_true.sum() == 0:
        return np.nan, np.nan
    y_true = deepcopy(y_true)
    y_est = deepcopy(y_est)
    # Absolute values
    y_true = np.abs(y_true)
    y_est = np.abs(y_est)

    # Normalize values
    y_true /= np.max(y_true)
    y_est /= np.max(y_est)

    auc_close = np.zeros(n_redraw)
    auc_far = np.zeros(n_redraw)

    source_mask = (y_true > epsilon).astype(int)

    numberOfActiveSources = int(np.sum(source_mask))
    numberOfDipoles = pos.shape[0]
    # Draw from the 20% of closest dipoles to sources (~100)
    closeSplit = int(round(numberOfDipoles / 5))
    # Draw from the 50% of furthest dipoles to sources
    farSplit = int(round(numberOfDipoles / 2))
    #
    distSortedIndices = find_indices_close_to_source(source_mask, pos)

    sourceIndices = np.where(source_mask == 1)[0]

    for n in range(n_redraw):
        selectedIndicesClose = np.concatenate(
            [
                sourceIndices,
                np.random.choice(
                    distSortedIndices[:closeSplit], size=numberOfActiveSources
                ),
            ]
        )
        selectedIndicesFar = np.concatenate(
            [
                sourceIndices,
                np.random.choice(
                    distSortedIndices[-farSplit:], size=numberOfActiveSources
                ),
            ]
        )
        # print(f'redraw {n}:\ny_true={y_true[selectedIndicesClose]}\y_est={y_est[selectedIndicesClose]}')
        fpr_close, tpr_close, _ = roc_curve(
            source_mask[selectedIndicesClose], y_est[selectedIndicesClose]
        )

        fpr_far, tpr_far, _ = roc_curve(
            source_mask[selectedIndicesFar], y_est[selectedIndicesFar]
        )

        auc_close[n] = auc(fpr_close, tpr_close)
        auc_far[n] = auc(fpr_far, tpr_far)

    auc_far = np.mean(auc_far)
    auc_close = np.mean(auc_close)

    return np.mean([auc_close, auc_far])


def find_indices_close_to_source(source_mask, pos):
    """Finds the dipole indices that are closest to the active sources.

    Parameters
    -----------
    simSettings : dict
        retrieved from the simulate_source function
    pos : numpy.ndarray
        list of all dipole positions in XYZ coordinates

    Return
    -------
    ordered_indices : numpy.ndarray
        ordered list of dipoles that are near active
        sources in ascending order with respect to their distance to the next source.
    """

    numberOfDipoles = pos.shape[0]

    sourceIndices = np.array([i[0] for i in np.argwhere(source_mask == 1)])

    min_distance_to_source = np.zeros(numberOfDipoles)

    # D = np.zeros((numberOfDipoles, len(sourceIndices)))
    # for i, idx in enumerate(sourceIndices):
    #     D[:, i] = np.sqrt(np.sum(((pos-pos[idx])**2), axis=1))
    # min_distance_to_source = np.min(D, axis=1)
    # min_distance_to_source[source_mask==1] = np.nan
    # numberOfNans = source_mask.sum()

    ###OLD
    numberOfNans = 0
    for i in range(numberOfDipoles):
        if source_mask[i] == 1:
            min_distance_to_source[i] = np.nan
            numberOfNans += 1
        elif source_mask[i] == 0:
            distances = np.sqrt(
                np.sum((pos[sourceIndices, :] - pos[i, :]) ** 2, axis=1)
            )
            min_distance_to_source[i] = np.min(distances)
        else:
            logger.error("source mask has invalid entries")
    # print('new: ', np.nanmean(min_distance_to_source), min_distance_to_source.shape)
    ###OLD

    ordered_indices = np.argsort(min_distance_to_source)

    return ordered_indices[:-numberOfNans]


def precision_recall_curve(y_true, y_pred):
    """
    Computes the precision-recall curve for binary classification.

    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth binary labels (0 or 1), shape (n_samples,).
    y_pred : np.ndarray
        Predicted probabilities or scores, shape (n_samples,).

    Returns:
    --------
    precision : np.ndarray
        Precision values for different threshold values.
    recall : np.ndarray
        Recall values for different threshold values.
    thresholds : np.ndarray
        Thresholds at which precision and recall are calculated.
    """
    # prepare data
    y_true = y_true != 0
    y_pred = abs(y_pred)

    if y_true.ndim == 2:
        y_true = y_true.mean(axis=1)
    if y_pred.ndim == 2:
        y_pred = y_pred.mean(axis=1)
    #  y_true and y_pred are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Sort predictions and corresponding true labels by predicted scores
    sorted_indices = np.argsort(-y_pred)
    y_true = y_true[sorted_indices]
    y_pred = y_pred[sorted_indices]

    # Calculate true positives and false positives cumulatively
    tp_cumsum = np.cumsum(y_true)
    fp_cumsum = np.cumsum(1 - y_true)

    # Total number of positive samples
    total_positives = np.sum(y_true)

    # Calculate precision and recall
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / total_positives

    # Include thresholds corresponding to each point in the curve
    thresholds = y_pred

    return precision, recall, thresholds


def compute_average_precision(precision, recall):
    """
    Computes the Average Precision (AP) from precision-recall values.
    Interpretation: AP is the area under the precision-recall curve.
    The precision recall curve is a plot of precision versus recall for different threshold values.

    Precision is also called positive predictive value and measures the proportion of true positives among all predicted positives.
    Recall is also called sensitivity and measures the proportion of true positives among all actual positives.

    The area under this curve is thus the average precision and is intuitively the probability that a randomly selected positive example is ranked higher than a randomly selected negative example.

    Parameters:
    -----------
    precision : np.ndarray
        Array of precision values.
    recall : np.ndarray
        Array of recall values.

    Returns:
    --------
    average_precision : float
        The average precision score.
    """
    #  precision and recall are numpy arrays
    precision = np.asarray(precision)
    recall = np.asarray(recall)

    # Append starting point for recall and precision (0,1)
    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([1.0], precision))

    # Calculate AP as the area under the precision-recall curve
    average_precision = np.sum((recall[1:] - recall[:-1]) * precision[1:])

    return average_precision
