import logging

import mne
import numpy as np

logger = logging.getLogger(__name__)

_EPS = 1e-12


def _weighted_minimum_norm(leadfield, data, weights, alpha):
    """Compute weighted minimum-norm estimate for a single timepoint.

    Implements the classic weighted minimum-norm estimator:
        x = W L^T (L W L^T + alpha I)^(-1) y
    where W is diagonal with positive entries given by ``weights``.
    """
    n_chans, n_sources = leadfield.shape
    data = np.asarray(data)
    if data.shape != (n_chans,):
        raise ValueError(
            f"`data` must have shape (n_chans,), got {data.shape} (n_chans={n_chans})."
        )

    weights = np.asarray(weights, dtype=float)
    if weights.shape != (n_sources,):
        raise ValueError(
            f"`weights` must have shape (n_sources,), got {weights.shape} (n_sources={n_sources})."
        )

    weights = np.maximum(weights, _EPS)
    # L @ W, where W is diagonal, can be done by column-wise scaling
    LW = leadfield * weights[np.newaxis, :]
    A = LW @ leadfield.T
    if alpha != 0:
        A = A + alpha * np.eye(n_chans, dtype=A.dtype)

    try:
        z = np.linalg.solve(A, data)
    except np.linalg.LinAlgError:
        z = np.linalg.pinv(A) @ data

    # W @ L^T @ z, where W is diagonal, can be done by element-wise scaling
    return weights * (leadfield.T @ z)


def focuss(stc, evoked, forward, alpha=0.01, max_iter=10, verbose=0):
    """FOCUSS algorithm.

    Parameters
    ----------
    stc : mne.SourceEstimate
        Source Estimate object.
    evoked : mne.EvokedArray
        Evoked EEG data object
    forward : mne.Forward
        The forward model
    alpha : float,
        Regularization parameter
    verbose : int
        Controls verbosity of the program

    Return
    ------
    stc_focuss : mne.SourceEstimate
        The new focussed source estimate
    """

    leadfield = forward["sol"]["data"]
    n_chans, n_dipoles = leadfield.shape
    D = stc.data
    M = evoked.data
    if D.shape[0] != n_dipoles:
        raise ValueError(f"stc has {D.shape[0]} sources but forward has {n_dipoles}.")
    if M.shape[0] != n_chans:
        raise ValueError(f"evoked has {M.shape[0]} channels but forward has {n_chans}.")
    if D.shape[1] != M.shape[1]:
        raise ValueError(
            f"stc and evoked must have the same number of timepoints, got {D.shape[1]} and {M.shape[1]}."
        )

    D_FOCUSS = np.zeros_like(D)
    if verbose:
        logger.info("FOCUSS:")

    for t in range(D.shape[1]):
        if verbose > 0:
            logger.info(f"Time step {t + 1}/{D.shape[1]}")

        x_last = np.asarray(D[:, t], dtype=float)
        weights = np.maximum(np.abs(x_last), _EPS)

        for i in range(max_iter):
            if verbose > 0:
                logger.info(f"Iteration {i + 1}/{max_iter}")

            x_new = _weighted_minimum_norm(leadfield, M[:, t], weights, alpha)

            rel_change = np.linalg.norm(x_new - x_last) / (
                np.linalg.norm(x_last) + _EPS
            )
            x_last = x_new
            weights = np.maximum(np.abs(x_last), _EPS)

            if rel_change < 1e-6:
                if verbose:
                    logger.info(f"Converged at iteration {i + 1}")
                break

        D_FOCUSS[:, t] = x_last

    stc_focuss = stc.copy()
    stc_focuss.data = D_FOCUSS
    return stc_focuss


def s_focuss(stc, evoked, forward, alpha=0.01, percentile=0.01, max_iter=10, verbose=0):
    """Shrinking FOCUSS algorithm. Based on Grech et al. (2008)

    Parameters
    ----------
    stc : mne.SourceEstimate
        Source Estimate object.
    evoked : mne.EvokedArray
        Evoked EEG data object
    forward : mne.Forward
        The forward model
    alpha : float,
        Regularization parameter
    verbose : int
        Controls verbosity of the program

    Return
    ------
    stc_focuss : mne.SourceEstimate
        The new focussed source estimate

    References
    ----------
    [1] Grech, R., Cassar, T., Muscat, J., Camilleri, K. P., Fabri, S. G.,
    Zervakis, M., ... & Vanrumste, B. (2008). Review on solving the inverse
    problem in EEG source analysis. Journal of neuroengineering and
    rehabilitation, 5(1), 1-33.

    """

    leadfield = forward["sol"]["data"]
    adjacency = mne.spatial_src_adjacency(forward["src"], verbose=verbose)
    adjacency = (
        adjacency.toarray() if hasattr(adjacency, "toarray") else np.asarray(adjacency)
    )
    n_chans, n_dipoles = leadfield.shape
    D = stc.data
    M = evoked.data

    if D.shape[0] != n_dipoles:
        raise ValueError(f"stc has {D.shape[0]} sources but forward has {n_dipoles}.")
    if M.shape[0] != n_chans:
        raise ValueError(f"evoked has {M.shape[0]} channels but forward has {n_chans}.")
    if D.shape[1] != M.shape[1]:
        raise ValueError(
            f"stc and evoked must have the same number of timepoints, got {D.shape[1]} and {M.shape[1]}."
        )

    D_FOCUSS = np.zeros_like(D)
    if verbose:
        logger.info("Shrinking FOCUSS:")

    for t in range(D.shape[1]):
        if verbose > 0:
            logger.info(f"Time step {t + 1}/{D.shape[1]}")

        x_last = np.asarray(D[:, t], dtype=float)
        weights = np.maximum(np.abs(x_last), _EPS)

        do_smoothing = True
        prev_sparsity = np.inf

        for i in range(max_iter):
            if verbose > 0:
                logger.info(f"Iteration {i + 1}/{max_iter}")

            x_new = _weighted_minimum_norm(leadfield, M[:, t], weights, alpha)

            if do_smoothing:
                x_smooth, sparsity = smooth(x_new, adjacency, percentile=percentile)
                if sparsity < n_chans or sparsity > prev_sparsity:
                    do_smoothing = False
                else:
                    prev_sparsity = sparsity
                    x_new = x_smooth

            rel_change = np.linalg.norm(x_new - x_last) / (
                np.linalg.norm(x_last) + _EPS
            )
            x_last = x_new
            weights = np.maximum(np.abs(x_last), _EPS)

            if rel_change < 1e-6:
                if verbose:
                    logger.info(f"Converged at iteration {i + 1}")
                break

        D_FOCUSS[:, t] = x_last

    stc_focuss = stc.copy()
    stc_focuss.data = D_FOCUSS
    return stc_focuss


def smooth(D_FOCUSS_t, adjacency, percentile=0.01):
    D_FOCUSS_t = np.asarray(D_FOCUSS_t)
    is_column_vector = D_FOCUSS_t.ndim == 2 and D_FOCUSS_t.shape[1] == 1
    x = D_FOCUSS_t[:, 0] if is_column_vector else D_FOCUSS_t.reshape(-1)

    adjacency = adjacency.toarray() if hasattr(adjacency, "toarray") else np.asarray(adjacency)
    n_dipoles = adjacency.shape[0]
    if x.shape[0] != n_dipoles:
        raise ValueError(
            f"adjacency has {n_dipoles} nodes but input has {x.shape[0]} entries."
        )

    max_val = np.abs(x).max()
    if max_val == 0:
        x_out = x.copy()
        return (x_out[:, np.newaxis] if is_column_vector else x_out), 0

    prominent_idc = np.where(np.abs(x) > max_val * percentile)[0]
    if len(prominent_idc) == 0:
        prominent_idc = np.array([int(np.argmax(np.abs(x)))], dtype=int)

    # Keep prominent sources and their neighbors. Note that MNE adjacency often has a
    # zero diagonal, so we add prominent indices explicitly.
    neighbor_mask = adjacency[prominent_idc, :].sum(axis=0) > 0
    neighbor_mask[prominent_idc] = True
    neighbor_cat = np.where(neighbor_mask)[0]

    x_smoothed = np.zeros_like(x)
    for idx in neighbor_cat:
        neighbor_idc = np.where(adjacency[idx, :] != 0)[0]
        if not np.any(neighbor_idc == idx):
            neighbor_idc = np.append(neighbor_idc, idx)
        x_smoothed[idx] = np.mean(x[neighbor_idc])

    sparsity = len(prominent_idc)
    return (x_smoothed[:, np.newaxis] if is_column_vector else x_smoothed), sparsity
