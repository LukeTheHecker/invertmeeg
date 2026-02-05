import logging
from copy import deepcopy

import numpy as np

logger = logging.getLogger(__name__)


def soft_threshold(x, alpha):
    """Applies the soft thresholding operator to x with threshold alpha.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Input array.
    alpha : float
        Threshold.

    Returns
    -------
    y : ndarray, shape (n,)
        Output array.
    """
    y = np.sign(x) * np.maximum(np.abs(x) - alpha, 0)
    return y


def calc_eloreta_D2(leadfield, noise_cov, alpha, stop_crit=0.005, verbose=0):
    """Algorithm that optimizes weight matrix D as described in
    Assessing interactions in the brain with exactlow-resolution electromagnetic tomography; Pascual-Marqui et al. 2011 and
    https://www.sciencedirect.com/science/article/pii/S1053811920309150
    """
    n_chans, n_dipoles = leadfield.shape
    # initialize weight matrix D with identity and some empirical shift (weights are usually quite smaller than 1)
    D = np.identity(n_dipoles)

    if verbose > 0:
        logger.info("Optimizing eLORETA weight matrix W...")
    cnt = 0
    while True:
        old_D = deepcopy(D)
        if verbose > 0:
            logger.debug(f"\trep {cnt + 1}")
        D_inv = np.linalg.inv(D)
        inner_term = np.linalg.inv(
            leadfield @ D_inv @ leadfield.T + alpha**2 * noise_cov
        )

        for v in range(n_dipoles):
            D[v, v] = np.sqrt(leadfield[:, v].T @ inner_term @ leadfield[:, v])

        averagePercentChange = np.abs(
            1 - np.mean(np.divide(np.diagonal(D), np.diagonal(old_D)))
        )

        if verbose > 0:
            logger.debug(f"averagePercentChange={100 * averagePercentChange:.2f} %")

        if averagePercentChange < stop_crit:
            if verbose > 0:
                logger.info("...converged...")
            break
        cnt += 1
    if verbose > 0:
        logger.info("...done!")
    return D
