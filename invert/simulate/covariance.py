import numpy as np


def compute_covariance(Y, cov_type="basic"):
    """Compute the covariance matrix of the data.

    Parameters
    ----------
    Y : numpy.ndarray
        The data matrix.
    cov_type : str
        The type of covariance matrix to compute. Options are 'basic' and 'SSM'. Default is 'basic'.

    Return
    ------
    C : numpy.ndarray
        The covariance matrix.
    """
    if cov_type == "basic":
        C = Y @ Y.T
    elif cov_type == "SSM":
        n_time = Y.shape[1]
        M_Y = Y.T @ Y
        YY = M_Y + 0.001 * (50 / n_time) * np.trace(M_Y) * np.eye(n_time)
        P_Y = (Y @ np.linalg.inv(YY)) @ Y.T
        C = P_Y.T @ P_Y
    elif cov_type == "riemann":
        msg = "Riemannian covariance is not yet implemented as a standalone function."
        raise NotImplementedError(msg)
    else:
        msg = "Covariance type not recognized. Use 'basic', 'SSM' or provide a custom covariance matrix."
        raise ValueError(msg)

    return C


def get_cov(n, corr_coef):
    """Generate a covariance matrix that is symmetric along the
    diagonal that correlates sources to a specified degree."""
    if corr_coef < 1:
        cov = np.ones((n, n)) * corr_coef + np.eye(n) * (1 - corr_coef)
        cov = np.linalg.cholesky(cov)
    else:
        # Make all signals be exactly the first one (perfectly coherent)
        cov = np.zeros((n, n))
        cov[:, 0] = 1
    return cov.T


def gen_correlated_sources(corr_coeff, T, Q):
    """Generate Q correlated sources with a specified correlation coefficient.
    The sources are generated as sinusoids with random frequencies and phases.

    Parameters
    ----------
    corr_coeff : float
        The correlation coefficient between the sources.
    T : int
        The number of time points in the sources.
    Q : int
        The number of sources to generate.

    Returns
    -------
    Y : numpy.ndarray
        The generated sources.
    """
    Cov = np.ones((Q, Q)) * corr_coeff + np.diag(
        np.ones(Q) * (1 - corr_coeff)
    )  # required covariance matrix
    freq = np.random.randint(10, 31, Q)  # random frequencies between 10Hz to 30Hz

    phases = 2 * np.pi * np.random.rand(Q)  # random phases
    t = np.linspace(10 * np.pi / T, 10 * np.pi, T)
    Signals = np.sqrt(2) * np.cos(
        2 * np.pi * freq[:, None] * t + phases[:, None]
    )  # the basic signals

    if corr_coeff < 1:
        A = np.linalg.cholesky(Cov).T  # cholesky Decomposition
        Y = A @ Signals
    else:  # Coherent Sources
        Y = np.tile(Signals[0, :], (Q, 1))

    return Y
