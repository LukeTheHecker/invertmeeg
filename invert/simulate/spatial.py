import mne
import numpy as np
from scipy.sparse import csr_matrix, vstack
from scipy.sparse.csgraph import laplacian


def build_adjacency(forward, verbose=0):
    """Build sparse adjacency matrix from an MNE forward solution.

    Parameters
    ----------
    forward : dict
        MNE forward solution object.
    verbose : int
        Verbosity level.

    Returns
    -------
    adjacency : csr_matrix
        Sparse adjacency matrix.
    """
    adjacency = mne.spatial_src_adjacency(forward["src"], verbose=verbose)
    return csr_matrix(adjacency)


def build_spatial_basis(adjacency, n_dipoles, min_order, max_order,
                        diffusion_smoothing=True, diffusion_parameter=0.1):
    """Build multi-order spatial basis from adjacency matrix.

    Parameters
    ----------
    adjacency : csr_matrix
        Sparse adjacency matrix.
    n_dipoles : int
        Number of dipoles (source space size).
    min_order : int
        Minimum smoothing order to include.
    max_order : int
        Maximum smoothing order (exclusive).
    diffusion_smoothing : bool
        Whether to use diffusion smoothing (True) or absolute Laplacian (False).
    diffusion_parameter : float
        Diffusion parameter alpha for smoothing.

    Returns
    -------
    sources : csr_matrix
        Stacked spatial basis (sparse).
    sources_dense : ndarray
        Dense version for fast indexing.
    gradient : csr_matrix
        The gradient/smoothing operator.
    """
    if diffusion_smoothing:
        gradient = np.identity(n_dipoles) - diffusion_parameter * laplacian(adjacency)
    else:
        gradient = abs(laplacian(adjacency))

    gradient = csr_matrix(gradient)

    # Build multi-order source basis using iterative sparse multiplication
    last_block = csr_matrix(np.identity(n_dipoles))
    blocks = [last_block]

    for _i in range(1, max_order):
        last_block = last_block @ gradient

        # Normalize each column (with guard against zero max)
        row_maxes = last_block.max(axis=0).toarray().flatten()
        row_maxes[row_maxes == 0] = 1.0
        last_block = last_block / row_maxes[np.newaxis]

        blocks.append(last_block)

    sources = vstack(blocks[min_order:])
    sources_dense = sources.toarray()

    return sources, sources_dense, gradient
