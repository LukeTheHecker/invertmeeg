from copy import deepcopy

import mne
import numpy as np
from scipy.sparse.csgraph import laplacian

from ..util import calc_residual_variance

_EPS = 1e-12


def _select_residual_components(s, *, max_components=5, energy_threshold=0.9):
    """Select how many residual components to use for pursuit scoring."""
    max_components = int(max_components)
    if max_components <= 0:
        return 1
    if s.size == 0 or float(s.sum()) <= 0:
        return 1
    energy = np.cumsum(s) / float(s.sum())
    n_components = int(np.searchsorted(energy, float(energy_threshold)) + 1)
    n_components = max(1, min(max_components, n_components))
    return n_components


def _ridge_refit_on_support(leadfield_support, data, *, reg_scale=1e-3):
    """Ridge-regularized refit on a restricted support using the dual form.

    Solves, for each timepoint, the Tikhonov-regularized least squares problem:
        argmin_x ||Lx - y||_2^2 + reg ||x||_2^2
    in the dual form:
        x = L^T (L L^T + reg I)^-1 y
    which is efficient when n_channels << n_active_sources.
    """
    n_chans, n_active = leadfield_support.shape
    if n_chans == 0 or n_active == 0:
        return np.zeros((n_active, data.shape[1]))

    C = leadfield_support @ leadfield_support.T
    # Scale regularization to the matrix spectrum to avoid unit issues.
    s = np.linalg.svd(C, compute_uv=False)
    s0 = float(s[0]) if s.size else 0.0
    if s0 == 0.0:
        return np.zeros((n_active, data.shape[1]))

    reg = float(reg_scale) * s0
    I = np.eye(n_chans, dtype=C.dtype)

    # Be robust to singular/ill-conditioned C.
    for _ in range(6):
        try:
            Z = np.linalg.solve(C + reg * I, data)
            break
        except np.linalg.LinAlgError:
            reg *= 10.0
    else:
        Z = np.linalg.pinv(C + reg * I) @ data

    return leadfield_support.T @ Z


def stampc(
    stc, evoked, forward, max_iter=25, K=1, rv_thresh=0.1, n_orders=0, verbose=0
):
    """Spatio-Temporal Matching Pursuit Contextualizer (STAMP-C)

    Parameters
    ----------
    stc : mne.SourceEstimate
        Source Estimate object.
    evoked : mne.EvokedArray
        Evoked EEG data object
    forward : mne.Forward
        The forward model
    verbose : int
        Controls verbosity of the program

    Return
    ------
    stc_stampc : mne.SourceEstimate
        Contextualized source estimate built by iteratively expanding a sparse
        support (matching pursuit on the residual) and refitting on that support
        with ridge regularization.
    """
    D = stc.data
    M = evoked.data
    # M = leadfield @ D
    # M -= M.mean(axis=0)
    n_dipoles, n_time = D.shape

    leadfield = forward["sol"]["data"].copy()
    n_chans, n_dipoles_fwd = leadfield.shape
    if n_dipoles_fwd != n_dipoles:
        raise ValueError(
            f"stc has {n_dipoles} sources but forward has {n_dipoles_fwd}."
        )
    if M.shape != (n_chans, n_time):
        raise ValueError(
            f"evoked.data must have shape ({n_chans}, {n_time}), got {M.shape}."
        )

    # Work on a re-referenced copy of the data for consistency with later
    # re-referencing of X_hat.
    M0 = M - M.mean(axis=0)

    leadfield -= leadfield.mean(axis=0)
    leadfield_norms = np.linalg.norm(leadfield, axis=0)
    leadfield_norms = np.maximum(leadfield_norms, _EPS)
    leadfield_norm = leadfield / leadfield_norms

    leadfields_norm = [
        leadfield_norm,
    ]
    neighbors_bases = [
        np.arange(n_dipoles, dtype=int),
    ]

    # Compute Leadfield bases and corresponding neighbors
    adjacency = mne.spatial_src_adjacency(forward["src"], verbose=0)
    for _order in range(n_orders):
        laplace_operator = laplacian(adjacency)

        leadfield_smooth = leadfield @ abs(laplace_operator)
        leadfield_smooth -= leadfield_smooth.mean(axis=0)
        leadfield_smooth_norms = np.linalg.norm(leadfield_smooth, axis=0)
        leadfield_smooth_norms = np.maximum(leadfield_smooth_norms, _EPS)
        leadfield_smooth_norm = leadfield_smooth / leadfield_smooth_norms

        neighbors_base = [np.where(adj != 0)[0] for adj in adjacency.toarray()]

        leadfields_norm.append(leadfield_smooth_norm)
        neighbors_bases.append(neighbors_base)

        adjacency = adjacency @ adjacency.T

    # Compute the re-weighting gamma factor from the
    # existing source estimate
    # Normalize each individual source
    time_norms = np.linalg.norm(D, axis=0)
    time_norms = np.maximum(time_norms, _EPS)
    y_hat_model = D / time_norms

    # Compute average source and normalize resulting vector
    gammas_model = np.mean(abs(y_hat_model), axis=1)
    gammas_model_max = gammas_model.max()
    if gammas_model_max > 0:
        gammas_model /= gammas_model_max
    else:
        gammas_model = np.ones_like(gammas_model)

    # Get initial orthogonal leadfield components
    R = deepcopy(M0)

    residual_norms = [
        1e99,
    ]

    if K <= 0:
        raise ValueError("K must be a positive integer.")
    K = min(int(K), n_dipoles)

    idc = np.array([], dtype=int)
    y_hat = np.zeros((n_dipoles, n_time))

    for _i in range(max_iter):
        # Calculate leadfield components of the dominant residual subspace.
        sigma_R = R @ R.T
        U, s, _ = np.linalg.svd(sigma_R, full_matrices=False)
        n_components = _select_residual_components(s, max_components=5, energy_threshold=0.9)
        n_components = min(n_components, U.shape[1])
        U_r = U[:, :n_components]

        # Select Gammas of Matching pursuit using the orthogonal leadfield:
        # gammas_mp = abs(leadfield_norm.T @ U[:, 0] )

        # Select the most informativ basis of Gammas
        gammas_bases = [
            np.linalg.norm(L_norm.T @ U_r, axis=1) for L_norm in leadfields_norm
        ]
        basis_idx = int(
            np.argmax([float(gammas_base.max()) for gammas_base in gammas_bases])
        )
        # basis_idx = np.argmax([np.mean(gammas_base) for gammas_base in gammas_bases])
        gammas_mp = gammas_bases[basis_idx]
        neighbors_base = neighbors_bases[basis_idx]

        gammas_mp_max = gammas_mp.max()
        if gammas_mp_max > 0:
            gammas_mp /= gammas_mp_max

        # Combine leadfield components with the source-gamma
        gammas = gammas_model * gammas_mp if gammas_mp_max > 0 else gammas_model
        # gammas = gammas_mp + (gammas_model * gammas_mp)
        # gammas = gammas_mp

        # Select the K dipoles with highest correlation (probability)
        idx = np.argsort(gammas)[-K:]
        if isinstance(neighbors_base, np.ndarray):
            idx_expanded = idx
        else:
            expanded = []
            for idxx in idx:
                expanded.append(int(idxx))
                expanded.extend(
                    np.asarray(neighbors_base[int(idxx)], dtype=int).ravel().tolist()
                )
            idx_expanded = np.asarray(expanded, dtype=int)

        # Add the new dipoles to the existing set of dipoles
        idc = np.unique(np.concatenate([idc, idx_expanded])).astype(int)

        # Inversion: ridge-regularized refit restricted to current support.
        leadfield_support = leadfield[:, idc]
        y_hat = np.zeros((n_dipoles, n_time))
        y_hat[idc] = _ridge_refit_on_support(leadfield_support, M0, reg_scale=1e-3)

        X_hat = leadfield @ y_hat
        # Rereference predicted EEG
        X_hat -= X_hat.mean(axis=0)

        # Calculate Residual
        R = M0 - X_hat

        # Calculate the norm of the EEG-Residual
        residual_norm = np.linalg.norm(R)
        residual_norms.append(residual_norm)
        # Calculate the percentage of residual variance
        rv = calc_residual_variance(X_hat, M0)
        # print(i, " Res var: ", round(rv, 2))
        if rv < rv_thresh or np.isclose(residual_norms[-2], residual_norms[-1]):
            break

    stc_stampc = stc.copy()
    stc_stampc.data = y_hat
    return stc_stampc
