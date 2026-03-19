import numpy as np

from invert.solvers.minimum_norm.sslofo import SolverSSLOFO


def _reference_compute_sloreta(leadfield, data, alpha):
    data_arr = np.asarray(data, dtype=float)
    data_was_vector = data_arr.ndim == 1
    if data_was_vector:
        data_arr = data_arr[:, np.newaxis]

    ll_t = leadfield @ leadfield.T
    system = ll_t + alpha * np.eye(leadfield.shape[0])
    try:
        inner_inv = np.linalg.inv(system)
    except np.linalg.LinAlgError:
        inner_inv = np.linalg.pinv(system)

    k_mne = leadfield.T @ inner_inv
    res_matrix_diag = np.sum(k_mne * leadfield.T, axis=1)
    res_matrix_diag = np.maximum(res_matrix_diag, 1e-12)
    j_sloreta = (k_mne @ data_arr) / np.sqrt(res_matrix_diag)[:, np.newaxis]
    return j_sloreta[:, 0] if data_was_vector else j_sloreta


def _reference_standardized_focuss_update(leadfield, data, weights, alpha):
    data_arr = np.asarray(data, dtype=float)
    lw = leadfield * weights
    system = lw @ leadfield.T + alpha * np.eye(leadfield.shape[0])
    try:
        inner_inv = np.linalg.inv(system)
    except np.linalg.LinAlgError:
        inner_inv = np.linalg.pinv(system)

    lt_inv = leadfield.T @ inner_inv
    k_eff = weights[:, np.newaxis] * lt_inv
    j = k_eff @ data_arr
    res_matrix_diag = np.sum(k_eff * leadfield.T, axis=1)
    res_matrix_diag = np.maximum(res_matrix_diag, 1e-12)
    return j / np.sqrt(res_matrix_diag)


def _reference_shrink_and_smooth(
    J, active_idx, adjacency, percentile, smoothing_weight
):
    max_val = np.abs(J).max()
    if max_val == 0:
        return np.ones(len(J), dtype=bool), J

    threshold = max_val * percentile
    prominent_local_idx = np.where(np.abs(J) > threshold)[0]
    if len(prominent_local_idx) == 0:
        prominent_local_idx = np.array([np.argmax(np.abs(J))])

    prominent_global_idx = active_idx[prominent_local_idx]
    neighbor_mask = adjacency[prominent_global_idx, :].sum(axis=0) > 0
    neighbor_global_idx = np.where(neighbor_mask)[0]
    keep_global_idx = np.unique(
        np.concatenate([prominent_global_idx, neighbor_global_idx])
    )
    prominent_mask = np.isin(active_idx, keep_global_idx)

    J_smoothed = J.copy()
    if smoothing_weight > 0:
        smooth_local_idx = np.where(prominent_mask)[0]
        smooth_global_idx = active_idx[smooth_local_idx]
        global_to_local = np.full(adjacency.shape[0], -1, dtype=np.int32)
        global_to_local[active_idx] = np.arange(len(active_idx))

        for local_idx, global_idx in zip(
            smooth_local_idx, smooth_global_idx, strict=False
        ):
            neighbor_global = adjacency[global_idx, :].nonzero()[0]
            neighbor_local = global_to_local[neighbor_global]
            neighbor_local = neighbor_local[neighbor_local >= 0]
            if len(neighbor_local) > 0:
                neighbor_avg = np.mean(J[neighbor_local])
                J_smoothed[local_idx] = (1 - smoothing_weight) * J[
                    local_idx
                ] + smoothing_weight * neighbor_avg

    return prominent_mask, J_smoothed


def _reference_apply_final_smoothing(J, adjacency):
    J_smoothed = J.copy()
    active_sources = np.where(np.abs(J) > 0)[0]

    for idx in active_sources:
        neighbors = np.where(adjacency[idx, :] == 1)[0]
        if len(neighbors) > 0:
            neighbor_vals = J[neighbors]
            J_smoothed[idx] = 0.7 * J[idx] + 0.3 * np.mean(neighbor_vals)

    return J_smoothed


def _reference_weighted_minimum_norm(leadfield, data, weights, alpha):
    data_arr = np.asarray(data, dtype=float)
    lw = leadfield * weights
    system = lw @ leadfield.T + alpha * np.eye(leadfield.shape[0])
    try:
        inner_inv = np.linalg.inv(system)
    except np.linalg.LinAlgError:
        inner_inv = np.linalg.pinv(system)

    j = weights[:, np.newaxis] * (leadfield.T @ (inner_inv @ data_arr))
    return j.squeeze()


def _reference_solve_single_timepoint(solver, data_t, alpha, adjacency):
    leadfield = solver.leadfield_full
    n_sources = leadfield.shape[1]
    j_sloreta = _reference_compute_sloreta(leadfield, data_t, solver.sloreta_alpha)

    w_full = np.abs(j_sloreta) ** solver.focuss_power
    w_full = np.maximum(w_full, 1e-12)

    active_idx = np.arange(n_sources)
    j_current = j_sloreta.copy()

    for _ in range(solver.max_iter):
        w_active = w_full[active_idx]
        j_new = _reference_standardized_focuss_update(
            leadfield[:, active_idx], data_t, w_active, alpha
        )

        prominent_mask, j_smoothed = _reference_shrink_and_smooth(
            j_new,
            active_idx,
            adjacency,
            solver.percentile_threshold,
            solver.smoothing_weight,
        )

        j_full = np.zeros(n_sources)
        j_full[active_idx] = j_smoothed
        change = np.linalg.norm(j_full - j_current) / (
            np.linalg.norm(j_current) + 1e-12
        )
        j_current = j_full.copy()
        new_active_idx = active_idx[prominent_mask]

        if change < solver.convergence_tol:
            break
        if len(new_active_idx) == 0:
            break
        if (
            solver.sparsity_threshold is not None
            and len(new_active_idx) < solver.sparsity_threshold
        ):
            break
        if len(new_active_idx) >= len(active_idx):
            break

        active_idx = new_active_idx
        w_full = np.abs(j_current) ** solver.focuss_power
        w_full = np.maximum(w_full, 1e-12)

    if solver.final_smoothing:
        return _reference_apply_final_smoothing(j_current, adjacency)
    return j_current


def _reference_solve_spatiotemporal(solver, data, alpha, adjacency):
    n_sources = solver.leadfield_full.shape[1]
    n_times = data.shape[1]
    window_size = n_times if solver.time_window is None else min(solver.time_window, n_times)

    solutions = []
    for t in range(window_size):
        solutions.append(
            _reference_solve_single_timepoint(solver, data[:, t], alpha, adjacency)
        )

    j_sum = np.sum(np.abs(solutions), axis=0)
    support_mask = j_sum > 0
    support_idx = np.where(support_mask)[0]
    if len(support_idx) == 0:
        return np.zeros((n_sources, n_times))

    source_mat = np.zeros((n_sources, n_times))
    leadfield_support = solver.leadfield_full[:, support_idx]
    weights = np.maximum(j_sum[support_idx] ** solver.focuss_power, 1e-12)
    j_all = _reference_weighted_minimum_norm(leadfield_support, data, weights, alpha)
    if j_all.ndim == 1:
        j_all = j_all[:, np.newaxis]
    source_mat[support_idx, :] = j_all
    return source_mat


def test_sslofo_standardized_update_matches_dense_reference():
    rng = np.random.RandomState(0)
    leadfield = rng.randn(9, 14)
    data = rng.randn(9)
    weights = np.maximum(np.abs(rng.randn(14)), 1e-3)
    alpha = 0.2

    solver = SolverSSLOFO(n_reg_params=1)
    actual = solver._standardized_focuss_update(leadfield, data, weights, alpha)
    expected = _reference_standardized_focuss_update(leadfield, data, weights, alpha)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_sslofo_sparse_smoothing_matches_dense_reference():
    solver = SolverSSLOFO(n_reg_params=1)
    adjacency = np.array(
        [
            [0, 1, 1, 0, 0, 0],
            [1, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
        ],
        dtype=float,
    )
    active_idx = np.array([0, 1, 3, 4, 5])
    j = np.array([0.1, -2.0, 0.3, 1.2, -0.5], dtype=float)

    actual_mask, actual_j = solver._shrink_and_smooth(
        j,
        active_idx,
        adjacency,
        percentile=0.2,
        smoothing_weight=0.35,
    )
    expected_mask, expected_j = _reference_shrink_and_smooth(
        j,
        active_idx,
        adjacency,
        percentile=0.2,
        smoothing_weight=0.35,
    )

    np.testing.assert_array_equal(actual_mask, expected_mask)
    np.testing.assert_allclose(actual_j, expected_j, rtol=1e-12, atol=1e-12)


def test_sslofo_final_smoothing_matches_dense_reference():
    solver = SolverSSLOFO(n_reg_params=1)
    adjacency = np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=float,
    )
    j = np.array([1.0, 0.5, 0.0, -0.2], dtype=float)

    actual = solver._apply_final_smoothing(j, adjacency)
    expected = _reference_apply_final_smoothing(j, adjacency)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_sslofo_spatiotemporal_matches_dense_reference(forward_model, simulated_evoked):
    solver = SolverSSLOFO(
        n_reg_params=1,
        max_iter=8,
        spatial_temporal=True,
        time_window=6,
        percentile_threshold=0.05,
        smoothing_weight=0.1,
        final_smoothing=False,
    )
    solver.make_inverse_operator(forward_model, simulated_evoked, alpha=0.1)

    alpha = solver.alphas[0]
    data = solver._sensor_transform @ simulated_evoked.data
    adjacency = solver.adjacency.toarray()

    actual = solver.apply_inverse_operator(simulated_evoked).data
    expected = _reference_solve_spatiotemporal(solver, data, alpha, adjacency)

    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)
