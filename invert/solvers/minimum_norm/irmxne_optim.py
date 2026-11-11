"""irMxNE majorization with safe fixed/vector group-coordinate subproblems.

The outer equations follow Strohmeier et al. (2016) and MNE's BSD-3-Clause
``iterative_mixed_norm_solver``. MNE's fixed-orientation coordinate-descent
backend is used directly. Vector subproblems reuse invert's block updates:
unlike MNE 1.11's BCD shortcut, their activity check does not assume that a
block's first orientation/first time sample is nonzero. No dependency globals
are patched and no temporal/orientation perturbation is introduced.
"""

from __future__ import annotations

import numpy as np
from mne.inverse_sparse.mxne_debiasing import compute_bias
from mne.inverse_sparse.mxne_optim import mixed_norm_solver

from .mxne import _bcd_pass, _compute_lipschitz, _dgap_l21, _groups_norm2


def _vector_mixed_norm(data, gain, alpha, *, n_orient, maxit, tol, active_set_size):
    """Convex group Lasso with a global primal-dual gap and safe block updates."""
    n_positions = gain.shape[1] // n_orient
    x = np.zeros((gain.shape[1], data.shape[1]))
    residual = data.copy()
    lipschitz = _compute_lipschitz(gain, n_orient)
    correlation = np.sqrt(_groups_norm2(gain.T @ data, n_orient))
    candidates = np.flatnonzero((lipschitz > 0) & (correlation > alpha))
    if candidates.size == 0:
        return x[:0], np.zeros(gain.shape[1], dtype=bool), 0.0, 0
    active = set(
        candidates[np.argsort(correlation[candidates])[-active_set_size:]].tolist()
    )
    gap = np.inf
    for iteration in range(1, maxit + 1):
        active = _bcd_pass(
            gain, x, residual, sorted(active), lipschitz, n_orient, alpha
        )
        if iteration % 10 == 0 or iteration == maxit:
            residual = data - gain @ x
            gap, _ = _dgap_l21(gain, x, residual, alpha, n_orient)
            gap = max(float(gap), 0.0)
            if gap <= tol:
                break
            correlation = np.sqrt(_groups_norm2(gain.T @ residual, n_orient))
            candidates = [
                j
                for j in range(n_positions)
                if j not in active and lipschitz[j] > 0 and correlation[j] > alpha
            ]
            candidates.sort(key=lambda j: correlation[j], reverse=True)
            active.update(candidates[:active_set_size])
    active_mask = np.repeat(_groups_norm2(x, n_orient) > 0, n_orient)
    return x[active_mask], active_mask, gap, iteration


def iterative_mixed_norm_solver_safe(
    data,
    gain,
    alpha,
    n_mxne_iter,
    *,
    maxit=2000,
    tol=1e-6,
    debias=True,
    n_orient=1,
    active_set_size=10,
):
    """Minimize .5||Y-GX||² + alpha*sum_j sqrt(||X_j||_F).

    Initialize weights at one (the convex MxNE solution), then solve the
    majorizing weighted group-Lasso problems with inverse weights
    ``2*sqrt(||X_j||_F)``. Exactly zero groups remain inactive, as in the
    reference algorithm. Histories/gaps refer to estimates before optional
    amplitude debiasing. An unmet inner duality-gap tolerance terminates the
    outer loop and is reported explicitly. ``outer_converged`` means a stable
    support and max coefficient change <= tol after converged subproblems;
    it does not certify the global minimum of the nonconvex objective.
    """
    data = np.asarray(data, dtype=float)
    gain = np.asarray(gain, dtype=float)
    if data.ndim != 2 or gain.ndim != 2 or data.shape[0] != gain.shape[0]:
        raise ValueError("data and gain must be two-dimensional with matching sensors.")
    if n_orient not in (1, 3) or gain.shape[1] % n_orient:
        raise ValueError("n_orient must be 1 or 3 and divide gain's column count.")
    if not np.isfinite(data).all() or not np.isfinite(gain).all():
        raise ValueError("data and gain must be finite.")
    if not np.isfinite(alpha) or alpha < 0 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("alpha must be nonnegative and tol positive, both finite.")
    for value in (n_mxne_iter, maxit, active_set_size):
        if int(value) != value or value < 1:
            raise ValueError(
                "Iteration counts and active_set_size must be positive integers."
            )
    if alpha == 0:
        estimate = np.linalg.lstsq(gain, data, rcond=None)[0]
        active = np.repeat(_groups_norm2(estimate, n_orient) > 0, n_orient)
        return (
            estimate[active],
            active,
            [float(0.5 * np.sum((data - gain @ estimate) ** 2))],
            {
                "inner_duality_gaps": np.array([0.0]),
                "inner_converged": np.array([True]),
                "inner_iterations": [None],
                "outer_iterations": 1,
                "outer_converged": True,
                "outer_stop_reason": "unregularized_least_squares",
                "outer_convergence_criterion": "Unregularized least-squares solve; no majorization necessary.",
                "convex_backend": "numpy_lstsq",
            },
        )

    x = np.zeros((gain.shape[1], data.shape[1]))
    active = np.ones(gain.shape[1], dtype=bool)
    weights = np.ones(gain.shape[1])
    history, gaps, inner_iterations = [], [], []
    stop = "outer_iteration_limit"
    outer_converged = False
    for _iteration in range(int(n_mxne_iter)):
        previous = x.copy()
        previous_active = active.copy()
        weighted_gain = gain[:, active] * weights[active]
        if n_orient == 1:
            fitted, local_active, _, gap = mixed_norm_solver(
                data,
                weighted_gain,
                alpha,
                maxit=int(maxit),
                tol=tol,
                active_set_size=min(int(active_set_size), weighted_gain.shape[1]),
                debias=False,
                n_orient=1,
                solver="cd",
                return_gap=True,
                verbose=False,
            )
            inner_iterations.append(None)  # MNE does not expose this count.
        else:
            fitted, local_active, gap, inner_count = _vector_mixed_norm(
                data,
                weighted_gain,
                alpha,
                n_orient=n_orient,
                maxit=int(maxit),
                tol=tol,
                active_set_size=int(active_set_size),
            )
            inner_iterations.append(inner_count)
        active[previous_active] = local_active
        x = np.zeros_like(x)
        x[active] = fitted * weights[active, None]
        group_norms = np.sqrt(_groups_norm2(x, n_orient))
        history.append(
            float(
                0.5 * np.sum((data - gain @ x) ** 2)
                + alpha * np.sum(np.sqrt(group_norms))
            )
        )
        gaps.append(float(gap))
        if not np.isfinite(gap) or gap > tol:
            stop = "inner_iteration_limit"
            break
        if not np.any(active):
            stop = "empty_support"
            outer_converged = True
            break
        if (
            _iteration >= 1
            and np.array_equal(active, previous_active)
            and np.max(np.abs(x - previous)) <= tol
        ):
            stop = "fixed_point"
            outer_converged = True
            break
        weights = np.repeat(2 * np.sqrt(group_norms), n_orient)

    fitted = x[active]
    if debias and np.any(active):
        fitted = (
            fitted
            * compute_bias(data, gain[:, active], fitted, n_orient=n_orient)[:, None]
        )
    diagnostics = {
        "inner_duality_gaps": np.asarray(gaps),
        "inner_converged": np.asarray([np.isfinite(g) and g <= tol for g in gaps]),
        "inner_iterations": inner_iterations,
        "outer_iterations": len(history),
        "outer_converged": outer_converged,
        "outer_stop_reason": stop,
        "outer_convergence_criterion": "Stable support and maximum coefficient change <= tol, after converged convex subproblems; not a global-optimum certificate.",
        "convex_backend": "mne_coordinate_descent"
        if n_orient == 1
        else "invert_safe_block_coordinate_descent",
    }
    return fitted, active, history, diagnostics
