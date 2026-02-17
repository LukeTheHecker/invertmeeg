"""Shared Sparse Bayesian Learning (SBL) iteration engine.

This module extracts the common EM iteration loop from the Champagne-family
solvers (Champagne, FlexChampagne, OmniChampagne) into a reusable function.

Champagne (standard rules), FlexChampagne, and OmniChampagne use this engine
directly.  Complex Champagne variants (AR-EM, TEM, noise learning) retain
their own loops.

Typical usage::

    from invert.solvers.bayesian.sbl_core import sbl_iterate, GammaUpdateRule

    result = sbl_iterate(
        L=leadfield,
        Y=data,
        noise_cov=noise_cov,
        update_rule="mackay",
    )
    active_set = result.active_set
    gammas = result.gammas
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def _cholesky_inv(A):
    """Fast inverse for symmetric positive (semi-)definite matrices.

    Falls back to np.linalg.inv if Cholesky fails (e.g. near-singular or
    not quite PSD due to numerical noise).
    """
    n = A.shape[0]
    try:
        L, low = cho_factor(A)
        return cho_solve((L, low), np.eye(n))
    except np.linalg.LinAlgError:
        # Try with small diagonal loading proportional to trace
        eps = max(1e-10 * abs(np.trace(A)) / n, 1e-20)
        try:
            L, low = cho_factor(A + eps * np.eye(n))
            return cho_solve((L, low), np.eye(n))
        except np.linalg.LinAlgError:
            return np.linalg.inv(A + eps * np.eye(n))


@dataclasses.dataclass
class SBLResult:
    """Result of an SBL iteration run."""

    active_set: np.ndarray
    """Indices of surviving (non-pruned) atoms."""
    gammas: np.ndarray
    """Gamma values for surviving atoms."""
    mu_x: np.ndarray
    """Posterior mean for surviving atoms, shape (n_active, n_times)."""
    Sigma_y_inv: np.ndarray
    """Inverse model covariance at convergence."""
    loss: float
    """Final negative log-likelihood."""
    n_iters: int
    """Number of iterations executed."""


# ---------------------------------------------------------------------------
# Built-in gamma update rules
# ---------------------------------------------------------------------------


def _gamma_mackay(
    mu_x: np.ndarray,
    gammas: np.ndarray,
    z_diag: np.ndarray,
    n_times: int,
) -> np.ndarray:
    upper = np.mean(mu_x**2, axis=1)
    return upper / (gammas * z_diag + 1e-20)


def _gamma_convexity(
    mu_x: np.ndarray,
    gammas: np.ndarray,
    z_diag: np.ndarray,
    n_times: int,
) -> np.ndarray:
    upper = np.mean(mu_x**2, axis=1)
    return np.sqrt(upper / (z_diag + 1e-20))


def _gamma_em(
    mu_x: np.ndarray,
    gammas: np.ndarray,
    z_diag: np.ndarray,
    n_times: int,
) -> np.ndarray:
    diag_sigma_x = gammas - gammas**2 * z_diag
    return diag_sigma_x + np.mean(mu_x**2, axis=1)


GammaUpdateRule = Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]
"""Signature: (mu_x, gammas, z_diag, n_times) -> gammas_new."""

_BUILTIN_RULES: dict[str, GammaUpdateRule] = {
    "mackay": _gamma_mackay,
    "convexity": _gamma_convexity,
    "mm": _gamma_convexity,
    "em": _gamma_em,
}


def _resolve_update_rule(rule: str | GammaUpdateRule) -> GammaUpdateRule:
    if callable(rule):
        return rule
    key = str(rule).lower()
    if key not in _BUILTIN_RULES:
        raise ValueError(
            f"Unknown update rule {rule!r}. "
            f"Supported: {sorted(_BUILTIN_RULES.keys())} or a callable."
        )
    return _BUILTIN_RULES[key]


# ---------------------------------------------------------------------------
# Main iteration engine
# ---------------------------------------------------------------------------


def sbl_iterate(
    L: np.ndarray,
    Y: np.ndarray,
    noise_cov: np.ndarray,
    update_rule: str | GammaUpdateRule = "mackay",
    *,
    max_iter: int = 2000,
    prune: bool = True,
    pruning_thresh: float = 1e-3,
    conv_crit: float = 1e-8,
    inverse_fn: Callable[[np.ndarray], np.ndarray] = _cholesky_inv,
) -> SBLResult:
    """Run a generic SBL iteration loop.

    Parameters
    ----------
    L : ndarray (n_chans, n_atoms)
        Leadfield (or extended dictionary).
    Y : ndarray (n_chans, n_times)
        Data matrix.
    noise_cov : ndarray (n_chans, n_chans)
        Noise covariance.
    update_rule : str or callable
        Gamma update rule.  Built-in: ``"mackay"``, ``"convexity"``/``"mm"``,
        ``"em"``.  Or a callable with signature
        ``(mu_x, gammas, z_diag, n_times) -> gammas_new``.
    max_iter : int
        Maximum EM iterations.
    prune : bool
        Whether to prune inactive atoms each iteration.
    pruning_thresh : float
        Relative threshold: atoms with gamma < pruning_thresh * max(gamma)
        are pruned.
    conv_crit : float
        Relative change in loss below which iteration stops.
    inverse_fn : callable
        Function to invert a matrix, e.g. ``np.linalg.inv`` or a
        regularized variant.

    Returns
    -------
    SBLResult
        Dataclass with ``active_set``, ``gammas``, ``mu_x``,
        ``Sigma_y_inv``, ``loss``, ``n_iters``.
    """
    rule_fn = _resolve_update_rule(update_rule)
    n_chans, n_atoms = L.shape
    n_times = Y.shape[1]

    gammas = np.ones(n_atoms)
    active_set = np.arange(n_atoms)
    L_act = L.copy()

    # Initial posterior
    Sigma_y = noise_cov + (L_act * gammas) @ L_act.T
    Sigma_y = 0.5 * (Sigma_y + Sigma_y.T)
    Sigma_y_inv = inverse_fn(Sigma_y)
    SiL = Sigma_y_inv @ L_act  # cached for z_diag and mu_x
    mu_x = (SiL.T @ Y) * gammas[:, None]

    # Compute initial loss so we have a valid value even if the loop breaks
    # on iteration 0 (e.g. all gammas go to zero immediately).
    _data_fit_init = float(np.trace(Sigma_y_inv @ Y @ Y.T) / n_times)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        _sign_init, _log_det_init = np.linalg.slogdet(Sigma_y)
    if _sign_init <= 0 or not np.isfinite(_log_det_init):
        _log_det_init = np.finfo(float).max / 2
    final_loss = _data_fit_init + float(_log_det_init)

    loss_prev: float | None = None
    n_iters = 0

    for i_iter in range(max_iter):
        n_iters = i_iter + 1

        # Fisher information diagonal: diag(L.T @ Sigma_y_inv @ L)
        # SiL is cached from posterior recompute (or initial)
        z_diag = np.sum(L_act * SiL, axis=0)

        # Gamma update (rule-specific)
        gammas_new = rule_fn(mu_x, gammas, z_diag, n_times)

        # Cleanup
        gammas_new[~np.isfinite(gammas_new)] = 0.0
        gammas_new = np.maximum(gammas_new, 0.0)
        if float(np.linalg.norm(gammas_new)) == 0.0:
            break

        # Pruning
        if prune:
            thresh = pruning_thresh * float(gammas_new.max())
            keep = np.where(gammas_new > thresh)[0]
            if keep.size == 0:
                break
            active_set = active_set[keep]
            gammas = gammas_new[keep]
            L_act = L_act[:, keep]
        else:
            gammas = gammas_new

        # Recompute posterior after pruning
        Sigma_y = noise_cov + (L_act * gammas) @ L_act.T
        Sigma_y = 0.5 * (Sigma_y + Sigma_y.T)
        Sigma_y_inv = inverse_fn(Sigma_y)
        SiL = Sigma_y_inv @ L_act  # cached for z_diag in next iteration
        mu_x = (SiL.T @ Y) * gammas[:, None]

        # Loss: negative log-marginal-likelihood
        # slogdet is faster than eigvalsh for just the log-determinant
        data_fit = float(np.trace(Sigma_y_inv @ Y @ Y.T) / n_times)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sign, log_det = np.linalg.slogdet(Sigma_y)
        if sign <= 0 or not np.isfinite(log_det):
            log_det = np.finfo(float).max / 2
        loss = data_fit + float(log_det)
        final_loss = loss

        # Convergence check
        if loss_prev is not None:
            rel_change = (loss_prev - loss) / (abs(loss_prev) + 1e-20)
            if rel_change > 0 and rel_change < conv_crit:
                loss_prev = loss
                break
        loss_prev = loss

    return SBLResult(
        active_set=active_set,
        gammas=gammas,
        mu_x=mu_x,
        Sigma_y_inv=Sigma_y_inv,
        loss=final_loss,
        n_iters=n_iters,
    )
