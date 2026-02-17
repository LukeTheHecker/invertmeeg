"""General-purpose covariance regularization utilities.

These functions are used by beamformers and can also serve non-beamformer
solvers that need covariance regularization (e.g., Bayesian, subspace methods).

Canonical imports::

    from invert.solvers.regularization import (
        diag_loading_trace,
        oas_shrink_covariance,
        condition_number_loaded_covariance,
        build_covariance_candidates,
    )

For backward compatibility, ``invert.solvers.beamformers.utils`` re-exports
all symbols from this module.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Primitive regularization operations
# ---------------------------------------------------------------------------


def diag_loading_trace(C: np.ndarray, beta: float) -> tuple[np.ndarray, float]:
    """Trace-scaled diagonal loading: C + (beta * tr(C)/m) I."""
    C = np.asarray(C)
    m = int(C.shape[0])
    tr = float(np.real(np.trace(C)))
    if not np.isfinite(tr) or tr <= 0:
        tr = float(np.real(np.mean(np.diag(C))))
    scale = tr / float(max(m, 1))
    lam = float(beta) * float(scale)
    return C + lam * np.eye(m, dtype=C.dtype), lam


def oas_shrink_covariance(C: np.ndarray, *, n_samples: int) -> tuple[np.ndarray, float]:
    """Oracle Approximating Shrinkage (OAS) toward scaled identity.

    Returns (shrunk_cov, shrinkage) where:
      C_oas = (1 - shrinkage) * C + shrinkage * tr(C)/m * I
    """
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"C must be square, got shape {C.shape}")
    C = 0.5 * (C + C.T)

    m = int(C.shape[0])
    n = int(max(n_samples, 2))
    tr = float(np.trace(C))
    tr2 = float(np.sum(C * C))  # == trace(C @ C) for symmetric C

    if m <= 1 or not np.isfinite(tr) or not np.isfinite(tr2):
        return C, 0.0

    mu = tr / float(m)
    den = (n + 1.0 - 2.0 / float(m)) * (tr2 - float(m) * mu * mu)
    if den <= 0:
        shrinkage = 0.0
    else:
        num = (1.0 - 2.0 / float(m)) * tr2 + tr * tr
        shrinkage = float(np.clip(num / den, 0.0, 1.0))

    shrunk = (1.0 - shrinkage) * C
    shrunk.flat[:: m + 1] += shrinkage * mu
    return shrunk, shrinkage


def condition_number_loaded_covariance(
    C: np.ndarray,
    *,
    cond_target: float,
    eps: float = 1e-15,
) -> tuple[np.ndarray, float]:
    """Choose lambda so cond(C + lambda*I) <= cond_target (via eigenvalue shift).

    The epsilon is applied *relative* to the largest eigenvalue so that
    this function works correctly regardless of the absolute scale of ``C``
    (e.g. MEG covariances with eigenvalues ~ 1e-23).
    """
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"C must be square, got shape {C.shape}")
    C = 0.5 * (C + C.T)
    m = int(C.shape[0])
    if m <= 1:
        return C, 0.0
    kappa = float(cond_target)
    if not np.isfinite(kappa) or kappa <= 1.0:
        raise ValueError(f"cond_target must be finite and > 1, got {cond_target!r}")

    evals = np.linalg.eigvalsh(C)
    lmax = float(np.max(evals))
    lmin = float(np.min(evals))
    lmax = max(lmax, 0.0)
    lmin = max(lmin, 0.0)
    # Use a relative epsilon so the floor scales with the matrix.
    eps_abs = float(eps) * max(lmax, np.finfo(float).tiny)
    if lmax <= eps_abs:
        return C + eps_abs * np.eye(m), eps_abs

    # Need lambda such that (lmax+lambda)/(lmin+lambda) <= kappa.
    if lmin > 0 and (lmax / lmin) <= kappa:
        lam = 0.0
    else:
        lam = (lmax - kappa * lmin) / (kappa - 1.0)
        lam = float(max(lam, 0.0))

    return C + lam * np.eye(m), lam


# ---------------------------------------------------------------------------
# Pipeline parsing + covariance candidate builder
# ---------------------------------------------------------------------------


def _parse_cov_reg_pipeline(cov_reg: str | list[str] | tuple[str, ...]) -> list[str]:
    """Normalize regularization pipeline tokens."""
    if isinstance(cov_reg, (list, tuple)):
        raw_tokens = [str(t).strip().lower() for t in cov_reg if str(t).strip()]
    else:
        text = str(cov_reg).strip().lower()
        for sep in (",", "|", "/", "->"):
            text = text.replace(sep, "+")
        raw_tokens = [tok.strip() for tok in text.split("+") if tok.strip()]

    if not raw_tokens:
        return ["oas"]

    normalized: list[str] = []
    for tok in raw_tokens:
        if tok in {"grid", "legacy", "maxeig"}:
            normalized.append("grid")
        elif tok in {"trace", "diag", "loading", "diag_loading"}:
            normalized.append("trace")
        elif tok in {"oas"}:
            normalized.append("oas")
        elif tok in {"cond", "cond_target", "kappa"}:
            normalized.append("cond")
        else:
            raise ValueError(
                "Unknown covariance regularization token "
                f"{tok!r}. Supported: grid, oas, trace, cond."
            )

    if "grid" in normalized and len(normalized) > 1:
        raise ValueError("cov_reg='grid' cannot be combined with other strategies.")
    return normalized


def build_covariance_candidates(
    *,
    C: np.ndarray,
    I: np.ndarray,
    alpha: str | float,
    get_alphas_fn,
    n_samples: int,
    cov_reg: str | list[str] | tuple[str, ...] = "oas",
    cov_reg_beta: float = 0.05,
    cov_reg_cond_target: float = 1e4,
) -> tuple[list[np.ndarray], list[float], dict[str, float]]:
    """Build one or more covariance matrices from a regularization policy."""
    if alpha != "auto":
        alphas = list(get_alphas_fn(reference=C))
        cov_mats = [C + float(alpha_eff) * I for alpha_eff in alphas]
        return cov_mats, [float(a) for a in alphas], {"mode": 0.0}

    pipeline = _parse_cov_reg_pipeline(cov_reg)
    if pipeline == ["grid"]:
        alphas = list(get_alphas_fn(reference=C))
        cov_mats = [C + float(alpha_eff) * I for alpha_eff in alphas]
        return cov_mats, [float(a) for a in alphas], {"mode": 0.0}

    C_reg = np.asarray(C, dtype=float)
    meta: dict[str, float] = {}
    for step in pipeline:
        if step == "oas":
            C_reg, shrink = oas_shrink_covariance(C_reg, n_samples=int(n_samples))
            meta["oas_shrinkage"] = float(shrink)
        elif step == "trace":
            C_reg, lam = diag_loading_trace(C_reg, float(cov_reg_beta))
            meta["diag_lambda"] = float(lam)
        elif step == "cond":
            C_reg, lam = condition_number_loaded_covariance(
                C_reg, cond_target=float(cov_reg_cond_target)
            )
            meta["cond_lambda"] = float(lam)

    alpha_eff = float(
        meta.get(
            "diag_lambda",
            meta.get("cond_lambda", 0.0),
        )
    )
    return [C_reg], [alpha_eff], meta
