"""Riemannian covariance transport for leadfield adaptation.

Adapts a template forward model to an individual subject using only their
sensor covariance — no MRI, no BEM rebuild, no optimization. The transport
matrix W maps the subject's sensor covariance structure toward the template's,
and its inverse warps the template leadfield to match the individual's head
geometry.

Two full-transport maps are supported:

    **Legacy whitening/recoloring** (default):
        ``W_full = C_tpl^{1/2} C_s^{-1/2}`` so that ``W_full C_s W_fullᵀ = C_tpl``.

    **Optimal transport / Bures map** (recommended):
        the unique symmetric positive-definite map ``W_full`` such that
        ``W_full C_s W_full = C_tpl``.

Two modes of use:

    **Leadfield transport** (recommended):
        ``fwd_pseudo = riemannian_leadfield_transport(C_data, fwd_tpl, ...)``
        Creates L_pseudo = W⁻¹ @ L_template.  The solver then adapts its
        regularization (depth weighting, noise normalization) to the warped
        geometry.

    **Data transport** (for pipelines that cannot modify the forward model):
        ``W = compute_transport_matrix(C_data, G_tpl, ...)``
        Yields x_aligned = W @ x_subject, so the template inverse operator
        can be applied directly.

Leadfield transport is preferred because the solver re-derives its spatial
priors from the pseudo-leadfield, whereas data transport forces the solver
to use template priors on warped data.

**Key parameters:**

    alpha : float in [0, 1]
        Partial transport strength between identity (0) and full transport (1).
        Full transport often over-corrects by forcing exact covariance alignment.
        A robust default is alpha≈0.50 (protocol sweep, 2026-03-18).

    reg : float
        Diagonal loading (fraction of trace/n) applied to the *data* covariance
        before computing matrix powers. Set to 0 if C_data is already
        well-conditioned (e.g., shrinkage).

    tpl_reg : float
        Diagonal loading (fraction of trace/n) applied to the template
        covariance. This defaults to 0.01 because ``G@Gᵀ`` is rank-deficient by
        construction (n_src > n_ch).

    transport : {'whiten_recolor', 'bures'}
        Choice of full transport map (see above).

    alpha_mode : {'blend', 'geodesic'}
        How to apply alpha:
        - ``blend``: ``W = (1-α) I + α W_full`` (legacy)
        - ``geodesic``: ``W = W_full^α`` (SPD only; recommended with Bures map)

    template_prior : dict | None
        Optional source prior used to build the template covariance
        ``C_tpl = G Σ0 Gᵀ`` (e.g. graph smoothness).

References
----------
- Sabbagh et al. (2020). "Predictive regression modeling with MEG/EEG:
  from source power to signals and back." NeuroImage.
- Rodrigues et al. (2019). "Riemannian Procrustes analysis: transfer
  learning for brain-computer interfaces." IEEE TBME.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy import sparse
    from scipy.sparse.linalg import splu
except Exception:  # pragma: no cover
    sparse = None  # type: ignore[assignment]
    splu = None  # type: ignore[assignment]


def _sym_mat_pow(A: np.ndarray, power: float, *, eps: float = 1e-12) -> np.ndarray:
    """Symmetric matrix power for (near) SPD inputs.

    Uses an eigen-decomposition on the symmetrized matrix and clips
    eigenvalues to ``eps`` for numerical stability.
    """
    A = np.asarray(A, dtype=float)
    A = 0.5 * (A + A.T)
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, eps)
    return (eigvecs * (eigvals**power)) @ eigvecs.T


def _compute_template_covariance(
    G_tpl: np.ndarray,
    template_prior: dict | None,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute template sensor covariance C_tpl = G Σ0 Gᵀ under a source prior.

    Parameters
    ----------
    G_tpl
        Template leadfield, shape (n_ch, n_src).
    template_prior
        If ``None``: Σ0 = I, giving C_tpl = G Gᵀ.

        Otherwise a dict with a ``kind`` field:

        - ``{'kind': 'diag', 'weights': w}`` where w has shape (n_src,).
          Uses Σ0 = diag(w).
        - ``{'kind': 'graph_smooth', 'adjacency': A, 'tau': τ}``
          Uses a smoothness prior induced by a source-graph Laplacian L:
          P = (I + τ L)^{-1}, and Σ0 = P Pᵀ (so C_tpl = (G P)(G P)ᵀ).
          Here ``adjacency`` is a symmetric sparse matrix (n_src × n_src).

    Returns
    -------
    C_tpl
        Sensor covariance, shape (n_ch, n_ch).
    """
    if template_prior is None:
        return G_tpl @ G_tpl.T

    kind = str(template_prior.get("kind", "")).lower()
    if kind == "diag":
        weights = np.asarray(template_prior.get("weights"), dtype=float)
        if weights.ndim != 1 or weights.shape[0] != G_tpl.shape[1]:
            raise ValueError(
                f"diag prior weights must have shape (n_src,), got {weights.shape}"
            )
        weights = np.maximum(weights, 0.0)
        G_w = G_tpl * np.sqrt(weights)[None, :]
        return G_w @ G_w.T

    if kind == "graph_smooth":
        if sparse is None or splu is None:  # pragma: no cover
            raise RuntimeError(
                "graph_smooth prior requires scipy.sparse + scipy.sparse.linalg"
            )
        adj = template_prior.get("adjacency")
        if adj is None:
            raise ValueError("graph_smooth prior requires 'adjacency'")
        if not sparse.issparse(adj):
            adj = sparse.csr_matrix(adj)
        adj = adj.tocsr()
        if adj.shape[0] != adj.shape[1] or adj.shape[0] != G_tpl.shape[1]:
            raise ValueError(
                f"adjacency must have shape (n_src, n_src), got {adj.shape}"
            )
        tau = float(template_prior.get("tau", 0.0))
        if tau < 0:
            raise ValueError(f"tau must be >= 0, got {tau}")
        if tau == 0.0:
            return G_tpl @ G_tpl.T

        # Laplacian L = D - A
        deg = np.asarray(adj.sum(axis=1), dtype=float).ravel()
        L = sparse.diags(deg, format="csr", dtype=float) - adj.astype(float)
        A = sparse.identity(L.shape[0], format="csr", dtype=float) + tau * L

        # Compute G_sm = G @ (I + τL)^{-1} without forming the inverse:
        # Solve (I + τL) X = G.T  ->  X = (I + τL)^{-1} G.T
        lu = splu(A.tocsc())
        X = lu.solve(np.asarray(G_tpl.T, dtype=float))  # (n_src, n_ch)
        G_sm = np.asarray(X, dtype=float).T
        # C_tpl = G_sm G_sm.T  corresponds to Σ0 = P P^T with P=(I+τL)^{-1}.
        return G_sm @ G_sm.T

    raise ValueError(f"Unknown template_prior kind: {kind!r}")


def compute_transport_matrix(
    C_data: np.ndarray,
    G_tpl: np.ndarray,
    alpha: float = 0.80,
    reg: float = 0.01,
    *,
    transport: str = "bures",
    alpha_mode: str = "geodesic",
    template_prior: dict | None = None,
    tpl_reg: float = 0.01,
    eps: float = 1e-12,
):
    """Compute the (partial) Riemannian transport matrix W.

    W maps data from the subject's sensor space to the template's:
        x_aligned = W @ x_subject

    Its inverse warps the template leadfield to the subject:
        L_pseudo = W⁻¹ @ L_template

    Parameters
    ----------
    C_data : ndarray, shape (n_ch, n_ch)
        Subject's sensor covariance.  Can be a raw sample covariance or a
        shrinkage-regularized estimate (e.g. Ledoit-Wolf).  If already
        regularized, set ``reg=0`` to skip additional diagonal loading on
        C_data.
    G_tpl : ndarray, shape (n_ch, n_src)
        Template leadfield matrix (``fwd_tpl['sol']['data']``).
    alpha : float, default 0.50
        Blending: ``W = alpha * W_full + (1 - alpha) * I``.
        - 0.0: no transport (identity)
        - 1.0: full Riemannian transport
        - 0.50: optimal robustness across SNR and recording duration
    reg : float, default 0.01
        Diagonal loading as fraction of trace/n, added to C_data before
        matrix square root.  Set to 0 if C_data is already well-conditioned
        (e.g. after Ledoit-Wolf shrinkage).  The template covariance
        G@G.T always receives 0.01 regularization regardless of this
        parameter, because it is rank-deficient by construction
        (n_dipoles > n_channels).

    transport : {'whiten_recolor', 'bures'}
        Which full transport to compute.

        - ``'whiten_recolor'`` (legacy): ``W_full = C_tpl^{1/2} C_s^{-1/2}``.
          Maps covariances by congruence: ``W C_s Wᵀ = C_tpl``.
        - ``'bures'`` (recommended): the unique symmetric positive-definite
          optimal transport / Bures map ``W_full`` such that
          ``W_full C_s W_full = C_tpl``.

    alpha_mode : {'blend', 'geodesic'}
        How to apply the partial-transport strength ``alpha``.

        - ``'blend'`` (legacy): ``W = (1-α) I + α W_full``.
        - ``'geodesic'`` (recommended, SPD only): ``W = W_full^α`` using a
          symmetric matrix power (requires ``transport='bures'``).

    template_prior : dict | None
        Optional source prior used to build ``C_tpl`` (see
        ``_compute_template_covariance``). Defaults to ``None`` (Σ0 = I).

    tpl_reg : float
        Diagonal loading applied to the template covariance as fraction of
        trace/n. Unlike ``reg``, this is *always* applied by default because
        ``G@Gᵀ`` is rank-deficient (n_src > n_ch).

    eps : float
        Eigenvalue floor used in symmetric matrix powers.

    Returns
    -------
    W : ndarray, shape (n_ch, n_ch)
        Transport matrix.
    """
    n = G_tpl.shape[0]

    # Template geometric covariance: what sensors see from unit-variance
    # uncorrelated sources through the template geometry.
    C_tpl = _compute_template_covariance(G_tpl, template_prior, eps=eps)
    C_tpl += tpl_reg * np.trace(C_tpl) / n * np.eye(n)
    C_tpl /= np.trace(C_tpl) / n  # normalize to trace = n

    # Subject covariance (copy to avoid mutation)
    C_s = C_data.copy()
    if reg > 0:
        C_s += reg * np.trace(C_s) / n * np.eye(n)
    C_s /= np.trace(C_s) / n  # normalize to trace = n

    transport = str(transport).lower()
    alpha_mode = str(alpha_mode).lower()

    if transport in {"whiten_recolor", "legacy"}:
        # Full transport: W_full = C_tpl^½ @ C_s^{-½}
        W_full = _sym_mat_pow(C_tpl, 0.5, eps=eps) @ _sym_mat_pow(
            C_s, -0.5, eps=eps
        )
    elif transport in {"bures", "ot"}:
        # Optimal transport / Bures map (unique SPD solution).
        sqrt_Cs = _sym_mat_pow(C_s, 0.5, eps=eps)
        invsqrt_Cs = _sym_mat_pow(C_s, -0.5, eps=eps)
        middle = sqrt_Cs @ C_tpl @ sqrt_Cs
        W_full = invsqrt_Cs @ _sym_mat_pow(middle, 0.5, eps=eps) @ invsqrt_Cs
        W_full = 0.5 * (W_full + W_full.T)
    else:
        raise ValueError(f"Unknown transport: {transport!r}")

    if alpha_mode == "blend":
        W = alpha * W_full + (1 - alpha) * np.eye(n)
    elif alpha_mode == "geodesic":
        if transport not in {"bures", "ot"}:
            raise ValueError(
                "alpha_mode='geodesic' requires transport='bures' (SPD map)"
            )
        W = _sym_mat_pow(W_full, float(alpha), eps=eps)
    else:
        raise ValueError(f"Unknown alpha_mode: {alpha_mode!r}")

    return W


def riemannian_leadfield_transport(
    C_data: np.ndarray,
    fwd_tpl,
    alpha: float = 0.80,
    reg: float = 0.01,
    *,
    transport: str = "bures",
    alpha_mode: str = "geodesic",
    template_prior: dict | None = None,
    tpl_reg: float = 0.01,
    eps: float = 1e-12,
):
    """Create a pseudo forward model via Riemannian leadfield transport.

    Computes L_pseudo = W⁻¹ @ L_template where W is the partial Riemannian
    transport matrix.  The returned forward model can be passed directly to
    any solver — the solver will adapt its regularization to the warped
    geometry.

    Parameters
    ----------
    C_data : ndarray, shape (n_ch, n_ch)
        Subject's sensor covariance (see ``compute_transport_matrix``).
    fwd_tpl : mne.Forward
        Template forward model.  Not modified; a shallow copy is returned.
    alpha : float, default 0.80
        Transport blending (see ``compute_transport_matrix``).
        Compromise between MLE and EMD from protocol sweep (2026-03-19):
        α=0.80 gives MNI MLE 20.2 mm and EMD 28.7 mm.  For best EMD,
        use α=1.0 (EMD 28.0, MLE 20.2).  For best head-coordinate MLE,
        use α=0.5-0.6 (MLE 16.9).
    reg : float, default 0.01
        Diagonal loading (see ``compute_transport_matrix``).

    Returns
    -------
    fwd_pseudo : mne.Forward
        Copy of ``fwd_tpl`` with leadfield replaced by L_pseudo.
        Source positions, adjacency, and source space are unchanged
        (they belong to the template).
    """
    if template_prior is not None:
        kind = str(template_prior.get("kind", "")).lower()
        if kind == "graph_smooth" and template_prior.get("adjacency") is None:
            # Lazily build adjacency from the template source space to keep the
            # module import lightweight (MNE config writes can be expensive).
            try:
                from invert.util.source_adjacency import build_source_adjacency
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "graph_smooth template_prior requires MNE + "
                    "invert.util.source_adjacency"
                ) from exc

            adjacency_type = template_prior.get("adjacency_type", "spatial")
            adjacency_distance = template_prior.get("adjacency_distance", 3e-3)
            verbose = template_prior.get("verbose", 0)
            adjacency = build_source_adjacency(
                fwd_tpl["src"],
                adjacency_type=adjacency_type,
                adjacency_distance=adjacency_distance,
                verbose=verbose,
            )
            template_prior = dict(template_prior)
            template_prior["adjacency"] = adjacency

    G_tpl = fwd_tpl['sol']['data']
    W = compute_transport_matrix(
        C_data,
        G_tpl,
        alpha=alpha,
        reg=reg,
        transport=transport,
        alpha_mode=alpha_mode,
        template_prior=template_prior,
        tpl_reg=tpl_reg,
        eps=eps,
    )
    G_pseudo = np.linalg.solve(W, G_tpl)

    fwd_pseudo = fwd_tpl.copy()
    fwd_pseudo['sol'] = dict(fwd_pseudo['sol'])
    fwd_pseudo['sol']['data'] = G_pseudo
    if fwd_pseudo.get("_orig_sol") is not None:
        fwd_pseudo["_orig_sol"] = G_pseudo
    return fwd_pseudo
