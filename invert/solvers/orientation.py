from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np
import numpy.typing as npt
from mne.io.constants import FIFF
from numpy.lib.stride_tricks import as_strided


@dataclass(frozen=True)
class LeadfieldView:
    """Lightweight view helpers for block-structured leadfields.

    Parameters
    ----------
    G
        Leadfield of shape ``(n_chans, n_orient * n_locs)``.
    n_orient
        Number of orientation components per location (typically 3).
    """

    G: npt.NDArray[np.floating]
    n_orient: int

    @property
    def n_locs(self) -> int:
        G = np.asarray(self.G)
        if G.ndim != 2:
            raise ValueError(f"G must be 2D, got ndim={G.ndim}")
        n_orient = int(self.n_orient)
        if n_orient <= 0:
            raise ValueError(f"n_orient must be > 0, got {n_orient}")
        n_cols = int(G.shape[1])
        if n_cols % n_orient != 0:
            raise ValueError(
                f"G has {n_cols} columns which is not divisible by n_orient={n_orient}"
            )
        return n_cols // n_orient

    def blocks(self) -> npt.NDArray[np.floating]:
        """Return a (view) of shape ``(n_chans, n_locs, n_orient)``.

        The mapping is ``blocks[:, i, j] == G[:, i*n_orient + j]``.
        """

        G = np.asarray(self.G)
        if G.ndim != 2:
            raise ValueError(f"G must be 2D, got ndim={G.ndim}")
        m = int(G.shape[0])
        n_locs = int(self.n_locs)
        n_orient = int(self.n_orient)

        # Use explicit stride construction to avoid copies for Fortran-contiguous
        # leadfields (MNE forwards are typically column-major).
        row_stride, col_stride = G.strides
        return as_strided(
            G,
            shape=(m, n_locs, n_orient),
            strides=(row_stride, col_stride * n_orient, col_stride),
            writeable=False,
        )


def estimate_orientation_pca(
    G_free: npt.NDArray[np.floating],
    Y: npt.NDArray[np.floating],
    *,
    reg: float,
    deterministic_sign: bool,
    time_chunk: int = 256,
) -> npt.NDArray[np.floating]:
    """Estimate a single, time-invariant orientation per location from data.

    Implements:
    1) Per-location ridge-stabilized LS to estimate a vector time course
    2) PCA on the estimated vector time course to get the dominant direction

    Parameters
    ----------
    G_free
        Free-orientation leadfield of shape ``(n_chans, 3*n_locs)``.
    Y
        Sensor data of shape ``(n_chans, n_times)``.
    reg
        Dimensionless ridge knob. The effective ridge is
        ``alpha = reg * median(trace(L_i^T L_i) / 3)``.
    deterministic_sign
        If True, flip each q_i so its largest-magnitude component is positive.
    time_chunk
        Chunk size for accumulating the PCA covariance without storing all time points.

    Returns
    -------
    q
        Unit vectors of shape ``(n_locs, 3)``.
    """

    G_free = np.asarray(G_free, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if G_free.ndim != 2:
        raise ValueError(f"G_free must be 2D, got ndim={G_free.ndim}")
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D, got ndim={Y.ndim}")

    m = int(G_free.shape[0])
    if int(Y.shape[0]) != m:
        raise ValueError(
            f"G_free and Y must have matching n_chans, got {m} and {Y.shape[0]}"
        )

    L = LeadfieldView(G_free, 3).blocks()  # (m, n, 3) view
    n_locs = int(L.shape[1])
    n_times = int(Y.shape[1])
    if n_locs <= 0:
        raise ValueError("Cannot estimate orientation with n_locs == 0")

    # A_i = L_i^T L_i (n, 3, 3)
    A = np.einsum("mni,mnj->nij", L, L, optimize=True)
    tr = np.trace(A, axis1=1, axis2=2) / 3.0
    scale = float(np.median(tr))
    eps = float(np.finfo(float).eps)
    scale = max(scale, eps)
    alpha = float(reg) * scale
    alpha = max(alpha, eps * scale)

    A_reg = A + alpha * np.eye(3, dtype=float)[np.newaxis, :, :]

    # Accumulate C_i = J_hat J_hat^T without storing J_hat for all times.
    C = np.zeros((n_locs, 3, 3), dtype=float)
    chunk = int(max(1, time_chunk))
    for start in range(0, n_times, chunk):
        Y_chunk = Y[:, start : start + chunk]
        B = np.einsum("mni,mt->nit", L, Y_chunk, optimize=True)  # (n, 3, t)
        J = np.linalg.solve(A_reg, B)  # (n, 3, t)
        C += np.einsum("nit,njt->nij", J, J, optimize=True)

    # Dominant eigenvector of C_i.
    _evals, evecs = np.linalg.eigh(C)
    q = evecs[:, :, -1]

    # Normalize defensively.
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    q = q / norms

    if deterministic_sign:
        idx = np.argmax(np.abs(q), axis=1)
        sel = q[np.arange(n_locs), idx]
        flip = np.where(sel < 0, -1.0, 1.0)
        q = q * flip[:, np.newaxis]

    return q


def reduce_free_to_scalar(
    G_free: npt.NDArray[np.floating],
    q: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Reduce a free-orientation leadfield to scalar using per-location q_i."""

    G_free = np.asarray(G_free, dtype=float)
    q = np.asarray(q, dtype=float)
    if G_free.ndim != 2:
        raise ValueError(f"G_free must be 2D, got ndim={G_free.ndim}")
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError(f"q must have shape (n_locs, 3), got {q.shape}")

    L = LeadfieldView(G_free, 3).blocks()  # (m, n, 3)
    if q.shape[0] != L.shape[1]:
        raise ValueError(
            f"q has n_locs={q.shape[0]} but G_free implies n_locs={L.shape[1]}"
        )
    return np.einsum("mni,ni->mn", L, q, optimize=True)


def ensure_surface_free_surf_ori(forward: mne.Forward) -> mne.Forward:
    """Ensure a surface free-orientation forward uses surf_ori=True basis.

    Keeps ``source_ori=FREE`` but rotates each local basis to (tangent, tangent, normal).
    """

    try:
        src_types = [str(s.get("type", "")) for s in forward["src"]]
    except Exception:
        return forward

    if not (len(src_types) == 2 and all(t == "surf" for t in src_types)):
        return forward
    if forward.get("source_ori") != FIFF.FIFFV_MNE_FREE_ORI:
        return forward
    if bool(forward.get("surf_ori", False)):
        return forward

    return mne.convert_forward_solution(
        forward,
        surf_ori=True,
        force_fixed=False,
        use_cps=True,
        verbose=0,
    )
