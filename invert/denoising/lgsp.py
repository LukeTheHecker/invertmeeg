from dataclasses import dataclass
from typing import Literal, Optional, Union

import mne
import numpy as np
from scipy.linalg import eigh
from sklearn.covariance import OAS

ArrayLike = np.ndarray
MNEObj = Union[mne.io.BaseRaw, mne.Epochs, mne.Evoked, mne.EvokedArray]


# ============================
# Utilities
# ============================


def _hann(n: int) -> np.ndarray:
    # symmetric Hann for perfect overlap-add with 50% step
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))


def _overlap_add(n_times: int, win_len: int, step: int):
    starts = list(range(0, max(1, n_times - win_len + 1), step))
    if len(starts) == 0 or starts[-1] != n_times - win_len:
        starts.append(max(0, n_times - win_len))
    return starts


def _safe_shrinkage_cov(X: np.ndarray) -> np.ndarray:
    # X: (n_ch, n_t) demeaned
    if X.shape[1] < X.shape[0]:  # too few samples -> fallback to diagonal
        s = np.var(X, axis=1, ddof=1)
        return np.diag(np.maximum(s, 1e-12))
    oas = OAS(assume_centered=True).fit(X.T)
    return oas.covariance_


def _spd_from_eig(Q: np.ndarray, d: np.ndarray) -> np.ndarray:
    return Q @ np.diag(np.maximum(d, 1e-12)) @ Q.T


def _inv_sqrt_spd(Q: np.ndarray, d: np.ndarray) -> np.ndarray:
    return Q @ np.diag(1.0 / np.sqrt(np.maximum(d, 1e-12))) @ Q.T


# ============================
# Base class
# ============================


@dataclass
class DenoiserBase:
    sfreq: Optional[float] = None
    window_size: float = 0.5  # seconds
    step_frac: float = 0.5  # overlap = 1 - step_frac
    use_windows: bool = True  # set False to process full signal

    def _ensure_sfreq(self, sfreq: Optional[float]):
        if self.sfreq is None:
            if sfreq is None:
                raise ValueError(
                    "sfreq must be provided (or set on class) for NumPy input."
                )
            self.sfreq = sfreq

    # ---- public API on NumPy ----
    def run(self, X: ArrayLike, sfreq: Optional[float] = None) -> ArrayLike:
        """
        X shape must be (n_channels, n_times).
        """
        self._ensure_sfreq(sfreq)
        X = np.asarray(X, dtype=float)
        assert X.ndim == 2, "X must be (n_channels, n_times)"
        n_ch, n_t = X.shape
        if not self.use_windows:
            return self._process_window(X)

        assert self.sfreq is not None
        win_len = max(8, int(round(self.window_size * self.sfreq)))
        step = max(1, int(round(win_len * self.step_frac)))
        if n_t < 2 * win_len:  # too short for sliding windows -> process once
            return self._process_window(X)

        w = _hann(win_len)
        starts = _overlap_add(n_t, win_len, step)
        Y = np.zeros_like(X)
        wsum = np.zeros(n_t)

        for s in starts:
            seg = X[:, s : s + win_len].copy()
            seg -= seg.mean(axis=1, keepdims=True)
            seg *= w  # apply window in time
            clean = self._process_window(seg)  # (n_ch, win_len)
            Y[:, s : s + win_len] += clean
            wsum[s : s + win_len] += w

        wsum = np.clip(wsum, 1e-12, None)
        Y /= wsum
        return Y

    # ---- public API for MNE ----
    def run_mne(self, obj: MNEObj):
        """
        Returns the same MNE type with data cleaned.
        """
        if isinstance(obj, mne.io.BaseRaw):
            self._ensure_sfreq(obj.info["sfreq"])
            data = obj.get_data()
            cleaned = self.run(data, sfreq=self.sfreq)
            return mne.io.RawArray(cleaned, obj.info)

        if isinstance(obj, mne.Epochs):
            self._ensure_sfreq(obj.info["sfreq"])
            X = obj.get_data()  # (n_ep, n_ch, n_t)
            cleaned = np.empty_like(X)
            for i in range(X.shape[0]):
                cleaned[i] = self.run(X[i], sfreq=self.sfreq)
            return mne.EpochsArray(cleaned, obj.info, events=obj.events, tmin=obj.tmin)

        if isinstance(obj, (mne.Evoked, mne.EvokedArray)):
            self._ensure_sfreq(obj.info["sfreq"])
            cleaned = self.run(obj.data, sfreq=self.sfreq)
            return mne.EvokedArray(cleaned, obj.info, tmin=obj.times[0])

        raise TypeError(f"Unsupported MNE object: {type(obj)}")

    # ---- to be implemented by subclasses ----
    def _process_window(self, Xw: ArrayLike) -> ArrayLike:
        raise NotImplementedError


# ============================
# Variant A: Leadfield-Guided Subspace Projection (LGSP)
# ============================


@dataclass
class LGSP(DenoiserBase):
    L: Optional[np.ndarray] = None  # (m, n_sources)
    sigma_prior: Literal["identity"] = "identity"
    lambda_ref: float = 1e-3  # ridge on C_ref
    alpha_model: float = 0.2  # blend with scaled I: Cref'=(1-a)Cref+a*trace(Cref)/m*I
    rank: Optional[int] = None  # keep r smallest generalized eigenvalues
    tau: Optional[float] = None  # or threshold on generalized eigenvalues
    center: bool = True
    shrink_data: bool = True

    def _build_Cref(self, n_ch: int) -> np.ndarray:
        if self.L is None:
            # fall back: isotropic brain covariance
            Cref = np.eye(n_ch)
        else:
            L = self.L
            if self.sigma_prior == "identity":
                # C_ref = L L^T (scaled) + lambda I
                Cref = L @ L.T
        # scale & regularize
        tr = np.trace(Cref) / max(1, n_ch)
        Cref = (1 - self.alpha_model) * Cref + self.alpha_model * tr * np.eye(
            Cref.shape[0]
        )
        Cref = Cref + self.lambda_ref * np.eye(Cref.shape[0])
        return Cref

    def _gevd_brain_projector(
        self, Cx: np.ndarray, Cref: np.ndarray, r: Optional[int], tau: Optional[float]
    ) -> np.ndarray:
        """
        Return the sensor-space projector P onto the 'brain-like' subspace
        spanned by generalized eigenvectors with the smallest eigenvalues.
        P is constructed as a B-orthogonal projector: P = V (V^T B V)^{-1} V^T B
        where columns of V are the chosen generalized eigenvectors and B=Cref.
        """
        # Solve symmetric definite generalized eigenproblem: Cx v = eta Cref v
        # eigh returns eigenvalues in ascending order for symmetric problems.
        eta, V = eigh(Cx, Cref, turbo=True)
        # choose modes (smallest eta are more brain-like)
        if r is None and tau is None:
            r = max(1, min(Cx.shape[0] - 1, int(round(0.7 * Cx.shape[0]))))
        if tau is not None:
            keep = np.where(eta <= tau)[0]
            if keep.size == 0:
                keep = np.arange(min(1, len(eta)))  # type: ignore[arg-type, type-var]
        else:
            keep = np.arange(min(r, len(eta)))  # type: ignore[arg-type, type-var]
        Vr = V[:, keep]  # (m, k)
        assert Cref is not None
        B = Cref
        G = Vr.T @ B @ Vr  # (k, k)
        Ginv = np.linalg.pinv(G)
        P = Vr @ Ginv @ Vr.T @ B  # (m, m) projector in B-metric
        # P is idempotent in B-metric: P^2 ≈ P (numerically)
        return P

    def _process_window(self, Xw: ArrayLike) -> ArrayLike:
        # Xw: (n_ch, n_t), mean already subtracted & windowed
        n_ch, _ = Xw.shape
        if self.center:
            Xw = Xw - Xw.mean(axis=1, keepdims=True)

        Cref = self._build_Cref(n_ch)
        # Shrinkage covariance of data (robust in low-sample windows)
        Cx = _safe_shrinkage_cov(Xw) if self.shrink_data else np.cov(Xw)

        P = self._gevd_brain_projector(Cx, Cref, self.rank, self.tau)
        return P @ Xw


# ============================
# Variant B: Source-space Reprojection & Blend (SRB)
# ============================


@dataclass
class SRB(DenoiserBase):
    L: Optional[np.ndarray] = None  # (m, n_sources) required for meaningful use
    mu: float = 1e-1  # Tikhonov on sources (||S||^2)
    beta: float = 0.5  # blend factor: X_clean = beta*L*W*X + (1-beta)*X
    center: bool = True
    cache_inverse: bool = True

    def __post_init__(self):
        self._W = None  # cached inverse

    def _compute_inverse(self, L: np.ndarray) -> np.ndarray:
        # W = (L^T L + mu I)^{-1} L^T
        LtL = L.T @ L
        n_src = LtL.shape[0]
        A = LtL + self.mu * np.eye(n_src)
        W = np.linalg.solve(A, L.T)
        return W

    def _ensure_inverse(self, n_ch: int):
        if self.L is None:
            # fall back: identity projection (no-op)
            self._W = None
            return
        if (self._W is None) or (self._W.shape[1] != self.L.shape[0]):
            self._W = self._compute_inverse(self.L)

    def _process_window(self, Xw: ArrayLike) -> ArrayLike:
        if self.center:
            Xw = Xw - Xw.mean(axis=1, keepdims=True)
        n_ch, _ = Xw.shape
        self._ensure_inverse(n_ch)
        if self._W is None:  # no leadfield provided
            return Xw
        S = self._W @ Xw  # (n_src, n_t)
        Xb = self.L @ S  # (n_ch, n_t) projection to brain-span
        return self.beta * Xb + (1.0 - self.beta) * Xw


# ============================
# Usage Examples
# ============================

# -- NumPy data (n_ch, n_times) --
# sfreq must be provided if not set on class
# L: (n_ch, n_src) leadfield

# LGSP example:
# lgsp = LGSP(L=L, window_size=0.5, step_frac=0.5, rank=3, lambda_ref=1e-3, alpha_model=0.3)
# X_clean = lgsp.run(X, sfreq=256)

# SRB example:
# srb = SRB(L=L, mu=1e-1, beta=0.6, use_windows=False)  # SRB often fine without windowing
# X_clean = srb.run(X, sfreq=256)

# -- MNE objects --
# lgsp_mne = lgsp.run_mne(raw)         # returns Raw
# srb_mne  = srb.run_mne(epochs)       # returns Epochs
