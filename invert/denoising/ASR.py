import logging
from typing import Union

import mne
import numpy as np

logger = logging.getLogger(__name__)


class ASR:
    """
    Artifact Subspace Reconstruction (ASR) implementation.

    Parameters
    ----------
    sfreq : float
        Sampling frequency (Hz). Required if input is np.ndarray.
    window_size : float
        Window length in seconds (default 0.5).
    step_size : float
        Step size as fraction of window (default 0.5 → 50% overlap).
    cutoff : float
        Standard deviation threshold (default 5.0).
    """

    def __init__(self, sfreq=None, window_size=0.5, step_size=0.5, cutoff=5.0):
        self.sfreq = sfreq
        self.window_size = window_size
        self.step_size = step_size
        self.cutoff = cutoff
        self.V = None
        self.lam_ref = None

    # -------------------------- core utilities --------------------------

    @staticmethod
    def _window_cov(X, win_len, step):
        """Compute covariance matrices from sliding windows."""
        n_ch, n_samp = X.shape
        step_len = int(win_len * (1 - step))
        covs = []
        for start in range(0, n_samp - win_len + 1, step_len):
            seg = X[:, start : start + win_len]
            seg -= seg.mean(axis=1, keepdims=True)
            covs.append(np.cov(seg))
        return np.array(covs)

    def _fit_reference(self, X):
        """Fit reference covariance and thresholds."""
        win_len = int(self.window_size * self.sfreq)
        covs = self._window_cov(X, win_len, self.step_size)
        C_ref = covs.mean(axis=0)
        self.V, lam, _ = np.linalg.svd(C_ref)
        log_lam = np.log10(lam + 1e-12)
        self.lam_ref = lam * 10 ** (self.cutoff * log_lam.std())
        return C_ref

    # -------------------------- main logic --------------------------

    def run(self, data: np.ndarray, calibration=None) -> np.ndarray:
        """
        Run ASR on numpy array data.

        Parameters
        ----------
        data : np.ndarray, shape (n_channels, n_times)
        calibration : np.ndarray or None
            Optional baseline data for reference fit.

        Returns
        -------
        cleaned : np.ndarray
            Denoised signal (same shape as input).
        """
        X = np.asarray(data)
        n_ch, n_samp = X.shape

        if self.sfreq is None:
            raise ValueError("Sampling frequency (sfreq) must be provided.")

        # For very short recordings, skip ASR
        win_len = int(self.window_size * self.sfreq)
        if n_samp < win_len * 2:
            logger.warning("[ASR] Too few samples for stable covariance estimation.")
            return X.copy()

        # Fit reference if not done
        if calibration is None:
            calibration = X
        self._fit_reference(calibration)

        # Process in windows
        step_len = int(win_len * (1 - self.step_size))
        cleaned = np.zeros_like(X)
        weights = np.zeros(n_samp)

        for start in range(0, n_samp - win_len + 1, step_len):
            seg = X[:, start : start + win_len]
            seg -= seg.mean(axis=1, keepdims=True)
            C = np.cov(seg)
            z = np.diag(self.V.T @ C @ self.V)
            z_clipped = np.minimum(z, self.lam_ref)
            C_clipped = self.V @ np.diag(z_clipped) @ self.V.T

            # Whitening and projection back
            eigvals, eigvecs = np.linalg.eigh(C)
            W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + 1e-12)) @ eigvecs.T
            S = np.linalg.cholesky(C_clipped + 1e-12 * np.eye(n_ch))
            seg_clean = S @ W @ seg

            cleaned[:, start : start + win_len] += seg_clean
            weights[start : start + win_len] += 1.0

        cleaned /= np.maximum(weights, 1e-12)
        return cleaned

    # -------------------------- mne wrapper --------------------------

    def run_mne(self, mne_obj: Union[mne.io.BaseRaw, mne.Epochs, mne.Evoked]):
        """
        Run ASR on an MNE object and return a cleaned copy.
        """
        if isinstance(mne_obj, mne.io.BaseRaw):
            data = mne_obj.get_data()
            self.sfreq = mne_obj.info["sfreq"]
            cleaned = self.run(data)
            new_raw = mne.io.RawArray(cleaned, mne_obj.info)
            return new_raw

        elif isinstance(mne_obj, mne.Epochs):
            self.sfreq = mne_obj.info["sfreq"]
            cleaned_data = []
            for ep in mne_obj.get_data():
                cleaned_data.append(self.run(ep.T).T)
            new_epochs = mne.EpochsArray(
                np.array(cleaned_data),
                mne_obj.info,
                events=mne_obj.events,
                tmin=mne_obj.tmin,
            )
            return new_epochs

        elif isinstance(mne_obj, mne.Evoked):
            self.sfreq = mne_obj.info["sfreq"]
            cleaned = self.run(mne_obj.data)
            return mne.EvokedArray(cleaned, mne_obj.info, tmin=mne_obj.times[0])

        else:
            raise TypeError(f"Unsupported MNE object type: {type(mne_obj)}")
