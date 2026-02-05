from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..base import BaseSolver, SolverMeta

logger = logging.getLogger(__name__)


class SolverDICS(BaseSolver):
    """Dynamic Imaging of Coherent Sources (DICS) beamformer.

    This implementation estimates the sensor cross-spectral density (CSD) in a
    frequency band, computes unit-gain spatial filters, and returns the
    estimated source power. Since the library interface expects a
    time-resolved SourceEstimate, the band power is replicated across time.

    References
    ----------
    [1] Gross, J., Kujala, J., Hämäläinen, M., Timmermann, L., Schnitzler, A.,
        & Salmelin, R. (2001). Dynamic imaging of coherent sources: Studying
        neural interactions in the human brain. PNAS, 98(2), 694-699.
    """

    meta = SolverMeta(
        slug="dics",
        full_name="Dynamic Imaging of Coherent Sources",
        category="Beamformers",
        description=(
            "Frequency-domain beamformer based on the sensor cross-spectral density "
            "in a band. Returns band-limited source power (replicated over time)."
        ),
        references=[
            "Gross, J., Kujala, J., Hämäläinen, M., Timmermann, L., Schnitzler, A., "
            "& Salmelin, R. (2001). Dynamic imaging of coherent sources: Studying "
            "neural interactions in the human brain. PNAS, 98(2), 694-699.",
        ],
    )

    def __init__(
        self,
        name: str = "DICS Beamformer",
        reduce_rank: bool = True,
        rank: str | int = "auto",
        **kwargs: Any,
    ) -> None:
        self.name = name
        self.fmin: float | None = None
        self.fmax: float | None = None
        self.csd: np.ndarray | None = None
        self.source_powers: list[np.ndarray] = []
        super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    @staticmethod
    def _compute_csd_from_data(
        data: np.ndarray,
        sfreq: float,
        fmin: float,
        fmax: float,
        n_fft: int | None = None,
        window: str = "hann",
    ) -> np.ndarray:
        """Estimate a single CSD matrix by averaging FFT bins in [fmin, fmax]."""
        n_chans, n_times = data.shape
        if n_times < 2:
            return np.eye(n_chans)

        if n_fft is None:
            # Ensure at least one FFT bin falls in [fmin, fmax] by requiring
            # frequency resolution <= (fmax - fmin), i.e. n_fft >= sfreq / (fmax - fmin).
            min_nfft_for_band = int(np.ceil(sfreq / max(fmax - fmin, 1e-10)))
            n_fft = int(2 ** int(np.ceil(np.log2(max(n_times, min_nfft_for_band)))))
        n_fft = max(int(n_fft), int(n_times))

        x = data - data.mean(axis=1, keepdims=True)
        if window == "hann":
            win = np.hanning(n_times)
            x = x * win[None, :]
        elif window not in {"boxcar", "rect", "none", None}:
            logger.warning("Unknown window '%s', using no window.", window)

        fft = np.fft.rfft(x, n=n_fft, axis=1)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sfreq)

        band = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        if band.size == 0:
            raise ValueError(
                f"No FFT bins in [{fmin}, {fmax}] Hz. "
                f"Try widening the band or increasing n_fft."
            )

        csd = np.zeros((n_chans, n_chans), dtype=np.complex128)
        for k in band:
            v = fft[:, k]
            csd += np.outer(v, np.conjugate(v))

        csd /= float(band.size)
        return csd

    def make_inverse_operator(  # type: ignore[override]
        self,
        forward,
        mne_obj,
        *args: Any,
        alpha: str | float = "auto",
        fmin: float = 8.0,
        fmax: float = 12.0,
        tmin: float | None = None,
        tmax: float | None = None,
        n_fft: int | None = None,
        window: str = "hann",
        **kwargs: Any,
    ):
        self.fmin = float(fmin)
        self.fmax = float(fmax)

        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)

        data = self.unpack_data_obj(mne_obj)
        sfreq = float(self.obj_info["sfreq"])

        if tmin is not None or tmax is not None:
            start = 0 if tmin is None else int(round((tmin - self.tmin) * sfreq))
            stop = data.shape[1] if tmax is None else int(round((tmax - self.tmin) * sfreq))
            start = int(np.clip(start, 0, data.shape[1]))
            stop = int(np.clip(stop, start + 1, data.shape[1]))
            data = data[:, start:stop]

        self.csd = self._compute_csd_from_data(
            data, sfreq, self.fmin, self.fmax, n_fft=n_fft, window=window
        )

        # Regularization scale based on the CSD (not the leadfield)
        self.get_alphas(reference=np.real(self.csd))

        leadfield = self.leadfield
        n_chans = leadfield.shape[0]

        self.source_powers = []
        for alpha_eff in self.alphas:
            csd_reg = self.csd + alpha_eff * np.eye(n_chans)
            csd_inv = self.robust_inverse(csd_reg)

            upper = csd_inv @ leadfield
            denom = np.sum(np.conjugate(leadfield) * upper, axis=0)
            denom = np.where(np.abs(denom) < 1e-15, 1e-15, denom)
            W = upper / denom

            power = np.real(np.sum(np.conjugate(W) * (self.csd @ W), axis=0))
            self.source_powers.append(power.astype(np.float64))

        return self

    def apply_inverse_operator(self, mne_obj):  # type: ignore[override]
        if not self.source_powers:
            raise RuntimeError(
                "Call make_inverse_operator() before apply_inverse_operator()."
            )

        data = self.unpack_data_obj(mne_obj)
        n_time = data.shape[1]

        if self.use_last_alpha and self.last_reg_idx is not None:
            idx = int(self.last_reg_idx)
        else:
            idx = 0

        power = self.source_powers[int(np.clip(idx, 0, len(self.source_powers) - 1))]
        source_mat = np.tile(power[:, None], (1, n_time))
        return self.source_to_object(source_mat)
