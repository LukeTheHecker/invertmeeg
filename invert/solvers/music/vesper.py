"""Vesper, a spectrally weighted search over cortical patches."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from invert.util.source_adjacency import build_source_adjacency

from ..base import BaseSolver, SolverMeta


def _patch_dictionary(adjacency):
    """Return unit-mass point sources and progressively wider diffusion atoms."""
    adjacency = adjacency.tocsr().astype(float)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    identity = sparse.eye(adjacency.shape[0], format="csr")
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    step = 0.5 * (identity + adjacency @ sparse.diags(1 / np.maximum(degree, 1)))
    patches = [identity]
    diffusion = identity
    for order in range(1, 11):
        diffusion = diffusion @ step
        if order in (1, 3, 6, 10):
            patches.append(diffusion)
    return sparse.hstack(patches, format="csr")


def _recalibrate_whitening(data, gain):
    """Correct baseline covariance error in the low-variance sensor directions."""
    centered = data - data.mean(axis=1, keepdims=True)
    n_sensors, n_times = data.shape
    values, vectors = np.linalg.eigh(centered @ centered.T / n_times)
    floor = max(np.median(values), np.finfo(float).tiny)
    bulk_edge = floor * (1 + np.sqrt(n_sensors / n_times)) ** 2
    correction = np.where(values < bulk_edge, np.clip(values / floor, 0.25, 4), 1)
    transform = (vectors * correction ** (-0.5 * 0.35)) @ vectors.T
    return transform @ data, transform @ gain


def _spectral_statistics(data, include_dc):
    """Pool covariance across frequency bands and estimate their noise levels."""
    n_sensors, n_times = data.shape
    if n_times < 16:
        sample_cov = data @ data.T
        scale = max(np.trace(sample_cov) / n_sensors, np.finfo(float).tiny)
        return sample_cov / scale, n_times, [], []

    spectrum = np.fft.rfft(data, norm="ortho")
    bins = np.arange(1, spectrum.shape[1])
    edges = np.flatnonzero(np.diff(np.floor(np.log2(bins / 4 + 1)))) + 1
    bands = np.split(bins, edges)

    band_covariances, degrees = [], []
    for band in bands:
        quadratures = np.concatenate(
            (spectrum[:, band].real, spectrum[:, band].imag), axis=1
        )
        degrees.append(quadratures.shape[1])
        band_covariances.append(2 * (quadratures @ quadratures.T) / degrees[-1])
    degrees = np.asarray(degrees)
    band_covariances = np.asarray(band_covariances)
    broadband = np.average(band_covariances, axis=0, weights=degrees)
    eigenvalues = np.maximum(np.linalg.eigvalsh(broadband), 0)
    mean = spectrum[:, 0].real / np.sqrt(n_times)
    floor = max(max(eigenvalues[-1], mean @ mean) * 1e-10, np.finfo(float).tiny)
    level = max(np.median(eigenvalues), floor)
    band_power = np.trace(band_covariances, axis1=1, axis2=2) / n_sensors
    high_frequency_noise = min(np.median(band_power[-2:]), 1.25 * level)
    centers = np.array([np.mean(band) for band in bands])
    noise = np.clip(
        high_frequency_noise * (centers[-1] / centers) ** 0.15, 0.5 * level, 5 * level
    )

    # Weight each observation by inverse noise power with modest extra weight
    # on bands containing a dominant spatial mode.
    excess = np.maximum(np.linalg.eigvalsh(band_covariances)[:, -1] / noise - 1, 0)
    relative_noise = noise / level
    weights = (1 + 0.04 * excess) / relative_noise
    weights *= degrees
    weights /= weights @ relative_noise
    band_covariances /= level
    covariance = np.einsum("i,ijk->jk", weights, band_covariances)
    effective_samples = 1 / np.sum((weights * relative_noise) ** 2 / degrees)
    if include_dc:
        covariance += 1.5 * np.outer(mean, mean) / level
    return covariance, effective_samples, bands, noise


def _projection_scores(gain, covariance, support, n_locations):
    """Score each possible addition by its incremental captured variance."""
    basis, _ = np.linalg.qr(gain[list(support)].T)
    directions = gain - (gain @ basis) @ basis.T
    norm2 = np.einsum("ij,ij->i", directions, directions)
    scores = np.einsum("ij,ij->i", directions @ covariance, directions)
    scores = np.divide(
        scores, norm2, out=np.full(len(gain), -np.inf), where=norm2 > 1e-4
    )
    for selected in support:
        scores[selected % n_locations :: n_locations] = -np.inf
    return scores, np.sum(basis * (covariance @ basis))


def _search(gain, covariance, n_locations, n_samples, beam_width, max_patches):
    """Grow competing supports and refine the best by one-atom exchanges."""
    beam = [()]
    best = ()
    best_score = -np.inf
    n_sensors = gain.shape[1]
    total = np.trace(covariance)
    complexity = 1 + 1.3 * np.sqrt(2 * np.log(len(gain)) / n_samples)
    for size in range(1, max_patches + 1):
        proposals = {}
        for support in beam:
            scores, baseline = _projection_scores(
                gain, covariance, support, n_locations
            )
            for candidate in np.argsort(scores)[-beam_width:]:
                if np.isfinite(scores[candidate]):
                    proposal = tuple(sorted((*support, int(candidate))))
                    proposals[proposal] = baseline + scores[candidate]
        if not proposals:
            break

        beam = sorted(proposals, key=proposals.__getitem__, reverse=True)[:beam_width]
        for original in beam[:2]:
            support = list(original)
            value = proposals[original]
            for _ in range(2):
                improved = False
                for index in range(size):
                    rest = support[:index] + support[index + 1 :]
                    scores, baseline = _projection_scores(
                        gain, covariance, rest, n_locations
                    )
                    candidate = int(np.argmax(scores))
                    if baseline + scores[candidate] > value + 1e-10:
                        value = baseline + scores[candidate]
                        support[index] = candidate
                        improved = True
                if not improved:
                    break
            proposals[tuple(sorted(support))] = value
        beam = sorted(proposals, key=proposals.__getitem__, reverse=True)[:beam_width]

        explained = proposals[beam[0]]
        residual = max(total - explained, 0)
        background = max(1.0, residual / max(1, n_sensors - size))
        objective = explained - size * background * complexity
        if objective > best_score:
            best_score = objective
            best = beam[0]
    return list(best)


def _recover_waveforms(data, demix, bands, noise):
    """Apply a bandwise Wiener filter to each demixed patch time course."""
    waves = demix @ data
    if not bands:
        return waves
    spectrum = np.fft.rfft(waves, norm="ortho")
    noise_gain = np.sum(demix**2, axis=1)
    for band, variance in zip(bands, noise, strict=True):
        power = np.mean(np.abs(spectrum[:, band]) ** 2, axis=1)
        excess = np.maximum(power - 0.5 * variance * noise_gain, 0)
        ratio = np.divide(excess, power, out=np.zeros_like(power), where=power > 0)
        spectrum[:, band] *= ratio[:, None]
    return np.fft.irfft(spectrum, n=data.shape[1], norm="ortho")


class SolverVesper(BaseSolver):
    """Recover focal and extended generators from whitened sensor recordings.

    Vesper uses a single graph dictionary and a spectrally weighted support
    objective. Each selected patch can share amplitude with anatomically
    adjacent alternatives whose sensor fields have similar explanatory power.

    Parameters
    ----------
    max_patches : int
        Upper bound on the number of latent patch waveforms.
    beam_width : int
        Number of concurrent support hypotheses per model order.
    include_dc : bool
        Use the recording mean as evidence for sustained sources. Set to
        False if the uncorrected mean is dominated by recording offsets.
    """

    meta = SolverMeta(
        slug="vesper",
        full_name="Vesper",
        category="Subspace Methods",
        description=(
            "Sparse multiscale cortical patch search with spectral noise "
            "calibration and local averaging of plausible patch shapes."
        ),
    )
    require_data = True
    require_recompute = True

    def __init__(
        self,
        *,
        max_patches: int = 7,
        beam_width: int = 12,
        include_dc: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if max_patches < 1 or beam_width < 1:
            msg = "Vesper requires positive max_patches and beam_width."
            raise ValueError(msg)
        self.name = "Vesper"
        self.max_patches = max_patches
        self.beam_width = beam_width
        self.include_dc = include_dc

    def make_inverse_operator(
        self, forward, mne_obj, *args, alpha="auto", noise_cov=None, **kwargs
    ):
        good_channels = set(mne_obj.ch_names) - set(mne_obj.info["bads"])
        keep = [name for name in forward.ch_names if name in good_channels]
        if len(keep) < 2:
            msg = "Vesper needs at least two shared good channels."
            raise ValueError(msg)
        if len(keep) != len(forward.ch_names):
            forward = forward.copy().pick_channels(keep, ordered=True)
        super().make_inverse_operator(forward, mne_obj, *args, alpha=alpha, **kwargs)
        self._whitener = self.prepare_whitened_forward(noise_cov)
        adjacency = build_source_adjacency(self.forward["src"])
        self._atoms = _patch_dictionary(adjacency)
        return self

    def apply_inverse_operator(self, mne_obj):
        data = self.unpack_data_obj(mne_obj)
        self.validate_operator_data_compatibility(data)
        gain = self._whitener.G_white
        data = self._whitener.sensor_transform @ data
        n_sensors, n_locations = gain.shape
        if not self.include_dc and data.shape[1] > 1:
            data = data - data.mean(axis=1, keepdims=True)
        if not np.any(data):
            return self.source_to_object(np.zeros((n_locations, data.shape[1])))
        if data.shape[1] >= 16:
            data, gain = _recalibrate_whitening(data, gain)

        covariance, n_samples, bands, noise = _spectral_statistics(
            data, self.include_dc
        )
        raw_fields = self._atoms.T @ gain.T
        norms = np.linalg.norm(raw_fields, axis=1)
        fields = np.divide(
            raw_fields,
            norms[:, None],
            out=np.zeros_like(raw_fields),
            where=norms[:, None] > 0,
        )
        support = _search(
            fields,
            covariance,
            n_locations,
            n_samples,
            self.beam_width,
            min(self.max_patches, max(1, n_sensors - 1)),
        )
        if not support:
            return self.source_to_object(np.zeros((n_locations, data.shape[1])))

        selected = self._atoms[:, support]
        demix = np.linalg.pinv(fields[support].T, rcond=1e-4)
        waves = _recover_waveforms(data, demix, bands, noise)
        estimate = selected.toarray() @ (waves / norms[support, None])

        # Condition on other selected patches when comparing nearby alternatives.
        # Keep the selected atom as the mode and average plausible shapes around it.
        for index, atom in enumerate(support):
            others = support[:index] + support[index + 1 :]
            scores, _ = _projection_scores(fields, covariance, others, n_locations)
            alignment = fields @ fields[atom]
            overlap = (selected[:, index].T @ self._atoms).toarray().ravel() > 0
            eligible = np.flatnonzero(
                (np.abs(alignment) > 0.985) & overlap & np.isfinite(scores)
            )
            if eligible.size == 0:
                continue
            tolerance = max(scores[atom] / max(1, n_samples), 1e-10)
            weights = np.exp(
                np.minimum((scores[eligible] - scores[atom]) / tolerance, 0)
            )
            weights /= weights.sum()
            mean_atom = self._atoms[:, eligible] @ (
                np.sign(alignment[eligible]) * weights / norms[eligible]
            )
            selected_contribution = selected[:, index].toarray() @ (
                waves[index : index + 1] / norms[atom]
            )
            estimate += 0.35 * (
                mean_atom[:, None] * waves[index] - selected_contribution
            )
        return self.source_to_object(estimate)
