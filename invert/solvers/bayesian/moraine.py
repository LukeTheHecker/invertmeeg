"""Moraine: spectral patch evidence above a diffuse cortical background."""

from __future__ import annotations

from operator import index

import numpy as np
from scipy import sparse
from scipy.linalg import null_space
from scipy.special import expit, logsumexp, softmax

from invert.util.source_adjacency import build_source_adjacency

from ..base import BaseSolver, SolverMeta


def _dictionary(graph):
    """Unit-mass point, diffusion and tapered geodesic disk atoms."""
    graph = sparse.csr_matrix(graph, dtype=float, copy=True)
    graph.setdiag(0)
    graph.eliminate_zeros()
    identity = sparse.eye(graph.shape[0], format="csr")
    degree = np.asarray(graph.sum(axis=0)).ravel()
    step = (graph + identity) @ sparse.diags(1 / (degree + 1))
    orders = [
        identity,
        (identity + step) * 0.5,
        step,
        step @ step,
        step @ step @ step @ step,
    ]
    near = (graph + identity).astype(bool).astype(float)
    two = (near @ near).astype(bool).astype(float)
    three = (two @ near).astype(bool).astype(float)
    for patch in [
        near + identity * 0.65,
        two + near * 0.8 + identity * 0.35,
        three + two * 1.2 + near,
    ]:
        orders.append(patch @ sparse.diags(1 / np.asarray(patch.sum(axis=0)).ravel()))
    return sparse.hstack(orders, format="csr"), (identity + step) * 0.5


def _bands(data):
    """Real-valued Fourier covariances with exact DC and Nyquist multiplicities."""
    n = data.shape[1]
    spectrum = np.fft.rfft(data, norm="ortho")
    edges = [i for i in (1, 5, 15, 40) if i < spectrum.shape[1]]
    bands = np.split(np.arange(spectrum.shape[1]), edges)
    multiplicity = np.full(spectrum.shape[1], 2)
    multiplicity[0] = 1
    if n % 2 == 0:
        multiplicity[-1] = 1
    sizes = np.array([multiplicity[b].sum() for b in bands])
    cov = np.stack(
        [
            ((spectrum[:, b] * multiplicity[b]) @ spectrum[:, b].conj().T).real / size
            for b, size in zip(bands, sizes, strict=True)
        ]
    )
    return spectrum, bands, sizes, cov


def _background(kernel, cov, sizes, fixed=None):
    """Separate diffuse cortical covariance from localized low-rank components."""
    sensors = kernel.shape[0]
    values, basis = np.linalg.eigh(kernel)
    values = np.maximum(values, 0)
    values /= max(values.mean(), np.finfo(float).tiny)
    high = cov[1:] if len(cov) > 1 else cov
    counts = sizes[1:] if len(cov) > 1 else sizes
    pooled = np.average(high, weights=counts, axis=0)
    pooled = basis.T @ pooled @ basis
    levels = np.array([0, 0.06, 0.16, 0.4, 1, 2.5, 6, 16, 50])
    edge = (1 + np.sqrt(sensors / sum(counts))) ** 2
    best = np.inf
    # Profile the overall noise scale and allow spiked covariance eigenvalues.
    for ratio in levels:
        diag = 1 + ratio * values
        gain = diag**-0.5
        eigen, modes = np.linalg.eigh(gain[:, None] * pooled * gain)
        eigen = np.maximum(eigen, 0)
        noise = max(np.median(eigen), np.max(eigen) * 1e-9, 1e-20)
        bulk = eigen <= noise * edge * 1.2
        bulk[0] = True
        noise = max(np.mean(eigen[bulk]), noise * 0.2)
        fitted = np.where(bulk, noise, np.maximum(eigen, noise))
        cost = np.log(diag).sum() + np.log(fitted).sum() + (eigen / fitted).sum()
        cost += (~bulk).sum() * (edge - 1 - np.log(edge))
        if cost < best:
            best = cost
            best_ratio = ratio
            transform = (basis * gain).T
            projector = modes[:, bulk].T
            bulk_power = np.clip(eigen[bulk] / noise, 0.3, 3)
    if fixed is not None:
        projector = null_space((transform @ fixed.T).T).T
        if not len(projector):
            projector = np.eye(sensors)
        sub_cov = (
            projector @ transform @ basis @ pooled @ basis.T @ transform.T @ projector.T
        )
        eigen, modes = np.linalg.eigh(sub_cov)
        projector = modes.T @ projector
        bulk_power = np.clip(eigen / max(np.median(eigen), 1e-18), 0.3, 3)
    white_cov = transform @ cov @ transform.T
    residual = projector @ white_cov @ projector.T
    floor = max(np.max(np.linalg.eigvalsh(white_cov)) * 1e-10, 1e-20)

    # Restrict nuisance fitting to directions outside the pooled signal space.
    # Diagonalize the two projected variance components simultaneously.
    sensor = 1 / (1 + best_ratio * values)
    projected_sensor = (projector * sensor) @ projector.T
    ev, u = np.linalg.eigh(projected_sensor)
    whitener = (u / np.sqrt(np.maximum(ev, 1e-12))).T
    cortical = (projector * (values * sensor)) @ projector.T
    ev, u = np.linalg.eigh(whitener @ cortical @ whitener.T)
    ev = np.maximum(ev, 0)
    restrict = u.T @ whitener
    empirical = np.diagonal(restrict @ residual @ restrict.T, axis1=1, axis2=2)
    ratios = np.asarray(levels)[:, None]
    variances = 1 + ratios * ev
    power = np.maximum((empirical @ (1 / variances).T) / len(ev), floor)
    nll = np.log(variances).sum(axis=1) + len(ev) * np.log(power)
    nll += (8 / sizes[:, None]) * (np.log1p(ratios.T) - np.log1p(best_ratio)) ** 2
    idx = nll.argmin(axis=1)
    ratio = np.asarray(levels)[idx]
    noise = power[np.arange(len(idx)), idx]
    # The recording mean supplies one observation. Borrow its noise shape from AC.
    if len(idx) > 1:
        ratio[0] = ratio[1]
        noise[0] = max(noise[-1], noise[1] * 0.5)
    noise = np.maximum(noise, floor)
    rescale = np.sqrt((1 + best_ratio * values) / (1 + ratio[:, None] * values))
    adjust = np.eye(sensors) + (projector.T * (bulk_power ** (-0.25) - 1)) @ projector
    return ((rescale[:, :, None] * adjust) @ transform) / np.sqrt(noise[:, None, None])


class _PatchEvidence:
    """Profile the variance of one patch after projecting out other generators."""

    def __init__(self, raw_fields, transform, covariance, degrees):
        self.raw = raw_fields
        self.transform = transform
        self.cov = transform @ covariance @ transform.transpose(0, 2, 1)
        self.degrees = degrees
        self.norm2 = np.empty((len(degrees), len(raw_fields)))
        self.energy = np.empty_like(self.norm2)
        for b, whitening in enumerate(transform):
            fields = raw_fields @ whitening.T
            self.norm2[b] = np.einsum("ij,ij->i", fields, fields)
            self.energy[b] = np.einsum("ij,ij->i", fields @ self.cov[b], fields)
        self.valid = np.all(
            self.norm2 > self.norm2.max(axis=1, keepdims=True) * 1e-24, axis=0
        )

    def scan(self, other, limit):
        """Twice the profile log-likelihood gain and conditional band variances."""
        raw = self.raw[:limit]
        norm2 = self.norm2[:, :limit].copy()
        energy = self.energy[:, :limit].copy()
        if other:
            # Expand h' P C P h using only the small basis of selected fields.
            # This avoids building a sensor-by-atoms array for every band and scan.
            for b, whitening in enumerate(self.transform):
                basis = np.linalg.qr(whitening @ self.raw[other].T)[0]
                cross = self.cov[b] @ basis
                overlap = raw @ (whitening.T @ basis)
                norm2[b] -= np.einsum("ij,ij->i", overlap, overlap)
                correction = 2 * (raw @ (whitening.T @ cross)) - overlap @ (
                    basis.T @ cross
                )
                energy[b] -= np.einsum("ij,ij->i", correction, overlap)
        eligible = self.valid[:limit] & np.all(
            norm2 > 0.015 * self.norm2[:, :limit], axis=0
        )
        norm2[:, ~eligible] = 1
        power = np.maximum(energy / norm2, 1)
        scores = self.degrees @ (power - 1 - np.log(power))
        scores[~eligible] = -np.inf
        return scores, norm2, power

    def fit(self, max_patches, limit, penalty):
        active = []
        minimum = penalty * (2 * np.log(limit) + len(self.degrees))
        for _ in range(max_patches):
            scores = self.scan(active, limit)[0]
            atom = int(np.argmax(scores))
            if not np.isfinite(scores[atom]) or (active and scores[atom] < minimum):
                break
            active.append(atom)
            for _ in range(3):
                changed = False
                for j, old in enumerate(active):
                    scores = self.scan(active[:j] + active[j + 1 :], limit)[0]
                    new = int(np.argmax(scores))
                    if scores[new] > scores[old] + 1e-5:
                        active[j] = new
                        changed = True
                if not changed:
                    break
        return active

    def support_score(self, active, penalty):
        if not active:
            return 0.0
        fields = self.transform @ self.raw[active].T
        basis = np.linalg.qr(fields)[0]
        subspace_cov = basis.transpose(0, 2, 1) @ self.cov @ basis
        power = np.maximum(np.linalg.eigvalsh(subspace_cov), 1)
        complexity = penalty * (2 * np.log(len(self.raw)) + len(self.degrees))
        return self.degrees @ (power - 1 - np.log(power)).sum(
            axis=1
        ) - complexity * len(active)

    def patch_posteriors(self, active, n_vertices):
        """Average over extent and position, with a shared point-source prior."""
        scans = [
            self.scan(active[:j] + active[j + 1 :], len(self.raw))
            for j in range(len(active))
        ]
        n_scales = len(self.raw) / n_vertices
        log_odds = 0.0
        for scores, _, _ in scans:
            evidence = scores / 4
            log_odds += (
                logsumexp(evidence[:n_vertices])
                - logsumexp(evidence)
                + np.log(n_scales)
            )
        focal_prob = expit(log_odds)
        temperature = 1 + 14 * (1 - focal_prob)
        posteriors = []
        for scores, norm2, power in scans:
            peak = np.max(scores)
            # Terms beyond this cutoff have individual relative weight below exp(-16).
            ids = np.flatnonzero(scores >= peak - 32 * temperature)
            prior = np.where(
                ids < n_vertices,
                1 + focal_prob * (n_scales - 1),
                max(1 - focal_prob, 1e-12),
            )
            weights = softmax((scores[ids] - peak) / (2 * temperature) + np.log(prior))
            amplitude = np.sqrt((power[:, ids] - 1) / norm2[:, ids])
            posteriors.append((ids, weights, amplitude))
        return posteriors

    def reconstruct(self, atoms, active, spectrum, bands):
        """Joint Wiener posterior from the first two moments of each patch."""
        posteriors = self.patch_posteriors(active, atoms.shape[0])
        output = np.zeros((atoms.shape[0], spectrum.shape[1]), dtype=complex)
        for b, band in enumerate(bands):
            if not active:
                break
            whitening = self.transform[b]
            # Sum the independent second moments, then add cross-patch covariance
            # through the posterior mean fields. Sign-align ambiguous orientations.
            atom_variance = np.zeros(atoms.shape[1])
            mode_fields = self.raw[active] @ whitening.T
            mean_fields, mean_shapes = [], []
            for i, (ids, weights, amplitude) in enumerate(posteriors):
                atom_variance[ids] += weights * amplitude[b] ** 2
                fields = self.raw[ids] @ whitening.T
                signed = weights * amplitude[b] * np.sign(fields @ mode_fields[i])
                mean_fields.append(fields.T @ signed)
                mean_shapes.append(atoms[:, ids] @ signed)
            h = np.stack(mean_fields, axis=1)
            z = np.stack(mean_shapes, axis=1)
            demix = np.linalg.pinv(
                (mode_fields / np.linalg.norm(mode_fields, axis=1, keepdims=True)).T,
                rcond=1e-5,
            )
            covariance = demix @ self.cov[b] @ demix.T
            std = np.sqrt(np.maximum(np.diag(covariance), 1e-15))
            correlation = covariance / (std[:, None] * std) - np.eye(len(active))
            correlation *= 0.65
            ids = np.flatnonzero(atom_variance)
            fields = self.raw[ids] @ whitening.T
            cross = atom_variance[ids, None] * fields
            sensor_cov = (
                np.eye(fields.shape[1]) + fields.T @ cross + h @ correlation @ h.T
            )
            residual = np.linalg.solve(sensor_cov, whitening @ spectrum[:, band])
            output[:, band] = atoms[:, ids] @ (cross @ residual) + z @ (
                correlation @ (h.T @ residual)
            )
        return output


class SolverMoraine(BaseSolver):
    """Estimate localized generator waveforms above a diffuse cortical field.

    The per-band nuisance covariance is the sum of a sensor noise term and the
    sensor projection of a mesh-smoothed cortical field. A provisional patch fit
    identifies directions to exclude when re-estimating these noise components.
    A second likelihood search localizes patches of unknown extent. Location
    and shape uncertainty enter the covariance of the final joint Wiener filter.

    Parameters
    ----------
    max_patches : int
        Maximum number of simultaneously active patch waveforms.
    include_dc : bool
        Include the recording mean as evidence for sustained cortical currents.
        Set to False to remove a constant recording offset.
    """

    meta = SolverMeta(
        slug="moraine",
        full_name="Moraine",
        category="Bayesian",
        description="Estimates multiscale patch currents against a frequency-dependent diffuse cortical background and averages over uncertain locations and shapes.",
    )
    require_data = True
    require_recompute = True

    def __init__(self, *, max_patches: int = 8, include_dc: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.name = "Moraine"
        self.max_patches = index(max_patches)
        if self.max_patches < 1:
            raise ValueError("max_patches must be a positive integer.")
        self.include_dc = include_dc

    def make_inverse_operator(
        self, forward, mne_obj, *args, alpha="auto", noise_cov=None, **kwargs
    ):
        """Prepare patch fields and whitening. Numeric alpha scales the model-order penalty."""
        good = set(mne_obj.ch_names) - set(mne_obj.info["bads"])
        keep = [name for name in forward.ch_names if name in good]
        if len(keep) < 2:
            raise ValueError("Moraine requires at least two shared good channels.")
        if len(keep) < len(forward.ch_names):
            forward = forward.copy().pick_channels(keep, ordered=True)
        self._penalty = 2.6 if alpha == "auto" else 2.6 * float(alpha)
        if not np.isfinite(self._penalty) or self._penalty < 0:
            raise ValueError("alpha must be 'auto' or a finite non-negative number.")
        super().make_inverse_operator(forward, mne_obj, *args, alpha=alpha, **kwargs)
        self._white = self.prepare_whitened_forward(noise_cov)
        self._atoms, smooth = _dictionary(build_source_adjacency(self.forward["src"]))
        gain = self._white.G_white.T
        self._fields = self._atoms.T @ gain
        ambient = smooth.T @ gain
        self._kernel = ambient.T @ ambient
        return self

    def apply_inverse_operator(self, mne_obj):
        data = self.unpack_data_obj(mne_obj)
        self.validate_operator_data_compatibility(data)
        data = self._white.sensor_transform @ data
        if not self.include_dc:
            data = data - data.mean(axis=1, keepdims=True)
        scale = np.max(np.abs(data))
        n_vertices, n_atoms = self._atoms.shape
        if scale == 0:
            return self.source_to_object(np.zeros((n_vertices, data.shape[1])))
        spectrum, bands, sizes, cov = _bands(data / scale)
        max_patches = min(self.max_patches, max(1, len(data) - 1))
        transforms = _background(self._kernel, cov, sizes)
        model = _PatchEvidence(self._fields, transforms, cov, sizes)
        active = model.fit(max_patches, n_atoms, self._penalty)
        if active:
            transforms = _background(self._kernel, cov, sizes, self._fields[active])
            model = _PatchEvidence(self._fields, transforms, cov, sizes)
            active = model.fit(max_patches, n_atoms, self._penalty)
            # The point-only restriction gives a second initialization when highly
            # coherent generators trap coordinate search on a broad patch.
            focal = model.fit(max_patches, n_vertices, self._penalty)
            if model.support_score(focal, self._penalty) > model.support_score(
                active, self._penalty
            ):
                active = focal
        estimate = model.reconstruct(self._atoms, active, spectrum, bands)
        sources = np.fft.irfft(estimate, n=data.shape[1], norm="ortho")
        return self.source_to_object(sources * scale)
