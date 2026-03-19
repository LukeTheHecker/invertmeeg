"""Neural Mass Model (Jansen-Rit) simulator for source-level EEG generation.

Implements the Jansen-Rit model (1995) to generate realistic epileptiform and
rhythmic brain source activity, then projects through a leadfield to sensors.

The simulation approach (Jansen-Rit NMM for source generation, region-growing
for spatial patches, forward projection through a leadfield) is inspired by
the DeepSIF project:
    https://github.com/bfinl/DeepSIF
    Sun et al. (2022). Deep neural networks constrained by neural mass models
    improve electrophysiological source imaging of spatiotemporal brain dynamics.
    PNAS, 119(31), e2201128119.

References
----------
Jansen & Rit (1995). Electroencephalogram and visual evoked potential
generation in a mathematical model of coupled cortical columns.
Biological Cybernetics, 73(4), 357-366.

Wendling et al. (2000). Epileptic fast activity can be explained by a
model of impaired GABAergic dendritic inhibition.
European Journal of Neuroscience, 12(4), 1371-1385.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import resample

from .batch import SimulationBatch
from .noise import apply_sensor_noise_single


def _sigmoid(v, e0=2.5, v0=6.0, r=0.56):
    """Sigmoid activation: converts mean membrane potential to firing rate.

    S(v) = 2*e0 / (1 + exp(r*(v0 - v)))
    """
    return 2.0 * e0 / (1.0 + np.exp(r * (v0 - v)))


def jansen_rit(
    duration_s: float = 2.0,
    dt: float = 0.5e-3,
    A: float = 3.25,
    B: float = 22.0,
    a: float = 100.0,
    b: float = 50.0,
    C: float = 135.0,
    p_mean: float = 220.0,
    p_std: float = 22.0,
    e0: float = 2.5,
    v0: float = 6.0,
    r: float = 0.56,
    warmup_s: float = 2.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one Jansen-Rit cortical column.

    Equations from Jansen & Rit (1995), eqs. 1-6:
        dy0/dt = y3
        dy3/dt = A*a*S(y1-y2) - 2a*y3 - a^2*y0
        dy1/dt = y4
        dy4/dt = A*a*(p(t) + C2*S(C1*y0)) - 2a*y4 - a^2*y1
        dy2/dt = y5
        dy5/dt = B*b*C4*S(C3*y0) - 2b*y5 - b^2*y2

    Parameters
    ----------
    duration_s : float
        Simulation duration in seconds (after warmup).
    dt : float
        Integration timestep in seconds.
    A, B : float
        Excitatory / inhibitory synaptic gains (mV).
    a, b : float
        Excitatory / inhibitory rate constants (1/s).
    C : float
        Connectivity constant. Internal couplings are C1=C, C2=0.8C,
        C3=C4=0.25C.
    p_mean, p_std : float
        Mean and std of external input pulse density (pulses/s).
    e0, v0, r : float
        Sigmoid parameters.
    warmup_s : float
        Warmup duration discarded from output (lets transients die out).
    rng : Generator, optional
        Numpy random generator.

    Returns
    -------
    t : ndarray, shape (n_steps,)
        Time vector in seconds.
    y_pyr : ndarray, shape (n_steps,)
        PSP of pyramidal population (y1 - y2), the EEG-like output.
    """
    # Delegate to coupled version with 1 column and no coupling
    t, tc = jansen_rit_coupled(
        n_columns=1, duration_s=duration_s, dt=dt,
        A_range=(A, A), B_range=(B, B), C_range=(C, C),
        p_mean_range=(p_mean, p_mean), p_std=p_std,
        coupling_strength=0.0, warmup_s=warmup_s,
        e0=e0, v0=v0, r=r, rng=rng,
    )
    return t, tc[0]


def jansen_rit_coupled(
    n_columns: int,
    duration_s: float = 2.0,
    dt: float = 0.5e-3,
    A_range: tuple[float, float] = (3.0, 3.6),
    B_range: tuple[float, float] = (18.0, 26.0),
    C_range: tuple[float, float] = (100.0, 150.0),
    p_mean_range: tuple[float, float] = (150.0, 280.0),
    p_std: float = 22.0,
    coupling_strength: float | tuple[float, float] = (10.0, 50.0),
    spike_mode: bool = False,
    A_spike: float = 5.0,
    warmup_s: float = 2.0,
    e0: float = 2.5,
    v0: float = 6.0,
    r: float = 0.56,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate multiple coupled Jansen-Rit cortical columns (vectorized).

    All columns are integrated simultaneously using numpy broadcasting.
    Columns interact via long-range excitatory connections: each column
    receives additional input proportional to the mean pyramidal firing rate
    of the other columns, weighted by a random coupling matrix.

    Parameters
    ----------
    n_columns : int
        Number of coupled cortical columns.
    duration_s : float
        Duration in seconds (after warmup).
    dt : float
        Integration timestep in seconds.
    A_range, B_range, C_range, p_mean_range : tuple
        Parameter ranges (uniform sampling per column).
    p_std : float
        Input noise std.
    coupling_strength : float or tuple
        Inter-column coupling weight (or range).
    spike_mode : bool
        If True, use elevated A for epileptiform spikes.
    A_spike : float
        Excitatory gain for spike mode.
    warmup_s : float
        Warmup duration discarded from output.
    e0, v0, r : float
        Sigmoid parameters.
    rng : Generator, optional

    Returns
    -------
    t : ndarray, shape (n_steps,)
    timecourses : ndarray, shape (n_columns, n_steps)
    """
    if rng is None:
        rng = np.random.default_rng()

    N = n_columns
    total_s = warmup_s + duration_s
    n_total = int(total_s / dt)
    n_warmup = int(warmup_s / dt)
    a, b = 100.0, 50.0

    # Per-column parameters (shape (N,))
    As = np.full(N, A_spike) if spike_mode else rng.uniform(*A_range, N)
    Bs = rng.uniform(*B_range, N)
    Cs = rng.uniform(*C_range, N)
    p_means = rng.uniform(*p_mean_range, N)
    C1, C2, C3, C4 = Cs, 0.8 * Cs, 0.25 * Cs, 0.25 * Cs

    # Coupling matrix
    if isinstance(coupling_strength, (int, float)):
        k_lo = k_hi = float(coupling_strength)
    else:
        k_lo, k_hi = coupling_strength
    if N > 1 and k_hi > 0:
        K = rng.uniform(k_lo, k_hi, (N, N))
        np.fill_diagonal(K, 0.0)
        mask = rng.random((N, N)) < 0.5
        np.fill_diagonal(mask, False)
        K *= mask
    else:
        K = np.zeros((N, N))

    # Pre-generate all noise at once
    p_noise = rng.normal(0, p_std, (N, n_total))
    p_noise += p_means[:, None]

    # State: (N, 6) — [y0, y1, y2, y3, y4, y5]
    state = np.zeros((N, 6))
    out = np.empty((N, n_total))

    # Pre-compute constants for the ODE
    Aa = As * a   # shape (N,)
    Bb = Bs * b
    a2 = a * a
    b2 = b * b
    two_a = 2.0 * a
    two_b = 2.0 * b
    BbC4 = Bb * C4

    for i in range(n_total):
        y0 = state[:, 0]
        y1 = state[:, 1]
        y2 = state[:, 2]
        y3 = state[:, 3]
        y4 = state[:, 4]
        y5 = state[:, 5]

        # Sigmoid firing rates — all columns at once
        S_pyr = _sigmoid(y1 - y2, e0, v0, r)
        S_e2p = _sigmoid(C1 * y0, e0, v0, r)
        S_i2p = _sigmoid(C3 * y0, e0, v0, r)

        # Coupling: K @ S_pyr
        p_ext = p_noise[:, i] + K @ S_pyr

        # ODE right-hand sides (vectorized over N columns)
        state[:, 0] += dt * y3
        state[:, 3] += dt * (Aa * S_pyr - two_a * y3 - a2 * y0)
        state[:, 1] += dt * y4
        state[:, 4] += dt * (Aa * (p_ext + C2 * S_e2p) - two_a * y4 - a2 * y1)
        state[:, 2] += dt * y5
        state[:, 5] += dt * (BbC4 * S_i2p - two_b * y5 - b2 * y2)

        out[:, i] = state[:, 1] - state[:, 2]

    out = out[:, n_warmup:]
    t = np.arange(out.shape[1]) * dt
    return t, out


class NMMSimulator:
    """Neural Mass Model simulator for source-space EEG.

    Generates source activity using Jansen-Rit neural mass models,
    assigns it to cortical patches, and projects through the leadfield.

    Follows the same API pattern as ``SimulationGenerator``:
    pre-computes a pool of NMM timecourses at init, then ``generate()``
    yields ``(x_batch, y_batch, info_df)`` batches indefinitely.

    Uses the same ``apply_sensor_noise_single`` noise pipeline as
    ``SimulationGenerator``, supporting all spatial correlation modes
    (realistic, cholesky, banded, diagonal, low_rank).

    Parameters
    ----------
    fwd : dict
        MNE forward solution (fixed orientation).
    batch_size : int
        Number of samples per batch.
    n_sources : int or tuple of int
        Number of active source patches (or range).
    duration_s : float
        Simulation duration in seconds.
    sfreq : float
        Output sampling frequency in Hz.
    dt : float
        NMM integration timestep in seconds.
    snr_range : tuple of float
        Signal-to-noise ratio range in dB.
    source_extent : int or tuple of int
        Size of each source patch in vertices (or range).
    spike_mode : bool
        Generate epileptiform spikes (elevated excitatory gain).
    coupling_strength : float or tuple of float
        Inter-column coupling weight (or range).
    correlation_mode : str or None
        Spatial noise correlation mode. Default ``"realistic"``.
    noise_color_coeff : float or tuple of float
        Spatial noise correlation strength.
    noise_temporal_beta : float or tuple of float
        Power-law exponent for temporal noise coloring (0 = white).
    noise_rank_deficiency : int or tuple of int
        Number of projected-out sensor dimensions.
    noise_low_rank_dim : int or tuple of int
        Latent rank for low_rank/realistic noise modes.
    apply_sensor_projector : bool
        Apply rank projector to signal and noise.
    estimate_noise_cov : bool
        Estimate noise covariance from independent baseline.
    noise_cov_n_baseline : int
        Baseline samples for covariance estimation.
    noise_cov_shrinkage : float
        Shrinkage factor for estimated covariance.
    return_noise_cov : bool
        Include per-sample noise covariance matrices in metadata.
    n_pool : int
        Number of NMM timecourses to pre-compute.
    max_pool_columns : int
        Maximum columns per coupled simulation when building the pool.
    random_seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        fwd,
        batch_size: int = 128,
        n_sources: int | tuple[int, int] = (1, 3),
        duration_s: float = 1.0,
        sfreq: float = 256.0,
        dt: float = 0.5e-3,
        snr_range: tuple[float, float] = (0.0, 10.0),
        source_extent: int | tuple[int, int] = (1, 30),
        spike_mode: bool = False,
        coupling_strength: float | tuple[float, float] = (10.0, 50.0),
        correlation_mode: str | None = "realistic",
        noise_color_coeff: float | tuple[float, float] = (0.25, 0.75),
        noise_temporal_beta: float | tuple[float, float] = 0.0,
        noise_rank_deficiency: int | tuple[int, int] = 0,
        noise_low_rank_dim: int | tuple[int, int] = (2, 8),
        apply_sensor_projector: bool = True,
        estimate_noise_cov: bool = True,
        noise_cov_n_baseline: int = 200,
        noise_cov_shrinkage: float = 0.05,
        return_noise_cov: bool = False,
        n_pool: int = 500,
        max_pool_columns: int = 5,
        random_seed: int | None = None,
    ):
        from copy import deepcopy

        from .spatial import build_adjacency

        self.fwd = fwd
        self.leadfield = deepcopy(fwd["sol"]["data"])
        self.n_chans, self.n_dipoles = self.leadfield.shape
        self.batch_size = batch_size
        self.duration_s = duration_s
        self.sfreq = sfreq
        self.dt = dt
        self.spike_mode = spike_mode
        self.coupling_strength = coupling_strength
        self.snr_range = snr_range
        self.n_pool = n_pool

        # Noise parameters
        self.correlation_mode = correlation_mode
        self.noise_color_coeff = noise_color_coeff
        self.noise_temporal_beta = noise_temporal_beta
        self.noise_rank_deficiency = noise_rank_deficiency
        self.noise_low_rank_dim = noise_low_rank_dim
        self.apply_sensor_projector = apply_sensor_projector
        self.estimate_noise_cov = estimate_noise_cov
        self.noise_cov_n_baseline = noise_cov_n_baseline
        self.noise_cov_shrinkage = noise_cov_shrinkage
        self.return_noise_cov = return_noise_cov

        self.rng = np.random.default_rng(random_seed)

        # Parse ranges
        if isinstance(n_sources, int):
            self.min_sources, self.max_sources = n_sources, n_sources
        else:
            self.min_sources, self.max_sources = n_sources

        if isinstance(source_extent, int):
            self.extent_range = (source_extent, source_extent)
        else:
            self.extent_range = source_extent

        # Build adjacency for region growing
        adjacency = build_adjacency(fwd, verbose=0)
        adjacency_csr = adjacency.tocsr()
        self.neighbors = [
            adjacency_csr.indices[
                adjacency_csr.indptr[i] : adjacency_csr.indptr[i + 1]
            ].astype(int)
            for i in range(self.n_dipoles)
        ]

        # Pre-compute pool of NMM timecourses
        self._precompute_pool(max_pool_columns)

    def _sample_param(self, value, n=1, dtype=float):
        """Sample n values from a scalar or (lo, hi) range."""
        if isinstance(value, (tuple, list)):
            lo, hi = value
            if np.issubdtype(np.dtype(dtype), np.integer):
                return self.rng.integers(int(lo), int(hi) + 1, size=n)
            return self.rng.uniform(lo, hi, n)
        return np.full(n, value, dtype=dtype)

    def _precompute_pool(self, max_columns: int):
        """Generate a pool of resampled NMM timecourses."""
        n_out = int(self.duration_s * self.sfreq)
        pool = []
        remaining = self.n_pool
        while remaining > 0:
            nc = min(remaining, max(self.max_sources, max_columns))
            _, tc = jansen_rit_coupled(
                n_columns=nc,
                duration_s=self.duration_s,
                dt=self.dt,
                spike_mode=self.spike_mode,
                coupling_strength=self.coupling_strength,
                rng=self.rng,
            )
            for j in range(nc):
                resampled = resample(tc[j], n_out)
                mx = np.max(np.abs(resampled))
                if mx > 0:
                    resampled /= mx
                pool.append(resampled)
            remaining -= nc
        self.tc_pool = np.array(pool[:self.n_pool])  # (n_pool, n_out)

    def _grow_patch(self, seed: int, size: int) -> np.ndarray:
        """Grow a connected patch from a seed vertex via BFS."""
        size = int(np.clip(size, 1, self.n_dipoles))
        region = [seed]
        region_set = {seed}
        frontier = list(self.neighbors[seed])
        self.rng.shuffle(frontier)
        fi = 0
        while len(region) < size and fi < len(frontier):
            v = int(frontier[fi])
            fi += 1
            if v in region_set:
                continue
            region.append(v)
            region_set.add(v)
            nbs = list(self.neighbors[v])
            self.rng.shuffle(nbs)
            frontier.extend(nbs)
        return np.array(region, dtype=int)

    def _gaussian_patch_weights(
        self, vertices: np.ndarray, center: int, sigma: float = 2.0,
    ) -> np.ndarray:
        """Compute Gaussian weights within a patch based on graph distance."""
        dists = {center: 0}
        queue = [center]
        qi = 0
        vset = set(vertices.tolist())
        while qi < len(queue):
            node = queue[qi]
            qi += 1
            for nb in self.neighbors[node]:
                nb = int(nb)
                if nb in vset and nb not in dists:
                    dists[nb] = dists[node] + 1
                    queue.append(nb)
        d = np.array([float(dists.get(int(v), len(vertices))) for v in vertices])
        w = np.exp(-d**2 / (2.0 * sigma**2))
        return w / w.max()

    def generate(self):
        """Generate batches of NMM simulations.

        Yields ``(x, y, info)`` tuples indefinitely, matching the
        ``SimulationGenerator`` API:

        - x : ndarray (batch_size, n_chans, n_timepoints) — noisy sensor data
        - y : ndarray (batch_size, n_dipoles, n_timepoints) — source activity
        - info : DataFrame with per-sample metadata
        """
        n_out = int(self.duration_s * self.sfreq)
        bs = self.batch_size

        while True:
            x_batch = np.empty((bs, self.n_chans, n_out))
            y_batch = np.empty((bs, self.n_dipoles, n_out))

            # Pre-sample noise parameters for the batch
            snr_levels = self.rng.uniform(*self.snr_range, bs)
            noise_color_coeffs = self._sample_param(self.noise_color_coeff, bs)
            temporal_betas = self._sample_param(self.noise_temporal_beta, bs)
            rank_losses = self._sample_param(
                self.noise_rank_deficiency, bs, dtype=int,
            )
            low_rank_dims = self._sample_param(
                self.noise_low_rank_dim, bs, dtype=int,
            )

            n_sources_list = []
            centers_list = []
            vertices_list = []
            snr_realized_list = []
            noise_scale_list = []
            noise_cov_model_list = []
            noise_cov_true_list = []
            noise_cov_est_list = []
            noise_cov_est_scaled_list = []

            for b in range(bs):
                n_src = int(
                    self.rng.integers(self.min_sources, self.max_sources + 1)
                )
                tc_idx = self.rng.choice(len(self.tc_pool), n_src, replace=False)
                tc = self.tc_pool[tc_idx]

                y = np.zeros((self.n_dipoles, n_out))
                centers = []
                patch_verts = []

                for i in range(n_src):
                    seed = int(self.rng.integers(0, self.n_dipoles))
                    extent = int(self.rng.integers(
                        self.extent_range[0], self.extent_range[1] + 1,
                    ))
                    verts = self._grow_patch(seed, extent)
                    weights = self._gaussian_patch_weights(verts, seed)
                    y[verts] += weights[:, None] * tc[i][None, :]
                    centers.append(seed)
                    patch_verts.append(verts)

                x_clean = self.leadfield @ y

                result = apply_sensor_noise_single(
                    x_clean,
                    self.n_chans,
                    n_out,
                    snr_db=float(snr_levels[b]),
                    correlation_mode=self.correlation_mode,
                    noise_color_coeff=float(noise_color_coeffs[b]),
                    temporal_beta=float(temporal_betas[b]),
                    rank_deficiency=int(rank_losses[b]),
                    low_rank_dim=int(low_rank_dims[b]),
                    apply_projector=self.apply_sensor_projector,
                    estimate_cov=self.estimate_noise_cov,
                    cov_n_baseline=self.noise_cov_n_baseline,
                    cov_shrinkage=self.noise_cov_shrinkage,
                    rng=self.rng,
                )

                x_batch[b] = result["x_noisy"]
                y_batch[b] = y

                n_sources_list.append(n_src)
                centers_list.append(centers)
                vertices_list.append(patch_verts)
                snr_realized_list.append(result["snr_realized"])
                noise_scale_list.append(result["noise_scale"])
                noise_cov_model_list.append(result["noise_cov_model"])
                noise_cov_true_list.append(result["noise_cov_true"])
                noise_cov_est_list.append(result["noise_cov_est"])
                noise_cov_est_scaled_list.append(result["noise_cov_est_scaled"])

            info_dict = {
                "n_sources": n_sources_list,
                "centers": centers_list,
                "source_vertices": vertices_list,
                "snr": snr_levels,
                "snr_realized": np.array(snr_realized_list),
                "noise_scale": np.array(noise_scale_list),
                "noise_color_coeff": noise_color_coeffs,
                "noise_temporal_beta": temporal_betas,
                "noise_rank_deficiency": rank_losses,
                "noise_low_rank_dim": low_rank_dims,
                "correlation_mode": self.correlation_mode,
                "n_timepoints": n_out,
                "sfreq": self.sfreq,
                "spike_mode": self.spike_mode,
            }

            if self.return_noise_cov:
                info_dict["noise_cov_model"] = noise_cov_model_list
                info_dict["noise_cov_true"] = noise_cov_true_list
                if self.estimate_noise_cov:
                    info_dict["noise_cov_est"] = noise_cov_est_list
                    info_dict["noise_cov_est_scaled"] = noise_cov_est_scaled_list

            yield SimulationBatch(
                x=x_batch,
                y=y_batch,
                info=pd.DataFrame(info_dict),
            )
