from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .covariance import get_cov
from .noise import (
    add_error,
    empirical_covariance,
    make_rank_projector,
    make_sensor_noise_covariance,
    powerlaw_noise,
    sample_sensor_noise,
    scale_noise_to_snr,
)
from .spatial import build_adjacency, build_spatial_basis


class SimulationGenerator:
    """Class-based EEG simulation generator with precomputed components.

    This generator creates realistic EEG simulations by:
    1. Generating spatially smooth source patterns
    2. Assigning colored noise timecourses to sources
    3. Projecting through the leadfield matrix
    4. Adding spatially/temporally colored sensor noise

    The class precomputes spatial smoothing operators and timecourses
    during initialization for faster batch generation.
    """

    def __init__(self, fwd, config: SimulationConfig | None = None, **kwargs):
        """Initialize the simulation generator.

        Parameters:
            fwd: MNE forward solution object
            config: SimulationConfig instance (optional)
            **kwargs: Configuration parameters (used if config is None)
        """
        # Initialize configuration
        if config is None:
            self.config = SimulationConfig(**kwargs)
        else:
            self.config = config

        # Store forward solution components
        self.fwd = fwd
        self.channel_types = np.array(fwd["info"].get_channel_types())
        self.leadfield_original = deepcopy(fwd["sol"]["data"])
        self.leadfield = deepcopy(self.leadfield_original)
        self.n_chans, self.n_dipoles = self.leadfield.shape

        # Initialize random number generator
        self.rng = np.random.default_rng(self.config.random_seed)

        # Parse n_sources parameter
        if isinstance(self.config.n_sources, (int, float)):
            n_sources_val = np.clip(self.config.n_sources, a_min=1, a_max=np.inf)
            self.min_sources, self.max_sources = int(n_sources_val), int(n_sources_val)
        else:
            self.min_sources, self.max_sources = self.config.n_sources

        # Precompute spatial smoothing operator
        self._precompute_spatial_operators()

        # Precompute timecourses
        self._precompute_timecourses()

        # Setup correlation sampling functions
        self._setup_correlation_samplers()

        # Normalize leadfield if requested
        if self.config.normalize_leadfield:
            self.leadfield /= np.linalg.norm(self.leadfield, axis=0)

    def _precompute_spatial_operators(self):
        """Precompute graph Laplacian and smoothing operators."""
        self.adjacency = build_adjacency(self.fwd, verbose=self.config.verbose)

        # Parse n_orders parameter
        if isinstance(self.config.n_orders, (tuple, list)):
            self.min_order, self.max_order = self.config.n_orders
            if self.min_order == self.max_order:
                self.max_order += 1
        else:
            self.min_order = 0
            self.max_order = self.config.n_orders

        self.sources, self.sources_dense, self.gradient = build_spatial_basis(
            self.adjacency,
            self.n_dipoles,
            self.min_order,
            self.max_order,
            diffusion_smoothing=self.config.diffusion_smoothing,
            diffusion_parameter=self.config.diffusion_parameter,
        )
        self.n_candidates = self.sources.shape[0]

    def _precompute_timecourses(self):
        """Precompute colored noise timecourses."""
        betas = self.rng.uniform(*self.config.beta_range, self.config.n_timecourses)

        time_courses = powerlaw_noise(
            betas,
            self.config.n_timepoints,
            n_signals=self.config.n_timecourses,
            rng=self.rng,
        )

        # Normalize to max(abs()) == 1
        self.time_courses = (time_courses.T / abs(time_courses).max(axis=1)).T

    def _setup_correlation_samplers(self):
        """Setup sampling functions for simulation hyperparameters."""

        def _sampler(value, n=1, dtype=float):
            if isinstance(value, (tuple, list)):
                lo, hi = value
                if np.issubdtype(np.dtype(dtype), np.integer):
                    return self.rng.integers(int(lo), int(hi) + 1, size=n)
                return self.rng.uniform(lo, hi, n)
            return np.full(n, value, dtype=dtype)

        isc = self.config.inter_source_correlation
        self.get_inter_source_correlation = lambda n=1: _sampler(isc, n=n, dtype=float)

        ncc = self.config.noise_color_coeff
        self.get_noise_color_coeff = lambda n=1: _sampler(ncc, n=n, dtype=float)

        ntb = self.config.noise_temporal_beta
        self.get_noise_temporal_beta = lambda n=1: _sampler(ntb, n=n, dtype=float)

        nrd = self.config.noise_rank_deficiency
        self.get_noise_rank_deficiency = lambda n=1: _sampler(nrd, n=n, dtype=int)

        nlr = self.config.noise_low_rank_dim
        self.get_noise_low_rank_dim = lambda n=1: _sampler(nlr, n=n, dtype=int)

    def _generate_smooth_background(self, batch_size):
        """Generate smooth background activity with 1/f^beta temporal dynamics.

        Uses vectorized FFT-based colored noise generation instead of
        per-dipole Python loops.

        Parameters:
            batch_size: Number of simulations to generate

        Returns:
            y_background: [batch_size, n_dipoles, n_timepoints] background activity
        """
        # Sample beta parameters for background
        if isinstance(self.config.background_beta, tuple):
            betas = self.rng.uniform(*self.config.background_beta, batch_size)
        else:
            betas = np.full(batch_size, self.config.background_beta)

        y_background_all = np.empty(
            (batch_size, self.n_dipoles, self.config.n_timepoints)
        )

        for b_idx, beta in enumerate(betas):
            # Vectorized: generate all dipole timecourses at once
            background_timecourses = powerlaw_noise(
                beta, self.config.n_timepoints, n_signals=self.n_dipoles, rng=self.rng
            )  # [n_dipoles, n_timepoints]

            # Apply spatial smoothing using gradient operator
            background_smooth = self.gradient @ background_timecourses

            # Normalize
            max_val = np.max(np.abs(background_smooth))
            if max_val > 0:
                background_smooth = background_smooth / max_val
            y_background_all[b_idx] = background_smooth

        return y_background_all

    def _setup_leadfield(self):
        """Get the leadfield matrix, optionally with forward model error."""
        if self.config.add_forward_error:
            return add_error(
                self.leadfield_original,
                self.config.forward_error,
                self.gradient,
                self.rng,
            )
        return self.leadfield

    def _generate_patches(self, batch_size):
        """Generate patch-based source activity.

        Returns:
            y_patches: [batch_size, n_dipoles, n_timepoints] patch activity
            selection: list of source index arrays
            amplitude_values: list of amplitude arrays
            inter_source_correlations: array of correlation values
            noise_color_coeffs: array of noise color coefficients
        """
        n_sources_batch = self.rng.integers(
            self.min_sources, self.max_sources + 1, batch_size
        )

        # Select source locations
        selection = [
            self.rng.integers(0, self.n_candidates, n) for n in n_sources_batch
        ]

        # Sample amplitudes and timecourses
        amplitude_values = [
            self.rng.uniform(*self.config.amplitude_range, n) for n in n_sources_batch
        ]
        timecourse_choices = [
            self.rng.choice(self.config.n_timecourses, n) for n in n_sources_batch
        ]
        amplitudes = [self.time_courses[choice].T for choice in timecourse_choices]

        # Apply inter-source correlations
        inter_source_correlations = self.get_inter_source_correlation(n=batch_size)
        noise_color_coeffs = self.get_noise_color_coeff(n=batch_size)

        source_covariances = [
            get_cov(n, isc)
            for n, isc in zip(n_sources_batch, inter_source_correlations, strict=False)
        ]
        amplitudes = [
            amp @ np.diag(amplitude_values[i]) @ cov
            for i, (amp, cov) in enumerate(zip(amplitudes, source_covariances, strict=False))
        ]

        # Generate patch activity using dense source matrix for fast indexing
        y_patches = np.stack(
            [
                (amplitudes[i] @ self.sources_dense[selection[i]]).T
                / len(amplitudes[i])
                for i in range(batch_size)
            ],
            axis=0,
        )

        return (
            y_patches,
            n_sources_batch,
            selection,
            amplitude_values,
            inter_source_correlations,
            noise_color_coeffs,
        )

    def _generate_background(self, batch_size, y_patches):
        """Mix background activity with patches if in mixture mode.

        Returns:
            y: [batch_size, n_dipoles, n_timepoints] combined activity
            alphas: mixing coefficients or None
        """
        if self.config.simulation_mode == "mixture":
            y_background = self._generate_smooth_background(batch_size)

            if isinstance(self.config.background_mixture_alpha, tuple):
                alphas = self.rng.uniform(
                    *self.config.background_mixture_alpha, batch_size
                )
            else:
                alphas = np.full(batch_size, self.config.background_mixture_alpha)

            alphas_bc = alphas[:, np.newaxis, np.newaxis]
            y = alphas_bc * y_background + (1 - alphas_bc) * y_patches
        else:
            y = y_patches
            alphas = None

        return y, alphas

    def _apply_noise(self, x, batch_size, noise_color_coeffs, modes_batch):
        """Apply realistic sensor noise with optional projection/rank loss.

        Returns
        -------
        x_noisy : np.ndarray
            Noisy EEG data [batch_size, n_channels, n_timepoints].
        noise_meta : dict
            Per-sample noise metadata (SNR, projector, covariance statistics).
        """
        snr_levels = self.rng.uniform(
            low=self.config.snr_range[0],
            high=self.config.snr_range[1],
            size=batch_size,
        )
        temporal_betas = self.get_noise_temporal_beta(n=batch_size)
        rank_losses = self.get_noise_rank_deficiency(n=batch_size)
        low_rank_dims = self.get_noise_low_rank_dim(n=batch_size)

        x_noisy_list = []
        realized_snrs = []
        projectors = []
        projector_ranks = []
        covariance_true = []
        covariance_est = []
        covariance_rank_true = []
        covariance_rank_est = []

        for (
            xx,
            snr_level,
            corr_mode,
            noise_color_level,
            temporal_beta,
            rank_loss,
            low_rank_dim,
        ) in zip(
            x,
            snr_levels,
            modes_batch,
            noise_color_coeffs,
            temporal_betas,
            rank_losses,
            low_rank_dims,
            strict=False,
        ):
            P, _ = make_rank_projector(
                self.n_chans, rank_deficiency=int(rank_loss), rng=self.rng
            )
            if self.config.apply_sensor_projector:
                signal = P @ xx
            else:
                signal = xx

            cov_spatial = make_sensor_noise_covariance(
                self.n_chans,
                mode=corr_mode,
                noise_color_coeff=float(noise_color_level),
                rng=self.rng,
                low_rank_dim=int(low_rank_dim),
            )

            noise = sample_sensor_noise(
                cov_spatial,
                self.config.n_timepoints,
                rng=self.rng,
                temporal_beta=float(temporal_beta),
            )
            if self.config.apply_sensor_projector:
                noise = P @ noise

            noise_scaled, _scale, realized_snr = scale_noise_to_snr(
                signal, noise, float(snr_level)
            )
            x_noisy = signal + noise_scaled

            # True covariance from realized noise in this sample.
            cov_true = empirical_covariance(noise_scaled, shrinkage=0.0, eps=0.0)

            # Estimated covariance from independent baseline noise.
            cov_est = None
            if self.config.estimate_noise_cov:
                baseline = sample_sensor_noise(
                    cov_spatial,
                    int(self.config.noise_cov_n_baseline),
                    rng=self.rng,
                    temporal_beta=float(temporal_beta),
                )
                if self.config.apply_sensor_projector:
                    baseline = P @ baseline
                cov_est = empirical_covariance(
                    baseline, shrinkage=float(self.config.noise_cov_shrinkage)
                )

            x_noisy_list.append(x_noisy)
            realized_snrs.append(realized_snr)
            projectors.append(P)
            projector_ranks.append(int(np.linalg.matrix_rank(P)))
            covariance_true.append(cov_true)
            covariance_rank_true.append(int(np.linalg.matrix_rank(cov_true)))
            covariance_est.append(cov_est)
            covariance_rank_est.append(
                int(np.linalg.matrix_rank(cov_est))
                if cov_est is not None
                else -1
            )

        x_noisy = np.stack(x_noisy_list, axis=0)
        noise_meta = {
            "snr_target": snr_levels,
            "snr_realized": np.asarray(realized_snrs, dtype=float),
            "projector": projectors,
            "projector_rank": np.asarray(projector_ranks, dtype=int),
            "noise_cov_true": covariance_true,
            "noise_cov_est": covariance_est,
            "noise_cov_rank_true": np.asarray(covariance_rank_true, dtype=int),
            "noise_cov_rank_est": np.asarray(covariance_rank_est, dtype=int),
            "noise_temporal_beta": np.asarray(temporal_betas, dtype=float),
            "noise_rank_deficiency": np.asarray(rank_losses, dtype=int),
            "noise_low_rank_dim": np.asarray(low_rank_dims, dtype=int),
        }
        return x_noisy, noise_meta

    def _build_metadata(
        self,
        batch_size,
        n_sources_batch,
        amplitude_values,
        inter_source_correlations,
        noise_color_coeffs,
        modes_batch,
        selection,
        alphas,
        noise_meta,
    ):
        """Build simulation metadata DataFrame."""
        info_dict = {
            "n_sources": n_sources_batch,
            "amplitudes": amplitude_values,
            "snr": noise_meta["snr_target"],
            "snr_realized": noise_meta["snr_realized"],
            "inter_source_correlations": inter_source_correlations,
            "n_orders": [[self.min_order, self.max_order]] * batch_size,
            "diffusion_parameter": [self.config.diffusion_parameter] * batch_size,
            "n_timepoints": [self.config.n_timepoints] * batch_size,
            "n_timecourses": [self.config.n_timecourses] * batch_size,
            "correlation_mode": modes_batch,
            "noise_color_coeff": noise_color_coeffs,
            "noise_temporal_beta": noise_meta["noise_temporal_beta"],
            "noise_rank_deficiency": noise_meta["noise_rank_deficiency"],
            "projector_rank": noise_meta["projector_rank"],
            "noise_cov_rank_true": noise_meta["noise_cov_rank_true"],
            "add_forward_error": [self.config.add_forward_error] * batch_size,
            "forward_error": [self.config.forward_error] * batch_size,
            "centers": selection,
            "simulation_mode": [self.config.simulation_mode] * batch_size,
        }

        if self.config.estimate_noise_cov:
            info_dict["noise_cov_rank_est"] = noise_meta["noise_cov_rank_est"]

        if self.config.return_noise_cov:
            info_dict["projector"] = noise_meta["projector"]
            info_dict["noise_cov_true"] = noise_meta["noise_cov_true"]
            if self.config.estimate_noise_cov:
                info_dict["noise_cov_est"] = noise_meta["noise_cov_est"]

        if self.config.simulation_mode == "mixture":
            info_dict.update(
                {
                    "background_beta": [self.config.background_beta] * batch_size,
                    "background_mixture_alpha": alphas,
                }
            )

        return pd.DataFrame(info_dict)

    def generate(self):
        """Generate batches of simulations.

        Yields:
            tuple: (x, y, info) where:
                - x: EEG data [batch_size, n_channels, n_timepoints]
                - y: Source activity [batch_size, n_dipoles, n_timepoints] (scaled)
                - info: DataFrame with simulation metadata
        """
        # Setup correlation modes
        if (
            isinstance(self.config.correlation_mode, str)
            and self.config.correlation_mode.lower() == "auto"
        ):
            correlation_modes = ["cholesky", "banded", "diagonal", "low_rank", None]
            modes_batch = self.rng.choice(correlation_modes, self.config.batch_size)
        else:
            modes_batch = [self.config.correlation_mode] * self.config.batch_size

        while True:
            leadfield = self._setup_leadfield()

            (
                y_patches,
                n_sources_batch,
                selection,
                amplitude_values,
                inter_source_correlations,
                noise_color_coeffs,
            ) = self._generate_patches(self.config.batch_size)

            y, alphas = self._generate_background(self.config.batch_size, y_patches)

            # Vectorized leadfield projection
            x = np.einsum("cd,bdt->bct", leadfield, y)

            x, noise_meta = self._apply_noise(
                x, self.config.batch_size, noise_color_coeffs, modes_batch
            )

            info = self._build_metadata(
                self.config.batch_size,
                n_sources_batch,
                amplitude_values,
                inter_source_correlations,
                noise_color_coeffs,
                modes_batch,
                selection,
                alphas,
                noise_meta,
            )

            output = (x, y, info)

            for _ in range(self.config.batch_repetitions):
                yield output
