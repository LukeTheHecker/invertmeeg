from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .covariance import get_cov
from .noise import add_error, add_white_noise, powerlaw_noise
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
        """Setup sampling functions for correlation parameters."""
        isc = self.config.inter_source_correlation
        if isinstance(isc, (tuple, list)):
            self.get_inter_source_correlation = lambda n=1: self.rng.uniform(
                isc[0], isc[1], n
            )
        else:
            self.get_inter_source_correlation = lambda n=1: np.full(n, isc)

        ncc = self.config.noise_color_coeff
        if isinstance(ncc, (tuple, list)):
            self.get_noise_color_coeff = lambda n=1: self.rng.uniform(ncc[0], ncc[1], n)
        else:
            self.get_noise_color_coeff = lambda n=1: np.full(n, ncc)

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
        """Apply sensor noise to EEG data.

        Returns:
            x: [batch_size, n_channels, n_timepoints] noisy EEG data
            snr_levels: array of SNR values used
        """
        snr_levels = self.rng.uniform(
            low=self.config.snr_range[0],
            high=self.config.snr_range[1],
            size=batch_size,
        )

        x = np.stack(
            [
                add_white_noise(
                    xx,
                    snr_level,
                    self.rng,
                    self.channel_types,
                    correlation_mode=corr_mode,
                    noise_color_coeff=noise_color_level,
                )
                for (xx, snr_level, corr_mode, noise_color_level) in zip(
                    x, snr_levels, modes_batch, noise_color_coeffs, strict=False
                )
            ],
            axis=0,
        )

        return x, snr_levels

    def _build_metadata(
        self,
        batch_size,
        n_sources_batch,
        amplitude_values,
        snr_levels,
        inter_source_correlations,
        noise_color_coeffs,
        selection,
        alphas,
    ):
        """Build simulation metadata DataFrame."""
        info_dict = {
            "n_sources": n_sources_batch,
            "amplitudes": amplitude_values,
            "snr": snr_levels,
            "inter_source_correlations": inter_source_correlations,
            "n_orders": [[self.min_order, self.max_order]] * batch_size,
            "diffusion_parameter": [self.config.diffusion_parameter] * batch_size,
            "n_timepoints": [self.config.n_timepoints] * batch_size,
            "n_timecourses": [self.config.n_timecourses] * batch_size,
            "correlation_mode": [self.config.correlation_mode] * batch_size,
            "noise_color_coeff": noise_color_coeffs,
            "centers": selection,
            "simulation_mode": [self.config.simulation_mode] * batch_size,
        }

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
            correlation_modes = ["cholesky", "banded", "diagonal", None]
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

            x, snr_levels = self._apply_noise(
                x, self.config.batch_size, noise_color_coeffs, modes_batch
            )

            info = self._build_metadata(
                self.config.batch_size,
                n_sources_batch,
                amplitude_values,
                snr_levels,
                inter_source_correlations,
                noise_color_coeffs,
                selection,
                alphas,
            )

            output = (x, y, info)

            for _ in range(self.config.batch_repetitions):
                yield output


