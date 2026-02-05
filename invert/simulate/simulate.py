from __future__ import annotations

import warnings
from copy import deepcopy

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .covariance import compute_covariance, gen_correlated_sources, get_cov
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
            for n, isc in zip(n_sources_batch, inter_source_correlations)
        ]
        amplitudes = [
            amp @ np.diag(amplitude_values[i]) @ cov
            for i, (amp, cov) in enumerate(zip(amplitudes, source_covariances))
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
                    x, snr_levels, modes_batch, noise_color_coeffs
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


def generator(
    fwd,
    use_cov=True,
    cov_type="basic",
    batch_size=1284,
    batch_repetitions=30,
    n_sources=10,
    n_orders=2,
    amplitude_range=(0.5, 1),
    n_timepoints=20,
    snr_range=(-5, 5),
    n_timecourses=5000,
    beta_range=(0, 3),
    return_mask=True,
    scale_data=True,
    return_info=False,
    add_forward_error=False,
    forward_error=0.1,
    remove_channel_dim=False,
    inter_source_correlation=(0.25, 0.75),
    diffusion_smoothing=True,
    diffusion_parameter=0.1,
    correlation_mode=None,
    noise_color_coeff=0.5,
    random_seed=None,
    normalize_leadfield=False,
    verbose=0,
):
    """
    .. deprecated::
        Use :class:`SimulationGenerator` instead.

    Parameters
    ----------
    fwd : object
        Forward solution object containing the source space and orientation information.
    use_cov : bool
        If True, a covariance matrix is used in the simulation. Default is True.
    batch_size : int
        Size of each batch of simulations. Default is 1284.
    batch_repetitions : int
        Number of repetitions of each batch. Default is 30.
    n_sources : int
        Number of sources in the brain from which activity is simulated. Default is 10.
    n_orders : int
        The order of the model used to generate time courses. Default is 2.
    amplitude_range : tuple
        Range of possible amplitudes for the simulated sources. Default is (0.001,1).
    n_timepoints : int
        Number of timepoints in each simulated time course. Default is 20.
    snr_range : tuple
        Range of signal to noise ratios (in dB) to be used in the simulations. Default is (-5, 5 dB).
    n_timecourses : int
        Number of unique time courses to simulate. Default is 5000.
    beta_range : tuple
        Range of possible power-law exponents for the power spectral density of the simulated sources. Default is (0, 3).
    return_mask : bool
        If True, the function will also return a mask of the sources used. Default is True.
    scale_data : bool
        If True, the EEG data will be scaled. Default is True.
    return_info : bool
        If True, the function will return a dictionary with information about the generated data. Default is False.
    add_forward_error : bool
        If True, the function will add an error to the forward model. Default is False.
    forward_error : float
        Amount of error to add to the forward model if 'add_forward_error' is True. Default is 0.1.
    remove_channel_dim : bool
        If True, the channel dimension will be removed from the output data. Default is False.
    inter_source_correlation : float|Tuple
        The level of correlation between different sources. Default is 0.5.
    diffusion_smoothing : bool
        Whether to use diffusion smoothing. Default is True.
    diffusion_parameter : float
        The diffusion parameter (alpha). Default is 0.1.
    correlation_mode : None/str
        correlation_mode : None/str
        None implies no correlation between the noise in different channels.
        'banded' : Colored banded noise, where channels closer to each other will be more correlated.
        'diagonal' : Channels have varying degrees of noise.
    noise_color_coeff : float
        The magnitude of spatial coloring of the noise.
    random_seed : None / int
        The random seed for replicable simulations
    verbose : int
        Level of verbosity for the function. Default is 0.

    Return
    ------
    x : numpy.ndarray
        The EEG data matrix.
    y : numpy.ndarray
        The source data matrix.
    """
    warnings.warn(
        "generator() is deprecated. Use SimulationGenerator instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    rng = np.random.default_rng(random_seed)
    channel_types = fwd["info"].get_channel_types()
    leadfield = deepcopy(fwd["sol"]["data"])
    leadfield_original = deepcopy(fwd["sol"]["data"])
    n_chans, n_dipoles = leadfield.shape

    if isinstance(n_sources, (int, float)):
        n_sources = [np.clip(n_sources, a_min=1, a_max=np.inf), n_sources]
    min_sources, max_sources = n_sources

    adjacency = build_adjacency(fwd, verbose=verbose)

    sources, sources_dense, gradient = build_spatial_basis(
        adjacency,
        n_dipoles,
        0 if not isinstance(n_orders, (tuple, list)) else n_orders[0],
        n_orders if not isinstance(n_orders, (tuple, list)) else n_orders[1],
        diffusion_smoothing=diffusion_smoothing,
        diffusion_parameter=diffusion_parameter,
    )

    # Parse n_orders for min/max
    if isinstance(n_orders, (tuple, list)):
        min_order, max_order = n_orders
        if min_order == max_order:
            max_order += 1
    else:
        min_order = 0
        max_order = n_orders

    del adjacency

    # Normalize columns of the leadfield
    if normalize_leadfield:
        leadfield /= np.linalg.norm(leadfield, axis=0)

    if isinstance(inter_source_correlation, (tuple, list)):

        def get_inter_source_correlation_fn(n=1):
            return rng.uniform(
                inter_source_correlation[0], inter_source_correlation[1], n
            )
    else:

        def get_inter_source_correlation_fn(n=1):
            return rng.uniform(inter_source_correlation, inter_source_correlation, n)

    if isinstance(noise_color_coeff, (tuple, list)):

        def get_noise_color_coeff_fn(n=1):
            return rng.uniform(noise_color_coeff[0], noise_color_coeff[1], n)
    else:

        def get_noise_color_coeff_fn(n=1):
            return rng.uniform(noise_color_coeff, noise_color_coeff, n)

    if isinstance(correlation_mode, str) and correlation_mode.lower() == "auto":
        correlation_modes = ["cholesky", "banded", "diagonal", None]
        correlation_modes = rng.choice(correlation_modes, batch_size)
    else:
        correlation_modes = [
            correlation_mode,
        ] * batch_size

    n_candidates = sources.shape[0]

    # Pre-compute random time courses using vectorized FFT
    betas = rng.uniform(*beta_range, n_timecourses)
    time_courses = powerlaw_noise(betas, n_timepoints, n_signals=n_timecourses, rng=rng)

    # Normalize time course to max(abs()) == 1
    time_courses = (time_courses.T / abs(time_courses).max(axis=1)).T

    while True:
        if add_forward_error:
            leadfield = add_error(leadfield_original, forward_error, gradient, rng)
        # select sources or source patches
        n_sources_batch = rng.integers(min_sources, max_sources + 1, batch_size)
        selection = [rng.integers(0, n_candidates, n) for n in n_sources_batch]

        amplitude_values = [rng.uniform(*amplitude_range, n) for n in n_sources_batch]
        choices = [rng.choice(n_timecourses, n) for n in n_sources_batch]
        amplitudes = [time_courses[choice].T for choice in choices]

        inter_source_correlations = get_inter_source_correlation_fn(n=batch_size)
        if not isinstance(noise_color_coeff, str):
            noise_color_coeffs = get_noise_color_coeff_fn(n=batch_size)
        else:
            noise_color_coeffs = [
                noise_color_coeff,
            ] * batch_size
        source_covariances = [
            get_cov(n, isc)
            for n, isc in zip(n_sources_batch, inter_source_correlations)
        ]
        amplitudes = [
            amp @ np.diag(amplitude_values[i]) @ cov
            for i, (amp, cov) in enumerate(zip(amplitudes, source_covariances))
        ]

        y = np.stack(
            [
                (amplitudes[i] @ sources_dense[selection[i]]) / len(amplitudes[i])
                for i in range(batch_size)
            ],
            axis=0,
        )

        # Project simulated sources through leadfield (vectorized)
        x = np.einsum("cd,bdt->bct", leadfield, np.swapaxes(y, 1, 2))

        # Add white noise to clean EEG
        snr_levels = rng.uniform(low=snr_range[0], high=snr_range[1], size=batch_size)
        x = np.stack(
            [
                add_white_noise(
                    xx,
                    snr_level,
                    rng,
                    channel_types,
                    correlation_mode=corr_mode,
                    noise_color_coeff=noise_color_level,
                )
                for (xx, snr_level, corr_mode, noise_color_level) in zip(
                    x, snr_levels, correlation_modes, noise_color_coeffs
                )
            ],
            axis=0,
        )

        if use_cov:
            # Calculate Covariance

            x = np.stack(
                [compute_covariance(xx, cov_type=cov_type) for xx in x], axis=0
            )

            # Normalize Covariance to abs. max. of 1
            x = np.stack([C / np.max(abs(C)) for C in x], axis=0)

            if not remove_channel_dim:
                x = np.expand_dims(x, axis=-1)
        else:
            if scale_data:
                x = np.stack([xx / np.max(abs(xx)) for xx in x], axis=0)
            # Reshape
            x = np.swapaxes(x, 1, 2)

        if return_mask:
            # (1) binary
            # Calculate mean source activity
            y = abs(y).mean(axis=1)
            # Masking the source vector (1-> active, 0-> inactive)
            y = (y > 0).astype(float)
        else:
            if scale_data:
                y = np.stack([(yy.T / np.max(abs(yy), axis=1)).T for yy in y], axis=0)

        if return_info:
            info = pd.DataFrame(
                dict(
                    n_sources=n_sources_batch,
                    amplitudes=amplitude_values,
                    snr=snr_levels,
                    inter_source_correlations=inter_source_correlations,
                    n_orders=[
                        [min_order, max_order],
                    ]
                    * batch_size,
                    diffusion_parameter=[
                        diffusion_parameter,
                    ]
                    * batch_size,
                    n_timepoints=[
                        n_timepoints,
                    ]
                    * batch_size,
                    n_timecourses=[
                        n_timecourses,
                    ]
                    * batch_size,
                    correlation_mode=[
                        correlation_mode,
                    ]
                    * batch_size,
                    noise_color_coeff=noise_color_coeffs,
                    centers=selection,
                )
            )
            output = (x, y, info)
        else:
            output = (x, y)

        for _ in range(batch_repetitions):
            yield output


def generator_simple(
    fwd, batch_size, corrs, T, n_sources, SNR_range, random_seed=42, return_info=True
):
    """
    .. deprecated::
        Use :class:`SimulationGenerator` instead.
    """
    warnings.warn(
        "generator_simple() is deprecated. Use SimulationGenerator instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    rng = np.random.default_rng(random_seed)
    leadfield = deepcopy(fwd["sol"]["data"])
    leadfield /= leadfield.std(axis=0)
    n_channels, n_dipoles = leadfield.shape

    while True:
        sim_info = list()
        X = np.zeros((batch_size, n_channels, T))
        y = np.zeros((batch_size, n_dipoles, T))
        corrs_batch = rng.uniform(corrs[0], corrs[1], batch_size)
        SNR_batch = rng.uniform(SNR_range[0], SNR_range[1], batch_size)
        indices = [
            rng.choice(fwd["sol"]["data"].shape[1], n_sources)
            for _ in range(batch_size)
        ]

        for i in range(batch_size):
            X[i], y[i] = generator_single_simple(
                leadfield,
                corrs_batch[i],
                T,
                n_sources,
                indices[i],
                SNR_batch[i],
                random_seed=random_seed,
            )
            d = dict(
                n_sources=n_sources,
                amplitudes=1,
                snr=SNR_batch[i],
                inter_source_correlations=corrs_batch[i],
                n_orders=[0, 0],
                diffusion_parameter=0,
                n_timepoints=T,
                n_timecourses=np.inf,
                iid_noise=True,
            )
            sim_info.append(d)
        if return_info:
            sim_info = pd.DataFrame(sim_info)
            yield X, y, sim_info
        else:
            yield X, y


def generator_single_simple(
    leadfield, corr, T, n_sources, indices, SNR, random_seed=42
):
    """
    .. deprecated::
        Use :class:`SimulationGenerator` instead.

    Parameters
    ----------
    leadfield : numpy.ndarray
        The leadfield matrix.
    corr : float
        The correlation coefficient between the sources.
    T : int
        The number of time points in the sources.
    n_sources : int
        The number of sources to generate.
    indices : list
        The indices of the sources to generate.
    SNR : float
        The signal to noise ratio.
    random_seed : int
        The random seed for replicable simulations.

    Return
    ------
    X : numpy.ndarray
        The simulated EEG data.
    y: numpy.ndarray
        The simulated source data.
    """
    warnings.warn(
        "generator_single_simple() is deprecated. Use SimulationGenerator instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    rng = np.random.default_rng(random_seed)

    S = gen_correlated_sources(corr, T, n_sources)
    M = leadfield[:, indices] @ S  # use Ground Truth Gain matrix
    n_channels, n_dipoles = leadfield.shape

    scale = np.max(abs(M))
    Ms = M * scale
    MEG_energy = np.trace(Ms @ Ms.T) / (n_channels * T)
    noise_var = MEG_energy / (10 ** (SNR / 10))
    Noise = rng.standard_normal((n_channels, T)) * np.sqrt(noise_var)
    X = Ms + Noise
    y = np.zeros((n_dipoles, T))
    y[indices, :] = S

    return X, y
