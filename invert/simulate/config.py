from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SimulationConfig(BaseModel):
    """Configuration for EEG simulation generator.

    Attributes:
        batch_size: Number of simulations per batch
        batch_repetitions: Number of times to repeat each batch
        n_sources: Number of active sources (int or tuple for range)
        n_orders: Smoothness order(s) for spatial patterns
        amplitude_range: Min/max amplitude for source activity
        n_timepoints: Number of time samples per simulation
        snr_range: Signal-to-noise ratio range in dB
        n_timecourses: Number of pre-generated timecourses
        beta_range: Power-law exponent range for 1/f noise
        add_forward_error: Whether to add perturbations to leadfield
        forward_error: Magnitude of forward model error
        inter_source_correlation: Correlation between sources (float or range)
        diffusion_smoothing: Whether to use diffusion-based smoothing
        diffusion_parameter: Smoothing strength (alpha parameter)
        correlation_mode: Spatial noise correlation pattern
        noise_color_coeff: Spatial noise correlation strength
        noise_temporal_beta: 1/f^beta temporal coloring of sensor noise
        noise_rank_deficiency: Number of projected-out sensor dimensions
        apply_sensor_projector: Apply projector to both signal and noise
        noise_low_rank_dim: Rank for low-rank spatial noise model
        return_noise_cov: Include per-sample noise covariance matrices in metadata
        estimate_noise_cov: Estimate covariance from baseline noise samples
        noise_cov_n_baseline: Number of baseline samples used for covariance estimate
        noise_cov_shrinkage: Shrinkage factor for estimated covariance
        random_seed: Random seed for reproducibility
        normalize_leadfield: Whether to normalize leadfield columns
        verbose: Verbosity level
        simulation_mode: Simulation mode ('patches' or 'mixture')
        background_beta: 1/f^beta exponent for smooth background
        background_mixture_alpha: Mixing coefficient alpha (higher = more background)
        source_spatial_model: Spatial source model for patch generation
        source_extent: Number of dipoles per source patch
        patch_smoothness_sigma: Gaussian smoothness (graph-hop units) within a patch
        patch_rank: Temporal rank per patch (1=single latent, 2=two latent components)
    """

    batch_size: int = Field(default=1284, ge=1)
    batch_repetitions: int = Field(default=1, ge=1)
    n_sources: int | tuple[int, int] = Field(
        default=(1, 5), description="Single value or (min, max) tuple"
    )
    n_orders: int | tuple[int, int] = Field(
        default=(0, 3), description="Smoothness order or (min, max) tuple"
    )
    amplitude_range: tuple[float, float] = Field(
        default=(0.5, 1.0), description="Source amplitude range"
    )
    n_timepoints: int = Field(default=20, ge=1)
    snr_range: tuple[float, float] = Field(
        default=(-5.0, 5.0), description="SNR range in dB"
    )
    n_timecourses: int = Field(default=5000, ge=1)
    beta_range: tuple[float, float] = Field(default=(0.0, 3.0))
    add_forward_error: bool = Field(default=False)
    forward_error: float = Field(default=0.1, ge=0.0)
    inter_source_correlation: float | tuple[float, float] = Field(default=(0.25, 0.75))
    diffusion_smoothing: bool = Field(default=True)
    diffusion_parameter: float = Field(default=0.1, ge=0.0)
    correlation_mode: (
        Literal["auto", "cholesky", "banded", "diagonal", "low_rank"] | None
    ) = Field(default=None)
    noise_color_coeff: float | tuple[float, float] = Field(default=(0.25, 0.75))
    noise_temporal_beta: float | tuple[float, float] = Field(
        default=0.0,
        description="Power-law exponent for sensor noise (0=white).",
    )
    noise_rank_deficiency: int | tuple[int, int] = Field(
        default=0,
        description="Projected-out sensor dimensions (rank loss).",
    )
    apply_sensor_projector: bool = Field(
        default=True,
        description="Apply simulated projector to signal and sensor noise.",
    )
    noise_low_rank_dim: int | tuple[int, int] = Field(
        default=4,
        description="Latent dimension for low-rank spatial noise.",
    )
    return_noise_cov: bool = Field(
        default=False,
        description="Store per-sample noise covariance matrices in metadata.",
    )
    estimate_noise_cov: bool = Field(
        default=True,
        description="Estimate noise covariance from baseline noise samples.",
    )
    noise_cov_n_baseline: int = Field(
        default=200,
        ge=8,
        description="Baseline time samples used for covariance estimation.",
    )
    noise_cov_shrinkage: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Shrinkage factor for covariance estimation.",
    )
    random_seed: int | None = Field(default=None)
    normalize_leadfield: bool = Field(default=False)
    verbose: int = Field(default=0, ge=0)

    # Mixture mode parameters (simple background + patches)
    simulation_mode: Literal["patches", "mixture"] = Field(
        default="patches",
        description="Simulation mode: 'patches' (sparse only) or 'mixture' (smooth background + patches)",
    )
    background_beta: float | tuple[float, float] = Field(
        default=(0.5, 2.0),
        description="1/f^beta exponent for smooth background temporal dynamics",
    )
    background_mixture_alpha: float | tuple[float, float] = Field(
        default=(0.7, 0.9),
        description="Mixing coefficient alpha: y = alpha*y_background + (1-alpha)*y_patches",
    )
    source_spatial_model: Literal["diffusion_basis", "contiguous_gaussian"] = Field(
        default="diffusion_basis",
        description="Spatial source model: legacy diffusion basis or contiguous Gaussian patches.",
    )
    source_extent: int | tuple[int, int] = Field(
        default=(1, 60),
        description="Source extent as number of dipoles (single value or min/max tuple).",
    )
    patch_smoothness_sigma: float | tuple[float, float] = Field(
        default=(0.8, 2.0),
        description="Gaussian smoothness in graph-hop units for contiguous patches.",
    )
    patch_rank: int | tuple[int, int] = Field(
        default=(1, 2),
        description="Temporal rank per patch (1 or 2).",
    )

    model_config = ConfigDict(frozen=False)
