from typing import Literal, Optional, Union

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
        random_seed: Random seed for reproducibility
        normalize_leadfield: Whether to normalize leadfield columns
        verbose: Verbosity level
        simulation_mode: Simulation mode ('patches' or 'mixture')
        background_beta: 1/f^beta exponent for smooth background
        background_mixture_alpha: Mixing coefficient alpha (higher = more background)
    """

    batch_size: int = Field(default=1284, ge=1)
    batch_repetitions: int = Field(default=1, ge=1)
    n_sources: Union[int, tuple[int, int]] = Field(
        default=(1, 5), description="Single value or (min, max) tuple"
    )
    n_orders: Union[int, tuple[int, int]] = Field(
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
    inter_source_correlation: Union[float, tuple[float, float]] = Field(
        default=(0.25, 0.75)
    )
    diffusion_smoothing: bool = Field(default=True)
    diffusion_parameter: float = Field(default=0.1, ge=0.0)
    correlation_mode: Optional[
        Union[Literal["auto", "cholesky", "banded", "diagonal"], None]
    ] = Field(default=None)
    noise_color_coeff: Union[float, tuple[float, float]] = Field(default=(0.25, 0.75))
    random_seed: Optional[int] = Field(default=None)
    normalize_leadfield: bool = Field(default=False)
    verbose: int = Field(default=0, ge=0)

    # Mixture mode parameters (simple background + patches)
    simulation_mode: Literal["patches", "mixture"] = Field(
        default="patches",
        description="Simulation mode: 'patches' (sparse only) or 'mixture' (smooth background + patches)",
    )
    background_beta: Union[float, tuple[float, float]] = Field(
        default=(0.5, 2.0),
        description="1/f^beta exponent for smooth background temporal dynamics",
    )
    background_mixture_alpha: Union[float, tuple[float, float]] = Field(
        default=(0.7, 0.9),
        description="Mixing coefficient alpha: y = alpha*y_background + (1-alpha)*y_patches",
    )

    model_config = ConfigDict(frozen=False)
