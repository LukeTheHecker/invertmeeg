from typing import Literal

from pydantic import BaseModel


class DatasetConfig(BaseModel):
    name: str
    description: str
    simulator_type: Literal["default", "nmm"] = "default"
    n_sources: int | tuple[int, int]
    n_orders: int | tuple[int, int] = 0
    snr_range: tuple[float, float]
    n_timepoints: int
    source_spatial_model: Literal["diffusion_basis", "contiguous_gaussian"] = (
        "diffusion_basis"
    )
    patch_rank: int | tuple[int, int] = (1, 2)
    simulation_mode: Literal["patches", "mixture"] = "patches"
    background_beta: float | tuple[float, float] | None = None
    background_mixture_alpha: float | tuple[float, float] | None = None
    amplitude_range: tuple[float, float] | None = None
    amplitude_distribution: Literal["uniform", "log_uniform"] | None = None
    correlation_mode: (
        Literal["auto", "banded", "diagonal", "low_rank", "cholesky", "realistic"]
        | None
    ) = None
    correlation_modes: (
        list[Literal["banded", "diagonal", "low_rank", "realistic"]] | None
    ) = None
    correlation_mode_weights: list[float] | None = None
    noise_temporal_beta: float | tuple[float, float] | None = None
    noise_low_rank_dim: int | tuple[int, int] | None = None
    source_extent: int | tuple[int, int] | None = None
    n_samples: int = 50

    # NMM-specific parameters (only used when simulator_type="nmm")
    duration_s: float = 1.0
    sfreq: float = 256.0
    spike_mode: bool = False
    coupling_strength: float | tuple[float, float] = (10.0, 50.0)
    n_pool: int = 500


BENCHMARK_DATASETS: dict[str, DatasetConfig] = {
    "focal_source": DatasetConfig(
        name="Focal Source",
        description="Single focal dipole source with moderate noise",
        n_sources=1,
        n_orders=0,
        snr_range=(3.0, 5.0),
        n_timepoints=200,
        correlation_mode="realistic",
        noise_low_rank_dim=(2, 8),
    ),
    "multi_source": DatasetConfig(
        name="Multi Source",
        description="Multiple focal dipole sources with low-to-moderate noise",
        n_sources=(2, 4),
        n_orders=0,
        snr_range=(1.0, 3.0),
        n_timepoints=200,
        correlation_mode="realistic",
        noise_low_rank_dim=(2, 8),
    ),
    "extended_source": DatasetConfig(
        name="Extended Source",
        description="Multiple contiguous Gaussian patch sources with spatial smoothing",
        n_sources=(2, 4),
        n_orders=(1, 3),
        snr_range=(1.0, 3.0),
        source_spatial_model="contiguous_gaussian",
        patch_rank=(1, 2),
        n_timepoints=200,
        correlation_mode="realistic",
        noise_low_rank_dim=(2, 8),
    ),
    "noisy": DatasetConfig(
        name="Noisy",
        description="Variable sources under challenging low-SNR conditions",
        n_sources=(2, 4),
        n_orders=(1, 3),
        snr_range=(-5.0, 0.0),
        source_spatial_model="contiguous_gaussian",
        patch_rank=(1, 2),
        n_timepoints=200,
        correlation_mode="realistic",
        noise_low_rank_dim=(2, 8),
    ),
    "nmm": DatasetConfig(
        name="Neural Mass Model",
        description=(
            "Jansen-Rit neural mass model sources with coupled cortical columns, "
            "realistic sensor noise, and contiguous spatial patches"
        ),
        simulator_type="nmm",
        n_sources=(2, 4),
        snr_range=(1.0, 3.0),
        n_timepoints=200,
        source_extent=(5, 30),
        correlation_mode="realistic",
        noise_low_rank_dim=(2, 8),
        duration_s=1.0,
        sfreq=256.0,
        spike_mode=False,
    ),
    "mixed_cortical_activity": DatasetConfig(
        name="Mixed Cortical Activity",
        description=(
            "Contiguous low-rank cortical patches on top of diffuse 1/f background "
            "with realistic sensor noise"
        ),
        simulator_type="default",
        n_sources=(2, 4),
        n_orders=0,
        snr_range=(1.0, 5.0),
        n_timepoints=256,
        source_spatial_model="contiguous_gaussian",
        patch_rank=(1, 2),
        simulation_mode="mixture",
        background_beta=(1.0, 2.5),
        background_mixture_alpha=(0.10, 0.30),
        amplitude_range=(0.2, 2.0),
        amplitude_distribution="log_uniform",
        correlation_mode="realistic",
        noise_temporal_beta=(0.3, 1.2),
        noise_low_rank_dim=(3, 10),
        source_extent=(3, 20),
        n_samples=100,
    ),
}


def create_datasets() -> dict[str, DatasetConfig]:
    return dict(BENCHMARK_DATASETS)
