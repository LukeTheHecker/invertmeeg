from typing import Literal

from pydantic import BaseModel


class DatasetConfig(BaseModel):
    name: str
    description: str
    n_sources: int | tuple[int, int]
    n_orders: int | tuple[int, int]
    snr_range: tuple[float, float]
    n_timepoints: int
    source_spatial_model: Literal["diffusion_basis", "contiguous_gaussian"] = "diffusion_basis"
    patch_rank: int | tuple[int, int] = (1, 2)
    n_samples: int = 50


BENCHMARK_DATASETS: dict[str, DatasetConfig] = {
    "focal_source": DatasetConfig(
        name="Focal Source",
        description="Single focal dipole source with moderate noise",
        n_sources=1,
        n_orders=0,
        snr_range=(3.0, 5.0),
        n_timepoints=50,
    ),
    "multi_source": DatasetConfig(
        name="Multi Source",
        description="Multiple focal dipole sources with low-to-moderate noise",
        n_sources=(2, 4),
        n_orders=0,
        snr_range=(1.0, 3.0),
        n_timepoints=50,
    ),
    "extended_source": DatasetConfig(
        name="Extended Source",
        description="Multiple contiguous Gaussian patch sources with spatial smoothing",
        n_sources=(2, 4),
        n_orders=(1, 3),
        snr_range=(1.0, 3.0),
        source_spatial_model="contiguous_gaussian",
        patch_rank=(1, 2),
        n_timepoints=50,
    ),
    "noisy": DatasetConfig(
        name="Noisy",
        description="Variable sources under challenging low-SNR conditions",
        n_sources=(1, 3),
        n_orders=(0, 2),
        snr_range=(-5.0, 0.0),
        n_timepoints=50,
    ),
}


def create_datasets() -> dict[str, DatasetConfig]:
    return dict(BENCHMARK_DATASETS)
