from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

_TORCH_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    import torch  # type: ignore[import-not-found]
    import torch.nn as nn  # type: ignore[import-not-found]
    import torch.nn.functional as F  # type: ignore[import-not-found]
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc


def _require_torch() -> None:
    if _TORCH_IMPORT_ERROR is not None:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for neural-network solvers. "
            'Install it via `pip install "invertmeeg[ann]"` (or install `torch` directly).'
        ) from _TORCH_IMPORT_ERROR


def get_torch_device() -> torch.device:
    _require_torch()
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def activation_from_name(name: str | None) -> nn.Module:
    _require_torch()
    value = (name or "linear").strip().lower()
    if value in {"linear", "identity", "none"}:
        return nn.Identity()
    if value == "tanh":
        return nn.Tanh()
    if value == "relu":
        return nn.ReLU()
    if value == "sigmoid":
        return nn.Sigmoid()
    if value == "softmax":
        return nn.Softmax(dim=-1)
    raise ValueError(f"Unknown activation function: {name!r}")


if TYPE_CHECKING:  # pragma: no cover
    import torch as _torch

    LossFn = Callable[[_torch.Tensor, _torch.Tensor], _torch.Tensor]
else:
    LossFn = Callable[[Any, Any], Any]


def loss_from_name(loss: str | LossFn) -> LossFn:
    _require_torch()
    if callable(loss):
        return loss
    value = loss.strip().lower()
    if value in {"cosine_similarity", "cosine"}:
        return lambda pred, target: (-F.cosine_similarity(pred, target, dim=-1)).mean()
    if value in {"mse", "mean_squared_error"}:
        mse = nn.MSELoss()
        return lambda pred, target: mse(pred, target)
    if value in {"mae", "mean_absolute_error", "l1"}:
        mae = nn.L1Loss()
        return lambda pred, target: mae(pred, target)
    raise ValueError(f"Unknown loss: {loss!r}")


def count_trainable_parameters(model: nn.Module) -> int:
    _require_torch()
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
