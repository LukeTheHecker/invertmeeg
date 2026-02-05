from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

_TORCH_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    import torch  # type: ignore[import-not-found]
    import torch.nn as nn  # type: ignore[import-not-found]
    import torch.nn.functional as F  # type: ignore[import-not-found]
    from torch.utils.data import (  # type: ignore[import-not-found]
        DataLoader,
        TensorDataset,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]

    class _NNModule:  # pragma: no cover
        pass

    class _NN:  # pragma: no cover
        Module = _NNModule

    nn = _NN()  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[misc,assignment]
    TensorDataset = None  # type: ignore[misc,assignment]
    _TORCH_IMPORT_ERROR = exc

logger = logging.getLogger(__name__)

_EPS = 1e-12


def _require_torch() -> None:
    if _TORCH_IMPORT_ERROR is not None:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for the contextual LSTM adapters. "
            'Install it via `pip install "invertmeeg[ann]"` (or install `torch` directly).'
        ) from _TORCH_IMPORT_ERROR


def _get_device() -> torch.device:
    _require_torch()
    assert torch is not None  # for type checkers
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _ContextLSTMNet(nn.Module):
    def __init__(self, n_features: int, n_units: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=int(n_features),
            hidden_size=int(n_units),
            batch_first=True,
        )
        self.out = nn.Linear(int(n_units), int(n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, features)
        out, _state = self.lstm(x)
        last = out[:, -1, :]
        return self.out(last)


if TYPE_CHECKING:  # pragma: no cover
    import torch as _torch

    LossFn = Callable[[_torch.Tensor, _torch.Tensor], _torch.Tensor]
else:
    LossFn = Callable[[Any, Any], Any]


def _loss_from_name(loss: str | LossFn) -> LossFn:
    _require_torch()
    assert torch is not None  # for type checkers
    assert F is not None  # for type checkers

    if callable(loss):
        return loss
    value = loss.strip().lower()
    if value in {"mse", "mean_squared_error"}:
        mse = nn.MSELoss()
        return lambda pred, target: mse(pred, target)
    if value in {"mae", "mean_absolute_error", "l1"}:
        mae = nn.L1Loss()
        return lambda pred, target: mae(pred, target)
    if value in {"cosine", "cosine_similarity"}:
        return lambda pred, target: (-F.cosine_similarity(pred, target, dim=-1)).mean()
    raise ValueError(f"Unknown loss: {loss!r}")


def _make_optimizer(
    optimizer: str | object, params, *, learning_rate: float
):  # -> torch.optim.Optimizer
    _require_torch()
    assert torch is not None  # for type checkers

    if isinstance(optimizer, str):
        value = optimizer.strip().lower()
        if value in {"adam", "adamw"}:
            opt_cls = torch.optim.Adam if value == "adam" else torch.optim.AdamW
            return opt_cls(params, lr=float(learning_rate))
        if value in {"sgd"}:
            return torch.optim.SGD(params, lr=float(learning_rate), momentum=0.9)
        raise ValueError(f"Unknown optimizer: {optimizer!r}")

    # Allow passing a torch optimizer instance or a factory.
    if hasattr(optimizer, "step") and hasattr(optimizer, "zero_grad"):
        return optimizer
    if callable(optimizer):
        return optimizer(params)

    raise TypeError(
        "`optimizer` must be a string, a torch optimizer instance, or a callable factory."
    )


def _train_model(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    epochs: int,
    steps_per_epoch: int | None,
    optimizer: str | object,
    loss: str | LossFn,
    fast: bool,
    verbose: int,
    device: torch.device,
) -> nn.Module:
    _require_torch()
    assert torch is not None  # for type checkers
    assert DataLoader is not None  # for type checkers
    assert TensorDataset is not None  # for type checkers
    assert F is not None  # for type checkers

    if x.ndim != 3:
        raise ValueError(f"x must be 3D (batch,time,features), got shape {x.shape}.")
    if y.ndim != 2:
        raise ValueError(f"y must be 2D (batch,features), got shape {y.shape}.")
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"x and y must have the same number of samples, got {x.shape[0]} and {y.shape[0]}."
        )

    n_samples = x.shape[0]
    if n_samples < 2:
        raise ValueError(
            "Not enough training samples. Reduce `lstm_look_back` or provide more time points."
        )

    # Train/val split (mirrors Keras' default validation_split=0.15).
    rng = np.random.RandomState(0)
    indices = rng.permutation(n_samples)
    val_size = max(1, int(round(0.15 * n_samples)))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    if len(train_idx) == 0:
        raise ValueError(
            "Training split would be empty. Reduce `lstm_look_back` or provide more time points."
        )

    x_train = torch.from_numpy(x[train_idx].astype(np.float32, copy=False))
    y_train = torch.from_numpy(y[train_idx].astype(np.float32, copy=False))
    x_val = torch.from_numpy(x[val_idx].astype(np.float32, copy=False))
    y_val = torch.from_numpy(y[val_idx].astype(np.float32, copy=False))

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
    )

    loss_fn = _loss_from_name(loss)
    opt = _make_optimizer(optimizer, model.parameters(), learning_rate=1e-3)

    # Early stopping similar to the original TF code.
    if fast:
        patience = 3
        min_delta = 0.01
        monitor = "val_cosine"
        best = -np.inf
    else:
        patience = 15
        min_delta = 0.0
        monitor = "val_loss"
        best = np.inf

    best_state = None
    bad_epochs = 0

    model.to(device)
    for epoch in range(int(epochs)):
        model.train()
        running = 0.0
        n_batches = 0

        for step, (xb, yb) in enumerate(train_loader):
            if steps_per_epoch is not None and step >= int(steps_per_epoch):
                break
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss_val = loss_fn(pred, yb)
            loss_val.backward()
            opt.step()

            running += float(loss_val.detach().cpu().item())
            n_batches += 1

        train_loss = running / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            val_losses = []
            val_coss = []
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_losses.append(float(loss_fn(pred, yb).detach().cpu().item()))
                val_coss.append(
                    float(F.cosine_similarity(pred, yb, dim=-1).mean().cpu().item())
                )

        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        val_cos = float(np.mean(val_coss)) if val_coss else float("-inf")

        if verbose > 0:
            logger.info(
                "Epoch %d/%d - loss=%.5f - val_loss=%.5f - val_cos=%.5f",
                epoch + 1,
                int(epochs),
                train_loss,
                val_loss,
                val_cos,
            )

        if monitor == "val_cosine":
            improved = val_cos > best + min_delta
            metric = val_cos
        else:
            improved = val_loss < best - min_delta
            metric = val_loss

        if improved:
            best = metric
            best_state = deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs > patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def contextualize(
    stc_instant,
    forward,
    lstm_look_back=80,
    num_units=128,
    num_epochs=100,
    steps_per_ep=25,
    batch_size=32,
    fast=True,
    optimizer="adam",
    loss="mean_squared_error",
    verbose=0,
):
    """
    Temporal contextualization of inverse solutions using Long-Short Term Memory
    (LSTM) Networks as described in [1].

    Parameters
    ----------
    stc_instant : mne.SourceEstimate
        The instantaneous source estimate object which shall be contextualized.
    forward : mne.Forward
        The forward model (used for dimensionality checks).
    lstm_look_back : int
        Number of time points to consider as context.
    num_units : int
        Number of LSTM cells (units).
    num_epochs : int
        Number of epochs to train the LSTM network.
    steps_per_ep : int
        Maximum gradient steps per epoch (approximate; may be capped by dataset size).
    batch_size : int
        Batch size for training.
    fast : bool
        If True, reduce model size and enable aggressive early stopping.
    optimizer : str / torch optimizer / callable
        Optimizer specifier. Strings supported: "adam", "adamw", "sgd".
    loss : str / callable
        Loss specifier. Strings supported: "mean_squared_error" (default), "mae", "cosine_similarity".
    verbose : int
        Controls verbosity of the program.

    Return
    ------
    stc_context : mne.SourceEstimate
        The contextualized source.

    References
    ----------
    [1] Dinh, C., Samuelsson, J. G., Hunold, A., Hämäläinen, M. S., & Khan, S.
        (2021). Contextual MEG and EEG source estimates using spatiotemporal
        LSTM networks. Frontiers in neuroscience, 15, 552666.
    """
    _require_torch()
    assert torch is not None  # for type checkers

    leadfield = forward["sol"]["data"]
    _, n_dipoles = leadfield.shape
    if stc_instant.data.shape[0] != n_dipoles:
        raise ValueError(
            f"stc has {stc_instant.data.shape[0]} sources but forward has {n_dipoles}."
        )
    if stc_instant.data.shape[1] <= int(lstm_look_back):
        raise ValueError(
            "Not enough time points for the requested look-back window. "
            f"Need > {int(lstm_look_back)} time points, got {stc_instant.data.shape[1]}."
        )

    if fast:
        num_epochs = 50
        num_units = 64

    stc_instant_unscaled = deepcopy(stc_instant.data)
    stc_instant_scaled = standardize_2(stc_instant.data)
    stc_epochs_train = deepcopy(stc_instant_scaled)[np.newaxis]
    x_train, y_train = prepare_training_data(stc_epochs_train, int(lstm_look_back))
    # LSTM expects (batch, time, features)
    x_train = np.swapaxes(x_train, 1, 2)

    device = _get_device()
    model = _ContextLSTMNet(n_features=int(n_dipoles), n_units=int(num_units))
    model = _train_model(
        model,
        x_train,
        y_train,
        batch_size=int(batch_size),
        epochs=int(num_epochs),
        steps_per_epoch=int(steps_per_ep) if steps_per_ep is not None else None,
        optimizer=optimizer,
        loss=loss,
        fast=bool(fast),
        verbose=int(verbose),
        device=device,
    )

    stc_lstm = np.zeros_like(stc_instant_scaled)
    stc_cmne = np.zeros_like(stc_instant_scaled)

    stc_lstm[:, : int(lstm_look_back)] = 1.0
    stc_cmne[:, : int(lstm_look_back)] = stc_instant_scaled[:, : int(lstm_look_back)]

    steps = stc_instant_scaled.shape[1] - int(lstm_look_back)

    model.eval()
    with torch.no_grad():
        for i in range(int(steps)):
            stc_prior = stc_cmne[:, i : i + int(lstm_look_back)]
            x = np.swapaxes(stc_prior[np.newaxis], 1, 2).astype(np.float32, copy=False)
            pred = model(torch.from_numpy(x).to(device)).cpu().numpy().reshape(-1)
            denom = float(np.abs(pred).max())
            mask = np.abs(pred) / (denom + _EPS)

            t = i + int(lstm_look_back)
            stc_lstm[:, t] = mask
            stc_cmne[:, t] = stc_instant_scaled[:, t] * mask

    stc_context_data = stc_instant_unscaled * stc_lstm
    stc_context = stc_instant.copy()
    stc_context.data = stc_context_data
    return stc_context


def contextualize_bd(
    stc_instant,
    forward,
    lstm_look_back=80,
    num_units=128,
    num_epochs=100,
    steps_per_ep=25,
    batch_size=32,
    fast=True,
    optimizer="adam",
    loss="mean_squared_error",
    verbose=0,
):
    """
    Bi-directional temporal contextualization of inverse solutions using
    Long-Short Term Memory (LSTM) Networks as described in [1] using both past
    and future time points.

    Notes
    -----
    This uses a single model trained on both the forward and time-reversed
    source sequence. The backward mask is used for early time points that lack
    sufficient past context.
    """
    _require_torch()
    assert torch is not None  # for type checkers

    leadfield = forward["sol"]["data"]
    _, n_dipoles = leadfield.shape
    if stc_instant.data.shape[0] != n_dipoles:
        raise ValueError(
            f"stc has {stc_instant.data.shape[0]} sources but forward has {n_dipoles}."
        )
    if stc_instant.data.shape[1] <= int(lstm_look_back):
        raise ValueError(
            "Not enough time points for the requested look-back window. "
            f"Need > {int(lstm_look_back)} time points, got {stc_instant.data.shape[1]}."
        )

    if fast:
        num_epochs = 50
        num_units = 64

    stc_instant_forward = standardize_2(stc_instant.data)
    stc_instant_backwards = stc_instant_forward[:, ::-1]

    stc_epochs_train_forward = deepcopy(stc_instant_forward)[np.newaxis]
    stc_epochs_train_backwards = deepcopy(stc_epochs_train_forward[:, :, ::-1])
    stc_epochs_train = np.concatenate(
        [stc_epochs_train_forward, stc_epochs_train_backwards], axis=0
    )

    x_train, y_train = prepare_training_data(stc_epochs_train, int(lstm_look_back))
    x_train = np.swapaxes(x_train, 1, 2)

    device = _get_device()
    model = _ContextLSTMNet(n_features=int(n_dipoles), n_units=int(num_units))
    model = _train_model(
        model,
        x_train,
        y_train,
        batch_size=int(batch_size),
        epochs=int(num_epochs),
        steps_per_epoch=int(steps_per_ep) if steps_per_ep is not None else None,
        optimizer=optimizer,
        loss=loss,
        fast=bool(fast),
        verbose=int(verbose),
        device=device,
    )

    stc_lstm_forward = np.zeros_like(stc_instant_forward)
    stc_cmne_forward = np.zeros_like(stc_instant_forward)
    stc_lstm_forward[:, : int(lstm_look_back)] = 1.0
    stc_cmne_forward[:, : int(lstm_look_back)] = stc_instant_forward[
        :, : int(lstm_look_back)
    ]

    stc_lstm_backwards = np.zeros_like(stc_instant_backwards)
    stc_cmne_backwards = np.zeros_like(stc_instant_backwards)
    stc_lstm_backwards[:, : int(lstm_look_back)] = 1.0
    stc_cmne_backwards[:, : int(lstm_look_back)] = stc_instant_backwards[
        :, : int(lstm_look_back)
    ]

    model.eval()
    with torch.no_grad():
        steps_forward = stc_instant_forward.shape[1] - int(lstm_look_back)
        for i in range(int(steps_forward)):
            stc_prior = stc_cmne_forward[:, i : i + int(lstm_look_back)]
            x = np.swapaxes(stc_prior[np.newaxis], 1, 2).astype(np.float32, copy=False)
            pred = model(torch.from_numpy(x).to(device)).cpu().numpy().reshape(-1)
            denom = float(np.abs(pred).max())
            mask = np.abs(pred) / (denom + _EPS)

            t = i + int(lstm_look_back)
            stc_lstm_forward[:, t] = mask
            stc_cmne_forward[:, t] = stc_instant_forward[:, t] * mask

        steps_back = stc_instant_backwards.shape[1] - int(lstm_look_back)
        for i in range(int(steps_back)):
            stc_prior = stc_cmne_backwards[:, i : i + int(lstm_look_back)]
            x = np.swapaxes(stc_prior[np.newaxis], 1, 2).astype(np.float32, copy=False)
            pred = model(torch.from_numpy(x).to(device)).cpu().numpy().reshape(-1)
            denom = float(np.abs(pred).max())
            mask = np.abs(pred) / (denom + _EPS)

            t = i + int(lstm_look_back)
            stc_lstm_backwards[:, t] = mask
            stc_cmne_backwards[:, t] = stc_instant_backwards[:, t] * mask

    stc_lstm_combined = deepcopy(stc_lstm_forward)
    stc_lstm_backwards_rev = stc_lstm_backwards[:, ::-1]
    stc_lstm_combined[:, : int(lstm_look_back)] = stc_lstm_backwards_rev[
        :, : int(lstm_look_back)
    ]

    stc_context_data = deepcopy(stc_instant.data)
    stc_context_data *= stc_lstm_combined
    stc_context = stc_instant.copy()
    stc_context.data = stc_context_data
    return stc_context


def rectify_norm(x):
    x = np.asarray(x)
    denom = float(np.abs(x).std())
    if denom == 0:
        return np.zeros_like(x)
    return (x - np.abs(x).mean()) / denom


def prepare_training_data(stc, lstm_look_back=20):
    if len(stc.shape) != 3:
        raise ValueError(
            "stc must be a 3D numpy.ndarray of shape (epochs, dipoles, time)"
        )
    n_samples, _, n_time = stc.shape
    if n_time <= int(lstm_look_back):
        raise ValueError(
            f"Need n_time > lstm_look_back, got n_time={n_time}, lstm_look_back={int(lstm_look_back)}."
        )

    x = []
    y = []
    for i in range(n_samples):
        for j in range(n_time - int(lstm_look_back)):
            x.append(stc[i, :, j : j + int(lstm_look_back)])
            y.append(stc[i, :, j + int(lstm_look_back)])

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)

    return x, y


def standardize(mat, mean=None, std=None):
    """0-center and scale data along sources (axis=1)."""
    if mean is None:
        mean = np.mean(mat, axis=1)

    if std is None:
        std = np.std(mat, axis=1)

    std = np.where(std == 0, 1, std)
    return np.transpose((mat.T - mean) / std)


def standardize_2(mat):
    mat_scaled = deepcopy(mat)
    for t, time_slice in enumerate(mat_scaled.T):
        denom = float(np.abs(time_slice).max())
        if denom == 0:
            mat_scaled[:, t] = 0.0
        else:
            mat_scaled[:, t] = time_slice / denom
    return mat_scaled
