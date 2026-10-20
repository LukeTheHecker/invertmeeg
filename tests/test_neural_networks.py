"""Exercise training and inference with the supported PyTorch/NumPy stack."""

import importlib

import numpy as np
import pytest

from invert import Solver
from invert.simulate import SimulationConfig

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("solver_id", ["fc", "covcnn", "lstm", "cnn"])
def test_neural_solver_train_and_predict(
    solver_id, forward_model, simulated_evoked, monkeypatch
):
    torch.manual_seed(42)
    solver = Solver(solver_id)
    module = importlib.import_module(type(solver).__module__)
    monkeypatch.setattr(module, "get_torch_device", lambda: torch.device("cpu"))
    config = SimulationConfig(
        batch_size=2,
        batch_repetitions=1,
        n_sources=1,
        n_orders=0,
        n_timepoints=20,
        random_seed=42,
    )
    solver.make_inverse_operator(
        forward_model,
        config,
        epochs=1,
        size_validation_set=2,
        loss="mse",
        n_dense_units=8,
    )
    stc = solver.apply_inverse_operator(simulated_evoked)

    assert len(solver.history["loss"]) == 1
    assert np.isfinite(solver.history["loss"]).all()
    assert np.isfinite(solver.history["val_loss"]).all()
    assert stc.data.shape == (
        forward_model["nsource"],
        simulated_evoked.data.shape[1],
    )
    assert np.isfinite(stc.data).all()
    assert np.any(stc.data != 0)
