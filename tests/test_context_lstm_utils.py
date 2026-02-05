import importlib.util

import numpy as np
import pytest


def test_prepare_training_data_alignment():
    from invert.adapters.context_lstm import prepare_training_data

    stc = np.array(
        [
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [10.0, 20.0, 30.0, 40.0, 50.0],
            ]
        ]
    )
    x, y = prepare_training_data(stc, lstm_look_back=2)

    assert x.shape == (3, 2, 2)
    assert y.shape == (3, 2)

    np.testing.assert_allclose(x[0], [[1.0, 2.0], [10.0, 20.0]])
    np.testing.assert_allclose(y[0], [3.0, 30.0])

    np.testing.assert_allclose(x[2], [[3.0, 4.0], [30.0, 40.0]])
    np.testing.assert_allclose(y[2], [5.0, 50.0])


def test_standardize_2_is_zero_safe():
    from invert.adapters.context_lstm import standardize_2

    mat = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, -2.0, 0.0],
        ]
    )
    scaled = standardize_2(mat)

    assert np.all(np.isfinite(scaled))
    np.testing.assert_allclose(scaled[:, 0], 0.0, atol=0.0)
    np.testing.assert_allclose(scaled[:, 2], 0.0, atol=0.0)
    np.testing.assert_allclose(scaled[:, 1], [0.5, -1.0], atol=1e-12)


def test_contextualize_requires_torch_when_missing():
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch is installed")

    from invert.adapters.context_lstm import contextualize, contextualize_bd

    forward = {"sol": {"data": np.zeros((1, 1))}}

    with pytest.raises(ImportError):
        contextualize(None, forward)
    with pytest.raises(ImportError):
        contextualize_bd(None, forward)
