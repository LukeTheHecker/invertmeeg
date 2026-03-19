import numpy as np
from scipy.sparse import csr_matrix

from invert.benchmark.runner import evaluate_all


def test_evaluate_all_uses_unsigned_correlation_by_default():
    adjacency = csr_matrix(
        np.array(
            [
                [1, 1, 0],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=int,
        )
    )
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.02, 0.0, 0.0],
        ]
    )
    y_true = np.array(
        [
            [0.0, 0.0],
            [1.0, -1.0],
            [0.0, 0.0],
        ]
    )
    y_pred = -y_true

    metrics_unsigned = evaluate_all(
        y_true,
        y_pred,
        adjacency,
        adjacency,
        pos,
        pos,
    )
    metrics_signed = evaluate_all(
        y_true,
        y_pred,
        adjacency,
        adjacency,
        pos,
        pos,
        correlation_signal="signed",
    )

    assert metrics_unsigned["correlation"] == 1.0
    assert metrics_signed["correlation"] == -1.0
