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


def test_tied_scores_are_invariant_to_vertex_order():
    adjacency = csr_matrix(np.ones((4, 4)))
    positions = np.eye(4)
    flat = np.ones((4, 1))
    for truth in np.eye(4):
        result = evaluate_all(
            truth[:, None], flat, adjacency, adjacency, positions, positions
        )
        assert result["average_precision"] == 0.25


def test_exact_and_local_waveforms_use_consistent_locations():
    adjacency = csr_matrix(np.ones((3, 3)))
    wave = np.array([-1.0, 1.0, -1.0, 1.0])
    truth = np.stack([wave, wave * 0, wave * 0])
    prediction = np.stack([wave * 0, wave, wave * 0])
    for epsilon in [0, 1e-12]:
        prediction[0] = epsilon * np.array([1, 1, -1, -1])
        scores = evaluate_all(
            truth, prediction, adjacency, adjacency, np.eye(3), np.eye(3)
        )
        np.testing.assert_allclose(scores["correlation"], 0, atol=1e-15)
        np.testing.assert_allclose(scores["local_correlation"], 1)
