import numpy as np
import ot
import pytest
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

from invert.benchmark.runner import _eval_mod
from invert.solvers.bayesian.sbl_core import _BUILTIN_RULES, sbl_iterate


def test_sparse_geometry_matches_dense_reference():
    rng = np.random.default_rng(812)
    pos = rng.normal(size=(10, 3))
    distances = cdist(pos, pos)
    graph = csr_matrix(distances < 2.5)
    truth = np.array([1, 0, 0.6, 0.1, 0, 0, 0, 0, 0, 0])
    pred = np.roll(truth, 3)
    dense = np.where(truth >= 0.25, truth, 0)
    dense /= dense.sum()
    expected = ot.emd2(dense, np.roll(dense, 3), distances)
    np.testing.assert_allclose(
        _eval_mod.eval_emd(None, truth, pred, positions=(pos, pos)), expected
    )
    np.testing.assert_allclose(
        _eval_mod._mesh_extent(pred, graph, pos),
        _eval_mod.compute_fwhm(pred, distances, graph.toarray()),
    )


@pytest.mark.parametrize("rule", ["mackay", "convexity", "em"])
def test_sbl_energy_compression_preserves_full_posterior(rule):
    rng = np.random.default_rng(390)
    gain = rng.normal(size=(5, 14))
    data = rng.normal(size=(5, 40))
    data[0] = data[1]
    args = (gain, data, np.eye(5))
    options = dict(max_iter=15, conv_crit=0.0, pruning_thresh=0.02)
    compressed = sbl_iterate(*args, update_rule=rule, **options)
    reference = sbl_iterate(*args, update_rule=_BUILTIN_RULES[rule], **options)
    np.testing.assert_array_equal(compressed.active_set, reference.active_set)
    for key in ["gammas", "mu_x", "loss", "Sigma_y_inv"]:
        np.testing.assert_allclose(
            getattr(compressed, key), getattr(reference, key), rtol=1e-10, atol=1e-12
        )


def test_vb_sbl_temporal_compression_preserves_second_moment():
    from invert.solvers.bayesian.vb_sbl import SolverVBSBL

    rng = np.random.default_rng(918)
    gain = rng.normal(size=(4, 11))
    data = rng.normal(size=(4, 80))
    solver = SolverVBSBL()
    noise = 0.4
    actual = solver._vb_sbl(
        gain,
        data,
        np.eye(4),
        noise,
        max_iter=1,
        prune=False,
        pruning_thresh=0,
        conv_crit=0,
    )
    norms = np.linalg.norm(gain, axis=0)
    leadfield = gain / norms
    inverse = np.linalg.inv(noise * np.eye(4) + leadfield @ leadfield.T)
    posterior = leadfield.T @ inverse @ data
    variance = 1 - np.diag(leadfield.T @ inverse @ leadfield)
    gamma = (
        solver.b0 + 0.5 * ((posterior**2).sum(axis=1) + data.shape[1] * variance)
    ) / (solver.a0 + 0.5 * data.shape[1])
    updated_inverse = np.linalg.inv(
        noise * np.eye(4) + (leadfield * gamma) @ leadfield.T
    )
    reference = (gamma / norms)[:, None] * (leadfield.T @ updated_inverse)
    np.testing.assert_allclose(actual, reference, rtol=1e-11, atol=1e-13)
