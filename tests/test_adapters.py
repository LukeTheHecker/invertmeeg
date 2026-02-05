import numpy as np


def _weighted_mne_expected(leadfield, y, weights, alpha):
    """Reference implementation used for adapter tests."""
    weights = np.maximum(np.asarray(weights, dtype=float), 1e-12)
    LW = leadfield * weights[np.newaxis, :]
    A = LW @ leadfield.T + alpha * np.eye(leadfield.shape[0])
    z = np.linalg.solve(A, y)
    return weights * (leadfield.T @ z)


class TestFocussAdapter:
    def test_single_iter_matches_weighted_mne(
        self, forward_model, simulated_evoked, simulated_stc
    ):
        """With max_iter=1, FOCUSS should equal a single weighted MNE update."""
        from invert.adapters.focuss import focuss

        alpha = 0.05
        stc_out = focuss(
            simulated_stc, simulated_evoked, forward_model, alpha=alpha, max_iter=1
        )

        L = forward_model["sol"]["data"]
        y = simulated_evoked.data[:, 0]
        x0 = simulated_stc.data[:, 0]
        expected = _weighted_mne_expected(L, y, np.abs(x0), alpha)

        np.testing.assert_allclose(stc_out.data[:, 0], expected, atol=1e-10, rtol=1e-10)


class TestSmooth:
    def test_keeps_prominent_without_self_edges(self):
        from invert.adapters.focuss import smooth

        adjacency = np.array(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ],
            dtype=int,
        )
        x = np.array([10.0, 0.0, 0.0])
        x_smooth, sparsity = smooth(x, adjacency, percentile=0.5)

        assert sparsity == 1
        assert x_smooth[0] != 0
        assert x_smooth[2] == 0

    def test_zero_input_is_stable(self):
        from invert.adapters.focuss import smooth

        adjacency = np.zeros((3, 3), dtype=int)
        x = np.zeros(3)
        x_smooth, sparsity = smooth(x, adjacency, percentile=0.5)
        np.testing.assert_allclose(x_smooth, 0.0, atol=0.0)
        assert sparsity == 0


class TestOtherAdapters:
    def test_s_focuss_runs_and_is_finite(self, forward_model, simulated_evoked, simulated_stc):
        from invert.adapters.focuss import s_focuss

        stc_out = s_focuss(
            simulated_stc,
            simulated_evoked,
            forward_model,
            alpha=0.05,
            percentile=0.5,
            max_iter=2,
        )
        assert stc_out.data.shape == simulated_stc.data.shape
        assert np.all(np.isfinite(stc_out.data))

    def test_stampc_does_not_mutate_forward(self, forward_model, simulated_evoked, simulated_stc):
        from invert.adapters.stamp import stampc

        leadfield_before = forward_model["sol"]["data"].copy()
        _ = stampc(
            simulated_stc,
            simulated_evoked,
            forward_model,
            max_iter=2,
            K=1,
            n_orders=0,
        )
        np.testing.assert_allclose(forward_model["sol"]["data"], leadfield_before, atol=0.0)

    def test_stampc_runs_with_orders_and_zero_seed(
        self, forward_model, simulated_evoked, simulated_stc
    ):
        from invert.adapters.stamp import stampc

        stc_zero = simulated_stc.copy()
        stc_zero.data[:] = 0.0

        stc_out = stampc(
            stc_zero,
            simulated_evoked,
            forward_model,
            max_iter=3,
            K=2,
            rv_thresh=0.1,
            n_orders=1,
        )

        assert stc_out.data.shape == simulated_stc.data.shape
        assert np.all(np.isfinite(stc_out.data))
