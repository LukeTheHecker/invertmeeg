"""Tests for the simulation / data generator utilities."""

import numpy as np

from invert.simulate import generator


class TestGeneratorDefaults:
    def test_output_types(self, forward_model):
        """Generator should yield numpy arrays."""
        x, y = next(generator(forward_model))
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_output_shapes(self, forward_model, dimensions):
        """Default generator shapes should match forward model dimensions."""
        n_chans, n_dipoles = dimensions
        x, y = next(generator(forward_model))
        # x is (batch, n_chans, n_chans, 1) by default (covariance mode)
        assert x.shape[1] == n_chans
        assert x.shape[2] == n_chans
        # y is (batch, n_dipoles)
        assert y.shape[1] == n_dipoles

    def test_output_finite(self, forward_model):
        """Generator output should be finite."""
        x, y = next(generator(forward_model))
        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(y))


class TestGeneratorCustom:
    def test_custom_params(self, forward_model, dimensions):
        """Generator with custom parameters should respect them."""
        n_chans, n_dipoles = dimensions
        params = dict(
            use_cov=False,
            batch_size=10,
            batch_repetitions=1,
            n_sources=3,
            n_orders=2,
            n_timepoints=30,
            snr_range=(5, 15),
            remove_channel_dim=True,
            random_seed=42,
            verbose=0,
        )
        x, y = next(generator(forward_model, **params))
        assert x.shape[0] == params["batch_size"]
        assert x.shape[1] == params["n_timepoints"]
        assert x.shape[2] == n_chans

    def test_reproducibility(self, forward_model):
        """Same random_seed should produce identical output."""
        params = dict(
            use_cov=False,
            batch_size=5,
            batch_repetitions=1,
            n_timepoints=10,
            random_seed=99,
            remove_channel_dim=True,
            verbose=0,
        )
        x1, y1 = next(generator(forward_model, **params))
        x2, y2 = next(generator(forward_model, **params))
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)


class TestGeneratorEdgeCases:
    def test_single_source(self, forward_model):
        """Generator should work with a single source."""
        params = dict(
            use_cov=False,
            batch_size=5,
            batch_repetitions=1,
            n_sources=1,
            n_timepoints=10,
            remove_channel_dim=True,
            verbose=0,
        )
        x, y = next(generator(forward_model, **params))
        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(y))
