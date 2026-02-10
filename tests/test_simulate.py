"""Tests for the simulation / data generator utilities."""

import numpy as np

from invert.simulate import SimulationConfig, SimulationGenerator


class TestSimulationGeneratorDefaults:
    def test_output_types(self, forward_model):
        """Generator should yield numpy arrays."""
        gen = SimulationGenerator(forward_model)
        x, y, info = next(gen.generate())
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_output_shapes(self, forward_model, dimensions):
        """Default generator shapes should match forward model dimensions."""
        n_chans, n_dipoles = dimensions
        gen = SimulationGenerator(forward_model)
        x, y, info = next(gen.generate())
        # x is (batch, n_channels, n_timepoints)
        assert x.shape[1] == n_chans
        # y is (batch, n_dipoles, n_timepoints)
        assert y.shape[1] == n_dipoles

    def test_output_finite(self, forward_model):
        """Generator output should be finite."""
        gen = SimulationGenerator(forward_model)
        x, y, info = next(gen.generate())
        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(y))


class TestSimulationGeneratorCustom:
    def test_custom_params(self, forward_model, dimensions):
        """Generator with custom parameters should respect them."""
        n_chans, n_dipoles = dimensions
        config = SimulationConfig(
            batch_size=10,
            batch_repetitions=1,
            n_sources=3,
            n_orders=2,
            n_timepoints=30,
            snr_range=(5, 15),
            random_seed=42,
        )
        gen = SimulationGenerator(forward_model, config=config)
        x, y, info = next(gen.generate())
        assert x.shape[0] == 10
        assert x.shape[1] == n_chans
        assert x.shape[2] == 30

    def test_reproducibility(self, forward_model):
        """Same random_seed should produce identical output."""
        config = SimulationConfig(
            batch_size=5,
            batch_repetitions=1,
            n_timepoints=10,
            random_seed=99,
        )
        gen1 = SimulationGenerator(forward_model, config=config)
        x1, y1, _ = next(gen1.generate())
        gen2 = SimulationGenerator(forward_model, config=config)
        x2, y2, _ = next(gen2.generate())
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)


class TestSimulationGeneratorEdgeCases:
    def test_single_source(self, forward_model):
        """Generator should work with a single source."""
        config = SimulationConfig(
            batch_size=5,
            batch_repetitions=1,
            n_sources=1,
            n_timepoints=10,
        )
        gen = SimulationGenerator(forward_model, config=config)
        x, y, info = next(gen.generate())
        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(y))
