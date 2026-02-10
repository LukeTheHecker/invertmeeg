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


class TestSimulationGeneratorRealism:
    def test_noise_covariance_metadata_shapes(self, forward_model, dimensions):
        """Noise covariance metadata should be present and well-shaped when requested."""
        n_chans, _ = dimensions
        config = SimulationConfig(
            batch_size=4,
            batch_repetitions=1,
            n_timepoints=40,
            correlation_mode="low_rank",
            noise_rank_deficiency=2,
            noise_cov_n_baseline=256,
            return_noise_cov=True,
            estimate_noise_cov=True,
            random_seed=7,
        )
        gen = SimulationGenerator(forward_model, config=config)
        _, _, info = next(gen.generate())

        for key in ["noise_cov_true", "noise_cov_est", "projector"]:
            assert key in info.columns

        for idx in range(config.batch_size):
            cov_true = info.iloc[idx]["noise_cov_true"]
            cov_est = info.iloc[idx]["noise_cov_est"]
            projector = info.iloc[idx]["projector"]
            assert cov_true.shape == (n_chans, n_chans)
            assert cov_est.shape == (n_chans, n_chans)
            assert projector.shape == (n_chans, n_chans)

            np.testing.assert_allclose(cov_true, cov_true.T, atol=1e-8)
            np.testing.assert_allclose(cov_est, cov_est.T, atol=1e-8)
            np.testing.assert_allclose(projector, projector.T, atol=1e-8)
            np.testing.assert_allclose(projector @ projector, projector, atol=1e-7)

            eigvals_true = np.linalg.eigvalsh(cov_true)
            eigvals_est = np.linalg.eigvalsh(cov_est)
            assert eigvals_true.min() > -1e-8
            assert eigvals_est.min() > -1e-8

    def test_rank_deficiency_reflected_in_metadata(self, forward_model, dimensions):
        """Projector-induced rank loss should reduce reported covariance rank."""
        n_chans, _ = dimensions
        rank_loss = 3
        config = SimulationConfig(
            batch_size=3,
            batch_repetitions=1,
            n_timepoints=80,
            correlation_mode="cholesky",
            noise_rank_deficiency=rank_loss,
            return_noise_cov=True,
            estimate_noise_cov=False,
            random_seed=13,
        )
        gen = SimulationGenerator(forward_model, config=config)
        _, _, info = next(gen.generate())

        assert np.all(info["projector_rank"].to_numpy() <= (n_chans - rank_loss))
        assert np.all(info["noise_cov_rank_true"].to_numpy() <= (n_chans - rank_loss))

    def test_snr_targets_are_realized(self, forward_model):
        """Realized SNR should closely match target SNR per sample."""
        config = SimulationConfig(
            batch_size=6,
            batch_repetitions=1,
            n_timepoints=60,
            snr_range=(2.0, 2.0),
            correlation_mode="banded",
            noise_temporal_beta=(0.0, 1.5),
            noise_rank_deficiency=(0, 2),
            random_seed=123,
        )
        gen = SimulationGenerator(forward_model, config=config)
        _, _, info = next(gen.generate())

        snr_target = info["snr"].to_numpy(dtype=float)
        snr_realized = info["snr_realized"].to_numpy(dtype=float)
        # Sampling / finite-length effects are small because we scale explicitly.
        assert np.max(np.abs(snr_target - snr_realized)) < 0.25

    def test_realistic_mode_is_reproducible(self, forward_model):
        """Reproducibility should include metadata statistics under realism settings."""
        config = SimulationConfig(
            batch_size=3,
            batch_repetitions=1,
            n_timepoints=30,
            simulation_mode="mixture",
            correlation_mode="auto",
            noise_temporal_beta=(0.0, 1.0),
            noise_rank_deficiency=(0, 2),
            return_noise_cov=False,
            random_seed=2026,
        )
        gen1 = SimulationGenerator(forward_model, config=config)
        x1, y1, info1 = next(gen1.generate())
        gen2 = SimulationGenerator(forward_model, config=config)
        x2, y2, info2 = next(gen2.generate())

        np.testing.assert_allclose(x1, x2)
        np.testing.assert_allclose(y1, y2)
        np.testing.assert_allclose(
            info1["snr_realized"].to_numpy(), info2["snr_realized"].to_numpy()
        )
        np.testing.assert_allclose(
            info1["projector_rank"].to_numpy(), info2["projector_rank"].to_numpy()
        )
