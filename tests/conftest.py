import mne
import numpy as np
import pytest

from invert.forward import create_forward_model, get_info


@pytest.fixture(scope="session")
def sensor_info():
    """Create sensor info with sfreq (independent of forward model)."""
    return get_info(kind="biosemi16")


@pytest.fixture(scope="session")
def forward_model(sensor_info):
    """Create a small forward model for testing (ico1 = fewest dipoles)."""
    fwd = create_forward_model(info=sensor_info, sampling="ico1")
    return fwd


@pytest.fixture(scope="session")
def leadfield(forward_model):
    """Extract the leadfield matrix from the forward model."""
    return forward_model["sol"]["data"].copy()


@pytest.fixture(scope="session")
def vertices(forward_model):
    """Extract vertex numbers from forward model."""
    return [forward_model["src"][0]["vertno"], forward_model["src"][1]["vertno"]]


@pytest.fixture(scope="session")
def dimensions(leadfield):
    """Return (n_channels, n_dipoles) tuple."""
    return leadfield.shape


@pytest.fixture(scope="session")
def simulated_evoked(sensor_info, forward_model, leadfield, vertices):
    """Create a simulated evoked response with known source and noise."""
    info = sensor_info
    n_chans, n_dipoles = leadfield.shape
    n_time = 20
    rng = np.random.RandomState(42)

    source_mat = rng.randn(n_dipoles, n_time)
    evoked_mat = leadfield @ source_mat
    evoked_mat -= evoked_mat.mean(axis=0)
    evoked_mat += rng.randn(*evoked_mat.shape) * evoked_mat.std()

    evoked = mne.EvokedArray(evoked_mat, info, verbose=0)
    evoked.set_eeg_reference("average", projection=True, verbose=0)
    evoked.apply_proj()
    return evoked


@pytest.fixture(scope="session")
def simulated_stc(leadfield, vertices):
    """Create a simulated SourceEstimate."""
    n_dipoles = leadfield.shape[1]
    n_time = 20
    rng = np.random.RandomState(42)
    source_mat = rng.randn(n_dipoles, n_time)
    return mne.SourceEstimate(source_mat, vertices, tmin=0, tstep=0.001)
