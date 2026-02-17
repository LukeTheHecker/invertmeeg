import mne
import numpy as np
import pytest

from invert.forward import create_forward_model, get_info


def pytest_addoption(parser):
    parser.addoption(
        "--update-baselines",
        action="store_true",
        default=False,
        help="Update solver_baselines.json instead of asserting against it.",
    )


@pytest.fixture(scope="session")
def update_baselines(request):
    return request.config.getoption("--update-baselines")


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


@pytest.fixture(scope="session")
def forward_model_free_surface(sensor_info):
    """Small *free-orientation* surface forward model for orientation tests."""
    fwd = create_forward_model(info=sensor_info, sampling="ico1", fixed_ori=False)
    return fwd


@pytest.fixture(scope="session")
def leadfield_free_surface(forward_model_free_surface):
    """Extract free-orientation leadfield (n_channels, 3*n_sources)."""
    return forward_model_free_surface["sol"]["data"].copy()


@pytest.fixture(scope="session")
def simulated_evoked_free_surface(
    sensor_info, forward_model_free_surface, leadfield_free_surface
):
    """Simulated evoked response using a free-orientation surface leadfield."""
    info = sensor_info
    n_chans, n_cols = leadfield_free_surface.shape
    n_time = 20
    rng = np.random.RandomState(42)

    source_mat = rng.randn(n_cols, n_time)
    evoked_mat = leadfield_free_surface @ source_mat
    evoked_mat -= evoked_mat.mean(axis=0)
    evoked_mat += rng.randn(*evoked_mat.shape) * evoked_mat.std()

    evoked = mne.EvokedArray(evoked_mat, info, verbose=0)
    evoked.set_eeg_reference("average", projection=True, verbose=0)
    evoked.apply_proj()
    return evoked


@pytest.fixture(scope="session")
def forward_model_discrete_free(sensor_info):
    """Small discrete (volume) forward model (free orientation) for offline tests.

    Uses a spherical conductor model to avoid downloading fsaverage assets.
    """
    info = sensor_info
    sphere = mne.make_sphere_model(info=info, verbose=0)
    # Try a few stable configurations; some parameter combos can yield NaNs.
    for radius in (0.06, 0.05, 0.04):
        for pos_mm in (30.0, 40.0, 50.0):
            src = mne.setup_volume_source_space(
                subject=None,
                pos=pos_mm,
                sphere=(0.0, 0.0, 0.0, radius),
                verbose=0,
            )
            fwd = mne.make_forward_solution(
                info,
                trans=None,
                src=src,
                bem=sphere,
                eeg=True,
                meg=False,
                verbose=0,
            )
            G = np.asarray(fwd["sol"]["data"], dtype=float)
            if np.isfinite(G).all():
                return fwd
    raise RuntimeError("Could not create a finite discrete forward solution for tests.")


@pytest.fixture(scope="session")
def simulated_evoked_discrete_free(sensor_info, forward_model_discrete_free):
    """Simulated evoked response using a discrete free-orientation leadfield."""
    info = sensor_info
    leadfield = np.asarray(forward_model_discrete_free["sol"]["data"], dtype=float)
    n_chans, n_cols = leadfield.shape
    n_time = 20
    rng = np.random.RandomState(42)

    source_mat = rng.randn(n_cols, n_time)
    evoked_mat = leadfield @ source_mat
    evoked_mat -= evoked_mat.mean(axis=0)
    evoked_mat += rng.randn(*evoked_mat.shape) * evoked_mat.std()

    evoked = mne.EvokedArray(evoked_mat, info, verbose=0)
    evoked.set_eeg_reference("average", projection=True, verbose=0)
    evoked.apply_proj()
    return evoked
