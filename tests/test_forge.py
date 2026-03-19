import mne
import numpy as np
import pytest

from invert import Solver


def _as_mne_cov(data: np.ndarray, ch_names: list[str]) -> mne.Covariance:
    return mne.Covariance(
        data=np.asarray(data, dtype=float),
        names=list(ch_names),
        bads=[],
        projs=[],
        nfree=1,
    )


def _make_forge(**kwargs):
    defaults = dict(
        n_starts=8,
        n_refine_iters=1,
        k_max=4,
        n_orders=2,
        neighbor_hops=1,
        max_iter_nlc=40,
        smoothing_orders=(0, 1, 2),
    )
    defaults.update(kwargs)
    return Solver("forge", **defaults)


def test_forge_smoke_with_noise_cov(forward_model, simulated_evoked, leadfield):
    n_chans = len(forward_model.ch_names)
    n_dipoles = int(leadfield.shape[1])
    n_time = int(simulated_evoked.data.shape[1])
    noise_cov = _as_mne_cov(np.eye(n_chans, dtype=float), forward_model.ch_names)

    solver = _make_forge()
    solver.make_inverse_operator(
        forward_model,
        simulated_evoked,
        alpha=0.1,
        noise_cov=noise_cov,
    )
    stc = solver.apply_inverse_operator(simulated_evoked)

    data = np.asarray(stc.data, dtype=float)
    assert data.shape == (n_dipoles, n_time)
    assert np.isfinite(data).all()
    assert np.any(data != 0.0)


@pytest.mark.parametrize("orientation", ["fixed", "pca"])
def test_forge_orientation_smoke(
    orientation, forward_model_free_surface, simulated_evoked_free_surface
):
    n_chans = len(forward_model_free_surface.ch_names)
    n_dipoles = int(forward_model_free_surface["nsource"])
    n_time = int(simulated_evoked_free_surface.data.shape[1])
    noise_cov = _as_mne_cov(
        np.eye(n_chans, dtype=float), forward_model_free_surface.ch_names
    )

    solver = _make_forge(orientation=orientation)
    solver.make_inverse_operator(
        forward_model_free_surface,
        simulated_evoked_free_surface,
        alpha=0.1,
        noise_cov=noise_cov,
    )
    stc = solver.apply_inverse_operator(simulated_evoked_free_surface)

    data = np.asarray(stc.data, dtype=float)
    assert data.shape == (n_dipoles, n_time)
    assert np.isfinite(data).all()
    assert np.any(data != 0.0)


def test_forge_handles_rank_deficient_noise_cov(forward_model, simulated_evoked):
    n_chans = len(forward_model.ch_names)
    diag = np.ones(n_chans, dtype=float)
    diag[-1] = 0.0
    noise_cov = _as_mne_cov(np.diag(diag), forward_model.ch_names)

    solver = _make_forge()
    solver.make_inverse_operator(
        forward_model,
        simulated_evoked,
        alpha=0.1,
        noise_cov=noise_cov,
    )
    stc = solver.apply_inverse_operator(simulated_evoked)

    data = np.asarray(stc.data, dtype=float)
    assert np.isfinite(data).all()
    assert np.any(data != 0.0)
