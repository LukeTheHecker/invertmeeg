import numpy as np
import pytest

from invert import Solver


def test_mca_smooth_sparse_full_smoke(forward_model, simulated_evoked, leadfield):
    n_dipoles = int(leadfield.shape[1])
    n_time = int(simulated_evoked.data.shape[1])

    solver = Solver("mca-smooth-sparse-full", n_reg_params=1)
    solver.make_inverse_operator(
        forward_model,
        simulated_evoked,
        alpha=0.1,
        epochs=1,
        n=2,
        k=2,
    )

    stc = solver.apply_inverse_operator(simulated_evoked)
    data = np.asarray(stc.data, dtype=float)
    assert data.shape == (n_dipoles, n_time)
    assert np.isfinite(data).all()
    assert np.any(data != 0)


@pytest.mark.parametrize("orientation", ["fixed", "pca"])
def test_mca_smooth_sparse_full_orientation_smoke(
    orientation, forward_model_free_surface, simulated_evoked_free_surface
):
    n_dipoles = int(forward_model_free_surface["nsource"])
    n_time = int(simulated_evoked_free_surface.data.shape[1])

    solver = Solver("mca-smooth-sparse-full", orientation=orientation, n_reg_params=1)
    solver.make_inverse_operator(
        forward_model_free_surface,
        simulated_evoked_free_surface,
        alpha=0.1,
        epochs=1,
        n=2,
        k=2,
    )
    stc = solver.apply_inverse_operator(simulated_evoked_free_surface)
    data = np.asarray(stc.data, dtype=float)
    assert data.shape == (n_dipoles, n_time)
    assert np.isfinite(data).all()
    assert np.any(data != 0)


def test_mca_smooth_sparse_full_rejects_free_orientation(
    forward_model_free_surface, simulated_evoked_free_surface
):
    solver = Solver("mca-smooth-sparse-full", orientation="free", n_reg_params=1)
    with pytest.raises(ValueError, match="does not support orientation='free'"):
        solver.make_inverse_operator(
            forward_model_free_surface,
            simulated_evoked_free_surface,
            alpha=0.1,
            epochs=1,
            n=2,
            k=2,
        )
