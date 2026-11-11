"""Native current directions and anatomical subject IDs must survive export."""

import mne
import numpy as np
import pytest
from mne.source_estimate import _make_stc

from invert.solvers.base import BaseSolver


@pytest.mark.parametrize("depth", [False, True])
def test_free_export_matches_mne_cartesian_conversion_and_reprojects(
    forward_model_free_surface, simulated_evoked_free_surface, depth
):
    solver = BaseSolver(orientation="free", use_depth_weighting=depth)
    solver.SUPPORTS_VECTOR_ORIENTATION = True
    solver.make_inverse_operator(
        forward_model_free_surface, simulated_evoked_free_surface
    )
    solver.unpack_data_obj(simulated_evoked_free_surface)
    n = solver.forward["nsource"]
    rng = np.random.default_rng(20261107)
    local = rng.normal(size=(n * 3, 4)) * 1e-8
    internal = local * solver.depth_weights[:, None] if depth else local
    actual = solver.source_to_object_vector(internal)
    vertices = [s["vertno"] for s in solver.forward["src"]]
    reference = _make_stc(
        local,
        vertices,
        tmin=solver.tmin,
        tstep=1 / solver.obj_info["sfreq"],
        subject=actual.subject,
        vector=True,
        source_nn=solver.forward["source_nn"],
        src_type="surface",
    )
    np.testing.assert_allclose(actual.data, reference.data, rtol=1e-13, atol=0)
    basis = solver.forward["source_nn"].reshape(n, 3, 3)
    assert not np.allclose(basis, np.eye(3))  # A trivial identity test would miss it.
    recovered = np.einsum("nki,nit->nkt", basis, actual.data).reshape(n * 3, -1)
    np.testing.assert_allclose(recovered, local, rtol=1e-10, atol=1e-21)
    native_gain = np.asarray(solver.forward["sol"]["data"])
    np.testing.assert_allclose(
        native_gain @ recovered, native_gain @ local, rtol=1e-10, atol=1e-20
    )
    np.testing.assert_allclose(
        np.linalg.norm(actual.data, axis=1),
        solver.source_to_object(internal).data,
        rtol=1e-13,
        atol=0,
    )


@pytest.mark.parametrize("vector", [False, True])
def test_source_subject_comes_from_forward_not_evoked_person_id(
    forward_model_free_surface, simulated_evoked_free_surface, vector
):
    fwd = forward_model_free_surface.copy()
    for src in fwd["src"]:
        src["subject_his_id"] = "individual-anatomy"
    evoked = simulated_evoked_free_surface.copy()
    evoked.info["subject_info"] = {"his_id": "recording-person-id"}
    solver = BaseSolver(orientation="free")
    solver.SUPPORTS_VECTOR_ORIENTATION = True
    solver.make_inverse_operator(fwd, evoked)
    solver.unpack_data_obj(evoked)
    current = np.zeros((fwd["nsource"] * 3, 2))
    stc = (
        solver.source_to_object_vector(current)
        if vector
        else solver.source_to_object(current)
    )
    assert stc.subject == "individual-anatomy"
    assert isinstance(stc, mne.VectorSourceEstimate if vector else mne.SourceEstimate)
