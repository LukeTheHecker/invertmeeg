import numpy as np

from invert import Solver


def test_get_alphas_updates_self():
    solver = Solver("MNE")
    solver.alpha = 0.1
    alphas = solver.get_alphas(reference=np.eye(3))
    assert solver.alphas == alphas


def test_dics_scales_alpha_to_csd(forward_model, simulated_evoked):
    solver = Solver("DICS")
    solver.make_inverse_operator(
        forward_model, simulated_evoked, alpha=0.1, fmin=8.0, fmax=12.0
    )
    assert solver.csd is not None
    expected_max_eig = float(
        np.linalg.svd(np.real(solver.csd), compute_uv=False).max()
    )
    assert np.isclose(float(solver.max_eig), expected_max_eig)


def test_champagne_auto_alpha_changes_operator(forward_model, simulated_evoked):
    solver = Solver("Champagne", n_reg_params=3)
    # Use a moderate grid to keep the test fast/stable.
    solver.r_values = np.asarray([1e-6, 1e-3, 1e-1], dtype=float)

    solver.make_inverse_operator(
        forward_model, simulated_evoked, alpha="auto", max_iter=50
    )
    ops = [op.data[0] for op in solver.inverse_operators]
    assert len(ops) == 3

    diffs = [
        float(np.linalg.norm(ops[0] - ops[1])),
        float(np.linalg.norm(ops[1] - ops[2])),
    ]
    assert max(diffs) > 1e-12


def test_recipsiicos_alpha_is_dimensionless(forward_model, simulated_evoked):
    solver = Solver("ReciPSIICOS")
    solver.make_inverse_operator(forward_model, simulated_evoked, alpha=0.1)
    assert np.allclose(np.asarray(solver.alphas, dtype=float), [0.1])

