import numpy as np

from invert.solver_registry import create_solver, get_solver_spec
from invert.solvers.bayesian.mem import _mem


def test_mem_runs_small():
    rng = np.random.default_rng(0)
    m, n, t = 10, 60, 8
    L = rng.standard_normal((m, n)) * 0.1
    Y = rng.standard_normal((m, t))

    J = _mem(Y, L, max_iter=8, batch_size=4, n_outer=2)
    assert J.shape == (n, t)
    assert np.isfinite(J).all()


def test_solver_registry_has_mem_and_cmem():
    mem_spec = get_solver_spec("mem")
    cmem_spec = get_solver_spec("cmem")
    assert mem_spec.class_name == "SolverMEM"
    assert cmem_spec.class_name == "SolverMEM"
    assert mem_spec.default_kwargs.get("clustered") is False
    assert cmem_spec.default_kwargs.get("clustered") is True

    mem_solver = create_solver("mem")
    cmem_solver = create_solver("cmem")
    assert mem_solver.__class__.__name__ == "SolverMEM"
    assert cmem_solver.__class__.__name__ == "SolverMEM"
    assert mem_solver.clustered is False
    assert cmem_solver.clustered is True
