import mne
import numpy as np

from invert import Solver


def _as_mne_cov(data: np.ndarray, ch_names: list[str]) -> mne.Covariance:
    return mne.Covariance(
        data=np.asarray(data, dtype=float),
        names=list(ch_names),
        bads=[],
        projs=[],
        nfree=1,
    )


def test_solver_factory_resolves_sissy_variants():
    assert Solver("sissy").__class__.__name__ == "SolverSISSY"
    assert Solver("apesissy").__class__.__name__ == "SolverAPESISSY"
    assert Solver("ape-sissy").__class__.__name__ == "SolverAPESISSY"


def test_sissy_runs_with_whitening(forward_model, simulated_evoked):
    n_chans = len(forward_model.ch_names)
    n_dipoles = int(forward_model["nsource"])
    noise_cov = _as_mne_cov(np.eye(n_chans, dtype=float), forward_model.ch_names)

    solver = Solver("sissy", n_reg_params=1)
    solver.make_inverse_operator(
        forward_model, simulated_evoked, alpha=0.1, noise_cov=noise_cov
    )
    stc = solver.apply_inverse_operator(
        simulated_evoked,
        max_iter=12,
        lambda_reg=0.5,
        alpha_l1=0.05,
    )

    assert stc.data.shape[0] == n_dipoles
    assert stc.data.shape[1] == simulated_evoked.data.shape[1]
    assert np.all(np.isfinite(stc.data))
    assert np.any(stc.data != 0.0)


def test_apesissy_tracks_adaptive_lambda(forward_model, simulated_evoked):
    n_chans = len(forward_model.ch_names)
    noise_cov = _as_mne_cov(np.eye(n_chans, dtype=float), forward_model.ch_names)

    solver = Solver("apesissy", n_reg_params=1)
    solver.make_inverse_operator(
        forward_model, simulated_evoked, alpha=0.1, noise_cov=noise_cov
    )
    stc = solver.apply_inverse_operator(
        simulated_evoked,
        max_iter=12,
        lambda_init=0.1,
        alpha_l1=0.05,
    )

    info = solver.last_solver_info
    assert np.all(np.isfinite(stc.data))
    assert np.any(stc.data != 0.0)
    assert "lambda_path" in info
    assert info["lambda_path"].size >= 1
    assert np.all(np.isfinite(info["lambda_path"]))
    assert float(info["lambda_last"]) >= 0.0
    assert float(info["discrepancy_c"]) > 0.0
