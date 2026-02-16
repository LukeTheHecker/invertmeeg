"""Diagnostic evaluation for cMEM solver.

Runs cMEM on simulated data and prints diagnostics at each stage:
MSP scores, parcellation stats, dual function convergence,
posterior probabilities, and localization error.
Compares against MNE and eLORETA reference solvers.
"""

import importlib.util as _ilu
import logging
import time
from pathlib import Path

import mne
import numpy as np

# Direct import of evaluate_all to avoid circular dependency
_spec = _ilu.spec_from_file_location(
    "invert.evaluate.evaluate",
    str(Path(__file__).resolve().parents[1] / "invert" / "evaluate" / "evaluate.py"),
)
_eval_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_eval_mod)
evaluate_all = _eval_mod.evaluate_all

from invert.forward import create_forward_model, get_info
from invert.simulate import SimulationConfig, SimulationGenerator
from invert.solvers.bayesian.cmem import (
    SolverCMEM,
    _compute_msp_coefficients,
    _compute_sources_batch,
    _data_driven_parcellation,
    _mem_dual_obj_grad,
    _optimize_lambda_batch,
)
from invert.util import pos_from_forward

mne.set_log_level("WARNING")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_diagnostic():
    print("=" * 70)
    print("cMEM Diagnostic Evaluation")
    print("=" * 70)

    # Setup
    info = get_info(kind="biosemi32")
    fwd = create_forward_model(sampling="ico2", info=info)
    leadfield = fwd["sol"]["data"]
    n_channels, n_dipoles = leadfield.shape
    pos = pos_from_forward(fwd)
    adjacency = mne.spatial_src_adjacency(fwd["src"], verbose=0)

    print(f"Forward model: {n_channels} channels, {n_dipoles} dipoles")

    # Create focal simulation
    config = SimulationConfig(
        batch_size=3,
        n_sources=(1, 2),
        n_orders=(0, 0),
        snr_range=(5.0, 5.0),
        n_timepoints=50,
        random_seed=42,
        estimate_noise_cov=True,
        return_noise_cov=True,
    )

    gen = SimulationGenerator(fwd, config=config)
    x_batch, y_batch, sim_info = next(gen.generate())

    for sample_idx in range(min(2, len(x_batch))):
        print(f"\n{'=' * 70}")
        print(f"Sample {sample_idx + 1}")
        print(f"{'=' * 70}")

        x = x_batch[sample_idx]
        y_true = y_batch[sample_idx]

        # Get noise cov
        noise_cov_arr = sim_info.iloc[sample_idx].get("noise_cov_est", None)
        if noise_cov_arr is not None:
            from invert.benchmark.runner import _make_mne_covariance

            noise_cov = _make_mne_covariance(noise_cov_arr, info, nfree=200)
        else:
            noise_cov = None

        evoked = mne.EvokedArray(x, info, tmin=0.0, verbose=0)

        # Ground truth info
        true_peak = np.argmax(np.max(np.abs(y_true), axis=1))
        print(f"True peak source: {true_peak}, position: {pos[true_peak]}")

        # --- Stage-by-stage diagnostics ---
        print("\n--- Stage 1: MSP Scores ---")
        # Whiten data as cMEM would
        solver_tmp = SolverCMEM()
        solver_tmp.make_inverse_operator(fwd, mne_obj=evoked, alpha="auto", noise_cov=noise_cov)
        # Get whitened data/leadfield
        wf = solver_tmp.prepare_whitened_forward(noise_cov)
        Y_w = wf.sensor_transform @ x
        L_w = solver_tmp.leadfield

        msp_scores = _compute_msp_coefficients(Y_w, L_w)
        msp_peak = np.argmax(msp_scores)
        msp_at_true = msp_scores[true_peak]
        print(f"MSP range: [{msp_scores.min():.4f}, {msp_scores.max():.4f}]")
        print(f"MSP peak source: {msp_peak} (score={msp_scores[msp_peak]:.4f})")
        print(f"MSP at true source: {msp_at_true:.4f}")
        dist_msp = np.linalg.norm(pos[msp_peak] - pos[true_peak])
        print(f"MSP peak distance to true: {dist_msp:.1f} mm")

        print("\n--- Stage 2: Parcellation ---")
        parcels = _data_driven_parcellation(Y_w, L_w, num_parcels=200, A=adjacency)
        unique_parcels = np.unique(parcels)
        parcel_sizes = [np.sum(parcels == p) for p in unique_parcels]
        print(f"Number of parcels: {len(unique_parcels)}")
        print(f"Parcel sizes: min={min(parcel_sizes)}, max={max(parcel_sizes)}, "
              f"median={np.median(parcel_sizes):.0f}")
        true_parcel = parcels[true_peak]
        print(f"True source in parcel {true_parcel} "
              f"(size={np.sum(parcels == true_parcel)})")

        print("\n--- Stage 3: Energy Scaling ---")
        J_mne = np.linalg.lstsq(L_w, Y_w, rcond=None)[0]
        for pid in list(unique_parcels[:5]):
            verts = np.where(parcels == pid)[0]
            eta = 0.05 * np.mean(J_mne[verts] ** 2)
            print(f"  Parcel {pid}: eta={eta:.2e}, n_verts={len(verts)}")

        print("\n--- Stage 4: Dual Function ---")
        # Build precomputed dict (same as _cmem does)
        precomputed = {}
        for parcel_id in unique_parcels:
            verts = np.where(parcels == parcel_id)[0]
            mu_k = np.zeros(len(verts))
            if adjacency is not None:
                from invert.solvers.bayesian.cmem import _compute_spatial_covariance
                W_k = _compute_spatial_covariance(verts, adjacency)
            else:
                W_k = np.eye(len(verts))
            eta_k = max(0.05 * np.mean(J_mne[verts] ** 2), 1e-30)
            Sigma_k = eta_k * W_k
            L_k = L_w[:, verts]
            msp_k = _compute_msp_coefficients(Y_w, L_k)
            alpha_k = np.clip(np.median(msp_k), 0.01, 0.99)
            precomputed[parcel_id] = {
                "L_k": L_k, "Sigma_k": Sigma_k, "mu_k": mu_k,
                "alpha_k": alpha_k, "vertices": verts,
            }

        # Test dual at lambda=0 and after optimization for one time point
        y_t = Y_w[:, 0]
        obj0, grad0 = _mem_dual_obj_grad(np.zeros(len(y_t)), y_t, precomputed)
        print(f"Dual at lambda=0: obj={obj0:.4f}, |grad|={np.linalg.norm(grad0):.4f}")

        lam_opt = _optimize_lambda_batch(Y_w[:, :1], precomputed, max_iter=100)
        obj_opt, grad_opt = _mem_dual_obj_grad(lam_opt[:, 0], y_t, precomputed)
        print(f"Dual after opt:   obj={obj_opt:.4f}, |grad|={np.linalg.norm(grad_opt):.4f}")
        print(f"Lambda range: [{lam_opt.min():.2e}, {lam_opt.max():.2e}]")

        print("\n--- Stage 5: Posterior Probabilities ---")
        J_batch = _compute_sources_batch(lam_opt, precomputed, n_dipoles)
        # Check per-parcel activation probabilities
        for pid in list(unique_parcels[:10]):
            verts = precomputed[pid]["vertices"]
            L_k = precomputed[pid]["L_k"]
            Sigma_k = precomputed[pid]["Sigma_k"]
            mu_k = precomputed[pid]["mu_k"]
            alpha_k = precomputed[pid]["alpha_k"]
            xi_k = L_k.T @ lam_opt[:, 0]
            Sigma_xi = Sigma_k @ xi_k
            E_k = np.dot(mu_k, xi_k) + 0.5 * np.dot(xi_k, Sigma_xi)
            exp_E = np.exp(np.clip(E_k, -500, 500))
            p_k = alpha_k * exp_E / ((1 - alpha_k) + alpha_k * exp_E)
            print(f"  Parcel {pid}: alpha={alpha_k:.3f}, E_k={E_k:.3f}, p_k={p_k:.4f}")

        print("\n--- Stage 6: cMEM Result ---")
        t0 = time.time()
        solver = SolverCMEM(num_parcels=200, max_iter=100)
        solver.make_inverse_operator(fwd, mne_obj=evoked, alpha="auto", noise_cov=noise_cov)
        stc = solver.apply_inverse_operator(evoked)
        elapsed = time.time() - t0
        y_pred = stc.data
        print(f"Time: {elapsed:.2f}s")

        pred_peak = np.argmax(np.max(np.abs(y_pred), axis=1))
        dist = np.linalg.norm(pos[pred_peak] - pos[true_peak])
        print(f"Predicted peak: {pred_peak}, true peak: {true_peak}")
        print(f"Localization error: {dist:.1f} mm")

        metrics = evaluate_all(y_true, y_pred, adjacency, adjacency, pos, pos)
        print(f"MLE={metrics['Mean_Localization_Error']:.1f}mm, EMD={metrics['EMD']:.1f}, "
              f"Corr={metrics['correlation']:.3f}")

        # --- Reference solvers ---
        print("\n--- Reference: eLORETA ---")
        from invert.solvers.minimum_norm.eloreta import SolverELORETA

        solver_elor = SolverELORETA()
        solver_elor.make_inverse_operator(fwd, mne_obj=evoked, alpha="auto", noise_cov=noise_cov)
        stc_elor = solver_elor.apply_inverse_operator(evoked)
        y_elor = stc_elor.data
        peak_elor = np.argmax(np.max(np.abs(y_elor), axis=1))
        dist_elor = np.linalg.norm(pos[peak_elor] - pos[true_peak])
        metrics_elor = evaluate_all(y_true, y_elor, adjacency, adjacency, pos, pos)
        print(f"Peak error: {dist_elor:.1f}mm")
        print(f"MLE={metrics_elor['Mean_Localization_Error']:.1f}mm, EMD={metrics_elor['EMD']:.1f}, "
              f"Corr={metrics_elor['correlation']:.3f}")

        print("\n--- Reference: MNE ---")
        from invert.solvers.minimum_norm.mne import SolverMNE

        solver_mne = SolverMNE()
        solver_mne.make_inverse_operator(fwd, mne_obj=evoked, alpha="auto", noise_cov=noise_cov)
        stc_mne = solver_mne.apply_inverse_operator(evoked)
        y_mne = stc_mne.data
        peak_mne = np.argmax(np.max(np.abs(y_mne), axis=1))
        dist_mne = np.linalg.norm(pos[peak_mne] - pos[true_peak])
        metrics_mne = evaluate_all(y_true, y_mne, adjacency, adjacency, pos, pos)
        print(f"Peak error: {dist_mne:.1f}mm")
        print(f"MLE={metrics_mne['Mean_Localization_Error']:.1f}mm, EMD={metrics_mne['EMD']:.1f}, "
              f"Corr={metrics_mne['correlation']:.3f}")


if __name__ == "__main__":
    run_diagnostic()
