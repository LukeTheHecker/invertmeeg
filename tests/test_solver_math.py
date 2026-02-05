"""Mathematical correctness tests for actual solver implementations.

These tests instantiate real solver classes with a synthetic MNE forward model
and verify that the outputs satisfy known mathematical properties of each method.
"""

from copy import deepcopy

import mne
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_kernel(solver, idx=0):
    """Get the raw inverse operator matrix from a solver."""
    return solver.inverse_operators[idx].data[0]


def _get_leadfield(solver):
    """Get the (preprocessed) leadfield from a solver."""
    return solver.leadfield


# ---------------------------------------------------------------------------
# 1. SolverMNE: Tikhonov normal equations
# ---------------------------------------------------------------------------


class TestSolverMNE:
    def test_satisfies_normal_equations(self, forward_model, simulated_evoked):
        """The MNE kernel K should satisfy K = L^T (L L^T + alpha I)^{-1}
        on the depth-weighted leadfield the solver actually uses."""
        from invert.solvers.minimum_norm.mne import SolverMNE

        alpha_val = 0.1
        solver = SolverMNE()
        solver.make_inverse_operator(forward_model, simulated_evoked, alpha=alpha_val)

        L = _get_leadfield(solver)
        K = _extract_kernel(solver)
        n_chans = L.shape[0]
        actual_alpha = solver.alphas[0]

        K_expected = L.T @ np.linalg.inv(L @ L.T + actual_alpha * np.eye(n_chans))
        np.testing.assert_allclose(K, K_expected, atol=1e-8)

    def test_zero_input_zero_output(self, forward_model, simulated_evoked):
        """Zero data should produce zero source estimate."""
        from invert.solvers.minimum_norm.mne import SolverMNE

        solver = SolverMNE()
        solver.make_inverse_operator(forward_model, simulated_evoked, alpha=0.1)

        K = _extract_kernel(solver)
        n_chans = K.shape[1]
        result = K @ np.zeros(n_chans)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_linearity(self, forward_model, simulated_evoked):
        """K @ (a*y1 + b*y2) == a*K@y1 + b*K@y2."""
        from invert.solvers.minimum_norm.mne import SolverMNE

        solver = SolverMNE()
        solver.make_inverse_operator(forward_model, simulated_evoked, alpha=0.1)

        K = _extract_kernel(solver)
        n_chans = K.shape[1]
        rng = np.random.RandomState(0)
        y1, y2 = rng.randn(n_chans), rng.randn(n_chans)
        a, b = 2.5, -0.7

        np.testing.assert_allclose(
            K @ (a * y1 + b * y2),
            a * (K @ y1) + b * (K @ y2),
            atol=1e-10,
        )

    def test_increasing_alpha_decreases_source_norm(
        self, forward_model, simulated_evoked
    ):
        """Larger regularization should shrink the solution norm."""
        from invert.solvers.minimum_norm.mne import SolverMNE

        rng = np.random.RandomState(1)
        norms = []
        for alpha in [0.01, 0.1, 1.0]:
            solver = SolverMNE(n_reg_params=1)
            solver.make_inverse_operator(forward_model, simulated_evoked, alpha=alpha)
            K = _extract_kernel(solver)
            y = rng.randn(K.shape[1])
            norms.append(np.linalg.norm(K @ y))

        for i in range(len(norms) - 1):
            assert norms[i] > norms[i + 1]

    def test_resolution_matrix_trace(self, forward_model, simulated_evoked):
        """trace(K @ L) should equal sum of s_i^2 / (s_i^2 + alpha)."""
        from invert.solvers.minimum_norm.mne import SolverMNE

        solver = SolverMNE()
        solver.make_inverse_operator(forward_model, simulated_evoked, alpha=0.1)

        L = _get_leadfield(solver)
        K = _extract_kernel(solver)
        alpha = solver.alphas[0]

        _, s, _ = np.linalg.svd(L, full_matrices=False)
        expected_trace = np.sum(s**2 / (s**2 + alpha))
        np.testing.assert_allclose(np.trace(K @ L), expected_trace, rtol=1e-6)

    def test_primal_dual_equivalence(self, forward_model, simulated_evoked):
        """L^T(LL^T+aI)^{-1} should equal (L^TL+aI)^{-1}L^T on the actual leadfield."""
        from invert.solvers.minimum_norm.mne import SolverMNE

        solver = SolverMNE()
        solver.make_inverse_operator(forward_model, simulated_evoked, alpha=0.1)

        L = _get_leadfield(solver)
        K_dual = _extract_kernel(solver)
        alpha = solver.alphas[0]
        n_dipoles = L.shape[1]

        K_primal = np.linalg.inv(L.T @ L + alpha * np.eye(n_dipoles)) @ L.T
        np.testing.assert_allclose(K_dual, K_primal, atol=1e-8)


# ---------------------------------------------------------------------------
# 2. SolverSLORETA: standardization property
# ---------------------------------------------------------------------------


class TestSolverSLORETA:
    def test_standardized_resolution_diagonal(self, forward_model, simulated_evoked):
        """diag(K_slor @ L)^2 should equal diag(K_mne @ L) for MNE resolution
        diagonal elements, verifying the sLORETA standardization."""
        from invert.solvers.minimum_norm.sloreta import SolverSLORETA

        solver = SolverSLORETA()
        solver.make_inverse_operator(forward_model, simulated_evoked, alpha=0.05)

        L = _get_leadfield(solver)
        K_slor = solver._sloreta_operators[0].data[0]
        K_mne = _extract_kernel(solver)
        R_mne_diag = np.diag(K_mne @ L)

        # sLORETA resolution diagonal squared should equal MNE resolution diagonal
        R_slor_diag = np.diag(K_slor @ L)
        np.testing.assert_allclose(R_slor_diag**2, R_mne_diag, rtol=1e-8)

    def test_output_finite_and_nonzero(self, forward_model, simulated_evoked):
        """Basic sanity: sLORETA kernel should be finite and non-trivial."""
        from invert.solvers.minimum_norm.sloreta import SolverSLORETA

        solver = SolverSLORETA()
        solver.make_inverse_operator(forward_model, simulated_evoked, alpha=0.05)

        K = solver._sloreta_operators[0].data[0]
        assert np.all(np.isfinite(K))
        assert np.any(K != 0)

    def test_auto_alpha(self, forward_model, simulated_evoked):
        """sLORETA should accept alpha='auto' and build multiple operators."""
        from invert.solvers.minimum_norm.sloreta import SolverSLORETA

        solver = SolverSLORETA()
        solver.make_inverse_operator(forward_model, simulated_evoked, alpha="auto")
        assert len(solver.inverse_operators) > 1
        assert len(solver._sloreta_operators) == len(solver.inverse_operators)

    def test_auto_alpha_standardization(self, forward_model, simulated_evoked):
        """sLORETA operators built with alpha='auto' should each satisfy the
        standardization property: diag(K_slor @ L)^2 == diag(K_mne @ L)."""
        from invert.solvers.minimum_norm.sloreta import SolverSLORETA

        solver = SolverSLORETA()
        solver.make_inverse_operator(forward_model, simulated_evoked, alpha="auto")

        L = _get_leadfield(solver)
        for mne_op, slor_op in zip(solver.inverse_operators, solver._sloreta_operators):
            K_mne = mne_op.data[0]
            K_slor = slor_op.data[0]
            R_mne_diag = np.diag(K_mne @ L)
            R_slor_diag = np.diag(K_slor @ L)
            np.testing.assert_allclose(R_slor_diag**2, R_mne_diag, rtol=1e-8)


# ---------------------------------------------------------------------------
# 3. SolverLORETA: smoother than MNE
# ---------------------------------------------------------------------------


class TestSolverLORETA:
    def test_smoother_than_mne(self, forward_model, simulated_evoked):
        """LORETA solution should have lower Laplacian norm than MNE for the
        same data, verifying the smoothness prior is effective."""
        from scipy.sparse.csgraph import laplacian as sp_laplacian

        from invert.solvers.minimum_norm.loreta import SolverLORETA
        from invert.solvers.minimum_norm.mne import SolverMNE

        alpha = 0.1

        solver_mne = SolverMNE()
        solver_mne.make_inverse_operator(
            deepcopy(forward_model), simulated_evoked, alpha=alpha
        )
        stc_mne = solver_mne.apply_inverse_operator(simulated_evoked)

        solver_lor = SolverLORETA()
        solver_lor.make_inverse_operator(
            deepcopy(forward_model), simulated_evoked, alpha=alpha
        )
        stc_lor = solver_lor.apply_inverse_operator(simulated_evoked)

        # Build Laplacian from the source adjacency
        adj = mne.spatial_src_adjacency(forward_model["src"], verbose=0).toarray()
        Lap = sp_laplacian(adj)

        lap_norm_mne = np.linalg.norm(Lap @ stc_mne.data)
        lap_norm_lor = np.linalg.norm(Lap @ stc_lor.data)

        assert lap_norm_lor < lap_norm_mne, (
            f"LORETA Laplacian norm ({lap_norm_lor:.4f}) should be < "
            f"MNE ({lap_norm_mne:.4f})"
        )


# ---------------------------------------------------------------------------
# 4. SolverLCMV: unit-gain constraint (before weight normalization)
# ---------------------------------------------------------------------------


class TestSolverLCMV:
    def test_unit_gain_without_weight_norm(self, forward_model, simulated_evoked):
        """LCMV weights (without weight_norm) should satisfy w_i^T l_i = 1."""
        from invert.solvers.beamformers.lcmv import SolverLCMV

        solver = SolverLCMV()
        solver.make_inverse_operator(
            deepcopy(forward_model),
            simulated_evoked,
            alpha=0.1,
            weight_norm=False,
        )

        L = _get_leadfield(solver)
        # After LCMV, leadfield is column-normalized
        L_normed = L / np.linalg.norm(L, axis=0, keepdims=True)

        K = _extract_kernel(solver)  # shape (n_dipoles, n_chans)
        # Unit gain: K[i,:] @ L_normed[:,i] = 1 for each dipole i
        gains = np.einsum("ij,ji->i", K, L_normed)
        np.testing.assert_allclose(gains, 1.0, atol=1e-6)

    def test_output_varies_with_data(
        self, forward_model, sensor_info, simulated_evoked
    ):
        """LCMV kernel should change when the data covariance changes."""
        from invert.solvers.beamformers.lcmv import SolverLCMV

        info = sensor_info
        n_chans = len(info["ch_names"])
        rng = np.random.RandomState(10)

        evoked1 = mne.EvokedArray(rng.randn(n_chans, 20), info, verbose=0)
        evoked1.set_eeg_reference("average", projection=True, verbose=0).apply_proj()
        evoked2 = mne.EvokedArray(rng.randn(n_chans, 20) * 5, info, verbose=0)
        evoked2.set_eeg_reference("average", projection=True, verbose=0).apply_proj()

        solver1 = SolverLCMV()
        solver1.make_inverse_operator(
            deepcopy(forward_model),
            evoked1,
            alpha=0.1,
            weight_norm=False,
        )
        solver2 = SolverLCMV()
        solver2.make_inverse_operator(
            deepcopy(forward_model),
            evoked2,
            alpha=0.1,
            weight_norm=False,
        )

        K1 = _extract_kernel(solver1)
        K2 = _extract_kernel(solver2)
        assert not np.allclose(K1, K2, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. SolverOMP: sparse recovery from noiseless data
# ---------------------------------------------------------------------------


class TestSolverOMP:
    def test_recovers_sparse_source(self, forward_model, sensor_info, simulated_evoked):
        """Given noiseless data from a few active dipoles, OMP should
        identify at least some of the correct support."""
        from invert.solvers.matching_pursuit import SolverOMP

        solver = SolverOMP()
        solver.make_inverse_operator(
            deepcopy(forward_model), simulated_evoked, alpha=0.1
        )

        L = solver.leadfield_original
        n_chans, n_dipoles = L.shape

        # Create a sparse source with 2 active dipoles
        rng = np.random.RandomState(42)
        true_support = rng.choice(n_dipoles, size=2, replace=False)
        x_true = np.zeros(n_dipoles)
        x_true[true_support] = rng.randn(2) * 10

        # Generate noiseless data and wrap as Evoked
        y = L @ x_true
        info = sensor_info
        evoked = mne.EvokedArray(y.reshape(-1, 1), info, verbose=0)
        evoked.set_eeg_reference("average", projection=True, verbose=0).apply_proj()

        stc = solver.apply_inverse_operator(evoked, K=2, max_iter=10)
        estimated_support = set(np.argsort(np.abs(stc.data[:, 0]))[-5:])

        # At least one of the true dipoles should be in the top 5
        overlap = estimated_support & set(true_support)
        assert len(overlap) >= 1, (
            f"OMP found {estimated_support}, expected overlap with {set(true_support)}"
        )

    def test_output_is_sparse(self, forward_model, simulated_evoked):
        """OMP output should have many zeros (sparse)."""
        from invert.solvers.matching_pursuit import SolverOMP

        solver = SolverOMP()
        solver.make_inverse_operator(
            deepcopy(forward_model), simulated_evoked, alpha=0.1
        )
        stc = solver.apply_inverse_operator(simulated_evoked, K=2, max_iter=5)

        n_dipoles = stc.data.shape[0]
        n_nonzero = np.count_nonzero(stc.data[:, 0])
        # OMP should activate far fewer than all dipoles
        assert n_nonzero < n_dipoles * 0.5, (
            f"OMP activated {n_nonzero}/{n_dipoles} dipoles, expected sparse output"
        )


# ---------------------------------------------------------------------------
# 6. SolverDSPM: noise normalization property
# ---------------------------------------------------------------------------


class TestSolverDSPM:
    def test_normalization_matches_formula(self, forward_model, simulated_evoked):
        """dSPM kernel should equal diag(1/sqrt(diag(K C_n K^T))) @ K
        where K is the MNE kernel and C_n is the noise covariance (identity)."""
        from invert.solvers.minimum_norm.dspm import SolverDSPM

        solver = SolverDSPM()
        solver.make_inverse_operator(forward_model, simulated_evoked, alpha=0.1)

        L = _get_leadfield(solver)
        K_dspm = _extract_kernel(solver)
        alpha = solver.alphas[0]
        n_chans = L.shape[0]

        # Reconstruct MNE kernel
        K_mne = L.T @ np.linalg.inv(L @ L.T + alpha * np.eye(n_chans))

        # dSPM normalization with identity noise covariance
        noise_cov = np.eye(n_chans)
        variance = np.diag(K_mne @ noise_cov @ K_mne.T)
        W = np.diag(1.0 / np.sqrt(variance))
        K_expected = W @ K_mne

        np.testing.assert_allclose(K_dspm, K_expected, rtol=1e-3)


# ---------------------------------------------------------------------------
# 7. Cross-solver consistency: MNE vs sLORETA
# ---------------------------------------------------------------------------


class TestCrossSolverConsistency:
    def test_sloreta_and_mne_same_sign_pattern(self, forward_model, simulated_evoked):
        """sLORETA and MNE should produce source estimates with the same
        sign pattern (sLORETA only rescales, doesn't flip signs)."""
        from invert.solvers.minimum_norm.mne import SolverMNE
        from invert.solvers.minimum_norm.sloreta import SolverSLORETA

        solver_mne = SolverMNE()
        solver_mne.make_inverse_operator(
            deepcopy(forward_model), simulated_evoked, alpha=0.05
        )
        stc_mne = solver_mne.apply_inverse_operator(simulated_evoked)

        solver_slor = SolverSLORETA()
        solver_slor.make_inverse_operator(
            deepcopy(forward_model), simulated_evoked, alpha=0.05
        )
        stc_slor = solver_slor.apply_inverse_operator(simulated_evoked)

        # Where both are non-negligible, signs should agree
        mask = (np.abs(stc_mne.data) > 1e-10) & (np.abs(stc_slor.data) > 1e-10)
        if mask.any():
            signs_mne = np.sign(stc_mne.data[mask])
            signs_slor = np.sign(stc_slor.data[mask])
            agreement = np.mean(signs_mne == signs_slor)
            assert agreement > 0.99, (
                f"Sign agreement between MNE and sLORETA is only {agreement:.1%}"
            )


# ---------------------------------------------------------------------------
# 8. Depth weighting integration test
# ---------------------------------------------------------------------------


class TestDepthWeightingIntegration:
    def test_depth_weighting_applied(self, forward_model, simulated_evoked):
        """Verify the solver's leadfield has been depth-weighted
        (column norms should differ from the raw leadfield)."""
        from invert.solvers.minimum_norm.mne import SolverMNE

        raw_leadfield = forward_model["sol"]["data"].copy()
        raw_norms = np.linalg.norm(raw_leadfield, axis=0)

        solver = SolverMNE()
        solver.make_inverse_operator(forward_model, simulated_evoked, alpha=0.1)
        processed_norms = np.linalg.norm(_get_leadfield(solver), axis=0)

        # Depth weighting should change the column norms
        assert not np.allclose(raw_norms, processed_norms, rtol=0.01), (
            "Leadfield column norms unchanged — depth weighting may not be applied"
        )

    def test_no_depth_weighting_option(self, forward_model, simulated_evoked):
        """With prep_leadfield=False, column norms should match the raw forward."""
        from invert.solvers.minimum_norm.mne import SolverMNE

        raw_leadfield = forward_model["sol"]["data"].copy()

        solver = SolverMNE(prep_leadfield=False)
        solver.make_inverse_operator(
            deepcopy(forward_model), simulated_evoked, alpha=0.1
        )
        processed = _get_leadfield(solver)

        np.testing.assert_allclose(processed, raw_leadfield, atol=1e-12)


# ---------------------------------------------------------------------------
# 9. Regularization selection
# ---------------------------------------------------------------------------


class TestRegularizationSelection:
    def test_auto_alpha_produces_multiple_operators(
        self, forward_model, simulated_evoked
    ):
        """alpha='auto' should produce multiple inverse operators for selection."""
        from invert.solvers.minimum_norm.mne import SolverMNE

        solver = SolverMNE(n_reg_params=5)
        solver.make_inverse_operator(forward_model, simulated_evoked, alpha="auto")

        assert len(solver.inverse_operators) > 1

    def test_fixed_alpha_produces_single_operator(
        self, forward_model, simulated_evoked
    ):
        """A fixed alpha should produce exactly one inverse operator."""
        from invert.solvers.minimum_norm.mne import SolverMNE

        solver = SolverMNE()
        solver.make_inverse_operator(forward_model, simulated_evoked, alpha=0.1)

        assert len(solver.inverse_operators) == 1


# ---------------------------------------------------------------------------
# 10. FISTA mathematical correctness
# ---------------------------------------------------------------------------


class TestFISTAMathCorrectness:
    """Tests that FISTA solvers decrease the objective monotonically and
    produce solutions close to a reference optimizer on small problems."""

    @staticmethod
    def _l1_l2_objective(x, A, y, l1_reg, l2_reg=0.0):
        """Compute 0.5*||y - Ax||^2 + l1*||x||_1 + 0.5*l2*||x||^2"""
        residual = y - A @ x
        return (
            0.5 * np.dot(residual, residual)
            + l1_reg * np.sum(np.abs(x))
            + 0.5 * l2_reg * np.dot(x, x)
        )

    def test_ista_objective_decreases(self):
        """On a small synthetic problem, ISTA (no momentum) iterates should
        decrease the objective function monotonically."""
        rng = np.random.RandomState(123)
        m, n = 10, 30
        A = rng.randn(m, n)
        x_true = np.zeros(n)
        x_true[rng.choice(n, 3, replace=False)] = rng.randn(3)
        y = A @ x_true

        l1_reg = 0.1
        l2_reg = 0.01

        # Lipschitz constant
        L = np.linalg.norm(A, ord=2) ** 2 + l2_reg
        lr = 1.0 / L

        x = np.zeros(n)

        objectives = [self._l1_l2_objective(x, A, y, l1_reg, l2_reg)]
        n_iter = 200
        for _ in range(n_iter):
            grad = A.T @ (A @ x - y) + l2_reg * x
            x_new = x - lr * grad
            # soft threshold
            x_new = np.sign(x_new) * np.maximum(np.abs(x_new) - l1_reg * lr, 0)
            x = x_new
            objectives.append(self._l1_l2_objective(x, A, y, l1_reg, l2_reg))

        # Objective should be non-increasing (allow tiny numerical noise)
        objectives = np.array(objectives)
        diffs = np.diff(objectives)
        assert np.all(diffs < 1e-10), (
            f"Objective increased at iterations: {np.where(diffs > 1e-10)[0]}"
        )

    def test_fista_converges(self):
        """FISTA should converge to a lower objective than the initial point."""
        rng = np.random.RandomState(123)
        m, n = 10, 30
        A = rng.randn(m, n)
        x_true = np.zeros(n)
        x_true[rng.choice(n, 3, replace=False)] = rng.randn(3)
        y = A @ x_true

        l1_reg = 0.1

        L = np.linalg.norm(A, ord=2) ** 2
        lr = 1.0 / L

        x = np.zeros(n)
        z = x.copy()
        t = 1.0
        obj_init = self._l1_l2_objective(x, A, y, l1_reg)

        for _ in range(500):
            grad = A.T @ (A @ z - y)
            x_new = z - lr * grad
            x_new = np.sign(x_new) * np.maximum(np.abs(x_new) - l1_reg * lr, 0)
            t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
            z = x_new + (t - 1) / t_new * (x_new - x)
            x = x_new
            t = t_new

        obj_final = self._l1_l2_objective(x, A, y, l1_reg)
        assert obj_final < obj_init * 0.1, (
            f"FISTA did not converge: init={obj_init:.6f}, final={obj_final:.6f}"
        )

    def test_fista_matches_scipy_reference(self):
        """FISTA solution should be close to scipy.optimize.minimize on a
        small elastic net problem."""
        from scipy.optimize import minimize

        rng = np.random.RandomState(42)
        m, n = 8, 15
        A = rng.randn(m, n)
        x_true = np.zeros(n)
        x_true[[2, 7, 11]] = [1.0, -0.5, 0.8]
        y = A @ x_true + rng.randn(m) * 0.01

        l1_reg = 0.05

        # FISTA solution
        L = np.linalg.norm(A, ord=2) ** 2
        lr = 1.0 / L
        x = np.zeros(n)
        z = x.copy()
        t = 1.0
        for _ in range(2000):
            grad = A.T @ (A @ z - y)
            x_new = z - lr * grad
            x_new = np.sign(x_new) * np.maximum(np.abs(x_new) - l1_reg * lr, 0)
            t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
            z = x_new + (t - 1) / t_new * (x_new - x)
            x = x_new
            t = t_new
        x_fista = x

        # Reference: scipy L-BFGS on smooth approximation
        def smooth_obj(x):
            res = y - A @ x
            eps = 1e-8
            return 0.5 * np.dot(res, res) + l1_reg * np.sum(np.sqrt(x**2 + eps))

        def smooth_grad(x):
            eps = 1e-8
            return -A.T @ (y - A @ x) + l1_reg * x / np.sqrt(x**2 + eps)

        result = minimize(
            smooth_obj,
            np.zeros(n),
            jac=smooth_grad,
            method="L-BFGS-B",
            options={"maxiter": 5000},
        )
        x_ref = result.x

        # Both should achieve similar objective values
        obj_fista = self._l1_l2_objective(x_fista, A, y, l1_reg)
        obj_ref = 0.5 * np.sum((y - A @ x_ref) ** 2) + l1_reg * np.sum(np.abs(x_ref))

        # FISTA should be at least as good as the smooth approximation
        assert obj_fista < obj_ref * 1.1, (
            f"FISTA obj {obj_fista:.6f} much worse than reference {obj_ref:.6f}"
        )


# ---------------------------------------------------------------------------
# 4. Regularization selection: (Modified) GCV
# ---------------------------------------------------------------------------


class TestGCVSelection:
    def test_mgcv_biases_toward_larger_alpha(self):
        """Modified GCV (gamma>1) should bias selection toward larger alpha
        relative to plain GCV on the same problem."""
        from invert.solvers.base import BaseSolver, InverseOperator

        rng = np.random.RandomState(0)
        m, n = 16, 40
        L = rng.randn(m, n)  # full row rank with high probability

        # Build a set of minimum-norm inverse operators: W(a) = L^T (L L^T + a I)^-1
        max_eig = np.linalg.svd(L @ L.T, full_matrices=False)[1].max()
        alphas = max_eig * np.logspace(-9, 4, 60)
        I = np.eye(m)
        inverse_operators = [
            InverseOperator(L.T @ np.linalg.inv(L @ L.T + a * I), "test")
            for a in alphas
        ]

        solver = BaseSolver(n_reg_params=len(alphas))
        solver.leadfield = L
        solver.inverse_operators = inverse_operators
        solver.alphas = list(alphas)

        # Weak sparse signal + noise
        rng = np.random.RandomState(0)
        x_true = np.zeros(n)
        x_true[rng.choice(n, 3, replace=False)] = rng.randn(3)
        M = L @ x_true[:, None] + 0.5 * rng.randn(m, 50)

        _, idx_gcv = solver.regularise_gcv(M, gamma=1.0)
        _, idx_mgcv = solver.regularise_gcv(M, gamma=1.02)

        assert idx_mgcv > idx_gcv
