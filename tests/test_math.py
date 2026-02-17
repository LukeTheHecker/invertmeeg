"""Mathematical correctness tests for inverse solvers.

These tests verify that the solvers satisfy known mathematical properties,
independent of MNE infrastructure where possible.
"""

import importlib.util
import pathlib

import numpy as np
import pytest

# Import evaluate.py directly (bypassing invert.evaluate.__init__ which has
# a broken import of invert.models.priors via evaluation.py).
_eval_path = (
    pathlib.Path(__file__).resolve().parent.parent
    / "invert"
    / "evaluate"
    / "evaluate.py"
)
_eval_spec = importlib.util.spec_from_file_location("_evaluate_standalone", _eval_path)
assert _eval_spec is not None and _eval_spec.loader is not None
_eval_mod = importlib.util.module_from_spec(_eval_spec)
_eval_spec.loader.exec_module(_eval_mod)

# ---------------------------------------------------------------------------
# Helpers – small synthetic forward problems (no MNE dependency)
# ---------------------------------------------------------------------------


def _random_leadfield(n_chans=20, n_dipoles=50, seed=0):
    """Return a random leadfield matrix L (n_chans, n_dipoles)."""
    rng = np.random.RandomState(seed)
    return rng.randn(n_chans, n_dipoles)


def _mne_kernel(L, alpha):
    """Compute the MNE inverse kernel: K = L^T (L L^T + alpha I)^{-1}."""
    n_chans = L.shape[0]
    return L.T @ np.linalg.inv(L @ L.T + alpha * np.eye(n_chans))


# ---------------------------------------------------------------------------
# 1. MNE identity: alpha -> 0, square invertible L => K @ L -> I
# ---------------------------------------------------------------------------


class TestMNEIdentity:
    def test_square_invertible(self):
        """When L is square & invertible and alpha~0, K @ L ~ I."""
        n = 20
        rng = np.random.RandomState(1)
        L = rng.randn(n, n)
        # Make well-conditioned
        L = L @ L.T + 0.1 * np.eye(n)
        alpha = 1e-12
        K = _mne_kernel(L, alpha)
        np.testing.assert_allclose(K @ L, np.eye(n), atol=1e-4)


# ---------------------------------------------------------------------------
# 2. MNE formula verification
# ---------------------------------------------------------------------------


class TestMNEFormula:
    @pytest.mark.parametrize("alpha", [0.01, 0.1, 1.0, 10.0])
    def test_formula(self, alpha):
        """Verify K = L^T (L L^T + alpha I)^{-1} matches manual computation."""
        L = _random_leadfield(15, 40, seed=2)
        n_chans = L.shape[0]

        K_expected = L.T @ np.linalg.inv(L @ L.T + alpha * np.eye(n_chans))
        K_actual = _mne_kernel(L, alpha)
        np.testing.assert_allclose(K_actual, K_expected, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. sLORETA normalization: unit-variance property
# ---------------------------------------------------------------------------


class TestSLORETANormalization:
    def test_standardized_resolution(self):
        """sLORETA normalizes each source by sqrt(resolution_diag), so the
        squared resolution diagonal should equal the sign of the MNE
        resolution diagonal (i.e., diag(K_slor @ L)^2 = |diag(K_mne @ L)| /
        |diag(K_mne @ L)| = 1 when resolution diagonal is positive).

        More precisely: K_slor = diag(1/sqrt(diag(K@L))) @ K, which means
        (K_slor @ L)[i,i] = (K@L)[i,i] / sqrt((K@L)[i,i]) = sqrt((K@L)[i,i]).
        So diag(K_slor @ L)^2 = diag(K_mne @ L).
        """
        L = _random_leadfield(15, 40, seed=3)
        L.shape[0]
        alpha = 0.05

        K_mne = _mne_kernel(L, alpha)
        R_diag = np.diag(K_mne @ L)
        # All resolution diagonal elements should be positive for MNE
        assert np.all(R_diag > 0), "MNE resolution diagonal should be positive"

        W_diag = np.sqrt(R_diag)
        K_slor = (K_mne.T / W_diag).T

        R_slor_diag = np.diag(K_slor @ L)
        # diag(K_slor @ L)[i] = R_diag[i] / sqrt(R_diag[i]) = sqrt(R_diag[i])
        np.testing.assert_allclose(R_slor_diag**2, R_diag, rtol=1e-10)


# ---------------------------------------------------------------------------
# 4. LORETA smoothness: should be smoother than MNE
# ---------------------------------------------------------------------------


class TestLORETASmoothness:
    def test_smoother_than_mne(self):
        """LORETA solution should have lower Laplacian norm than MNE."""
        n_chans, n_dipoles = 15, 40
        L = _random_leadfield(n_chans, n_dipoles, seed=4)
        alpha = 0.5

        # Build a simple 1-D Laplacian for the dipoles
        Lap = (
            np.diag(np.ones(n_dipoles)) * 2
            - np.diag(np.ones(n_dipoles - 1), 1)
            - np.diag(np.ones(n_dipoles - 1), -1)
        )

        # MNE kernel
        K_mne = _mne_kernel(L, alpha)

        # LORETA kernel: inv(L^T L + alpha Lap^T Lap) @ L^T
        LTL = L.T @ L
        BLapTLapB = Lap.T @ Lap
        K_lor = np.linalg.inv(LTL + alpha * BLapTLapB) @ L.T

        # Generate test data
        rng = np.random.RandomState(5)
        y = rng.randn(n_chans, 1)

        s_mne = K_mne @ y
        s_lor = K_lor @ y

        lap_norm_mne = np.linalg.norm(Lap @ s_mne)
        lap_norm_lor = np.linalg.norm(Lap @ s_lor)

        assert lap_norm_lor < lap_norm_mne, (
            f"LORETA Laplacian norm ({lap_norm_lor:.4f}) should be less than "
            f"MNE ({lap_norm_mne:.4f})"
        )


# ---------------------------------------------------------------------------
# 5. Beamformer unit-gain constraint
# ---------------------------------------------------------------------------


class TestBeamformerUnitGain:
    def test_lcmv_unit_gain(self):
        """LCMV weight for source i applied to leadfield col i should yield 1."""
        n_chans, n_dipoles = 20, 50
        L = _random_leadfield(n_chans, n_dipoles, seed=6)

        # Normalize leadfield columns (as the actual solver does)
        L_normed = L / np.linalg.norm(L, axis=0, keepdims=True)

        # Simulate data covariance
        rng = np.random.RandomState(7)
        data = rng.randn(n_chans, 200)
        data -= data.mean(axis=1, keepdims=True)
        C = data @ data.T
        alpha = 0.1
        C_inv = np.linalg.inv(C + alpha * np.eye(n_chans))

        # LCMV weights
        upper = C_inv @ L_normed
        lower = np.einsum("ij,jk,ki->i", L_normed.T, C_inv, L_normed)
        W = upper / lower  # (n_chans, n_dipoles)

        # Unit-gain check: w_i^T @ l_i = 1 for each dipole i
        gains = np.einsum("ji,ji->i", W, L_normed)
        np.testing.assert_allclose(gains, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 6. Noise-free sparse recovery (OMP-style)
# ---------------------------------------------------------------------------


class TestSparseRecovery:
    def test_omp_noiseless(self):
        """With noiseless data and known sparse source, greedy recovery should
        identify correct support."""
        n_chans, n_dipoles = 30, 100
        L = _random_leadfield(n_chans, n_dipoles, seed=8)
        # Normalize columns
        L = L / np.linalg.norm(L, axis=0, keepdims=True)

        rng = np.random.RandomState(9)
        true_support = np.array([10, 42, 77])
        x_true = np.zeros(n_dipoles)
        x_true[true_support] = rng.randn(len(true_support)) * 5

        y = L @ x_true  # noiseless

        # Simple OMP implementation
        r = y.copy()
        omega = []
        for _ in range(len(true_support)):
            corr = np.abs(L.T @ r)
            idx = np.argmax(corr)
            omega.append(idx)
            L_sel = L[:, omega]
            x_hat = np.linalg.lstsq(L_sel, y, rcond=None)[0]
            r = y - L_sel @ x_hat

        assert set(omega) == set(true_support), (
            f"OMP found {sorted(omega)}, expected {sorted(true_support)}"
        )


# ---------------------------------------------------------------------------
# 7. Regularization monotonicity
# ---------------------------------------------------------------------------


class TestRegularizationMonotonicity:
    def test_increasing_alpha_decreases_norm(self):
        """Increasing alpha should decrease the solution norm."""
        L = _random_leadfield(15, 40, seed=10)
        rng = np.random.RandomState(11)
        y = rng.randn(15, 1)

        alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
        norms = []
        for alpha in alphas:
            K = _mne_kernel(L, alpha)
            s = K @ y
            norms.append(np.linalg.norm(s))

        for i in range(len(norms) - 1):
            assert norms[i] > norms[i + 1], (
                f"Norm at alpha={alphas[i]} ({norms[i]:.4f}) should be > "
                f"norm at alpha={alphas[i + 1]} ({norms[i + 1]:.4f})"
            )


# ---------------------------------------------------------------------------
# 8. Symmetry
# ---------------------------------------------------------------------------


class TestSymmetry:
    def test_symmetric_input_symmetric_output(self):
        """A symmetric leadfield with symmetric input should give symmetric output."""
        n = 10
        rng = np.random.RandomState(12)
        A = rng.randn(n, n)
        L = A + A.T  # symmetric square leadfield
        alpha = 0.1

        y = np.ones((n, 1))
        K = _mne_kernel(L, alpha)
        K @ y

        # Check that source estimate is also symmetric in some sense:
        # since L is symmetric and y is constant, K @ y should be related
        # to the row sums of K, which inherit L's symmetry
        # Specifically: K = L (LL + aI)^{-1}, and L symmetric => K symmetric
        np.testing.assert_allclose(K, K.T, atol=1e-10)


# ---------------------------------------------------------------------------
# 9. Zero input -> zero (or near-zero) output
# ---------------------------------------------------------------------------


class TestZeroInput:
    def test_zero_data_zero_source(self):
        """Zero data should produce zero source estimate."""
        L = _random_leadfield(15, 40, seed=13)
        alpha = 0.1
        K = _mne_kernel(L, alpha)
        y = np.zeros((15, 1))
        s = K @ y
        np.testing.assert_allclose(s, 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# 10. Thresholding function
# ---------------------------------------------------------------------------


class TestThresholding:
    def test_keeps_k_largest(self):
        """thresholding should keep exactly k non-zero entries."""
        from invert.util.util import thresholding

        x = np.array([1, -5, 3, -2, 4])
        result = thresholding(x, 3)
        assert np.count_nonzero(result) == 3
        # The 3 largest magnitudes are 5, 4, 3
        assert result[1] == -5
        assert result[4] == 4
        assert result[2] == 3

    def test_k_zero_returns_zeros(self):
        from invert.util.util import thresholding

        x = np.array([1.0, 2.0, 3.0])
        result = thresholding(x, 0)
        np.testing.assert_array_equal(result, 0.0)

    def test_k_geq_len_returns_copy(self):
        from invert.util.util import thresholding

        x = np.array([1.0, -2.0, 3.0])
        result = thresholding(x, 5)
        np.testing.assert_array_equal(result, x)

    def test_preserves_signs(self):
        from invert.util.util import thresholding

        x = np.array([-10, 5, -3, 1])
        result = thresholding(x, 2)
        assert result[0] == -10
        assert result[1] == 5

    def test_list_input(self):
        from invert.util.util import thresholding

        result = thresholding([3, 1, 2], 1)
        assert result[0] == 3
        assert result[1] == 0
        assert result[2] == 0


# ---------------------------------------------------------------------------
# 11. Residual variance
# ---------------------------------------------------------------------------


class TestResidualVariance:
    def test_perfect_reconstruction(self):
        """If M_hat == M, residual variance should be 0."""
        from invert.util.util import calc_residual_variance

        M = np.random.RandomState(20).randn(5, 10)
        assert calc_residual_variance(M, M) == pytest.approx(0.0)

    def test_zero_estimate(self):
        """If M_hat is zero, residual variance should be 100%."""
        from invert.util.util import calc_residual_variance

        M = np.random.RandomState(21).randn(5, 10)
        M_hat = np.zeros_like(M)
        assert calc_residual_variance(M_hat, M) == pytest.approx(100.0)

    def test_non_negative(self):
        from invert.util.util import calc_residual_variance

        rng = np.random.RandomState(22)
        M = rng.randn(5, 10)
        M_hat = rng.randn(5, 10)
        assert calc_residual_variance(M_hat, M) >= 0


# ---------------------------------------------------------------------------
# 12. Triangle area (Heron's formula)
# ---------------------------------------------------------------------------


class TestTriangleArea:
    def test_known_triangle(self):
        """3-4-5 right triangle has area 6."""
        from invert.util.util import calc_area_tri

        assert calc_area_tri(3, 4, 5) == pytest.approx(6.0)

    def test_equilateral(self):
        from invert.util.util import calc_area_tri

        area = calc_area_tri(2, 2, 2)
        assert area == pytest.approx(np.sqrt(3), rel=1e-10)

    def test_degenerate_triangle(self):
        """Collinear points give zero area."""
        from invert.util.util import calc_area_tri

        assert calc_area_tri(1, 2, 3) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 13. Euclidean distance
# ---------------------------------------------------------------------------


class TestEuclideanDistance:
    def test_same_point(self):
        from invert.util.util import euclidean_distance

        A = np.array([1, 2, 3])
        assert euclidean_distance(A, A) == pytest.approx(0.0)

    def test_known_distance(self):
        from invert.util.util import euclidean_distance

        A = np.array([0, 0])
        B = np.array([3, 4])
        assert euclidean_distance(A, B) == pytest.approx(5.0)

    def test_symmetry(self):
        from invert.util.util import euclidean_distance

        rng = np.random.RandomState(30)
        A, B = rng.randn(3), rng.randn(3)
        assert euclidean_distance(A, B) == pytest.approx(euclidean_distance(B, A))


# ---------------------------------------------------------------------------
# 14. L-Curve corner finding
# ---------------------------------------------------------------------------


class TestFindCorner:
    def test_obvious_corner(self):
        """An L-shaped curve should find the corner near the elbow."""
        from invert.util.util import find_corner

        # Construct an L-shape: source_power decreases, residual increases
        source_power = np.array([10, 9, 8, 5, 1.0, 0.9, 0.8, 0.7])
        residual = np.array([0.1, 0.2, 0.3, 0.5, 5.0, 6.0, 7.0, 8.0])
        idx = find_corner(source_power, residual)
        # Corner should be around index 3 or 4
        assert 2 <= idx <= 5

    def test_two_points(self):
        """With fewer than 3 points, return last index."""
        from invert.util.util import find_corner

        idx = find_corner(np.array([1, 2]), np.array([3, 4]))
        assert idx == 1


# ---------------------------------------------------------------------------
# 15. MNE kernel is a left-inverse in the limit
# ---------------------------------------------------------------------------


class TestMNELeftInverse:
    def test_recovery_of_data(self):
        """K @ L @ x should approximate x when alpha is small and n_chans >= n_dipoles."""
        n = 15
        L = _random_leadfield(n, n, seed=40)
        # Well-conditioned square L
        L = L + 0.5 * np.eye(n)
        alpha = 1e-10
        K = _mne_kernel(L, alpha)
        rng = np.random.RandomState(41)
        x = rng.randn(n, 1)
        np.testing.assert_allclose(K @ L @ x, x, atol=1e-3)


# ---------------------------------------------------------------------------
# 16. SVD truncation preserves energy
# ---------------------------------------------------------------------------


class TestSVDTruncation:
    def test_energy_fraction(self):
        """Truncated SVD should capture expected fraction of energy."""
        rng = np.random.RandomState(50)
        M = rng.randn(20, 100)
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        np.sum(S**2)
        for k in [1, 5, 10]:
            partial_energy = np.sum(S[:k] ** 2)
            M_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
            recon_energy = np.linalg.norm(M_k, "fro") ** 2
            np.testing.assert_allclose(recon_energy, partial_energy, rtol=1e-10)


# ---------------------------------------------------------------------------
# 17. Tikhonov solution equivalence (dual form)
# ---------------------------------------------------------------------------


class TestTikhonovDualForm:
    def test_primal_dual_equivalence(self):
        """MNE kernel via L^T(LL^T+aI)^{-1} should equal (L^TL+aI)^{-1}L^T."""
        L = _random_leadfield(15, 40, seed=60)
        alpha = 0.5
        n_chans, n_dipoles = L.shape

        # Primal form: (L^T L + alpha I)^{-1} L^T
        K_primal = np.linalg.inv(L.T @ L + alpha * np.eye(n_dipoles)) @ L.T

        # Dual form: L^T (L L^T + alpha I)^{-1}
        K_dual = _mne_kernel(L, alpha)

        np.testing.assert_allclose(K_primal, K_dual, atol=1e-10)


# ---------------------------------------------------------------------------
# 18. Resolution matrix properties
# ---------------------------------------------------------------------------


class TestResolutionMatrix:
    def test_trace_equals_effective_dof(self):
        """trace(K @ L) should equal sum of eigenvalues s_i^2/(s_i^2+alpha)."""
        L = _random_leadfield(15, 40, seed=70)
        alpha = 0.5
        K = _mne_kernel(L, alpha)
        R = K @ L

        _, s, _ = np.linalg.svd(L, full_matrices=False)
        expected_trace = np.sum(s**2 / (s**2 + alpha))
        np.testing.assert_allclose(np.trace(R), expected_trace, rtol=1e-8)

    def test_resolution_eigenvalues_bounded(self):
        """All eigenvalues of the resolution matrix should be in [0, 1]."""
        L = _random_leadfield(15, 40, seed=71)
        alpha = 0.1
        K = _mne_kernel(L, alpha)
        R = K @ L
        np.linalg.eigvalsh(R @ R.T)  # R is not symmetric, use R R^T
        # Singular values of R should be <= 1
        _, sv, _ = np.linalg.svd(R, full_matrices=False)
        assert np.all(sv <= 1.0 + 1e-10)


# ---------------------------------------------------------------------------
# 19. Linearity of inverse operator
# ---------------------------------------------------------------------------


class TestLinearity:
    def test_superposition(self):
        """K @ (a*y1 + b*y2) == a*K@y1 + b*K@y2."""
        L = _random_leadfield(15, 40, seed=80)
        alpha = 0.3
        K = _mne_kernel(L, alpha)
        rng = np.random.RandomState(81)
        y1 = rng.randn(15, 1)
        y2 = rng.randn(15, 1)
        a, b = 2.5, -1.3
        lhs = K @ (a * y1 + b * y2)
        rhs = a * (K @ y1) + b * (K @ y2)
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)


# ---------------------------------------------------------------------------
# 20. Depth weighting normalization
# ---------------------------------------------------------------------------


class TestDepthWeighting:
    def test_columns_unit_norm_at_degree_one(self):
        """With degree=1, depth_weight_fixed should produce unit-norm columns."""
        from invert.solvers.base import BaseSolver

        rng = np.random.RandomState(90)
        L = rng.randn(20, 50)
        L_dw, _ = BaseSolver.depth_weight_fixed(L, degree=1.0)
        col_norms = np.linalg.norm(L_dw, axis=0)
        np.testing.assert_allclose(col_norms, 1.0, atol=1e-10)

    def test_degree_zero_is_identity(self):
        """With degree=0, depth weighting should not change L."""
        from invert.solvers.base import BaseSolver

        rng = np.random.RandomState(91)
        L = rng.randn(20, 50)
        L_dw, _ = BaseSolver.depth_weight_fixed(L, degree=0.0)
        # degree=0 => norms = ||col||^0 = 1, so L_dw = L / 1 = L
        np.testing.assert_allclose(L_dw, L, atol=1e-12)


# ---------------------------------------------------------------------------
# 21. MNE kernel data-fit residual decreases with decreasing alpha
# ---------------------------------------------------------------------------


class TestResidualMonotonicity:
    def test_decreasing_alpha_decreases_residual(self):
        """Decreasing alpha should decrease data-fit residual ||L K y - y||."""
        L = _random_leadfield(15, 40, seed=100)
        rng = np.random.RandomState(101)
        y = rng.randn(15, 1)

        alphas = [10.0, 1.0, 0.1, 0.01, 0.001]
        residuals = []
        for alpha in alphas:
            K = _mne_kernel(L, alpha)
            residual = np.linalg.norm(L @ K @ y - y)
            residuals.append(residual)

        for i in range(len(residuals) - 1):
            assert residuals[i] >= residuals[i + 1] - 1e-10, (
                f"Residual at alpha={alphas[i]} ({residuals[i]:.6f}) should be >= "
                f"residual at alpha={alphas[i + 1]} ({residuals[i + 1]:.6f})"
            )


# ---------------------------------------------------------------------------
# 22. Woodbury identity for MNE
# ---------------------------------------------------------------------------


class TestWoodburyIdentity:
    def test_woodbury_mne(self):
        """Verify Woodbury identity: (L^TL+aI)^{-1} = 1/a(I - L^T(LL^T+aI)^{-1}L)."""
        L = _random_leadfield(15, 40, seed=110)
        alpha = 0.5
        n_chans, n_dipoles = L.shape

        lhs = np.linalg.inv(L.T @ L + alpha * np.eye(n_dipoles))
        rhs = (
            np.eye(n_dipoles)
            - L.T @ np.linalg.inv(L @ L.T + alpha * np.eye(n_chans)) @ L
        ) / alpha

        np.testing.assert_allclose(lhs, rhs, atol=1e-10)


# ---------------------------------------------------------------------------
# 23. Data covariance correctness
# ---------------------------------------------------------------------------


class TestDataCovariance:
    """Tests for BaseSolver.data_covariance()."""

    def test_uncentered_matches_gram(self):
        """center=False, ddof=0 should give Y @ Y.T / n_times."""
        from invert.solvers.base import BaseSolver

        rng = np.random.RandomState(0)
        Y = rng.randn(5, 30)
        C = BaseSolver.data_covariance(Y, center=False, ddof=0)
        expected = (Y @ Y.T) / Y.shape[1]
        np.testing.assert_allclose(C, expected, atol=1e-14)

    def test_centered_removes_mean(self):
        """center=True should subtract per-channel mean before computing."""
        from invert.solvers.base import BaseSolver

        rng = np.random.RandomState(1)
        Y = rng.randn(4, 50) + 10.0  # large offset
        C = BaseSolver.data_covariance(Y, center=True, ddof=0)

        Y_c = Y - Y.mean(axis=1, keepdims=True)
        expected = (Y_c @ Y_c.T) / Y.shape[1]
        np.testing.assert_allclose(C, expected, atol=1e-14)

    def test_bessel_correction(self):
        """ddof=1 should divide by (n_times - 1), not n_times."""
        from invert.solvers.base import BaseSolver

        rng = np.random.RandomState(2)
        Y = rng.randn(3, 20)
        C_ddof0 = BaseSolver.data_covariance(Y, center=True, ddof=0)
        C_ddof1 = BaseSolver.data_covariance(Y, center=True, ddof=1)

        # ddof=1 should give a slightly larger covariance
        ratio = np.trace(C_ddof1) / np.trace(C_ddof0)
        expected_ratio = Y.shape[1] / (Y.shape[1] - 1)
        np.testing.assert_allclose(ratio, expected_ratio, rtol=1e-10)

    def test_single_timepoint_does_not_divide_by_zero(self):
        """n_times=1, ddof=1 should clamp denominator to 1."""
        from invert.solvers.base import BaseSolver

        Y = np.array([[1.0], [2.0], [3.0]])
        C = BaseSolver.data_covariance(Y, center=False, ddof=1)
        assert np.all(np.isfinite(C))
        # With denom clamped to 1, should be Y@Y.T / 1
        np.testing.assert_allclose(C, Y @ Y.T, atol=1e-15)

    def test_rank_deficient_input(self):
        """When n_times < n_chans, output covariance should be rank-deficient."""
        from invert.solvers.base import BaseSolver

        rng = np.random.RandomState(3)
        n_chans, n_times = 10, 3
        Y = rng.randn(n_chans, n_times)
        C = BaseSolver.data_covariance(Y, center=False, ddof=0)

        # Rank should be at most n_times
        s = np.linalg.svd(C, compute_uv=False)
        effective_rank = np.sum(s > s[0] * 1e-10)
        assert effective_rank <= n_times

    def test_output_is_psd(self):
        """Covariance matrix should be positive semi-definite."""
        from invert.solvers.base import BaseSolver

        rng = np.random.RandomState(4)
        Y = rng.randn(8, 100)
        C = BaseSolver.data_covariance(Y, center=True, ddof=1)

        eigvals = np.linalg.eigvalsh(C)
        assert np.all(eigvals >= -1e-12), (
            f"Covariance has negative eigenvalue: {eigvals.min():.2e}"
        )

    def test_output_is_symmetric(self):
        """Covariance should be symmetric."""
        from invert.solvers.base import BaseSolver

        rng = np.random.RandomState(5)
        Y = rng.randn(6, 40)
        C = BaseSolver.data_covariance(Y, center=True, ddof=1)
        np.testing.assert_allclose(C, C.T, atol=1e-15)

    def test_1d_input(self):
        """1D input should be treated as single-timepoint column."""
        from invert.solvers.base import BaseSolver

        y = np.array([1.0, 2.0, 3.0])
        C = BaseSolver.data_covariance(y, center=False, ddof=0)
        expected = np.outer(y, y)
        np.testing.assert_allclose(C, expected, atol=1e-15)


# ---------------------------------------------------------------------------
# 24. Whitener orthogonality
# ---------------------------------------------------------------------------


class TestWhitenerOrthogonality:
    """Verify W @ C_n @ W.T ≈ I in the retained subspace."""

    def test_whitened_noise_is_identity(self):
        """After whitening, noise covariance in retained subspace should be I."""
        from invert.solvers.base import BaseSolver

        rng = np.random.RandomState(10)
        n = 8
        # Random PD noise covariance
        A = rng.randn(n, n)
        C_n = A @ A.T + 0.1 * np.eye(n)
        C_n = 0.5 * (C_n + C_n.T)

        W = BaseSolver.compute_sensor_whitener(C_n, rank_tol=1e-12, eps=1e-15)
        whitened = W @ C_n @ W.T
        np.testing.assert_allclose(whitened, np.eye(W.shape[0]), atol=1e-10)

    def test_rank_truncation_preserves_subspace(self):
        """Whitener should drop zero-eigenvalue dimensions."""
        from invert.solvers.base import BaseSolver

        # Covariance with rank 3 out of 5
        V = np.eye(5)
        eigvals = np.array([5.0, 2.0, 1.0, 0.0, 0.0])
        C_n = V @ np.diag(eigvals) @ V.T

        W = BaseSolver.compute_sensor_whitener(C_n, rank_tol=1e-12, eps=1e-15)
        assert W.shape == (3, 5), f"Expected (3, 5), got {W.shape}"
        whitened = W @ C_n @ W.T
        np.testing.assert_allclose(whitened, np.eye(3), atol=1e-10)

    def test_whitener_at_meg_scale(self):
        """Whitener should work correctly at MEG scale (~1e-26 eigenvalues)."""
        from invert.solvers.base import BaseSolver

        rng = np.random.RandomState(20)
        n = 6
        V = np.linalg.qr(rng.randn(n, n))[0]
        # MEG-scale eigenvalues
        eigvals = np.array([1e-23, 5e-24, 1e-24, 5e-25, 1e-25, 1e-26])
        C_n = V @ np.diag(eigvals) @ V.T
        C_n = 0.5 * (C_n + C_n.T)

        W = BaseSolver.compute_sensor_whitener(C_n, rank_tol=1e-6, eps=1e-15)
        whitened = W @ C_n @ W.T
        r = W.shape[0]
        np.testing.assert_allclose(whitened, np.eye(r), atol=1e-6)

    def test_identity_cov_is_noop(self):
        """Whitening with identity noise cov should preserve dimensions."""
        from invert.solvers.base import BaseSolver

        C_n = np.eye(5)
        W = BaseSolver.compute_sensor_whitener(C_n, rank_tol=1e-12, eps=1e-15)
        assert W.shape == (5, 5)
        np.testing.assert_allclose(W @ W.T, np.eye(5), atol=1e-12)


# ---------------------------------------------------------------------------
# 25. GCV / L-curve edge cases
# ---------------------------------------------------------------------------


class TestGCVEdgeCases:
    """Edge case tests for regularization selection methods."""

    @staticmethod
    def _build_solver_with_operators(L, alphas):
        """Helper: build a BaseSolver with MNE-style operators at given alphas."""
        from invert.solvers.base import BaseSolver, InverseOperator

        m = L.shape[0]
        I = np.eye(m)
        ops = [
            InverseOperator(L.T @ np.linalg.inv(L @ L.T + a * I), "test")
            for a in alphas
        ]
        solver = BaseSolver(n_reg_params=len(alphas))
        solver.leadfield = L
        solver.inverse_operators = ops
        solver.alphas = list(alphas)
        return solver

    def test_gcv_all_inf_falls_back_to_midpoint(self):
        """When all GCV values are inf, should fall back to middle index."""
        from invert.solvers.base import BaseSolver, InverseOperator

        # Create operators where trace(H) >= n for all, making DOF <= 0
        n = 3
        solver = BaseSolver(n_reg_params=3)
        solver.leadfield = np.eye(n)
        # Operators that make trace(A) = n (identity), so DOF = n - gamma*n = 0
        solver.inverse_operators = [
            InverseOperator(np.eye(n), "test") for _ in range(3)
        ]
        solver.alphas = [1e-3, 1e-1, 1e1]

        rng = np.random.RandomState(0)
        M = rng.randn(n, 10)
        _, idx = solver.regularise_gcv(M, gamma=1.0)
        # Should fall back to midpoint (index 1 for 3 operators)
        assert idx == 1

    def test_gcv_single_operator(self):
        """Single operator should return index 0 without error."""
        rng = np.random.RandomState(0)
        L = rng.randn(8, 20)
        solver = self._build_solver_with_operators(L, [0.1])
        M = rng.randn(8, 15)
        _, idx = solver.regularise_gcv(M, gamma=1.0)
        assert idx == 0

    def test_gcv_selects_finite_over_inf(self):
        """If some GCV values are inf and some finite, should pick finite."""
        from invert.solvers.base import BaseSolver, InverseOperator

        n = 5
        rng = np.random.RandomState(42)
        L = rng.randn(n, 10)
        I = np.eye(n)

        # First operator: near-zero alpha -> trace(H) ≈ n -> DOF ≈ 0 -> inf
        # Last operator: large alpha -> low trace -> finite GCV
        alphas = [1e-15, 0.1, 10.0]
        ops = [
            InverseOperator(L.T @ np.linalg.inv(L @ L.T + a * I), "test")
            for a in alphas
        ]
        solver = BaseSolver(n_reg_params=3)
        solver.leadfield = L
        solver.inverse_operators = ops
        solver.alphas = alphas

        M = rng.randn(n, 20)
        _, idx = solver.regularise_gcv(M, gamma=1.0)
        # Should NOT pick index 0 (which gives inf)
        assert idx > 0


class TestLCurveEdgeCases:
    def test_two_points_returns_last(self):
        """With fewer than 3 points, find_corner should return last index."""
        from invert.util import find_corner

        idx = find_corner(np.array([1.0, 2.0]), np.array([2.0, 1.0]))
        assert idx == 1

    def test_collinear_points(self):
        """Collinear points (no corner) should still return a valid index."""
        from invert.util import find_corner

        # Points on a straight line
        source_power = np.linspace(10, 1, 10)
        residual = np.linspace(1, 10, 10)
        idx = find_corner(source_power, residual)
        assert 0 <= idx < len(source_power)

    def test_monotonic_curve(self):
        """Monotonically decreasing curve should not crash."""
        from invert.util import find_corner

        source_power = np.logspace(2, -2, 20)
        residual = np.logspace(-2, 2, 20)
        idx = find_corner(source_power, residual)
        assert 0 <= idx < len(source_power)

    def test_sharp_corner_detected(self):
        """A curve with a clear corner should identify it."""
        from invert.util import find_corner

        # L-shaped curve: flat residual then steep rise
        residual = np.concatenate([np.linspace(1, 1.01, 10), np.linspace(1.1, 10, 10)])
        source_power = np.concatenate([np.linspace(100, 10, 10), np.linspace(9, 1, 10)])
        idx = find_corner(source_power, residual)
        # Corner should be near the transition (around index 9-11)
        assert 5 <= idx <= 15, f"Corner at {idx}, expected near 10"


# ---------------------------------------------------------------------------
# 26. InverseOperator shape validation
# ---------------------------------------------------------------------------


class TestInverseOperatorValidation:
    def test_valid_shape_accepted(self):
        """Correct shape should not raise."""
        from invert.solvers.base import InverseOperator

        mat = np.zeros((50, 20))
        op = InverseOperator(mat, "test", expected_shape=(50, 20))
        assert op.data[0].shape == (50, 20)

    def test_wrong_shape_rejected(self):
        """Wrong shape should raise ValueError."""
        from invert.solvers.base import InverseOperator

        mat = np.zeros((50, 20))
        with pytest.raises(ValueError, match="expected shape"):
            InverseOperator(mat, "test", expected_shape=(40, 20))

    def test_1d_array_rejected(self):
        """1D array should raise ValueError when expected_shape given."""
        from invert.solvers.base import InverseOperator

        mat = np.zeros(50)
        with pytest.raises(ValueError, match="expected 2D"):
            InverseOperator(mat, "test", expected_shape=(50, 1))

    def test_no_expected_shape_accepts_anything(self):
        """Without expected_shape, any shape is accepted (backward compat)."""
        from invert.solvers.base import InverseOperator

        op = InverseOperator(np.zeros((3, 4)), "test")
        assert op.data[0].shape == (3, 4)

        op2 = InverseOperator(np.zeros(10), "test")
        assert op2.data[0].shape == (10,)


# ---------------------------------------------------------------------------
# Evaluation metric properties
# ---------------------------------------------------------------------------


class TestEMDThreshold:
    """Test that EMD threshold parameter works as documented."""

    def test_threshold_zeros_small_values(self):
        """Values below threshold * max should be zeroed before EMD."""
        eval_emd = _eval_mod.eval_emd

        n = 20
        rng = np.random.RandomState(42)
        pos = rng.randn(n, 3)
        M = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)

        # Distribution with one large peak and small background
        v1 = np.zeros(n)
        v1[0] = 1.0
        v1[1:5] = 0.1  # below 0.25 threshold → should be zeroed

        v2 = np.zeros(n)
        v2[0] = 1.0

        # With threshold=0.25, v1's small values are zeroed → v1 ≈ v2 → EMD ≈ 0
        emd_with_thresh = eval_emd(M, v1.copy(), v2.copy(), threshold=0.25)
        # Without threshold, the small mass at indices 1-4 must be transported
        emd_no_thresh = eval_emd(M, v1.copy(), v2.copy(), threshold=0.0)

        assert emd_with_thresh < emd_no_thresh

    def test_threshold_zero_preserves_all(self):
        """threshold=0 should keep all non-zero values."""
        eval_emd = _eval_mod.eval_emd

        n = 10
        M = np.ones((n, n)) - np.eye(n)  # uniform distance
        v1 = np.ones(n) / n
        v2 = np.ones(n) / n

        emd_val = eval_emd(M, v1, v2, threshold=0.0)
        assert np.isfinite(emd_val)
        assert emd_val == pytest.approx(0.0, abs=1e-10)

    def test_identical_distributions_zero_emd(self):
        """EMD between identical distributions should be 0."""
        eval_emd = _eval_mod.eval_emd

        n = 15
        rng = np.random.RandomState(7)
        pos = rng.randn(n, 3)
        M = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
        v = np.abs(rng.randn(n))
        v[v < 0.3 * v.max()] = 0  # ensure some zeros

        emd_val = eval_emd(M, v.copy(), v.copy())
        assert emd_val == pytest.approx(0.0, abs=1e-8)


class TestMLEProperties:
    """Basic mathematical properties of MLE (Mean Localization Error)."""

    @staticmethod
    def _make_simple_problem(n=50, seed=0):
        """Create a simple source space with adjacency."""

        rng = np.random.RandomState(seed)
        pos = rng.randn(n, 3)
        # Simple adjacency: connect nearest neighbors within threshold
        from scipy.spatial.distance import cdist

        D = cdist(pos, pos)
        adj = (D < np.percentile(D[D > 0], 20)).astype(float)
        np.fill_diagonal(adj, 0)
        from scipy.sparse import csr_matrix

        return pos, csr_matrix(adj), D

    def test_perfect_reconstruction_zero_error(self):
        """MLE(y, y) should be 0."""
        eval_mean_localization_error = _eval_mod.eval_mean_localization_error

        pos, adj, D = self._make_simple_problem()
        y = np.zeros(len(pos))
        y[10] = 1.0  # single source

        mle = eval_mean_localization_error(y, y, adj, adj, pos, pos, D, mode="dle")
        assert mle == pytest.approx(0.0, abs=1e-10)

    def test_mle_nonnegative(self):
        """MLE should always be >= 0."""
        eval_mean_localization_error = _eval_mod.eval_mean_localization_error

        pos, adj, D = self._make_simple_problem()
        y_true = np.zeros(len(pos))
        y_true[5] = 1.0
        y_pred = np.zeros(len(pos))
        y_pred[20] = 1.0

        mle = eval_mean_localization_error(
            y_true, y_pred, adj, adj, pos, pos, D, mode="dle"
        )
        assert np.isfinite(mle)
        assert mle >= 0

    def test_dle_is_symmetric(self):
        """DLE mode should give the same error regardless of argument order."""
        eval_mean_localization_error = _eval_mod.eval_mean_localization_error

        pos, adj, D = self._make_simple_problem()
        y1 = np.zeros(len(pos))
        y1[5] = 1.0
        y2 = np.zeros(len(pos))
        y2[20] = 1.0

        mle_fwd = eval_mean_localization_error(
            y1, y2, adj, adj, pos, pos, D, mode="dle"
        )
        mle_rev = eval_mean_localization_error(
            y2, y1, adj, adj, pos, pos, D, mode="dle"
        )
        assert mle_fwd == pytest.approx(mle_rev, abs=1e-10)


class TestCorrProperties:
    """Basic properties of per-timepoint correlation."""

    def test_perfect_correlation(self):
        """corr(y, y) should be 1 for all timepoints."""
        corr = _eval_mod.corr

        rng = np.random.RandomState(42)
        y = rng.randn(50, 10)
        r = corr(y, y)
        np.testing.assert_allclose(r, 1.0, atol=1e-10)

    def test_scaled_correlation(self):
        """corr(y, 2*y) should be 1 (Pearson is scale-invariant)."""
        corr = _eval_mod.corr

        rng = np.random.RandomState(42)
        y = rng.randn(50, 10)
        r = corr(y, 2 * y)
        np.testing.assert_allclose(r, 1.0, atol=1e-10)

    def test_nan_input_returns_nan(self):
        """NaN in input should return NaN."""
        corr = _eval_mod.corr

        y = np.ones((10, 5))
        y_nan = y.copy()
        y_nan[0, 0] = np.nan
        assert np.isnan(corr(y, y_nan))
