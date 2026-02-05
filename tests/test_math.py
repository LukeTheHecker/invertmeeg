"""Mathematical correctness tests for inverse solvers.

These tests verify that the solvers satisfy known mathematical properties,
independent of MNE infrastructure where possible.
"""

import numpy as np
import pytest

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
