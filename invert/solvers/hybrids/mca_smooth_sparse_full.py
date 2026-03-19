from __future__ import annotations

import mne
import numpy as np
from scipy.sparse import csc_matrix, identity
from scipy.sparse import diags
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh, splu

from invert.util import build_source_adjacency

from ..base import BaseSolver, InverseOperator, SolverMeta

# Cache Laplacian low-frequency eigenmodes per process to amortize runs.
_GFT_CACHE: dict[tuple[int, float, int, int, float], tuple[np.ndarray, np.ndarray]] = {}


class SolverMCASmoothSparseFull(BaseSolver):
    """MCA-style solver: diffusion background + FLEX patch residual (full output).

    This is designed for mixtures like ``mixed_cortical_activity`` and dynamic
    patch sources like ``nmm`` where *temporal correlation* is penalized if the
    solver returns only a sparse support estimate. We therefore explicitly
    estimate a smooth background component and a patch component and return
    their sum.

    The algorithm is an alternating scheme:
    1) Diffusion background estimate using the same form as the simulator:
       x_bg = S u with S = (I - αL)^p, u iid. This yields a prior covariance
       Σ_bg ∝ S S^T and recovers smooth 1/f background without overly spreading
       patch mass (important for AP/EMD/MLE thresholding).
    2) Patch estimate on the sensor residual using FLEX-GreedyML's patch
       dictionary and greedy ML support search.
    """

    meta = SolverMeta(
        acronym="MCA-Full",
        full_name="MCA Smooth Background + Connected Sparse Patches (Full)",
        category="Hybrids",
        description=(
            "Alternating decomposition into a graph-smooth background component "
            "and a FLEX-GreedyML patch component on the residual; returns the full source "
            "reconstruction (background + patches) to improve temporal correlation "
            "on mixture and NMM-style datasets."
        ),
        references=[
            "Starck, J.-L., Elad, M., & Donoho, D. (2005). Image decomposition via the combination of sparse representations and a variational approach. IEEE TIP.",
            "Lukas Hecker (2026). Unpublished.",
        ],
    )

    def __init__(self, name: str = "MCA Smooth+Sparse (Full)", **kwargs):
        self.name = str(name)
        super().__init__(**kwargs)
        self.require_recompute = True
        self.require_data = True
        self._source_mat: np.ndarray | None = None

    @staticmethod
    def _build_laplacian(forward, *, adjacency_distance: float) -> csc_matrix:
        adjacency = build_source_adjacency(
            forward["src"],
            adjacency_type="spatial",
            adjacency_distance=float(adjacency_distance),
            verbose=0,
        )
        L = laplacian(adjacency).astype(float)
        diag = np.asarray(L.diagonal(), dtype=float)
        scale = float(np.mean(diag)) if diag.size else 1.0
        if np.isfinite(scale) and scale > 0:
            L = L / scale
        return csc_matrix(L)

    @staticmethod
    def _smooth_bg_estimate_laplacian_tikhonov(
        G_w: np.ndarray,
        Y_w: np.ndarray,
        *,
        L: csc_matrix,
        lam: float,
        ridge: float,
        jitter: float = 1e-9,
    ) -> np.ndarray:
        """Compute graph-smooth background MAP estimate for all time points."""
        n_eff, n_src = G_w.shape
        lam = float(max(lam, 0.0))
        ridge = float(max(ridge, 1e-12))

        M = lam * L + float(ridge) * identity(n_src, format="csc", dtype=float)
        lu = splu(M)

        # X = M^{-1} G^T  (n_src x n_eff)
        X = lu.solve(G_w.T)
        K = G_w @ X  # (n_eff x n_eff)
        A = np.eye(n_eff, dtype=float) + K
        A.flat[:: n_eff + 1] += float(jitter)

        Z = np.linalg.solve(A, Y_w)  # (n_eff x T)
        y_bg = X @ Z  # (n_src x T)
        return np.asarray(y_bg, dtype=float)

    @staticmethod
    def _diffusion_operator(
        *,
        L: csc_matrix,
        diffusion_parameter: float,
        order: int,
    ):
        order = int(max(order, 0))
        alpha = float(diffusion_parameter)
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError(f"diffusion_parameter must be finite and >= 0, got {alpha}")
        n = int(L.shape[0])
        I = identity(n, format="csc", dtype=float)
        if order == 0 or alpha == 0:
            return I
        S = I - alpha * L
        return S**order

    @staticmethod
    def _bic_score(*, rss: float, df: float, n_obs: int, eps: float = 1e-20) -> float:
        rss = float(rss)
        df = float(df)
        n_obs = int(max(n_obs, 1))
        if not np.isfinite(rss) or rss < 0:
            return float("inf")
        if not np.isfinite(df) or df < 0:
            df = 0.0
        return float(n_obs * np.log(rss / float(n_obs) + eps) + df * np.log(float(n_obs)))

    @staticmethod
    def _diffusion_bg_estimate_bic(
        G_w: np.ndarray,
        Y_w: np.ndarray,
        *,
        S: csc_matrix,
        lam_grid: np.ndarray,
        bic_improvement_min: float = 0.0,
        min_effective_support_frac: float = 0.0,
    ) -> tuple[np.ndarray, float, float, float]:
        """Estimate background x_bg = S u by ridge on u, selecting lambda by BIC.

        Returns
        -------
        x_hat : (n_src, T)
        best_lam : float
        best_bic : float
        null_bic : float
        """
        n_eff, n_src = G_w.shape
        T = int(Y_w.shape[1])
        if T <= 0:
            return np.zeros((n_src, 0), dtype=float), float("nan"), float("inf"), float("inf")

        # Sensor dictionary for the latent u: y ≈ (G S) u
        GS = (S.T @ G_w.T).T  # (n_eff, n_src)
        I = np.eye(n_eff, dtype=float)

        null_rss = float(np.sum(Y_w * Y_w))
        null_bic = SolverMCASmoothSparseFull._bic_score(
            rss=null_rss, df=0.0, n_obs=int(n_eff * T)
        )

        best_bic = float("inf")
        best_x = np.zeros((n_src, T), dtype=float)
        best_lam = float("nan")

        for lam in np.asarray(lam_grid, dtype=float).reshape(-1):
            if not np.isfinite(lam) or lam <= 0:
                continue
            C = GS @ GS.T + float(lam) * I  # (n_eff, n_eff)
            C.flat[:: n_eff + 1] += 1e-12 * float(np.trace(C)) / max(1, n_eff)
            try:
                CinvY = np.linalg.solve(C, Y_w)  # (n_eff, T)
            except np.linalg.LinAlgError:
                continue

            u_hat = GS.T @ CinvY  # (n_src, T)
            x_hat = S @ u_hat  # (n_src, T)
            fit = G_w @ x_hat  # (n_eff, T)
            resid = Y_w - fit
            rss = float(np.sum(resid * resid))

            # Effective degrees of freedom of the ridge fit in sensor space.
            # This ridge problem is solved independently for each timepoint, so
            # the total df must scale with T.
            CinvGS = np.linalg.solve(C, GS)  # (n_eff, n_src)
            df_per_tp = float(np.sum(GS * CinvGS))
            df = df_per_tp * float(T)
            bic = SolverMCASmoothSparseFull._bic_score(
                rss=rss, df=df, n_obs=int(n_eff * T)
            )
            if bic < best_bic:
                best_bic = bic
                best_x = np.asarray(x_hat, dtype=float)
                best_lam = float(lam)

        # Accept background only if it improves BIC over the null (x_bg=0).
        min_improve = float(max(bic_improvement_min, 0.0))
        if not (best_bic + min_improve < null_bic):
            return np.zeros((n_src, T), dtype=float), float("nan"), best_bic, null_bic

        # Reject if the estimated background is too focal (it would likely
        # harm patch-only datasets by adding false positives).
        min_frac = float(max(min_effective_support_frac, 0.0))
        if min_frac > 0.0:
            a = np.mean(np.abs(best_x), axis=1)
            denom = float(np.sum(a * a) + 1e-30)
            eff = float((np.sum(a) ** 2) / denom)  # effective number of vertices
            eff_frac = eff / float(max(n_src, 1))
            if not (eff_frac >= min_frac):
                return (
                    np.zeros((n_src, T), dtype=float),
                    float("nan"),
                    best_bic,
                    null_bic,
                )

        return best_x, best_lam, best_bic, null_bic

    @staticmethod
    def _gft_lowrank_bg_estimate_bic(
        G_w: np.ndarray,
        Y_w: np.ndarray,
        *,
        L: csc_matrix,
        adjacency_distance: float,
        basis_dim: int,
        diffusion_parameter: float,
        diffusion_order: int,
        vertex_mask: np.ndarray | None = None,
        lam_grid: np.ndarray,
        bic_improvement_min: float = 0.0,
        min_effective_support_frac: float = 0.0,
    ) -> tuple[np.ndarray, float, float, float]:
        """Low-frequency (graph Fourier) background model fit by ridge + BIC.

        This constrains the background to a low-dimensional smooth subspace so it
        cannot explain localized patch residuals (critical for ``extended_source``).
        """
        n_eff, n_src = G_w.shape
        T = int(Y_w.shape[1])
        if T <= 0:
            return np.zeros((n_src, 0), dtype=float), float("nan"), float("inf"), float("inf")

        k = int(max(0, basis_dim))
        if k <= 0:
            null_rss = float(np.sum(Y_w * Y_w))
            null_bic = SolverMCASmoothSparseFull._bic_score(
                rss=null_rss, df=0.0, n_obs=int(n_eff * T)
            )
            return np.zeros((n_src, T), dtype=float), float("nan"), float("inf"), null_bic
        k = int(min(k, max(1, n_src - 1)))  # eigsh requires k < n

        # Cache signature: adjacency_distance alone is not sufficient to uniquely
        # identify a Laplacian across different forwards with the same n_src.
        # Include a cheap numeric fingerprint to reduce accidental collisions.
        L_abs_sum = float(np.sum(np.abs(L.data))) if getattr(L, "data", None) is not None else 0.0
        cache_key = (
            int(n_src),
            float(adjacency_distance),
            int(k),
            int(getattr(L, "nnz", 0)),
            float(np.round(L_abs_sum, 6)),
        )
        cached = _GFT_CACHE.get(cache_key)
        if cached is None:
            # Smallest eigenvalues correspond to smoothest eigenvectors.
            vals, vecs = eigsh(L, k=k, which="SM")
            vals = np.asarray(vals, dtype=float)
            vecs = np.asarray(vecs, dtype=float)
            _GFT_CACHE[cache_key] = (vals, vecs)
        else:
            vals, vecs = cached

        # Apply diffusion scaling in the eigenbasis: (I - a L)^p v_i = (1 - a*λ_i)^p v_i
        a = float(diffusion_parameter)
        p = int(max(diffusion_order, 0))
        scale = np.ones_like(vals, dtype=float)
        if p > 0 and a != 0.0:
            scale = np.power(1.0 - a * vals, p, dtype=float)
        B = vecs * scale.reshape(1, -1)  # (n_src, k)
        if vertex_mask is not None:
            vm = np.asarray(vertex_mask, dtype=float).reshape(-1)
            if vm.shape[0] != n_src:
                raise ValueError(
                    f"vertex_mask must have shape ({n_src},), got {vm.shape}"
                )
            B = B * vm.reshape(-1, 1)

        A = G_w @ B  # (n_eff, k)
        AtA = A.T @ A  # (k, k)
        AtY = A.T @ Y_w  # (k, T)
        null_rss = float(np.sum(Y_w * Y_w))
        null_bic = SolverMCASmoothSparseFull._bic_score(
            rss=null_rss, df=0.0, n_obs=int(n_eff * T)
        )

        best_bic = float("inf")
        best_lam = float("nan")
        best_x = np.zeros((n_src, T), dtype=float)

        for lam in np.asarray(lam_grid, dtype=float).reshape(-1):
            if not np.isfinite(lam) or lam <= 0:
                continue
            M = AtA.copy()
            M.flat[:: k + 1] += float(lam)
            try:
                C = np.linalg.solve(M, AtY)  # (k, T)
            except np.linalg.LinAlgError:
                continue
            fit = A @ C
            resid = Y_w - fit
            rss = float(np.sum(resid * resid))

            # df_per_tp = tr(AtA (AtA + lam I)^{-1})
            try:
                X = np.linalg.solve(M, AtA)
            except np.linalg.LinAlgError:
                continue
            df_per_tp = float(np.trace(X))
            df = df_per_tp * float(T)
            bic = SolverMCASmoothSparseFull._bic_score(
                rss=rss, df=df, n_obs=int(n_eff * T)
            )
            if bic < best_bic:
                best_bic = bic
                best_lam = float(lam)
                best_x = B @ C

        min_improve = float(max(bic_improvement_min, 0.0))
        if not (best_bic + min_improve < null_bic):
            return np.zeros((n_src, T), dtype=float), float("nan"), best_bic, null_bic

        min_frac = float(max(min_effective_support_frac, 0.0))
        if min_frac > 0.0:
            a_map = np.mean(np.abs(best_x), axis=1)
            denom = float(np.sum(a_map * a_map) + 1e-30)
            eff = float((np.sum(a_map) ** 2) / denom)
            eff_frac = eff / float(max(n_src, 1))
            if not (eff_frac >= min_frac):
                return np.zeros((n_src, T), dtype=float), float("nan"), best_bic, null_bic

        return np.asarray(best_x, dtype=float), best_lam, best_bic, null_bic

    @staticmethod
    def _flex_greedy_patch_estimate_refit(
        G_w: np.ndarray,
        Y_res: np.ndarray,
        forward,
        *,
        k_max: int,
        n_starts: int,
        n_refine_iters: int,
        penalty_mode: str,
        n_orders: int,
        diffusion_parameter: float,
        adjacency_distance: float,
        refit_ridge: float = 1e-6,
    ) -> np.ndarray:
        """FLEX support selection + timecourse refit in selected patch subspace.

        Compared to the direct inverse construction used in SolverFlexGreedyML, this
        performs an explicit refit of the K latent timecourses for the selected
        spatial maps. This tends to improve temporal correlation (corr) without
        sacrificing AP/EMD/MLE.
        """
        from ..music.flex_greedy_ml import SolverFlexGreedyML
        from ..music.greedy_ml import greedy_ml_search

        n_eff, n_dipoles = G_w.shape
        T = int(Y_res.shape[1])

        leadfields, gradients = SolverFlexGreedyML._build_patch_dictionary(
            G_w,
            forward,
            int(n_orders),
            float(diffusion_parameter),
            float(adjacency_distance),
        )
        n_total_orders = len(leadfields)
        G_ext = np.hstack(leadfields)
        n_ext = int(G_ext.shape[1])

        Gram = G_ext.T @ G_ext
        eps_gram = 1e-10 * np.trace(Gram) / max(n_ext, 1)
        Gram[np.diag_indices_from(Gram)] += eps_gram

        R = G_ext.T @ Y_res
        Q = R @ R.T
        total_var = float(np.sum(Y_res**2))

        def _excluded_fn(selected):
            blocked = set()
            for e in selected:
                dipole = int(e % n_dipoles)
                for o in range(n_total_orders):
                    blocked.add(o * n_dipoles + dipole)
            return blocked

        selected_ext = greedy_ml_search(
            Gram,
            Q,
            total_var,
            int(n_eff),
            int(T),
            int(n_ext),
            int(k_max),
            int(n_starts),
            int(n_refine_iters),
            str(penalty_mode),
            excluded_fn=_excluded_fn,
        )

        if not selected_ext:
            return np.zeros((n_dipoles, T), dtype=float)

        # Spatial maps in source space for each selected atom (n_dipoles x K).
        sel_orders = [int(e // n_dipoles) for e in selected_ext]
        sel_dipoles = [int(e % n_dipoles) for e in selected_ext]
        K = len(selected_ext)

        M_map = np.zeros((n_dipoles, K), dtype=float)
        for i, (o, d) in enumerate(zip(sel_orders, sel_dipoles, strict=False)):
            M_map[:, i] = gradients[o][d].toarray().ravel()

        # Refit latent timecourses Z in the selected K-dimensional subspace:
        #   Y ≈ (G_w M_map) Z
        A = G_w @ M_map  # (n_eff, K)
        AtA = A.T @ A  # (K, K)
        scale = float(np.trace(AtA)) / float(max(K, 1))
        lam = float(max(refit_ridge, 0.0)) * float(max(scale, 1e-12))
        AtA.flat[:: K + 1] += lam
        Z = np.linalg.solve(AtA, A.T @ Y_res)  # (K, T)

        y_patch = M_map @ Z  # (n_dipoles, T)
        return np.asarray(y_patch, dtype=float)

    @staticmethod
    def _flex_greedy_patch_estimate_inverse(
        G_w: np.ndarray,
        Y_res: np.ndarray,
        forward,
        *,
        k_max: int,
        n_starts: int,
        n_refine_iters: int,
        penalty_mode: str,
        n_orders: int,
        diffusion_parameter: float,
        adjacency_distance: float,
    ) -> np.ndarray:
        """Patch estimate using FLEX-GreedyML's exact inverse-operator construction.

        Important: this mirrors ``SolverFlexGreedyML``'s core logic, but runs on a
        provided residual (already whitened) rather than building a persistent
        inverse operator from the full data.
        """
        from ..music.flex_greedy_ml import SolverFlexGreedyML
        from ..music.greedy_ml import greedy_ml_search

        n_eff, n_dipoles = G_w.shape
        T = int(Y_res.shape[1])

        leadfields, gradients = SolverFlexGreedyML._build_patch_dictionary(
            G_w,
            forward,
            int(n_orders),
            float(diffusion_parameter),
            float(adjacency_distance),
        )
        n_total_orders = len(leadfields)
        G_ext = np.hstack(leadfields)
        n_ext = int(G_ext.shape[1])

        Gram = G_ext.T @ G_ext
        eps_gram = 1e-10 * np.trace(Gram) / max(n_ext, 1)
        Gram[np.diag_indices_from(Gram)] += eps_gram

        R = G_ext.T @ Y_res
        Q = R @ R.T
        total_var = float(np.sum(Y_res**2))

        def _excluded_fn(selected):
            blocked = set()
            for e in selected:
                dipole = int(e % n_dipoles)
                for o in range(n_total_orders):
                    blocked.add(o * n_dipoles + dipole)
            return blocked

        selected_ext = greedy_ml_search(
            Gram,
            Q,
            total_var,
            int(n_eff),
            int(T),
            int(n_ext),
            int(k_max),
            int(n_starts),
            int(n_refine_iters),
            str(penalty_mode),
            excluded_fn=_excluded_fn,
        )

        if not selected_ext:
            return np.zeros((n_dipoles, T), dtype=float)

        selected_ext_arr = np.array(selected_ext, dtype=np.intp)
        G_sel = G_ext[:, selected_ext_arr]

        # ML inverse operator in the selected extended subspace.
        inv_ext = np.linalg.lstsq(G_sel, np.eye(n_eff), rcond=None)[0]  # (K, n_eff)

        sel_orders = [int(e // n_dipoles) for e in selected_ext]
        sel_dipoles = [int(e % n_dipoles) for e in selected_ext]
        M = np.zeros((n_dipoles, len(selected_ext)), dtype=float)
        for i, (o, d) in enumerate(zip(sel_orders, sel_dipoles, strict=False)):
            M[:, i] = gradients[o][d].toarray().ravel()

        inverse_operator_w = M @ inv_ext  # (n_dipoles, n_eff)
        y_patch = inverse_operator_w @ Y_res  # (n_dipoles, T)
        return np.asarray(y_patch, dtype=float)

    def make_inverse_operator(
        self,
        forward,
        mne_obj=None,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
        # Alternation
        n_mca_iters: int = 1,
        auto_bg: bool = True,
        mixture_variance_threshold: float = 1e5,
        # Background
        bg_mode: str = "gft_lowrank",  # "gft_lowrank" | "diffusion" | "laplacian_tikhonov" | "none"
        bg_adjacency_distance: float = 3e-3,
        bg_diffusion_parameter: float = 0.1,
        bg_diffusion_order: int = 1,
        bg_lambda_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
        bg_bic_improvement_min: float = 25.0,
        bg_min_effective_support_frac: float = 0.02,
        bg_exclude_patch_mass_frac: float = 0.9,
        bg_basis_dim: int = 64,
        # (Legacy) Laplacian-Tikhonov params (only used when bg_mode="laplacian_tikhonov")
        bg_lambda: float = 10.0,
        bg_ridge: float = 1.0,
        # FLEX patch estimation on residual
        k_max: int = 5,
        n_starts: int = 50,
        n_refine_iters: int = 3,
        penalty_mode: str = "bic_per_timepoint",
        n_orders: int = 3,
        diffusion_parameter: float = 0.1,
        patch_adjacency_distance: float = 3e-3,
        patch_refit_ridge: float = 1e-6,
        # Optional post-smoothing of the patch estimate (selected by BIC on sensors).
        patch_post_diffusion_parameter: float = 0.15,
        patch_post_smoothing_orders: tuple[int, ...] = (0, 1, 2),
        # Return mode
        return_components: str = "sum",  # "sum" | "bg" | "patch"
        **kwargs,
    ):
        super().make_inverse_operator(forward, mne_obj, *args, alpha=alpha, **kwargs)
        wf = self.prepare_whitened_forward(noise_cov)

        if int(getattr(self, "_n_orient", 1)) != 1:
            raise ValueError(
                f"{self.name}: free/vector orientation is not supported. "
                "Use orientation='fixed' or orientation='pca'."
            )

        Y = self.unpack_data_obj(mne_obj)
        # Background is modeled explicitly; BIC comparison decides whether to include it.
        mode = str(bg_mode).lower()
        use_bg = mode != "none"
        if bool(auto_bg) and use_bg:
            try:
                mean_var = float(np.mean(np.var(Y, axis=1, ddof=1)))
            except Exception:
                mean_var = float("nan")
            if not (np.isfinite(mean_var) and mean_var > 0):
                use_bg = False

        Y_w = wf.sensor_transform @ Y  # (n_eff, T)
        G_w = wf.G_white  # (n_eff, n_src)
        n_eff, n_src = G_w.shape
        if n_eff <= 1 or n_src <= 1:
            raise ValueError(f"{self.name}: invalid shapes G_w={G_w.shape}")

        L = self._build_laplacian(
            forward, adjacency_distance=float(bg_adjacency_distance)
        )

        y_bg = np.zeros((n_src, int(Y_w.shape[1])), dtype=float)
        y_patch = np.zeros_like(y_bg)

        n_mca = int(max(1, n_mca_iters))
        for _ in range(n_mca):
            # Patch given current background.
            Y_res = Y_w - (G_w @ y_bg)
            y_patch = self._flex_greedy_patch_estimate_inverse(
                G_w,
                Y_res,
                forward,
                k_max=int(k_max),
                n_starts=int(n_starts),
                n_refine_iters=int(n_refine_iters),
                penalty_mode=str(penalty_mode),
                n_orders=int(n_orders),
                diffusion_parameter=float(diffusion_parameter),
                adjacency_distance=float(patch_adjacency_distance),
            )

            # Background given current patch (fit only residual).
            if use_bg:
                Y_bg_fit = Y_w - (G_w @ y_patch)
                if mode == "laplacian_tikhonov":
                    y_bg = self._smooth_bg_estimate_laplacian_tikhonov(
                        G_w,
                        Y_bg_fit,
                        L=L,
                        lam=float(bg_lambda),
                        ridge=float(bg_ridge),
                    )
                elif mode in {"diffusion", "gft_lowrank"}:
                    lam_grid = np.asarray(bg_lambda_grid, dtype=float)

                    # Exclude patch-dominated vertices from the background model.
                    vertex_mask = None
                    mass_frac = float(bg_exclude_patch_mass_frac)
                    if np.isfinite(mass_frac) and 0.0 < mass_frac < 1.0:
                        a_patch = np.mean(np.abs(y_patch), axis=1)
                        total = float(np.sum(a_patch))
                        if np.isfinite(total) and total > 0:
                            order = np.argsort(a_patch)[::-1]
                            cum = np.cumsum(a_patch[order])
                            cutoff = mass_frac * total
                            k_excl = int(np.searchsorted(cum, cutoff, side="left") + 1)
                            k_excl = int(np.clip(k_excl, 0, max(0, n_src - 1)))
                            if k_excl > 0:
                                vertex_mask = np.ones(n_src, dtype=float)
                                vertex_mask[order[:k_excl]] = 0.0

                    if mode == "diffusion":
                        S_bg = self._diffusion_operator(
                            L=L,
                            diffusion_parameter=float(bg_diffusion_parameter),
                            order=int(bg_diffusion_order),
                        )
                        if vertex_mask is not None:
                            S_bg = diags(vertex_mask, format="csc", dtype=float) @ S_bg
                        y_bg, _lam, _bic, _bic0 = self._diffusion_bg_estimate_bic(
                            G_w,
                            Y_bg_fit,
                            S=S_bg,
                            lam_grid=lam_grid,
                            bic_improvement_min=float(bg_bic_improvement_min),
                            min_effective_support_frac=float(bg_min_effective_support_frac),
                        )
                    else:
                        y_bg, _lam, _bic, _bic0 = self._gft_lowrank_bg_estimate_bic(
                            G_w,
                            Y_bg_fit,
                            L=L,
                            adjacency_distance=float(bg_adjacency_distance),
                            basis_dim=int(bg_basis_dim),
                            diffusion_parameter=float(bg_diffusion_parameter),
                            diffusion_order=int(bg_diffusion_order),
                            vertex_mask=vertex_mask,
                            lam_grid=lam_grid,
                            bic_improvement_min=float(bg_bic_improvement_min),
                            min_effective_support_frac=float(bg_min_effective_support_frac),
                        )
                else:
                    raise ValueError(
                        f"{self.name}: bg_mode must be 'gft_lowrank', 'diffusion', "
                        f"'laplacian_tikhonov', or 'none', got {bg_mode!r}"
                    )
            else:
                y_bg.fill(0.0)

            # Optional post-smoothing selection for the patch component.
            # This is particularly helpful for NMM/extended patches: it trades a bit
            # of focality for better patch extent, improving EMD/MLE and sometimes corr.
            orders = tuple(int(o) for o in patch_post_smoothing_orders)
            orders = tuple(o for o in orders if o >= 0)
            if any(o != 0 for o in orders):
                Y_res_post = Y_w - (G_w @ y_bg)
                S_post = self._diffusion_operator(
                    L=L,
                    diffusion_parameter=float(patch_post_diffusion_parameter),
                    order=1,
                )
                best_bic = float("inf")
                best_patch = y_patch
                n_obs = int(n_eff * int(Y_w.shape[1]))
                for o in orders:
                    if o == 0:
                        patch_o = y_patch
                    else:
                        patch_o = np.asarray((S_post**int(o)) @ y_patch, dtype=float)
                    resid = Y_res_post - (G_w @ patch_o)
                    rss = float(np.sum(resid * resid))
                    # Approx df by (support size) * T; timecourses are free per vertex.
                    a = np.mean(np.abs(patch_o), axis=1)
                    thr = 1e-3 * float(np.max(a)) if a.size else 0.0
                    support = int(np.sum(a > max(thr, 1e-12)))
                    df = float(max(1, support)) * float(max(1, int(Y_w.shape[1])))
                    bic = self._bic_score(rss=rss, df=df, n_obs=n_obs)
                    if bic < best_bic:
                        best_bic = bic
                        best_patch = patch_o
                y_patch = np.asarray(best_patch, dtype=float)

        mode = str(return_components).lower()
        if mode not in {"sum", "bg", "patch"}:
            raise ValueError(
                f"{self.name}: return_components must be one of "
                f"{{'sum','bg','patch'}}, got {return_components!r}"
            )
        if mode == "bg":
            self._source_mat = np.asarray(y_bg, dtype=float)
        elif mode == "patch":
            self._source_mat = np.asarray(y_patch, dtype=float)
        else:
            self._source_mat = np.asarray(y_bg + y_patch, dtype=float)

        # Dummy operator for API consistency (apply override returns stored output).
        inv = np.zeros((n_src, n_eff), dtype=float) @ wf.sensor_transform
        self.inverse_operators = [InverseOperator(inv, self.name)]
        return self

    def apply_inverse_operator(self, mne_obj) -> mne.SourceEstimate:  # type: ignore[override]
        if self._source_mat is None:
            return super().apply_inverse_operator(mne_obj)
        return self.source_to_object(self._source_mat)
