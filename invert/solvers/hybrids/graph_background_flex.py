from __future__ import annotations

import mne
import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta
from .mca_smooth_sparse_full import SolverMCASmoothSparseFull


class SolverGraphBackgroundFLEX(SolverMCASmoothSparseFull):
    """Background-first graph demixing followed by FLEX patch recovery.

    The key design choice is to estimate a *strictly smooth* nuisance component
    first, then localize the salient patch residual. This matches benchmarks like
    ``mixed_cortical_activity`` where the evaluated target is the patch-only
    component, not the diffuse background.

    Compared to ``SolverMCASmoothSparseFull``:
    - the background is fit before the sparse patch stage
    - only the patch component is returned
    - the background regularization grid excludes near-zero lambdas so the
      low-rank graph model cannot collapse into a patch-like fit
    """

    meta = SolverMeta(
        acronym="GB-FLEX",
        full_name="Graph Background-FLEX",
        category="Hybrids",
        description=(
            "Background-first solver that explains smooth graph-spectral nuisance "
            "activity before applying FLEX-GreedyML to the residual patch component."
        ),
        references=[
            "Starck, J.-L., Elad, M., & Donoho, D. (2005). Image decomposition via the combination of sparse representations and a variational approach. IEEE TIP.",
            "Lukas Hecker (2026). Unpublished.",
        ],
    )

    def __init__(self, name: str = "Graph Background+FLEX", **kwargs):
        self.name = str(name)
        super().__init__(name=name, **kwargs)

    @staticmethod
    def _resolve_basis_dim(n_vertices: int, basis_dim: int | float | None) -> int:
        if basis_dim is None:
            return n_vertices
        if isinstance(basis_dim, float):
            if not 0.0 < basis_dim <= 1.0:
                raise ValueError(
                    f"bg_basis_dim as float must be in (0, 1], got {basis_dim}"
                )
            return max(2, int(np.ceil(basis_dim * n_vertices)))
        basis_dim_int = int(basis_dim)
        if basis_dim_int < 2:
            raise ValueError(f"bg_basis_dim must be >= 2, got {basis_dim_int}")
        return min(n_vertices, basis_dim_int)

    def make_inverse_operator(
        self,
        forward,
        mne_obj=None,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
        bg_adjacency_distance: float = 3e-3,
        bg_basis_dim: int | float | None = 64,
        bg_diffusion_parameter: float = 0.1,
        bg_diffusion_order: int = 1,
        bg_lambda_grid: tuple[float, ...] = (1.0, 10.0, 100.0),
        bg_bic_improvement_min: float = 0.0,
        bg_min_effective_support_frac: float = 0.0,
        k_max: int = 5,
        n_starts: int = 50,
        n_refine_iters: int = 3,
        penalty_mode: str = "bic_per_timepoint",
        n_orders: int = 3,
        diffusion_parameter: float = 0.1,
        patch_adjacency_distance: float = 3e-3,
        patch_refit_timecourses: bool = False,
        patch_refit_ridge: float = 1e-6,
        patch_post_diffusion_parameter: float = 0.15,
        patch_post_smoothing_order: int = 5,
        **kwargs,
    ):
        BaseSolver.make_inverse_operator(
            self, forward, mne_obj, *args, alpha=alpha, **kwargs
        )
        wf = self.prepare_whitened_forward(noise_cov)

        if int(getattr(self, "_n_orient", 1)) != 1:
            raise ValueError(
                f"{self.name}: free/vector orientation is not supported. "
                "Use orientation='fixed' or orientation='pca'."
            )

        Y = self.unpack_data_obj(mne_obj)
        Y_w = wf.sensor_transform @ Y
        G_w = wf.G_white
        n_eff, n_src = G_w.shape
        if n_eff <= 1 or n_src <= 1:
            raise ValueError(f"{self.name}: invalid shapes G_w={G_w.shape}")

        L = self._build_laplacian(
            forward, adjacency_distance=float(bg_adjacency_distance)
        )
        basis_dim = self._resolve_basis_dim(int(n_src), bg_basis_dim)
        y_bg, _lam, _bic, _bic0 = self._gft_lowrank_bg_estimate_bic(
            G_w,
            Y_w,
            L=L,
            adjacency_distance=float(bg_adjacency_distance),
            basis_dim=int(basis_dim),
            diffusion_parameter=float(bg_diffusion_parameter),
            diffusion_order=int(bg_diffusion_order),
            vertex_mask=None,
            lam_grid=np.asarray(bg_lambda_grid, dtype=float),
            bic_improvement_min=float(bg_bic_improvement_min),
            min_effective_support_frac=float(bg_min_effective_support_frac),
        )

        Y_res = Y_w - (G_w @ y_bg)
        if bool(patch_refit_timecourses):
            y_patch = self._flex_greedy_patch_estimate_refit(
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
                refit_ridge=float(patch_refit_ridge),
            )
        else:
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

        smooth_order = int(max(patch_post_smoothing_order, 0))
        if smooth_order > 0:
            S_post = self._diffusion_operator(
                L=L,
                diffusion_parameter=float(patch_post_diffusion_parameter),
                order=1,
            )
            y_patch = np.asarray((S_post**smooth_order) @ y_patch, dtype=float)

        self._source_mat = np.asarray(y_patch, dtype=float)
        inv = np.zeros((n_src, n_eff), dtype=float) @ wf.sensor_transform
        self.inverse_operators = [InverseOperator(inv, self.name)]
        return self

    def apply_inverse_operator(self, mne_obj) -> mne.SourceEstimate:  # type: ignore[override]
        if self._source_mat is None:
            return super().apply_inverse_operator(mne_obj)
        return self.source_to_object(self._source_mat)
