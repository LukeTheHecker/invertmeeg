from __future__ import annotations

import mne
import numpy as np
from scipy.sparse import identity as sparse_identity
from scipy.sparse import kron as sparse_kron
from scipy.sparse.csgraph import laplacian

from invert.util import build_source_adjacency

from ..base import BaseSolver, SolverMeta
from .flex_greedy_ml import SolverFlexGreedyML


class SolverFlexGreedyMLDiffuse(BaseSolver):
    """FLEX-GreedyML followed by graph diffusion smoothing.

    FLEX-GreedyML localizes peaks very well on the NMM benchmark but tends to
    reconstruct patches that are too focal. This wrapper keeps the same sparse
    subspace search and then diffuses the recovered source estimate on the
    cortical graph to better match patch extent.
    """

    meta = SolverMeta(
        acronym="FLEX-GreedyML-Diffuse",
        full_name="FLEX Greedy Maximum-Likelihood with Diffusion Smoothing",
        category="Subspace Methods",
        description=(
            "Runs FLEX-GreedyML and then applies calibrated graph diffusion "
            "smoothing to better match spatial patch extent."
        ),
        references=[
            "Wax, M., & Kailath, T. (1985). Detection of signals by information theoretic criteria. IEEE Trans. ASSP, 33(2), 387-392.",
            "Lukas Hecker (2026). Unpublished.",
        ],
    )

    def __init__(
        self,
        name: str = "FLEX-GreedyML-Diffuse",
        *,
        diffusion_parameter: float = 0.15,
        smoothing_order: int = 4,
        adjacency_distance: float = 3e-3,
        flex_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        self.name = name
        self.diffusion_parameter = float(diffusion_parameter)
        self.smoothing_order = int(max(smoothing_order, 0))
        self.adjacency_distance = float(adjacency_distance)
        self._flex_kwargs = dict(flex_kwargs or {})
        self._child_solver: SolverFlexGreedyML | None = None
        self._smoothing_operator = None
        super().__init__(**kwargs)
        self.require_recompute = True
        self.require_data = True

    def _make_child_solver(self) -> SolverFlexGreedyML:
        child = SolverFlexGreedyML(
            verbose=self.verbose,
            regularisation_method=self.regularisation_method,
            n_reg_params=self.n_reg_params,
            prep_leadfield=self.prep_leadfield,
            use_depth_weighting=self.use_depth_weighting,
            use_last_alpha=self.use_last_alpha,
            rank=self.rank,
            depth_weighting=self.depth_weighting,
            reduce_rank=self.reduce_rank,
            plot_reg=self.plot_reg,
            common_average_reference=self.common_average_reference,
            gcv_gamma=self.gcv_gamma,
            mgcv_gamma=self.mgcv_gamma,
            rgcv_gamma=self.rgcv_gamma,
            r1gcv_gamma=self.r1gcv_gamma,
            sensor_scaling=self.sensor_scaling,
            sensor_scaling_eps=self.sensor_scaling_eps,
            orientation=self.orientation,
            orientation_pca_reg=self.orientation_pca_reg,
            orientation_pca_deterministic_sign=self.orientation_pca_deterministic_sign,
            **self._flex_kwargs,
        )
        child.data_covariance = self.data_covariance  # type: ignore[assignment]
        return child

    def _build_smoothing_operator(self, child: SolverFlexGreedyML):
        n_cols = int(child.leadfield.shape[1])
        n_orient = int(getattr(child, "_n_orient", 1))
        if n_orient <= 0:
            n_orient = 1
        if n_cols % n_orient != 0:
            raise ValueError(
                f"{self.name}: leadfield has {n_cols} columns which is not divisible "
                f"by n_orient={n_orient}."
            )
        n_locs = int(n_cols // n_orient)

        adjacency = build_source_adjacency(
            child.forward["src"],
            adjacency_type="spatial",
            adjacency_distance=self.adjacency_distance,
            verbose=0,
        )
        LL = laplacian(adjacency)
        if LL.shape != (n_locs, n_locs):
            raise ValueError(
                f"{self.name}: Laplacian has shape {LL.shape}, expected {(n_locs, n_locs)}."
            )
        if n_orient != 1:
            LL = sparse_kron(
                LL, sparse_identity(n_orient, format="csr"), format="csr"
            )
        I = sparse_identity(LL.shape[0], format="csr", dtype=float)
        if self.smoothing_order == 0:
            return I
        return (I - self.diffusion_parameter * LL) ** self.smoothing_order

    def make_inverse_operator(
        self,
        forward,
        mne_obj=None,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
        **kwargs,
    ):
        child = self._make_child_solver()
        child.make_inverse_operator(
            forward, mne_obj, *args, alpha=alpha, noise_cov=noise_cov, **kwargs
        )
        self._child_solver = child
        self._smoothing_operator = self._build_smoothing_operator(child)
        self.forward = child.forward
        self.leadfield = child.leadfield
        self._n_orient = getattr(child, "_n_orient", 1)
        self.made_inverse_operator = True
        return self

    def apply_inverse_operator(self, mne_obj):  # type: ignore[override]
        if self._child_solver is None or self._smoothing_operator is None:
            raise RuntimeError(
                f"{self.name}: make_inverse_operator must be called before apply."
            )
        stc = self._child_solver.apply_inverse_operator(mne_obj)
        smoothed = np.asarray(self._smoothing_operator @ stc.data, dtype=float)
        return self._child_solver.source_to_object(smoothed)
