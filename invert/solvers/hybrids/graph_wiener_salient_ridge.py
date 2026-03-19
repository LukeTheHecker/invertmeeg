from __future__ import annotations

import logging

import mne
import numpy as np
from scipy.sparse.csgraph import laplacian

from invert.simulate import SimulationConfig, SimulationGenerator
from invert.util import (
    build_source_adjacency,
    normalize_saliency,
    salient_center_target,
)

from ..base import SolverMeta
from .graph_wiener_smooth_sparse_adaptive_robust import (
    SolverGraphWienerSmoothSparseAdaptiveRobustEB,
)

logger = logging.getLogger(__name__)


class SolverGraphWienerSalientRidge(SolverGraphWienerSmoothSparseAdaptiveRobustEB):
    """Salient-region refinement on top of GW-SSAR.

    The base `GW-SSAR` estimate is used as a physics-informed feature map. A
    small ridge model is trained on simulated data to predict dominant source
    centers from that saliency pattern. The predicted center scores are then
    diffused over the cortical graph and used as a soft gate on the base source
    estimate, which suppresses diffuse background activity while preserving the
    main patches.
    """

    meta = SolverMeta(
        acronym="GW-SR",
        full_name="Graph Wiener Salient Ridge",
        category="Minimum Norm",
        description=(
            "GW-SSAR with a simulation-trained salient-region gate that focuses "
            "reconstruction on dominant cortical patches and downweights diffuse background."
        ),
        internal=True,
    )

    def __init__(
        self,
        name: str = "Graph Wiener Salient Ridge",
        *,
        ridge_alpha: float = 10.0,
        diffusion_tau: float = 2.0,
        gate_floor: float = 0.0,
        gate_exponent: float = 1.0,
        train_samples: int = 24,
        dominant_ratio: float = 0.6,
        cumulative_mass: float = 0.8,
        **kwargs,
    ):
        self.ridge_alpha = float(ridge_alpha)
        self.diffusion_tau = float(diffusion_tau)
        self.gate_floor = float(gate_floor)
        self.gate_exponent = float(gate_exponent)
        self.train_samples = int(train_samples)
        self.dominant_ratio = float(dominant_ratio)
        self.cumulative_mass = float(cumulative_mass)
        self._ridge_coef: np.ndarray | None = None
        self._ridge_x_mean: np.ndarray | None = None
        self._ridge_y_mean: np.ndarray | None = None
        self._heat_evecs: np.ndarray | None = None
        self._heat_weights: np.ndarray | None = None
        self._gate_ready = False
        super().__init__(name=name, **kwargs)

    def _make_training_base_solver(self) -> SolverGraphWienerSmoothSparseAdaptiveRobustEB:
        return SolverGraphWienerSmoothSparseAdaptiveRobustEB(
            taus=self.taus,
            n_basis=self.n_basis,
            sensor_cov_shrinkage_min=self.sensor_cov_shrinkage_min,
            sensor_cov_shrinkage_max=self.sensor_cov_shrinkage_max,
            sensor_cov_shrinkage_snr_ref=self.sensor_cov_shrinkage_snr_ref,
            w_ridge_min=self.w_ridge_min,
            w_ridge_max=self.w_ridge_max,
            w_ridge_snr_ref=self.w_ridge_snr_ref,
            w_sum_floor=self.w_sum_floor,
            time_whiten_cov=self.time_whiten_cov,
            time_cov_shrinkage=self.time_cov_shrinkage,
            time_cov_eps=self.time_cov_eps,
            n_outer_iters=self.n_outer_iters,
            n_gamma_iters=self.n_gamma_iters,
            gamma_damping=self.gamma_damping,
            gamma_floor=self.gamma_floor,
            learn_noise_scale=self.learn_noise_scale,
            noise_scale_floor=self.noise_scale_floor,
            noise_scale_damping=self.noise_scale_damping,
            gamma_prior_a=self.gamma_prior_a,
            gamma_prior_b=self.gamma_prior_b,
        )

    @staticmethod
    def _fit_dual_ridge(
        X: np.ndarray,
        Y: np.ndarray,
        *,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must be 2D arrays.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"X/Y sample mismatch: {X.shape[0]} vs {Y.shape[0]}"
            )

        x_mean = X.mean(axis=0)
        y_mean = Y.mean(axis=0)
        Xc = X - x_mean[np.newaxis, :]
        Yc = Y - y_mean[np.newaxis, :]

        gram = Xc @ Xc.T
        gram.flat[:: gram.shape[0] + 1] += float(alpha)
        dual = np.linalg.solve(gram, Yc)
        coef = Xc.T @ dual
        return coef, x_mean, y_mean

    @staticmethod
    def _make_noise_covariance(
        cov: np.ndarray | None,
        info: mne.Info,
    ) -> mne.Covariance | None:
        if cov is None:
            return None
        cov = np.asarray(cov, dtype=float)
        ch_names = list(info["ch_names"])
        if cov.shape != (len(ch_names), len(ch_names)):
            return None
        return mne.Covariance(
            data=cov,
            names=ch_names,
            bads=list(info.get("bads", [])),
            projs=list(info.get("projs", [])),
            nfree=1,
        )

    @staticmethod
    def _default_train_config(
        *,
        n_timepoints: int,
        train_samples: int,
        random_seed: int | None,
    ) -> SimulationConfig:
        return SimulationConfig(
            batch_size=int(train_samples),
            n_sources=(1, 4),
            n_orders=0,
            snr_range=(-3.0, 4.0),
            n_timepoints=int(n_timepoints),
            source_spatial_model="contiguous_gaussian",
            patch_rank=(1, 2),
            simulation_mode="mixture",
            background_beta=(1.0, 2.5),
            background_mixture_alpha=(0.4, 0.7),
            amplitude_range=(0.2, 2.0),
            amplitude_distribution="log_uniform",
            correlation_mode="realistic",
            noise_temporal_beta=(0.3, 1.2),
            noise_low_rank_dim=(3, 10),
            source_extent=(10, 80),
            estimate_noise_cov=True,
            return_noise_cov=True,
            random_seed=random_seed,
        )

    def _coerce_train_config(
        self,
        salient_train_config,
        *,
        n_timepoints: int,
    ) -> SimulationConfig:
        if salient_train_config is None:
            return self._default_train_config(
                n_timepoints=n_timepoints,
                train_samples=self.train_samples,
                random_seed=None,
            )
        if isinstance(salient_train_config, SimulationConfig):
            return salient_train_config.model_copy(
                update={
                    "batch_size": int(self.train_samples),
                    "n_timepoints": int(n_timepoints),
                    "estimate_noise_cov": True,
                    "return_noise_cov": True,
                }
            )
        if isinstance(salient_train_config, dict):
            cfg = dict(salient_train_config)
            cfg.update(
                {
                    "batch_size": int(self.train_samples),
                    "n_timepoints": int(n_timepoints),
                    "estimate_noise_cov": True,
                    "return_noise_cov": True,
                }
            )
            return SimulationConfig(**cfg)
        raise TypeError(
            "salient_train_config must be a SimulationConfig, dict, or None."
        )

    def _prepare_heat_kernel(self, forward) -> None:
        adjacency = build_source_adjacency(forward["src"], verbose=0)
        graph_laplacian = laplacian(adjacency, normed=False).astype(float).toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(graph_laplacian)
        positive = eigenvalues[eigenvalues > 1e-12]
        eig_scale = float(np.median(positive)) if positive.size else 1.0
        lam_scaled = eigenvalues / float(max(eig_scale, 1e-12))
        self._heat_evecs = np.asarray(eigenvectors, dtype=float)
        self._heat_weights = np.exp(-float(self.diffusion_tau) * lam_scaled)

    def _ensure_gate_model(
        self,
        forward,
        *,
        info: mne.Info,
        salient_train_config,
        n_timepoints: int,
    ) -> None:
        if self._gate_ready:
            return

        train_config = self._coerce_train_config(
            salient_train_config,
            n_timepoints=n_timepoints,
        )
        train_gen = SimulationGenerator(forward, config=train_config)
        train_batch = next(train_gen.generate())
        x_train, sim_info = train_batch.x, train_batch.info
        info = info.copy()

        features: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for i in range(int(train_config.batch_size)):
            evoked = mne.EvokedArray(x_train[i], info, tmin=0.0, verbose=0)
            noise_cov = self._make_noise_covariance(
                sim_info.iloc[i].get("noise_cov_est_scaled"),
                info,
            )
            base_solver = self._make_training_base_solver()
            base_solver.make_inverse_operator(
                forward,
                evoked,
                alpha="auto",
                noise_cov=noise_cov,
            )
            stc = base_solver.apply_inverse_operator(evoked)
            saliency = normalize_saliency(np.mean(np.abs(stc.data), axis=1), mode="max")
            target = salient_center_target(
                sim_info.iloc[i],
                saliency.shape[0],
                amplitude_ratio=self.dominant_ratio,
                cumulative_mass=self.cumulative_mass,
            )
            if np.any(target > 0):
                features.append(saliency)
                targets.append(target)

        if not features:
            logger.warning("Salient gate training produced no valid targets; falling back to GW-SSAR.")
            self._gate_ready = False
            return

        X = np.asarray(features, dtype=float)
        Y = np.asarray(targets, dtype=float)
        coef, x_mean, y_mean = self._fit_dual_ridge(X, Y, alpha=self.ridge_alpha)
        self._ridge_coef = coef
        self._ridge_x_mean = x_mean
        self._ridge_y_mean = y_mean
        self._prepare_heat_kernel(forward)
        self._gate_ready = True

    def _predict_gate(self, base_source_mat: np.ndarray) -> np.ndarray:
        if (
            not self._gate_ready
            or self._ridge_coef is None
            or self._ridge_x_mean is None
            or self._ridge_y_mean is None
            or self._heat_evecs is None
            or self._heat_weights is None
        ):
            return np.ones(int(base_source_mat.shape[0]), dtype=float)

        saliency = normalize_saliency(
            np.mean(np.abs(base_source_mat), axis=1),
            mode="max",
        )
        center_scores = (
            (saliency - self._ridge_x_mean) @ self._ridge_coef + self._ridge_y_mean
        )
        center_scores = np.clip(center_scores, a_min=0.0, a_max=None)
        center_scores = normalize_saliency(center_scores, mode="sum")
        if not np.any(center_scores > 0):
            return np.ones_like(saliency)

        gate = self._heat_evecs @ (self._heat_weights * (self._heat_evecs.T @ center_scores))
        gate = normalize_saliency(np.clip(gate, a_min=0.0, a_max=None), mode="max")
        gate = float(self.gate_floor) + (1.0 - float(self.gate_floor)) * gate
        return np.power(gate, float(self.gate_exponent))

    def make_inverse_operator(
        self,
        forward,
        mne_obj=None,
        *args,
        alpha="auto",
        salient_train_config: SimulationConfig | dict | None = None,
        noise_cov: mne.Covariance | None = None,
        verbose=0,
        **kwargs,
    ):
        super().make_inverse_operator(
            forward,
            mne_obj,
            *args,
            alpha=alpha,
            noise_cov=noise_cov,
            verbose=verbose,
            **kwargs,
        )
        n_timepoints = int(self.unpack_data_obj(mne_obj).shape[1]) if mne_obj is not None else 256
        self._ensure_gate_model(
            forward,
            info=mne_obj.info,
            salient_train_config=salient_train_config,
            n_timepoints=n_timepoints,
        )
        return self

    def apply_inverse_operator(self, mne_obj) -> mne.SourceEstimate:
        data = self.unpack_data_obj(mne_obj)
        self.validate_operator_data_compatibility(data)
        base_source_mat, _idx = self._compute_source_mat(data)
        gate = self._predict_gate(base_source_mat)
        stc = self.source_to_object(base_source_mat * gate[:, np.newaxis])
        return stc
