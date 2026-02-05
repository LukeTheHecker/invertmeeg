from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverSESAME(BaseSolver):
    """SESAME-style dipole localization via Sequential Monte Carlo (SMC).

    This is a lightweight, discrete-grid implementation inspired by SESAME:
    it approximates the posterior over a small set of dipole locations using
    sequential importance sampling with resampling. Dipole positions are
    restricted to the vertices in the provided source space.

    Notes
    -----
    - The algorithm is stochastic; use `random_state` for reproducibility.
    - The forward model is converted to fixed orientation by BaseSolver.
    - The output is sparse with `n_dipoles` active vertices.
    """

    meta = SolverMeta(
        acronym="SESAME",
        full_name="Sequential Semi-Analytic Monte Carlo Estimation",
        category="Dipole Fitting",
        description=(
            "SESAME-style sequential Monte Carlo dipole fitting on a discrete source grid."
        ),
        references=[
            "Sommariva, S., & Sorrentino, A. (2014). Sequential Monte Carlo samplers for semi-linear inverse problems and application to magnetoencephalography. Inverse Problems, 30(11), 114020.",
        ],
    )

    def __init__(self, name: str = "SESAME", **kwargs: Any) -> None:
        self.name = name
        self.selected_dipoles: np.ndarray | None = None
        super().__init__(**kwargs)

    @staticmethod
    def _explained_energy_scores(leadfield: np.ndarray, data: np.ndarray) -> np.ndarray:
        norms = np.sum(leadfield * leadfield, axis=0)
        norms = np.where(norms <= 0, 1e-15, norms)
        proj = leadfield.T @ data
        return np.sum(proj * proj, axis=1) / norms

    def make_inverse_operator(  # type: ignore[override]
        self,
        forward,
        mne_obj,
        *args: Any,
        alpha: str | float = "auto",
        n: int | str | None = None,
        n_dipoles: int | str | None = None,
        n_particles: int = 64,
        n_candidates: int = 256,
        resample_threshold: float = 0.5,
        max_dipoles: int = 4,
        min_rel_rss_improvement: float = 0.02,
        noise_var: float | str = "auto",
        random_state: int | None = 0,
        **kwargs: Any,
    ):
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)

        data = self.unpack_data_obj(mne_obj)
        data = data - data.mean(axis=0, keepdims=True)

        leadfield = self.leadfield
        n_chans, n_sources = leadfield.shape

        # Decide number of dipoles to fit.
        force_n = n is not None or n_dipoles is not None
        if n_dipoles is None:
            n_dipoles = n
        if n_dipoles is None or (
            isinstance(n_dipoles, str) and n_dipoles.lower() == "auto"
        ):
            # Conservative default for benchmarks: estimate and cap.
            n_est = int(self.estimate_n_sources(data, method="enhanced"))
            n_dipoles_int = int(np.clip(n_est, 1, int(max_dipoles)))
        else:
            n_dipoles_int = max(int(n_dipoles), 1)
        n_dipoles_int = min(n_dipoles_int, n_sources)

        if isinstance(noise_var, str) and noise_var == "auto":
            # Start with per-entry variance; refined per-iteration using RSS spread.
            noise_var_val = float(np.var(data))
        else:
            noise_var_val = float(noise_var)
        noise_var_val = max(noise_var_val, 1e-15)

        rng = np.random.RandomState(random_state)

        particles: list[np.ndarray] = [
            np.array([], dtype=int) for _ in range(int(n_particles))
        ]
        weights = np.full(len(particles), 1.0 / len(particles), dtype=np.float64)

        residual_best = data
        best_rss = float(np.sum(residual_best * residual_best))
        best_dipoles = np.array([], dtype=int)

        for _k in range(int(n_dipoles_int)):
            scores = self._explained_energy_scores(leadfield, residual_best)
            n_candidates_eff = int(np.clip(n_candidates, 8, n_sources))
            cand = np.argsort(scores)[-n_candidates_eff:]
            cand_scores = np.maximum(scores[cand], 0.0)
            if float(cand_scores.sum()) <= 0:
                cand_scores = np.ones_like(cand_scores)
            cand_p = cand_scores / float(cand_scores.sum())

            new_particles: list[np.ndarray] = []
            residuals: list[np.ndarray] = []
            rss_list = np.zeros(len(particles), dtype=np.float64)

            # Always include the MAP proposal in the particle set for stability.
            best_cand = int(cand[int(np.argmax(scores[cand]))])

            for i in range(len(particles)):
                existing = set(particles[i].tolist())
                if i == 0 and best_cand not in existing:
                    new_idx = best_cand
                else:
                    new_idx = None
                    for _attempt in range(16):
                        proposal = int(rng.choice(cand, p=cand_p))
                        if proposal not in existing:
                            new_idx = proposal
                            break
                if new_idx is None:
                    for proposal in cand[::-1]:
                        if int(proposal) not in existing:
                            new_idx = int(proposal)
                            break
                if new_idx is None:
                    new_idx = int(np.argmax(scores))

                dipoles = np.append(particles[i], new_idx).astype(int)
                A = leadfield[:, dipoles]
                G = A.T @ A
                reg = 1e-12 * np.eye(G.shape[0])
                moments = np.linalg.solve(G + reg, A.T @ data)
                resid = data - A @ moments

                residuals.append(resid)
                new_particles.append(dipoles)

                rss_list[i] = float(np.sum(resid * resid))

            rss_min = float(np.min(rss_list))
            if not force_n:
                # Stop early if the next dipole doesn't improve the fit enough.
                rel_improve = (best_rss - rss_min) / max(best_rss, 1e-15)
                if rel_improve < float(min_rel_rss_improvement):
                    break

            # Convert RSS to stable log-weights (scale-free).
            rss_sorted = np.sort(rss_list)
            rss_med = float(rss_sorted[len(rss_sorted) // 2])
            temp = max(
                rss_med - rss_min, noise_var_val * n_chans * data.shape[1] * 1e-6, 1e-12
            )
            loglikes = -(rss_list - rss_min) / temp

            logw = np.log(weights + 1e-300) + loglikes
            logw -= float(np.max(logw))
            weights = np.exp(logw)
            weights /= float(np.sum(weights))

            particles = new_particles

            ess = 1.0 / float(np.sum(weights * weights))
            if ess < float(resample_threshold) * len(particles):
                idx = rng.choice(
                    np.arange(len(particles)), size=len(particles), p=weights
                )
                particles = [particles[int(j)].copy() for j in idx]
                residuals = [residuals[int(j)].copy() for j in idx]
                weights = np.full(
                    len(particles), 1.0 / len(particles), dtype=np.float64
                )

            best_idx = int(np.argmin(rss_list))
            residual_best = residuals[best_idx]
            best_dipoles = particles[best_idx].astype(int)
            best_rss = float(rss_list[best_idx])

        if best_dipoles.size == 0:
            # Fallback: match ECD (single best dipole)
            scores = self._explained_energy_scores(leadfield, data)
            best_dipoles = np.array([int(np.argmax(scores))], dtype=int)

        A = leadfield[:, best_dipoles]
        G = A.T @ A
        reg = 1e-12 * np.eye(G.shape[0])
        pinv = np.linalg.solve(G + reg, A.T)  # (k, n_chans)

        kernel = np.zeros((n_sources, n_chans), dtype=np.float64)
        for row, src_idx in enumerate(best_dipoles.tolist()):
            kernel[int(src_idx), :] += pinv[row, :]

        self.selected_dipoles = best_dipoles
        self.inverse_operators = [InverseOperator(kernel, self.name)]
        return self
