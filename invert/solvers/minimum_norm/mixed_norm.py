"""Reweighted and time-frequency mixed norms with reference numerical cores.

TF and scalar coordinate-descent optimizers are delegated to MNE (BSD-3-Clause).
Channel alignment, whitening, orientation handling and SourceEstimate creation
remain in BaseSolver. The dependency on ``mne.inverse_sparse.mxne_optim`` is
intentional: it is the authors' maintained implementation of these algorithms.
See tests/test_mixed_norm_extensions.py for equation and adapter checks.
"""

from __future__ import annotations

import mne
import numpy as np
from mne.inverse_sparse.mxne_optim import (
    _Phi,
    iterative_tf_mixed_norm_solver,
    norm_epsilon_inf,
    tf_mixed_norm_solver,
)
from mne.proj import make_projector

from ..base import BaseSolver, SolverMeta
from .irmxne_optim import iterative_mixed_norm_solver_safe
from .mxne import _compute_depth_weights, _norm_l2inf


class _MixedNormBase(BaseSolver):
    SUPPORTS_VECTOR_ORIENTATION = True

    def __init__(
        self, name, *, max_iter=2000, tol=1e-6, depth=0.8, debias=True, **kwargs
    ):
        self.name = name
        self.max_iter = max_iter
        self.tol = tol
        self.depth = depth
        self.debias = debias
        super().__init__(**kwargs)

    @staticmethod
    def _make_sensor_projector(projs, ch_names, bads):
        return make_projector(projs, ch_names, bads=bads)

    def make_inverse_operator(
        self, forward, *args, alpha=0.2, noise_cov=None, **kwargs
    ):
        """Prepare the forward; alpha is a fraction of the convex zero threshold.

        ``alpha='auto'`` uses the explicit default 0.2, not GCV. The numerical
        problem uses unit-Frobenius data and unit maximum block leadfield norm,
        after optional depth weights. This defines the nonconvex penalty's scale
        unambiguously and makes changing measurement units harmless.
        """
        self.regularization_ = 0.2 if alpha == "auto" else float(alpha)
        self._validate_alpha(self.regularization_)
        super().make_inverse_operator(
            forward, *args, alpha=self.regularization_, **kwargs
        )
        self.prepare_whitened_forward(noise_cov)
        self.inverse_operators = []  # These are data-dependent nonlinear solves.
        return self

    @staticmethod
    def _validate_alpha(alpha):
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be finite and between 0 and 1.")

    def _prepare_problem(self, mne_obj, alpha, max_iter, tol, depth):
        if not self.made_inverse_operator:
            raise RuntimeError("Call make_inverse_operator before applying the solver.")
        alpha = self.regularization_ if alpha is None else float(alpha)
        self._validate_alpha(alpha)
        max_iter = self.max_iter if max_iter is None else max_iter
        tol = self.tol if tol is None else tol
        depth = self.depth if depth is None else depth
        if int(max_iter) != max_iter or max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be positive and finite.")
        if not np.isfinite(depth) or depth < 0:
            raise ValueError("depth must be nonnegative and finite.")
        data = self.unpack_data_obj(mne_obj)
        self.validate_operator_data_compatibility(data)
        M = self._sensor_transform @ data
        G = self.leadfield.copy()
        if not np.isfinite(M).all() or not np.isfinite(G).all():
            raise ValueError("Data and leadfield must be finite.")
        n_orient = int(getattr(self, "_n_orient", 1))
        weights = (
            _compute_depth_weights(G, n_orient, depth) if depth else np.ones(G.shape[1])
        )
        G /= weights
        blocks = G.reshape(G.shape[0], -1, n_orient)
        gain_scale = float(np.linalg.norm(blocks, axis=(0, 2)).max())
        data_scale = float(np.linalg.norm(M))
        self._source_scale = (
            data_scale / max(gain_scale, np.finfo(float).tiny) / weights
        )
        if gain_scale:
            G /= gain_scale
        if data_scale:
            M /= data_scale
        self.last_solver_info = {
            "backend": "mne.inverse_sparse.mxne_optim",
            "mne_version": mne.__version__,
            "alpha_fraction": alpha,
            "data_scale": data_scale,
            "gain_scale": gain_scale,
            "depth": depth,
            "n_orient": n_orient,
        }
        return M, G, alpha, int(max_iter), float(tol), n_orient

    def _finish(self, X, active, history, n_times, *, gap=None, tol=None):
        estimate = np.zeros((self.leadfield.shape[1], n_times))
        estimate[active] = X * self._source_scale[active, None]
        self.source_estimate_ = estimate
        self.active_set_ = np.asarray(active, dtype=bool)
        self.objective_history_ = np.asarray(history, dtype=float)
        self.last_solver_info.update(
            active_positions=int(self.active_set_.sum() // self._n_orient),
            objective_history=self.objective_history_,
            objective_before_debias=True,
        )
        if gap is not None:
            self.last_solver_info.update(
                duality_gap=float(gap), converged=bool(gap <= tol)
            )
        # Reweighted TF-MxNE does not expose inner convergence; do not invent it.
        return self.source_to_object(estimate)

    def apply_inverse_operator_vector(self, mne_obj, **kwargs):
        """Return vector currents using BaseSolver's separate vector-output API."""
        self.apply_inverse_operator(mne_obj, **kwargs)
        return self.source_to_object_vector(self.source_estimate_)

    def _trivial(self, M, G, alpha):
        if not np.any(M) or not np.any(G) or alpha == 1:
            return self._finish(
                np.empty((0, M.shape[1])), np.zeros(G.shape[1], bool), [], M.shape[1]
            )
        if alpha == 0:
            X = np.linalg.lstsq(G, M, rcond=None)[0]
            return self._finish(X, np.ones(G.shape[1], bool), [], M.shape[1])
        return None


class SolverIRMxNE(_MixedNormBase):
    """Iterative reweighted MxNE (Strohmeier et al., 2016).

    Minimize 0.5 ||Y-GX||_F^2 + lambda sum_j ||X_j||_F**0.5 by
    reweighted convex MxNE subproblems. Groups contain time and orientation.
    ``n_mxne_iter=1`` is the convex initialization, not the nonconvex optimum.
    Alpha, objective histories and tolerance refer to the normalized problem
    documented in make_inverse_operator. Convex subproblem gaps and outer
    iterate convergence are reported separately in last_solver_info.
    """

    meta = SolverMeta(
        acronym="irMxNE",
        full_name="Iterative Reweighted Mixed-Norm Estimate",
        category="Sparse / Mixed Norm",
        description="Nonconvex L2,0.5 group sparsity with checked weighted group-Lasso subproblems.",
        references=[
            "Strohmeier et al. (2016). The iterative reweighted Mixed-Norm Estimate for spatio-temporal MEG/EEG source reconstruction. IEEE TMI. doi:10.1109/TMI.2016.2553445."
        ],
    )

    def __init__(self, name="irMxNE", *, n_mxne_iter=50, **kwargs):
        self.n_mxne_iter = n_mxne_iter
        super().__init__(name, **kwargs)

    def apply_inverse_operator(
        self,
        mne_obj,
        alpha=None,
        *,
        n_mxne_iter=None,
        max_iter=None,
        tol=None,
        depth=None,
        debias=None,
    ):
        M, G, alpha, max_iter, tol, n_orient = self._prepare_problem(
            mne_obj, alpha, max_iter, tol, depth
        )
        n_iter = self.n_mxne_iter if n_mxne_iter is None else n_mxne_iter
        if int(n_iter) != n_iter or n_iter < 1:
            raise ValueError("n_mxne_iter must be a positive integer.")
        trivial = self._trivial(M, G, alpha)
        if trivial is not None:
            return trivial
        lambda_max = _norm_l2inf(G.T @ M, n_orient)
        self.last_solver_info["lambda"] = alpha * lambda_max
        X, active, history, diagnostics = iterative_mixed_norm_solver_safe(
            M,
            G,
            alpha * lambda_max,
            int(n_iter),
            maxit=max_iter,
            tol=tol,
            active_set_size=min(10, G.shape[1] // n_orient),
            debias=self.debias if debias is None else debias,
            n_orient=n_orient,
        )
        self.last_solver_info.update(diagnostics)
        self.last_solver_info.update(
            backend="invert MM with MNE scalar CD / project vector BCD",
            converged=diagnostics["outer_converged"]
            and all(diagnostics["inner_converged"]),
        )
        return self._finish(X, active, history, M.shape[1])


class SolverTFMxNE(_MixedNormBase):
    """Time-frequency mixed norms (Gramfort et al., 2013).

    Solve 0.5||Y-G Z Phi||_F^2 + lambda_s ||Z||_21 + lambda_t ||Z||_1
    with orientation-group norms for vector forwards. Window sizes and steps
    are in samples. ``n_tfmxne_iter>1`` uses MNE's nonconvex reweighted TF
    extension. Physical time order is retained; there is no temporal PCA.
    Epochs follow the standard BaseSolver evoked-average semantics.
    """

    meta = SolverMeta(
        acronym="TF-MxNE",
        full_name="Time-Frequency Mixed-Norm Estimate",
        category="Sparse / Mixed Norm",
        description="Spatial and time-frequency sparsity using MNE's reference STFT optimizer.",
        references=[
            "Gramfort et al. (2013). Time-frequency mixed-norm estimates: sparse M/EEG imaging with non-stationary source activations. NeuroImage. doi:10.1016/j.neuroimage.2012.12.051."
        ],
    )

    def __init__(
        self,
        name="TF-MxNE",
        *,
        l1_ratio=0.1,
        wsize=64,
        tstep=4,
        n_tfmxne_iter=1,
        **kwargs,
    ):
        self.l1_ratio = l1_ratio
        self.wsize = wsize
        self.tstep = tstep
        self.n_tfmxne_iter = n_tfmxne_iter
        super().__init__(name, **kwargs)

    def apply_inverse_operator(
        self,
        mne_obj,
        alpha=None,
        *,
        l1_ratio=None,
        wsize=None,
        tstep=None,
        n_tfmxne_iter=None,
        max_iter=None,
        tol=None,
        depth=None,
        debias=None,
    ):
        M, G, alpha, max_iter, tol, n_orient = self._prepare_problem(
            mne_obj, alpha, max_iter, tol, depth
        )
        ratio = self.l1_ratio if l1_ratio is None else l1_ratio
        windows = np.atleast_1d(self.wsize if wsize is None else wsize)
        steps = np.atleast_1d(self.tstep if tstep is None else tstep)
        n_iter = self.n_tfmxne_iter if n_tfmxne_iter is None else n_tfmxne_iter
        if not np.isfinite(ratio) or not 0 <= ratio < 1:
            raise ValueError("l1_ratio must be finite and in [0, 1).")
        if int(n_iter) != n_iter or n_iter < 1:
            raise ValueError("n_tfmxne_iter must be a positive integer.")
        if (
            windows.ndim != 1
            or steps.shape != windows.shape
            or not np.isfinite(windows).all()
            or not np.isfinite(steps).all()
            or np.any(windows < 4)
            or np.any(windows % 4)
            or np.any(steps < 2)
            or np.any(steps % 2)
            or np.any(steps > windows / 2)
            or np.any(windows % steps)
        ):
            raise ValueError(
                "wsize must be multiples of 4; matching even tstep must divide wsize and be <= wsize/2."
            )
        windows, steps = windows.astype(int), steps.astype(int)
        trivial = self._trivial(M, G, alpha)
        if trivial is not None:
            return trivial
        n_coefs = (np.ceil(M.shape[1] / steps).astype(int)) * (windows // 2 + 1)
        phi = _Phi(windows, steps, n_coefs, M.shape[1])
        lambda_max = norm_epsilon_inf(G, M, phi, ratio, n_orient)
        space, time = alpha * lambda_max * (1 - ratio), alpha * lambda_max * ratio
        self.last_solver_info.update(
            lambda_space=space, lambda_time=time, wsize=windows, tstep=steps
        )
        options = dict(
            wsize=windows,
            tstep=steps,
            n_orient=n_orient,
            maxit=max_iter,
            tol=tol,
            debias=self.debias if debias is None else debias,
            verbose=False,
        )
        if n_iter == 1:
            X, active, history, gap = tf_mixed_norm_solver(
                M, G, space, time, return_gap=True, **options
            )
        else:
            X, active, history = iterative_tf_mixed_norm_solver(
                M, G, space, time, int(n_iter), **options
            )
            gap = None
        return self._finish(X, active, history, M.shape[1], gap=gap, tol=tol)
