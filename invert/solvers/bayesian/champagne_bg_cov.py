import mne
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian

from invert.util import build_source_adjacency

from .champagne import SolverChampagne


class SolverChampagneBgCov(SolverChampagne):
    """Champagne with an augmented noise covariance that includes smooth background.

    For support-only evaluation on mixed_cortical_activity, we often want to
    treat the smooth diffuse background as nuisance. This variant estimates a
    smooth-background sensor covariance via a graph diffusion operator and adds
    it to the sensor noise covariance before whitening.
    """

    def __init__(self, name="Champagne (BG Cov)", **kwargs):
        super().__init__(name=name, **kwargs)
        self.require_recompute = True
        self.require_data = True

    def make_inverse_operator(  # type: ignore[override]
        self,
        forward,
        mne_obj=None,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
        bg_diffusion_parameter=0.1,
        bg_adjacency_distance=3e-3,
        bg_scale="trace_match",
        bg_weight=1.0,
        **kwargs,
    ):
        if noise_cov is None:
            # Fall back to standard Champagne behavior.
            return super().make_inverse_operator(
                forward,
                mne_obj,
                *args,
                alpha=alpha,
                noise_cov=noise_cov,
                **kwargs,
            )

        # Build base state (sets self.forward/self.leadfield, aligns channels, etc.).
        super().make_inverse_operator(
            forward,
            mne_obj,
            *args,
            alpha=alpha,
            noise_cov=noise_cov,
            **kwargs,
        )

        # Rebuild using augmented covariance. This is a bit redundant but keeps
        # the change localized and leverages SolverChampagne's implementation.
        # Reset so we don't carry operators from the warm-up call.
        self.inverse_operators = []

        # Compute smooth-background covariance in sensor space: C_bg = (G S)(G S)^T
        G = np.asarray(self.leadfield, dtype=float)  # (n_chans, n_src)
        n_chans, n_src = G.shape
        adjacency = build_source_adjacency(
            self.forward["src"],
            adjacency_type="spatial",
            adjacency_distance=float(bg_adjacency_distance),
            verbose=0,
        )
        L = laplacian(adjacency).astype(float).tocsr()
        S = csr_matrix(np.eye(n_src, dtype=float)) - float(bg_diffusion_parameter) * L
        GS = (S.T @ G.T).T
        C_bg = GS @ GS.T

        # Scale to the data covariance trace (optional).
        if str(bg_scale).lower() == "trace_match" and mne_obj is not None:
            Y = self.unpack_data_obj(mne_obj)
            Cx = (Y @ Y.T) / max(1, int(Y.shape[1]))
            tr_bg = float(np.trace(C_bg))
            tr_x = float(np.trace(Cx))
            scale = (tr_x / tr_bg) if (tr_bg > 0 and tr_x > 0) else 1.0
        else:
            scale = 1.0

        C_aug = np.asarray(noise_cov.data, dtype=float) + float(bg_weight) * float(scale) * C_bg
        noise_cov_aug = mne.Covariance(
            data=C_aug,
            names=list(noise_cov.ch_names),
            bads=list(noise_cov.get("bads", [])),
            projs=list(noise_cov.get("projs", [])),
            nfree=int(getattr(noise_cov, "nfree", 1) or 1),
        )

        return super().make_inverse_operator(
            forward,
            mne_obj,
            *args,
            alpha=alpha,
            noise_cov=noise_cov_aug,
            **kwargs,
        )

