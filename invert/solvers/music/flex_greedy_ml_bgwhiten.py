import mne
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian

from invert.util import build_source_adjacency

from ..base import BaseSolver, InverseOperator, SolverMeta
from .flex_greedy_ml import SolverFlexGreedyML
from .greedy_ml import greedy_ml_search


def _invsqrt_psd(M: np.ndarray, *, eps: float = 1e-10) -> np.ndarray:
    """Compute (M + ridge I)^(-1/2) for a symmetric PSD matrix (small n)."""
    M = np.asarray(M, dtype=float)
    M = 0.5 * (M + M.T)
    n = int(M.shape[0])
    ridge = float(eps) * float(np.trace(M)) / max(1, n)
    evals, evecs = np.linalg.eigh(M + ridge * np.eye(n))
    evals = np.maximum(evals, ridge)
    inv_sqrt = (evecs * (1.0 / np.sqrt(evals))[None, :]) @ evecs.T
    return 0.5 * (inv_sqrt + inv_sqrt.T)


def _smooth_operator(forward, *, diffusion_parameter: float, adjacency_distance: float) -> csr_matrix:
    """S = I - a*L using the same graph Laplacian construction as the simulator."""
    adjacency = build_source_adjacency(
        forward["src"],
        adjacency_type="spatial",
        adjacency_distance=float(adjacency_distance),
        verbose=0,
    )
    L = laplacian(adjacency).astype(float)
    n = int(L.shape[0])
    S = csr_matrix(np.eye(n, dtype=float)) - float(diffusion_parameter) * csr_matrix(L)
    return S.tocsr()


class SolverFlexGreedyMLBgWhiten(BaseSolver):
    """FLEX-GreedyML with graph-smooth background whitening (GLS).

    Model motivation: mixed_cortical_activity effectively contains a diffuse smooth
    background that we want to treat as nuisance when recovering salient patches.
    If background is modeled as y_bg(t) = S w(t), w iid, then in whitened sensor
    space its covariance is C_bg ∝ (G S)(G S)^T. Whitening by C_bg implements a
    generalized least squares (colored-noise ML) detector for sparse patches.
    """

    meta = SolverMeta(
        acronym="FLEX-GreedyML-BGWhite",
        full_name="FLEX Greedy Maximum-Likelihood (Background-Whitened)",
        category="Subspace Methods",
        description=(
            "FLEX-GreedyML after an additional whitening transform that treats "
            "graph-smooth background activity as colored nuisance (GLS/ML)."
        ),
        references=SolverFlexGreedyML.meta.references,
        internal=True,
    )

    def __init__(self, name="FLEX-GreedyML-BGWhite", **kwargs):
        self.name = name
        super().__init__(**kwargs)

    def make_inverse_operator(
        self,
        forward,
        mne_obj=None,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
        # Sparse search
        k_max=5,
        n_starts=50,
        n_refine_iters=3,
        penalty_mode="bic_support_only",
        # Patch dictionary
        n_orders=3,
        diffusion_parameter=0.1,
        adjacency_distance=3e-3,
        # Background model (graph smoothing operator used to form C_bg)
        bg_diffusion_parameter=0.1,
        bg_adjacency_distance=3e-3,
        bg_eps=1e-10,
        **kwargs,
    ):
        super().make_inverse_operator(forward, mne_obj, *args, alpha=alpha, **kwargs)
        wf = self.prepare_whitened_forward(noise_cov)

        data = self.unpack_data_obj(mne_obj)
        data_w = wf.sensor_transform @ data
        G_w = wf.G_white

        n_eff, n_dipoles = G_w.shape
        T = int(data_w.shape[1])

        # --- Background whitening in sensor space ---
        S_bg = _smooth_operator(
            forward,
            diffusion_parameter=float(bg_diffusion_parameter),
            adjacency_distance=float(bg_adjacency_distance),
        )
        # Compute GS without densifying S_bg.
        GS = (S_bg.T @ G_w.T).T  # (n_eff, n_dipoles)
        C_bg = GS @ GS.T
        # Match global scale to observed covariance (robust to unknown alpha).
        Cx = (data_w @ data_w.T) / max(1, T)
        tr_bg = float(np.trace(C_bg))
        tr_x = float(np.trace(Cx))
        scale = (tr_x / tr_bg) if (tr_bg > 0 and tr_x > 0) else 1.0
        W_bg = _invsqrt_psd(scale * C_bg, eps=float(bg_eps))

        data_bw = W_bg @ data_w
        G_bw = W_bg @ G_w

        # --- Build extended dictionary on background-whitened leadfield ---
        leadfields, gradients = SolverFlexGreedyML._build_patch_dictionary(
            G_bw,
            forward,
            int(n_orders),
            float(diffusion_parameter),
            float(adjacency_distance),
        )
        n_total_orders = len(leadfields)

        G_ext = np.hstack(leadfields)  # (n_eff, n_total_orders * n_dipoles)
        n_ext = int(G_ext.shape[1])

        Gram = G_ext.T @ G_ext
        eps_gram = 1e-10 * np.trace(Gram) / max(n_ext, 1)
        Gram[np.diag_indices_from(Gram)] += eps_gram

        R = G_ext.T @ data_bw
        Q = R @ R.T
        total_var = float(np.sum(data_bw**2))

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
            n_ext,
            int(k_max),
            int(n_starts),
            int(n_refine_iters),
            str(penalty_mode),
            excluded_fn=_excluded_fn,
        )

        sel_orders = [int(e // n_dipoles) for e in selected_ext]
        sel_dipoles = [int(e % n_dipoles) for e in selected_ext]

        selected_ext_arr = np.asarray(selected_ext, dtype=np.intp)
        G_sel = G_ext[:, selected_ext_arr]
        inv_ext = np.linalg.lstsq(G_sel, np.eye(n_eff), rcond=None)[0]  # (K, n_eff)

        M = np.zeros((n_dipoles, len(selected_ext)), dtype=float)
        for i, (o, d) in enumerate(zip(sel_orders, sel_dipoles, strict=False)):
            M[:, i] = gradients[o][d].toarray().ravel()

        # Mapping: data -> wf.sensor_transform -> W_bg -> coefficients.
        sensor_transform = W_bg @ wf.sensor_transform
        inverse_operator_w = M @ inv_ext  # (n_dipoles, n_eff)

        self.inverse_operators = [
            InverseOperator(inverse_operator_w @ sensor_transform, self.name),
        ]
        return self

