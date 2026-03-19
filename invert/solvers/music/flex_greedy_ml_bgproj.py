import mne
import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh

from invert.util import build_source_adjacency
from invert.util.source_adjacency import source_space_signature

from ..base import BaseSolver, InverseOperator, SolverMeta
from .greedy_ml import greedy_ml_search
from .flex_greedy_ml import SolverFlexGreedyML


_LOW_MODE_CACHE: dict[tuple[str, int], np.ndarray] = {}


def _low_graph_modes(forward, *, k_low: int) -> np.ndarray:
    """Return the k_low lowest-frequency Laplacian eigenvectors (graph Fourier modes)."""
    if k_low <= 0:
        return np.zeros((0, 0), dtype=float)

    sig = source_space_signature(forward["src"])
    cache_key = (sig, int(k_low))
    cached = _LOW_MODE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    adjacency = build_source_adjacency(forward["src"], verbose=0)
    lap = laplacian(adjacency).astype(float)
    n = int(lap.shape[0])
    k_low = int(max(1, min(int(k_low), n)))

    if k_low >= n:
        return np.eye(n, dtype=float)

    v0 = np.ones(n, dtype=float)
    v0 /= np.linalg.norm(v0)
    _, U = eigsh(lap, k=k_low, which="SM", v0=v0)
    U = np.real(U)
    _LOW_MODE_CACHE[cache_key] = U
    return U


def _orth_proj_complement(B: np.ndarray, *, rcond: float = 1e-6) -> tuple[np.ndarray, int]:
    """Compute P = I - QQ^T where Q spans col(B). Returns (P, rank(B))."""
    n_eff = int(B.shape[0])
    if B.size == 0:
        return np.eye(n_eff, dtype=float), 0

    U, s, _ = np.linalg.svd(B, full_matrices=False)
    if s.size == 0:
        return np.eye(n_eff, dtype=float), 0

    tol = float(np.max(s)) * float(rcond)
    r = int(np.sum(s > tol))
    if r <= 0:
        return np.eye(n_eff, dtype=float), 0
    Q = U[:, :r]
    P = np.eye(n_eff, dtype=float) - Q @ Q.T
    return P, r


class SolverFlexGreedyMLBgProj(BaseSolver):
    """FLEX-GreedyML with explicit smooth-background nuisance projection.

    mixed_cortical_activity includes a diffuse, smooth background component. For
    salient-region recovery, treating that background as a nuisance subspace and
    projecting it out before sparse patch selection can reduce localization bias.
    """

    meta = SolverMeta(
        acronym="FLEX-GreedyML-BGProj",
        full_name="FLEX Greedy Maximum-Likelihood (Background-Projected)",
        category="Subspace Methods",
        description=(
            "FLEX-GreedyML on a diffusion patch dictionary, preceded by projection "
            "out of a low graph-frequency (smooth) nuisance subspace to suppress "
            "diffuse background activity."
        ),
        references=SolverFlexGreedyML.meta.references,
        internal=True,
    )

    def __init__(self, name="FLEX-GreedyML-BGProj", **kwargs):
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
        n_starts=60,
        n_refine_iters=3,
        penalty_mode="bic_support_only",
        # Patch dictionary
        n_orders=5,
        diffusion_parameter=0.12,
        adjacency_distance=3e-3,
        # Background projection
        bg_gft_rank=6,
        bg_rcond=1e-6,
        **kwargs,
    ):
        super().make_inverse_operator(forward, mne_obj, *args, alpha=alpha, **kwargs)
        wf = self.prepare_whitened_forward(noise_cov)

        data = self.unpack_data_obj(mne_obj)
        data_w = wf.sensor_transform @ data
        G_w = wf.G_white

        n_eff, n_dipoles = G_w.shape
        T = data_w.shape[1]

        # --- Build smooth background sensor subspace and project it out ---
        U_low = _low_graph_modes(forward, k_low=int(bg_gft_rank))
        if U_low.size:
            B = G_w @ U_low  # (n_eff, k_low)
        else:
            B = np.zeros((n_eff, 0), dtype=float)
        P, bg_rank_eff = _orth_proj_complement(B, rcond=float(bg_rcond))

        data_p = P @ data_w
        G_p = P @ G_w

        # Effective observation dimension after projection.
        n_eff_eff = int(max(1, n_eff - bg_rank_eff))

        # --- Build extended dictionary on the projected leadfield ---
        leadfields, gradients = SolverFlexGreedyML._build_patch_dictionary(
            G_p,
            forward,
            int(n_orders),
            float(diffusion_parameter),
            float(adjacency_distance),
        )
        n_total_orders = len(leadfields)  # 1 + n_orders

        G_ext = np.hstack(leadfields)  # (n_eff, n_total_orders * n_dipoles)
        n_ext = int(G_ext.shape[1])

        Gram = G_ext.T @ G_ext
        eps_gram = 1e-10 * np.trace(Gram) / max(n_ext, 1)
        Gram[np.diag_indices_from(Gram)] += eps_gram

        R = G_ext.T @ data_p
        Q = R @ R.T
        total_var = float(np.sum(data_p**2))

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
            n_eff_eff,
            int(T),
            n_ext,
            int(k_max),
            int(n_starts),
            int(n_refine_iters),
            str(penalty_mode),
            excluded_fn=_excluded_fn,
        )

        # --- Map back to original source space via gradients ---
        sel_orders = [int(e // n_dipoles) for e in selected_ext]
        sel_dipoles = [int(e % n_dipoles) for e in selected_ext]

        selected_ext_arr = np.asarray(selected_ext, dtype=np.intp)
        G_sel = G_ext[:, selected_ext_arr]
        inv_ext = np.linalg.lstsq(G_sel, np.eye(n_eff), rcond=None)[0]  # (K, n_eff)

        M = np.zeros((n_dipoles, len(selected_ext)), dtype=float)
        for i, (o, d) in enumerate(zip(sel_orders, sel_dipoles, strict=False)):
            M[:, i] = gradients[o][d].toarray().ravel()

        # Include the projection explicitly: coefficients are fit to P @ data_w.
        inverse_operator_w = (M @ inv_ext) @ P  # (n_dipoles, n_eff)

        self.inverse_operators = [
            InverseOperator(inverse_operator_w @ wf.sensor_transform, self.name),
        ]
        return self
