import mne
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian

from invert.util import build_source_adjacency

from ..base import BaseSolver, InverseOperator, SolverMeta
from .greedy_ml import greedy_ml_search


class SolverFlexGreedyML(BaseSolver):
    """Source localization via multi-start greedy ML on an extended dictionary
    of single-dipole and diffusion-smoothed patch leadfields.

    At each vertex, either a single dipole or one of its patch orders is
    selected (mutually exclusive). Uses BIC for model order selection.

    Scales as O(n_starts * n_orders * n_sources * k_max).

    References
    ----------
    [1] Wax, M., & Kailath, T. (1985). Detection of signals by information
        theoretic criteria. IEEE Trans. ASSP, 33(2), 387-392.
    """

    meta = SolverMeta(
        acronym="FLEX-GreedyML",
        full_name="FLEX Greedy Maximum-Likelihood",
        category="Subspace Methods",
        description=(
            "Source localization via multi-start greedy ML on an extended "
            "dictionary of single-dipole and patch leadfields with BIC "
            "model order selection. At each vertex, only one patch order "
            "can be active."
        ),
        references=[
            "Wax, M., & Kailath, T. (1985). Detection of signals by information theoretic criteria. IEEE Trans. ASSP, 33(2), 387-392.",
        ],
    )

    def __init__(self, name="FLEX-GreedyML", **kwargs):
        self.name = name
        super().__init__(**kwargs)

    def make_inverse_operator(
        self,
        forward,
        mne_obj=None,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
        k_max=5,
        n_starts=50,
        n_refine_iters=3,
        penalty_mode="bic_per_timepoint",
        n_orders=3,
        diffusion_parameter=0.1,
        adjacency_distance=3e-3,
        **kwargs,
    ):
        super().make_inverse_operator(forward, mne_obj, *args, alpha=alpha, **kwargs)
        wf = self.prepare_whitened_forward(noise_cov)

        data = self.unpack_data_obj(mne_obj)
        data_w = wf.sensor_transform @ data
        G_w = wf.G_white

        n_eff, n_dipoles = G_w.shape
        T = data_w.shape[1]

        # --- Build extended dictionary ---
        leadfields, gradients = self._build_patch_dictionary(
            G_w,
            forward,
            n_orders,
            diffusion_parameter,
            adjacency_distance,
        )
        n_total_orders = len(leadfields)  # 1 + n_orders

        # Stack: [order0_all_dipoles | order1_all_dipoles | ...]
        G_ext = np.hstack(leadfields)  # (n_eff, n_total_orders * n_dipoles)
        n_ext = G_ext.shape[1]

        Gram = G_ext.T @ G_ext
        eps_gram = 1e-10 * np.trace(Gram) / max(n_ext, 1)
        Gram[np.diag_indices_from(Gram)] += eps_gram
        R = G_ext.T @ data_w
        Q = R @ R.T
        total_var = np.sum(data_w**2)

        # Mutual exclusivity: selecting extended index e blocks all other
        # orders at the same dipole position.
        def _excluded_fn(selected):
            blocked = set()
            for e in selected:
                dipole = e % n_dipoles
                for o in range(n_total_orders):
                    blocked.add(o * n_dipoles + dipole)
            return blocked

        selected_ext = greedy_ml_search(
            Gram,
            Q,
            total_var,
            n_eff,
            T,
            n_ext,
            k_max,
            n_starts,
            n_refine_iters,
            penalty_mode,
            excluded_fn=_excluded_fn,
        )

        # --- Map back to original source space via gradients ---
        # Each selected ext index e -> (order, dipole)
        sel_orders = [e // n_dipoles for e in selected_ext]
        sel_dipoles = [e % n_dipoles for e in selected_ext]

        selected_ext_arr = np.array(selected_ext, dtype=np.intp)
        G_sel = G_ext[:, selected_ext_arr]
        inv_ext = np.linalg.lstsq(G_sel, np.eye(n_eff), rcond=None)[0]  # (K, n_eff)

        # Gradient mapping: (n_dipoles, K)
        # M[:, i] = gradient[order_i][:, dipole_i]
        M = np.zeros((n_dipoles, len(selected_ext)))
        for i, (o, d) in enumerate(zip(sel_orders, sel_dipoles, strict=False)):
            M[:, i] = gradients[o][d].toarray().ravel()

        # Full inverse: sensor -> original source space
        inverse_operator_w = M @ inv_ext  # (n_dipoles, n_eff)

        self.inverse_operators = [
            InverseOperator(inverse_operator_w @ wf.sensor_transform, self.name),
        ]
        return self

    @staticmethod
    def _build_patch_dictionary(
        G_w, forward, n_orders, diffusion_parameter, adjacency_distance
    ):
        """Build extended dictionary with diffusion-smoothed patch leadfields.

        Returns
        -------
        leadfields : list of (n_eff, n_dipoles) arrays
        gradients : list of (n_dipoles, n_dipoles) sparse matrices
        """
        n_dipoles = G_w.shape[1]
        I = np.identity(n_dipoles)

        leadfields = [G_w.copy()]
        gradients = [csr_matrix(I)]

        if n_orders == 0:
            return leadfields, gradients

        adjacency = build_source_adjacency(
            forward["src"],
            adjacency_type="spatial",
            adjacency_distance=adjacency_distance,
            verbose=0,
        )

        LL = laplacian(adjacency)
        smoothing_op = csr_matrix(I - diffusion_parameter * LL)

        for i in range(n_orders):
            S_i = smoothing_op ** (i + 1)
            leadfields.append(G_w @ S_i.toarray())
            gradients.append(gradients[0] @ S_i)

        # Normalize gradients row-wise
        for i in range(len(gradients)):
            row_sums = gradients[i].sum(axis=1).ravel()
            scaling = 1.0 / np.maximum(np.abs(np.asarray(row_sums).ravel()), 1e-12)
            gradients[i] = csr_matrix(gradients[i].multiply(scaling.reshape(-1, 1)))

        return leadfields, gradients
