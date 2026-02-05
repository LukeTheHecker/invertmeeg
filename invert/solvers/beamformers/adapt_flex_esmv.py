from copy import deepcopy

import mne
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian

from ..base import BaseSolver, InverseOperator, SolverMeta


class SolverAdaptFlexESMV(BaseSolver):
    """Flexible-Extent Adaptive Eigenspace Minimum Variance Beamformer.

    Extends AdaptiveESMV with diffusion-smoothed leadfield dictionaries
    (the FLEX approach from Hecker et al. 2023). For each source location,
    the beamformer is evaluated at multiple smoothness orders (point source,
    small patch, medium patch, ...) and the order yielding the highest
    beamformer output power is selected. The corresponding gradient matrix
    maps the smoothed source estimate back to the original source grid.

    This allows the beamformer to adapt its spatial resolution per source:
      - Isolated point sources: order 0 (standard beamformer, no smoothing)
      - Extended patch sources: higher order (diffusion-smoothed leadfield)

    The core beamformer at each order uses AdaptiveESMV's gap-adaptive
    Wiener eigenspace projection.

    Parameters
    ----------
    n_orders : int
        Number of smoothing orders (0 = point only, 3 = up to 3 diffusion steps).
    diffusion_parameter : float
        Diffusion constant for the smoothing operator S = I - α·L_graph.
    adjacency_type : str
        "spatial" (graph neighbors) or "distance" (Euclidean threshold).
    """

    meta = SolverMeta(
        slug="adapt_flex_esmv",
        full_name="Adaptive Flexible-Extent ESMV",
        category="Beamformers",
        description=(
            "Adaptive multi-order diffusion-smoothed extension of an ESMV beamformer "
            "that selects the smoothing order per source location."
        ),
        references=[
            "Lukas Hecker (2025). Unpublished.",
            "Jonmohamadi, Y., Poudel, G., Innes, C., Weiss, D., Krueger, R., & Jones, R. "
            "(2014). Comparison of beamformers for EEG source signal reconstruction. "
            "Biomedical Signal Processing and Control, 14, 175-188.",
        ],
    )

    def __init__(
        self,
        name="AdaptFlexESMV Beamformer",
        reduce_rank=True,
        rank="auto",
        n_orders=3,
        diffusion_parameter=0.1,
        adjacency_type="spatial",
        adjacency_distance=3e-3,
        **kwargs,
    ):
        self.name = name
        self.n_orders = n_orders
        self.diffusion_parameter = diffusion_parameter
        self.adjacency_type = adjacency_type
        self.adjacency_distance = adjacency_distance
        self.is_prepared = False
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def _prepare_flex(self):
        """Build multi-order smoothed leadfields and gradient matrices."""
        n_dipoles = self.leadfield.shape[1]
        I_src = np.identity(n_dipoles)

        self.leadfields = [deepcopy(self.leadfield)]
        self.gradients = [csr_matrix(I_src)]

        if self.n_orders == 0:
            self.is_prepared = True
            return

        if self.adjacency_type == "spatial":
            adjacency = mne.spatial_src_adjacency(self.forward["src"], verbose=0)
        else:
            adjacency = mne.spatial_dist_adjacency(
                self.forward["src"], self.adjacency_distance, verbose=None
            )

        LL = laplacian(adjacency)
        smoothing_op = csr_matrix(I_src - self.diffusion_parameter * LL)

        for i in range(self.n_orders):
            S_i = smoothing_op ** (i + 1)
            new_lf = self.leadfields[0] @ S_i
            new_grad = self.gradients[0] @ S_i
            self.leadfields.append(new_lf)
            self.gradients.append(new_grad)

        # Normalize gradients row-wise
        for i in range(len(self.gradients)):
            row_sums = np.asarray(self.gradients[i].sum(axis=1)).ravel()
            scaling = 1.0 / np.maximum(np.abs(row_sums), 1e-12)
            self.gradients[i] = csr_matrix(
                self.gradients[i].multiply(scaling.reshape(-1, 1))
            )

        self.is_prepared = True

    def make_inverse_operator(self, forward, mne_obj, *args, alpha="auto", **kwargs):
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)

        if not self.is_prepared:
            self._prepare_flex()

        n_chans, n_dipoles = self.leadfield.shape
        n_orders = len(self.leadfields)
        epsilon = 1e-15

        y = data
        I = np.identity(n_chans)
        y -= y.mean(axis=1, keepdims=True)
        C = self.data_covariance(y, center=False, ddof=1)

        self.alphas = self.get_alphas(reference=C)

        inverse_operators = []
        for alpha_reg in self.alphas:
            C_reg = C + alpha_reg * I

            # Eigendecomposition for adaptive Wiener (computed once)
            eigvals, eigvecs = np.linalg.eigh(C_reg)
            idx = np.argsort(eigvals)[::-1]
            eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

            n_comp = self.estimate_n_sources(C_reg, method="auto")
            sigma2 = np.mean(eigvals[n_comp:]) if n_comp < n_chans else epsilon

            # Adaptive exponent from eigenvalue gap
            if n_comp < n_chans:
                gap = eigvals[n_comp - 1] / (eigvals[n_comp] + epsilon)
                p = max(1.0, np.log(gap + 1.0))
            else:
                p = 1.0

            wiener = (eigvals / (eigvals + sigma2 + epsilon)) ** p
            P_adapt = eigvecs @ (np.diag(wiener) @ eigvecs.T)

            C_inv = self.robust_inverse(C_reg)

            # Compute beamformer weights and output power for each order
            powers = np.zeros((n_orders, n_dipoles))
            weights_per_order = []

            for k in range(n_orders):
                lf_k = self.leadfields[k]
                lf_k_norm = lf_k / (
                    np.linalg.norm(lf_k, axis=0, keepdims=True) + epsilon
                )

                C_inv_lf = C_inv @ lf_k_norm
                diag_el = np.einsum("ij,ji->i", lf_k_norm.T, C_inv_lf)
                W_mv = C_inv_lf / (diag_el + epsilon)

                # Adaptive eigenspace projection
                W_k = P_adapt @ W_mv

                # Output power: mean squared beamformer output per dipole
                s_k = W_k.T @ y  # (n_dipoles, n_times)
                powers[k] = np.mean(s_k**2, axis=1)

                weights_per_order.append(W_k)

            # For each dipole, pick the order with maximum output power
            best_order = np.argmax(powers, axis=0)  # (n_dipoles,)

            # Build final inverse operator using gradient mapping
            # For dipole i with best order k:
            #   The gradient G[k][i,:] maps the smoothed activation back to
            #   the original source grid. Combined with weight w_i^k:
            #   inv_op[j, :] += G[k][i,j] * w_i^{k,T}
            inv_op = np.zeros((n_dipoles, n_chans))
            for k in range(n_orders):
                mask = best_order == k
                if not np.any(mask):
                    continue
                dipole_idx = np.where(mask)[0]

                W_sel = weights_per_order[k][:, dipole_idx]  # (n_chans, n_sel)
                G_sel = self.gradients[k][dipole_idx].toarray()  # (n_sel, n_dipoles)

                inv_op += G_sel.T @ W_sel.T

            inverse_operators.append(inv_op)

        self.inverse_operators = [
            InverseOperator(op, self.name) for op in inverse_operators
        ]
        return self
