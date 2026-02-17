import mne
import numpy as np
from scipy.spatial.distance import cdist

from ...util.util import pos_from_forward
from ..base import InverseOperator, SolverMeta
from .base_beamformer import BaseBeamformer
from .utils import build_covariance_candidates


class SolverMCMV(BaseBeamformer):
    """Class for the Multiple Constrained Minimum Variance (MCMV) Beamformer
    inverse solution [1].

    MCMV extends LCMV by applying multiple linear constraints simultaneously.
    This improves robustness in the presence of correlated sources.

    References
    ----------
    [1] Nunes, A. S., Moiseev, A., Kozhemiako, N., Cheung, T., Ribary, U., &
    Doesburg, S. M. (2020). Multiple constrained minimum variance beamformer
    (MCMV) performance in connectivity analyses. NeuroImage, 208, 116386.

    """

    meta = SolverMeta(
        slug="mcmv",
        full_name="Multiple Constrained Minimum Variance",
        category="Beamformers",
        description=(
            "Multiple-constrained extension of LCMV that imposes additional linear "
            "constraints (e.g., to improve robustness with correlated sources)."
        ),
        references=[
            "Nunes, A. S., Moiseev, A., Kozhemiako, N., Cheung, T., Ribary, U., "
            "& Doesburg, S. M. (2020). Multiple constrained minimum variance "
            "beamformer (MCMV) performance in connectivity analyses. "
            "NeuroImage, 208, 116386.",
        ],
    )

    def __init__(self, name="MCMV Beamformer", reduce_rank=True, rank="auto", **kwargs):
        self.name = name
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(
        self,
        forward,
        mne_obj=None,
        *args,
        weight_norm=True,
        noise_cov: mne.Covariance | None = None,
        alpha="auto",
        cov_reg: str = "oas",
        cov_reg_beta: float = 0.05,
        cov_reg_cond_target: float = 1e4,
        k_constraints=3,
        constraint_mode="distance",
        constraint_indices=None,
        n_preloc_peaks=5,
        verbose=0,
        **kwargs,
    ):
        """Calculate inverse operator using MCMV formula.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        weight_norm : bool
            Normalize the filter weight matrix W to unit length of the columns.
        alpha : float
            The regularization parameter.
        k_constraints : int
            Number of constraints per source. When k=1, reduces to LCMV.
            For k>1, includes k-1 nearest neighbors in constraint matrix.
            Ignored when constraint_mode="lcmv_preloc" (auto-determined)
            or when constraint_indices is provided.
        constraint_mode : str
            How to select constraint sources for each grid point:
            - "distance": k-1 physically nearest sources (Euclidean, default)
            - "similarity": k-1 most correlated leadfield columns
            - "lcmv_preloc": LCMV pre-localization peaks as constraints
            Ignored when constraint_indices is provided.
        constraint_indices : list of array-like, optional
            User-supplied constraint indices, one entry per source. Each
            entry is an array of source indices where the first element is
            the target source and the rest are constraint sources. When
            provided, constraint_mode and k_constraints are ignored.
        n_preloc_peaks : int
            Number of LCMV power peaks to use as constraints (only for
            constraint_mode="lcmv_preloc"). Default 5.

        Return
        ------
        self : object returns itself for convenience

        Notes
        -----
        Implements the MCMV formula: w = C_inv @ G @ inv(G.T @ C_inv @ G) @ f
        where G is the constraint matrix (m x k) and f is the constraint vector.
        """
        super().make_inverse_operator(forward, mne_obj, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)

        wf = self.prepare_whitened_forward(noise_cov)
        leadfield = wf.G_white.copy()
        sensor_transform = wf.sensor_transform
        leadfield /= np.linalg.norm(leadfield, axis=0)
        n_chans, n_dipoles = leadfield.shape

        self.weight_norm = weight_norm
        self.constraint_mode = constraint_mode

        y = sensor_transform @ data
        I = np.identity(n_chans)

        # Recompute regularization based on the max eigenvalue of the Covariance
        # Matrix (opposed to that of the leadfield)
        y -= y.mean(axis=1, keepdims=True)
        C = self.data_covariance(y, center=False, ddof=1)
        cov_mats, self.alphas, cov_meta = build_covariance_candidates(
            C=C,
            I=I,
            alpha=self.alpha,
            get_alphas_fn=self.get_alphas,
            n_samples=int(y.shape[1]),
            cov_reg=cov_reg,
            cov_reg_beta=float(cov_reg_beta),
            cov_reg_cond_target=float(cov_reg_cond_target),
        )
        if "oas_shrinkage" in cov_meta:
            self._cov_reg_oas_shrinkage = float(cov_meta["oas_shrinkage"])

        # Build per-source constraint index lists
        if constraint_indices is not None:
            constraint_indices_per_source = [
                np.asarray(idx, dtype=int) for idx in constraint_indices
            ]
        else:
            constraint_indices_per_source = self._build_constraint_indices(
                forward=forward,
                leadfield=leadfield,
                C=C,
                cov_mats=cov_mats,
                k_constraints=k_constraints,
                constraint_mode=constraint_mode,
                n_preloc_peaks=n_preloc_peaks,
                n_dipoles=n_dipoles,
            )

        inverse_operators = []
        for cov_mat in cov_mats:
            C_inv = self.robust_inverse(cov_mat)
            W = np.zeros((n_chans, n_dipoles))

            for i in range(n_dipoles):
                indices = constraint_indices_per_source[i]
                k = len(indices)

                if k == 1:
                    G = leadfield[:, i : i + 1]
                    f = np.array([1.0])
                else:
                    G = leadfield[:, indices]
                    f = np.zeros(k)
                    f[0] = 1.0

                # Apply MCMV formula: w = C_inv @ G @ inv(G.T @ C_inv @ G) @ f
                G_T_C_inv = G.T @ C_inv
                G_T_C_inv_G = G_T_C_inv @ G

                # Robust inverse of the k×k matrix
                try:
                    G_T_C_inv_G_inv = np.linalg.inv(G_T_C_inv_G)
                except np.linalg.LinAlgError:
                    G_T_C_inv_G_inv = np.linalg.pinv(G_T_C_inv_G)

                w = C_inv @ G @ G_T_C_inv_G_inv @ f
                W[:, i] = w

            if self.weight_norm:
                W /= np.linalg.norm(W, axis=0)

            # Map back to raw sensor space
            inverse_operator = (sensor_transform.T @ W).T
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]

        return self

    def _build_constraint_indices(
        self,
        *,
        forward,
        leadfield,
        C,
        cov_mats,
        k_constraints,
        constraint_mode,
        n_preloc_peaks,
        n_dipoles,
    ):
        """Return list of constraint index arrays, one per source.

        Each entry is an array where the first element is the target source
        and the remaining are constraint sources.
        """
        if constraint_mode == "similarity":
            k = min(k_constraints, n_dipoles)
            if k <= 1:
                return [np.array([i]) for i in range(n_dipoles)]
            leadfield_norm = leadfield / (np.linalg.norm(leadfield, axis=0) + 1e-10)
            sim = leadfield_norm.T @ leadfield_norm
            result = []
            for i in range(n_dipoles):
                row = sim[i, :].copy()
                row[i] = -np.inf
                neighbors = np.argsort(row)[::-1][: k - 1]
                result.append(np.concatenate([[i], neighbors]))
            return result

        elif constraint_mode == "distance":
            k = min(k_constraints, n_dipoles)
            if k <= 1:
                return [np.array([i]) for i in range(n_dipoles)]
            pos = pos_from_forward(forward, verbose=0)
            n_sources = len(pos)
            n_orient = n_dipoles // n_sources
            dists = cdist(pos, pos)
            result = []
            for i in range(n_dipoles):
                src_i = i // n_orient
                row = dists[src_i, :].copy()
                row[src_i] = np.inf
                # Find nearest sources, then map back to column indices
                src_neighbors = np.argsort(row)[: k - 1]
                col_neighbors = src_neighbors * n_orient + (i % n_orient)
                result.append(np.concatenate([[i], col_neighbors]))
            return result

        elif constraint_mode == "lcmv_preloc":
            # Compute LCMV power map using the first covariance candidate
            C_reg = cov_mats[0]
            C_inv = self.robust_inverse(C_reg)
            CiL = C_inv @ leadfield  # (n_chans, n_dipoles)

            # power_i = (CiL_i^T C CiL_i) / (l_i^T CiL_i)^2
            numerator = np.einsum("ji,jk,ki->i", CiL, C, CiL)  # CiL^T C CiL diag
            denominator = np.einsum("ji,ji->i", leadfield, CiL)  # l^T CiL diag
            denominator = denominator**2
            denominator = np.maximum(denominator, 1e-30)
            power = numerator / denominator

            # Find local maxima using source-space adjacency
            adjacency = mne.spatial_src_adjacency(forward["src"], verbose=0)
            peak_indices = _find_local_maxima(power, adjacency, n_preloc_peaks)

            # For each source: constraints = [i] + peaks (excluding i)
            result = []
            for i in range(n_dipoles):
                peers = [p for p in peak_indices if p != i]
                result.append(np.array([i] + peers, dtype=int))
            return result

        else:
            raise ValueError(
                f"Unknown constraint_mode={constraint_mode!r}. "
                "Expected 'similarity', 'distance', or 'lcmv_preloc'."
            )


def _find_local_maxima(values, adjacency, n_peaks):
    """Find top local maxima in a source-space scalar map.

    A vertex is a local maximum if its value is >= all adjacent vertices.
    Returns indices of the top ``n_peaks`` local maxima sorted by value.
    """
    from scipy.sparse import issparse

    n = len(values)
    if issparse(adjacency):
        adj_csr = adjacency.tocsr()
    else:
        from scipy.sparse import csr_array

        adj_csr = csr_array(adjacency)
    is_max = np.ones(n, dtype=bool)
    for i in range(n):
        row_start = adj_csr.indptr[i]
        row_end = adj_csr.indptr[i + 1]
        neighbors = adj_csr.indices[row_start:row_end]
        if len(neighbors) > 0 and np.any(values[neighbors] > values[i]):
            is_max[i] = False
    maxima_idx = np.where(is_max)[0]
    # Sort by descending value and take top n_peaks
    order = np.argsort(values[maxima_idx])[::-1]
    return list(maxima_idx[order[:n_peaks]])
