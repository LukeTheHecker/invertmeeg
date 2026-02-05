import logging
from copy import deepcopy

import mne
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverAlternatingProjections(BaseSolver):
    """Class for the Alternating Projections inverse solution [1] with flexible
        extent estimation (FLEX-AP). This approach combines the AP-approach by
        Adler et al. [1] with dipoles with flexible extents, e.g., FLEX-MUSIC
        (Hecker 2023, unpublished).

    References
    ---------
    [1] Adler, A., Wax, M., & Pantazis, D. (2022, March). Brain Source
    Localization by Alternating Projection. In 2022 IEEE 19th International
    Symposium on Biomedical Imaging (ISBI) (pp. 1-5). IEEE.

    """

    meta = SolverMeta(
        acronym="AP",
        full_name="Alternating Projections",
        category="Subspace Methods",
        description=(
            "Alternating-projection source localization on the signal subspace, "
            "extended here with flexible-extent (FLEX-AP) patch estimation."
        ),
        references=[
            "Adler, A., Wax, M., & Pantazis, D. (2022). Brain Source Localization by Alternating Projection. In 2022 IEEE 19th International Symposium on Biomedical Imaging (ISBI) (pp. 1–5). IEEE.",
            "Hecker, L., Tebartz van Elst, L., & Kornmeier, J. (2023). Source localization using recursively applied and projected MUSIC with flexible extent estimation. Frontiers in Neuroscience, 17, 1170862.",
        ],
    )

    def __init__(
        self, name="Flexible Alternative Projections", scale_leadfield=False, **kwargs
    ):
        self.name = name
        self.is_prepared = False
        self.scale_leadfield = scale_leadfield
        return super().__init__(**kwargs)

    def make_inverse_operator(
        self,
        forward,
        mne_obj,
        *args,
        n_orders=3,
        alpha="auto",
        n="enhanced",
        k="auto",
        refine_solution=True,
        max_iter=1000,
        diffusion_smoothing=True,
        diffusion_parameter=0.1,
        adjacency_type="spatial",
        adjacency_distance=3e-3,
        depth_weights=None,
        **kwargs,
    ):
        """Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        alpha : float
            The regularization parameter.
        n : int/ str
            Number of eigenvalues to use.
                int: The number of eigenvalues to use.
                "L": L-curve method for automated selection.
                "drop": Selection based on relative change of eigenvalues.
                "auto": Combine L and drop method
                "mean": Selects the eigenvalues that are larger than the mean of all eigs.
        k : int
            Number of recursions.

        max_iter : int
            Maximum number of iterations during refinement.
        diffusion_smoothing : bool
            Whether to use diffusion smoothing. Default is True.
        diffusion_parameter : float
            The diffusion parameter (alpha). Default is 0.1.
        adjacency_type : str
            The type of adjacency. "spatial" -> based on graph neighbors. "distance" -> based on distance
        adjacency_distance : float
            The distance at which neighboring dipoles are considered neighbors.
        depth_weights : numpy.ndarray
            The depth weights to use for depth weighting the leadfields. If None, no depth weighting is applied.

        Return
        ------
        self : object returns itself for convenience
        """
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)

        self.diffusion_smoothing = diffusion_smoothing
        self.diffusion_parameter = diffusion_parameter
        self.n_orders = n_orders
        self.adjacency_type = adjacency_type
        self.adjacency_distance = adjacency_distance
        data = self.unpack_data_obj(mne_obj)

        if not self.is_prepared:
            self.prepare_flex()
        inverse_operator = self.make_ap(
            data,
            n,
            k,
            max_iter=max_iter,
            refine_solution=refine_solution,
            depth_weights=depth_weights,
        )
        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name),
        ]
        return self

    def make_ap(
        self,
        y,
        n,
        k,
        refine_solution=True,
        max_iter=1000,
        covariance_type="AP",
        depth_weights=None,
        apply_depth_weights=True,
    ):
        """Create the FLEX-MUSIC inverse solution to the EEG data.

        Parameters
        ----------
        y : numpy.ndarray
            EEG data matrix (channels, time)
        n : int/ str
            Number of eigenvectors to use or "auto" for l-curve method.
        k : int
            Number of recursions.
        refine_solution : bool
            If True: Re-visit each selected candidate and check if there is a
            better alternative.
        depth_weights : numpy.ndarray
            The depth weights to use for depth weighting the leadfields. If None, no depth weighting is applied.
        apply_depth_weights : bool
            Whether to apply depth weights to the leadfields.

        Return
        ------
        x_hat : numpy.ndarray
            Source data matrix (sources, time)
        """
        n_chans, n_dipoles = self.leadfield.shape
        y.shape[1]

        leadfields = self.leadfields
        n_orders = len(self.leadfields)
        if k == "auto":
            k = n_chans

        C = y @ y.T
        U, D, _ = np.linalg.svd(C, full_matrices=False)

        # Estimate number of components
        if not isinstance(n, int):
            n_comp = self.estimate_n_sources(y, method=n)
        else:
            n_comp = n

        # MUSIC TYPE
        if covariance_type == "MUSIC":
            Q = np.identity(n_chans)

            Us = U[:, :n_comp]

            # MUSIC subspace-based Covariance
            C = Us @ Us.T

        elif covariance_type == "AP":  # Normal covariance
            mu = 0  # 1e-3
            C = y @ y.T + mu * np.trace(np.matmul(y, y.T)) * np.eye(
                n_chans
            )  # Array Covariance matrix
        else:
            msg = f"covariance_type must be MUSIC or AP but is {covariance_type}"
            raise AttributeError(msg)

        S_AP = []
        # Initialization:  search the 1st source location over the entire
        # dipoles topographies space
        ap_val1 = np.zeros((n_orders, n_dipoles))
        for nn in range(n_orders):
            L = leadfields[nn]
            # norm_1 = np.diag(L.T @ C @ L)
            norm_1 = np.einsum("ij,jk,ki->i", L.T, C, L)
            norm_2 = np.diag(
                L.T @ L
            )  # not necessary since leadfields were L2-normalized before
            ap_val1[nn, :] = norm_1 / norm_2
        self.mu = ap_val1
        best_order, best_dipole = np.unravel_index(np.argmax(ap_val1), ap_val1.shape)
        S_AP.append([best_order, best_dipole])

        # (b) Now, add one source at a time
        for _ii in range(1, n_comp):
            ap_val2 = np.zeros((n_orders, n_dipoles))
            # Compose current leadfield components
            A = np.stack(
                [leadfields[order][:, dipole] for order, dipole in S_AP], axis=1
            )
            P_A = A @ np.linalg.pinv(A.T @ A) @ A.T
            Q = np.identity(P_A.shape[0]) - P_A
            # QCQ = Q @ C @ Q
            # QCQ = np.dot(Q, C).dot(Q)
            QC = Q @ C
            for nn in range(n_orders):
                L = leadfields[nn]
                # QL = np.dot(Q, L)
                # ap_val2[nn] = np.sum(L * QCQ.dot(L), axis=0) / np.sum(L * QL, axis=0)  # fast, but unstable
                ap_val2[nn] = np.sum(L * (QC @ (Q @ L)), axis=0) / np.sum(
                    L * (Q @ L), axis=0
                )  # fast and stable

                # Old, slow
                # upper = np.diag(L.T @ QCQ @ L)
                # lower = np.diag(L.T @ Q @ L)
                # ap_val2[nn] = upper / lower

            # Select the best candidate, unless it is already in the set
            # best_order, best_dipole = np.unravel_index(np.argmax(ap_val2), ap_val2.shape)
            # S_AP.append( [best_order, best_dipole] )
            select_idx = -1
            while True:
                best_order, best_dipole = np.unravel_index(
                    np.argsort(ap_val2.flatten())[select_idx], ap_val2.shape
                )
                candidate = [best_order, best_dipole]
                if candidate not in S_AP:
                    S_AP.append([best_order, best_dipole])
                    break
                logger.debug("rerolled AP candidate")
                select_idx -= 1

            if len(S_AP) != len(set(tuple(row) for row in S_AP)):
                logger.warning("Found duplicate candidates in AP!!!!")

        # Phase 2: refinement
        S_AP_2 = deepcopy(S_AP)
        if len(S_AP) > 1 and refine_solution:
            # best_vals = np.zeros(n_comp)
            for _iter in range(max_iter):
                S_AP_2_Prev = deepcopy(S_AP_2)
                for q in range(len(S_AP)):
                    S_AP_TMP = S_AP_2.copy()
                    S_AP_TMP.pop(q)

                    A = np.stack(
                        [leadfields[order][:, dipole] for order, dipole in S_AP_TMP],
                        axis=1,
                    )
                    P_A = A @ np.linalg.pinv(A.T @ A) @ A.T
                    Q = np.identity(P_A.shape[0]) - P_A
                    # QCQ = np.dot(Q, C).dot(Q)
                    QC = Q @ C
                    ap_val2 = np.zeros((n_orders, n_dipoles))
                    for nn in range(n_orders):
                        # New, fast
                        L = leadfields[nn]
                        # QL = np.dot(Q, L)

                        # ap_val2[nn] = np.sum(L * QCQ.dot(L), axis=0) / np.sum(L * QL, axis=0)  # fast, but unstable
                        ap_val2[nn] = np.sum(L * (QC @ (Q @ L)), axis=0) / np.sum(
                            L * (Q @ L), axis=0
                        )  # fast and stable

                        # Old, slow
                        # upper = np.diag(L.T @ QCQ @ L)
                        # lower = np.diag(L.T @ Q @ L)
                        # ap_val2[nn] = upper / lower

                    # Select the best candidate, unless it is already in the set
                    # best_order, best_dipole = np.unravel_index(np.argmax(ap_val2), ap_val2.shape)
                    # S_AP_2[q] = [best_order, best_dipole]
                    select_idx = -1
                    while True:
                        best_order, best_dipole = np.unravel_index(
                            np.argsort(ap_val2.flatten())[select_idx], ap_val2.shape
                        )
                        candidate = [best_order, best_dipole]
                        if (
                            candidate not in S_AP_2[:q]
                            and candidate not in S_AP_2[q + 1 :]
                        ):
                            S_AP_2[q] = [best_order, best_dipole]
                            break
                        logger.debug("rerolled AP candidate")
                        select_idx -= 1

                    # S_AP_2[q] = [best_order, best_dipole]
                    if len(S_AP_2) != len(set(tuple(row) for row in S_AP_2)):
                        logger.warning("Found duplicate candidates in AP refinement")
                    # print(f"refinement: adding new value {best_val} at idx {best_dipole}, best_order {best_order}")
                    # best_vals[q] = best_val

                if S_AP_2 == S_AP_2_Prev:  # and iter>0:
                    break
        self.initial_candidates = S_AP
        self.candidates = S_AP_2
        source_covariance = np.sum(
            [
                np.squeeze(self.gradients[order][dipole].toarray())
                for order, dipole in S_AP_2
            ],
            axis=0,
        )

        # Prior-Cov based version 2: Use the selected smooth patches as source covariance priors
        nonzero = np.where(source_covariance != 0)[0]
        inverse_operator = np.zeros((n_dipoles, n_chans))

        source_covariance = csr_matrix(np.diag(source_covariance[nonzero]))
        L = self.leadfield[:, nonzero]
        if depth_weights is not None:
            L = L * depth_weights[nonzero]

        # Version 8: Lower rank MNE
        n = len(S_AP_2)
        source_covariance = np.identity(n)
        L = np.stack([leadfields[order][:, dipole] for order, dipole in S_AP_2], axis=1)
        gradients = np.stack(
            [self.gradients[order][dipole].toarray() for order, dipole in S_AP_2],
            axis=1,
        )[0]
        # print(source_covariance.shape, L.shape, gradients.shape)
        inverse_operator = (
            gradients.T
            @ source_covariance
            @ L.T
            @ np.linalg.pinv(L @ source_covariance @ L.T)
        )
        # print(inverse_operator.shape)
        return inverse_operator

    def prepare_flex(self):
        """Create the dictionary of increasingly smooth sources unless
        self.n_orders==0. Flexibly selects diffusion parameter, too.

        Parameters
        ----------

        """

        n_dipoles = self.leadfield.shape[1]
        I = np.identity(n_dipoles)
        self.leadfields = [
            deepcopy(self.leadfield),
        ]
        self.gradients = [
            csr_matrix(I),
        ]

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

        if self.diffusion_parameter == "auto":
            alphas = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175]
            smoothing_operators = [csr_matrix(I - alpha * LL) for alpha in alphas]
        else:
            smoothing_operators = [
                csr_matrix(I - self.diffusion_parameter * LL),
            ]

        for smoothing_operator in smoothing_operators:
            for i in range(self.n_orders):
                smoothing_operator_i = (
                    smoothing_operator ** (i + 1)
                )  # csr_matrix(np.linalg.matrix_power(smoothing_operator.toarray(), i+1))
                new_leadfield = self.leadfields[0] @ smoothing_operator_i
                new_gradient = self.gradients[0] @ smoothing_operator_i

                # Scaling? Not sure...
                if self.scale_leadfield:
                    # new_leadfield -= new_leadfield.mean(axis=0)
                    new_leadfield /= np.linalg.norm(new_leadfield, axis=0)

                self.leadfields.append(new_leadfield)
                self.gradients.append(new_gradient)
        # scale and transform gradients
        for i in range(len(self.gradients)):
            row_sums = (
                self.gradients[i].sum(axis=1).ravel()
            )  # Compute the sum of each row
            scaling_factors = 1 / row_sums
            self.gradients[i] = csr_matrix(
                self.gradients[i].multiply(scaling_factors.reshape(-1, 1))
            )

        self.is_prepared = True
