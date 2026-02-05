import logging
from copy import deepcopy

import mne
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverAdaptiveAlternatingProjections(BaseSolver):
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
        acronym="AAP",
        full_name="Adaptive Alternating Projections",
        category="Subspace Methods",
        description=(
            "Adaptive variant of alternating-projection source localization with "
            "flexible-extent patch estimation."
        ),
        references=[
            "Lukas Hecker 2025, unpublished",
            "Adler, A., Wax, M., & Pantazis, D. (2022). Brain Source Localization by Alternating Projection. In 2022 IEEE 19th International Symposium on Biomedical Imaging (ISBI) (pp. 1–5). IEEE.",
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
        stop_crit=0.95,
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
        stop_crit : float
            Criterion to stop recursions. The lower, the more dipoles will be
            incorporated.
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

        self.diffusion_smoothing = (diffusion_smoothing,)
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
        stop_crit : float
            Criterion to stop recursions. The lower, the more dipoles will be
            incorporated.
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

        # leadfield -= leadfield.mean(axis=0)

        leadfields = self.leadfields
        n_orders = len(self.leadfields)
        if k == "auto":
            k = n_chans
        # Assert common average reference
        # y -= y.mean(axis=0)
        max_rank = 4
        min_change = 1.001

        # Compute Data Covariance
        if not isinstance(n, int):
            n_comp = self.estimate_n_sources(y, method=n)
        else:
            n_comp = n

        # MUSIC TYPE
        if covariance_type == "MUSIC":
            C = y @ y.T
            Q = np.identity(n_chans)
            U, D, _ = np.linalg.svd(C, full_matrices=False)

            Us = U[:, :n_comp]

            # MUSIC subspace-based Covariance
            C = Us @ Us.T

        elif covariance_type == "AP":  # Normal covariance
            mu = 0  # 1e-3
            C = y @ y.T + mu * np.trace(np.matmul(y, y.T)) * np.eye(
                y.shape[0]
            )  # Array Covariance matrix
        else:
            msg = f"covariance_type must be MUSIC or AP but is {covariance_type}"
            raise AttributeError(msg)

        S_AP = []
        # Initialization:  search the 1st source location over the entire
        # dipoles topographies space
        np.zeros((n_orders, n_dipoles))
        adjacency = mne.spatial_src_adjacency(self.forward["src"], verbose=0).toarray()
        L = leadfields[0]
        cluster_indices = []
        # initial_candidate
        norm_1 = np.einsum("ij,jk,ki->i", L.T, C, L)
        norm_2 = np.diag(
            L.T @ L
        )  # not necessary since leadfields were L2-normalized before
        result = norm_1 / norm_2

        last_power = np.max(result)
        cluster_indices.append(np.argmax(result))
        cluster_indices = np.array(cluster_indices)
        logger.debug(f"First index is {cluster_indices[0]} ({cluster_indices})")
        # print(f"adjacency={adjacency}")
        while True:
            new_pot_neighbors = list(
                set(
                    np.concatenate(
                        [np.where(adjacency[ci] == 1)[0] for ci in cluster_indices]
                    )
                )
            )
            new_pot_neighbors = np.setdiff1d(new_pot_neighbors, cluster_indices)
            traces = []
            for neighbor in new_pot_neighbors:
                # composite leafield of cluster_indices and neighbor
                L = np.hstack(
                    [
                        leadfields[0][:, cluster_indices],
                        leadfields[0][:, neighbor : neighbor + 1],
                    ]
                )
                U, _, _ = np.linalg.svd(L, full_matrices=False)
                U = U[:, :max_rank]
                norm_1 = np.einsum("ij,jk,ki->i", U.T, C, U)
                norm_2 = np.diag(
                    U.T @ U
                )  # not necessary since leadfields were L2-normalized before
                result = norm_1 / norm_2
                tr_current = np.sum(abs(result))
                traces.append(tr_current)

            if np.max(traces) >= last_power * min_change:
                cluster_indices = np.append(
                    cluster_indices, new_pot_neighbors[np.argmax(traces)]
                )
                last_power = np.sum(abs(result))
                logger.debug(
                    f"Added new dipole at {cluster_indices[-1]} because power was higher by {100 * np.max(traces) / last_power:.2f}  than {last_power}"
                )
            else:
                logger.debug("Stopping")
                break

        S_AP.append(cluster_indices)
        # print("S_AP: ", np.concatenate(S_AP))
        for ii in range(1, n_comp):
            A = np.stack(
                [leadfields[0][:, dipole] for dipole in np.concatenate(S_AP)], axis=1
            )
            P_A = A @ np.linalg.pinv(A.T @ A) @ A.T
            Q = np.identity(P_A.shape[0]) - P_A
            QC = Q @ C

            result = np.sum(U * (QC @ (Q @ U)), axis=0) / np.sum(
                U * (Q @ U), axis=0
            )  # fast and stable
            last_power = np.max(result)
            cluster_indices = np.array([np.argmax(result)])
            logger.debug(
                f"First index of source {ii} is {cluster_indices[0]} ({cluster_indices})"
            )

            while True:
                new_pot_neighbors = list(
                    set(
                        np.concatenate(
                            [np.where(adjacency[ci] == 1)[0] for ci in cluster_indices]
                        )
                    )
                )
                new_pot_neighbors = np.setdiff1d(new_pot_neighbors, cluster_indices)

                traces = []
                for neighbor in new_pot_neighbors:
                    # composite leafield of cluster_indices and neighbor
                    L = np.hstack(
                        [
                            leadfields[0][:, cluster_indices],
                            leadfields[0][:, neighbor : neighbor + 1],
                        ]
                    )
                    U, _, _ = np.linalg.svd(L, full_matrices=False)
                    U = U[:, :max_rank]

                    A = np.stack(
                        [leadfields[0][:, dipole] for dipole in np.concatenate(S_AP)],
                        axis=1,
                    )
                    P_A = A @ np.linalg.pinv(A.T @ A) @ A.T
                    Q = np.identity(P_A.shape[0]) - P_A
                    QC = Q @ C

                    result = np.sum(U * (QC @ (Q @ U)), axis=0) / np.sum(
                        U * (Q @ U), axis=0
                    )  # fast and stable

                    tr_current = np.sum(abs(result))
                    traces.append(tr_current)

                if np.max(traces) >= last_power * min_change:
                    cluster_indices = np.append(
                        cluster_indices, new_pot_neighbors[np.argmax(traces)]
                    )
                    last_power = np.sum(abs(result))
                    logger.debug(
                        f"Added new dipole at {cluster_indices[-1]} because power was higher by {100 * np.max(traces) / last_power:.2f}  than {last_power}"
                    )
                else:
                    logger.debug("Stopping")
                    break

            S_AP.append(cluster_indices)

        # refinement missing

        inverse_operator = np.zeros((n_dipoles, n_chans))

        nonzero = np.concatenate(S_AP)

        L = self.leadfield[:, nonzero]
        if depth_weights is not None:
            L = L * depth_weights[nonzero]

        # # Version 2: MNE vanilla
        inverse_operator[nonzero, :] = np.linalg.pinv(L.T @ L) @ L.T

        return inverse_operator

    # def prepare_flex(self):
    #     ''' Create the dictionary of increasingly smooth sources unless self.n_orders==0.

    #     Parameters
    #     ----------

    #     '''
    #     n_dipoles = self.leadfield.shape[1]
    #     I = np.identity(n_dipoles)
    #     if self.adjacency_type == "spatial":
    #         adjacency = mne.spatial_src_adjacency(self.forward['src'], verbose=0)
    #     else:
    #         adjacency = mne.spatial_dist_adjacency(self.forward['src'], self.adjacency_distance, verbose=None)

    #     LL = laplacian(adjacency)
    #     self.leadfields = [deepcopy(self.leadfield), ]
    #     self.gradients = [csr_matrix(I),]

    #     if self.diffusion_smoothing:
    #         smoothing_operator = csr_matrix(I - self.diffusion_parameter * LL)
    #     else:
    #         smoothing_operator = csr_matrix(abs(LL))

    #     for _ in range(self.n_orders):
    #         new_leadfield = self.leadfields[-1] @ smoothing_operator
    #         new_gradient = self.gradients[-1] @ smoothing_operator

    #         # Scaling? Not sure...
    #         if self.scale_leadfield:
    #             new_leadfield -= new_leadfield.mean(axis=0)
    #             new_leadfield /= np.linalg.norm(new_leadfield, axis=0)

    #         self.leadfields.append( new_leadfield )
    #         self.gradients.append( new_gradient )
    #     # scale and transform gradients
    #     for i in range(self.n_orders+1):
    #         # self.gradients[i] = self.gradients[i].toarray() / self.gradients[i].toarray().max(axis=0)
    #         row_max = self.gradients[i].max(axis=1).toarray().ravel()
    #         scaling_factors = 1 / row_max
    #         self.gradients[i] = csr_matrix(self.gradients[i].multiply(scaling_factors.reshape(-1, 1)))

    #     self.is_prepared = True

    def prepare_flex(self):
        """Create the dictionary of increasingly smooth sources unless
        self.n_orders==0. Flexibly selects diffusion parameter, too.

        Parameters
        ----------

        """
        n_dipoles = self.leadfield.shape[1]
        I = np.identity(n_dipoles)
        if self.adjacency_type == "spatial":
            adjacency = mne.spatial_src_adjacency(self.forward["src"], verbose=0)
        else:
            adjacency = mne.spatial_dist_adjacency(
                self.forward["src"], self.adjacency_distance, verbose=None
            )

        LL = laplacian(adjacency)
        self.leadfields = [
            deepcopy(self.leadfield),
        ]
        self.gradients = [
            csr_matrix(I),
        ]

        # if self.diffusion_smoothing:
        #     smoothing_operator = csr_matrix(I - self.diffusion_parameter * LL)
        # else:
        #     smoothing_operator = csr_matrix(abs(LL))
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
            # self.gradients[i] = self.gradients[i].toarray() / self.gradients[i].toarray().max(axis=0)
            # row_max = self.gradients[i].max(axis=1).toarray().ravel()
            # scaling_factors = 1 / row_max
            row_sums = (
                self.gradients[i].sum(axis=1).ravel()
            )  # Compute the sum of each row
            scaling_factors = 1 / row_sums
            self.gradients[i] = csr_matrix(
                self.gradients[i].multiply(scaling_factors.reshape(-1, 1))
            )

        self.is_prepared = True
