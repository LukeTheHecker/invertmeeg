from copy import deepcopy

import mne
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian

from ..base import BaseSolver, InverseOperator, SolverMeta


class SolverFLEXMUSIC(BaseSolver):
    """Class for the RAP Multiple Signal Classification with flexible extent
        estimation (FLEX-MUSIC, [1]).

    References
    ---------
    [1] Hecker, L., Tebartz van Elst, L., & Kornmeier, J. (2023). Source
    localization using recursively applied and projected MUSIC with flexible
    extent estimation. Frontiers in Neuroscience, 17, 1170862.

    """

    meta = SolverMeta(
        acronym="FLEX-MUSIC",
        full_name="Flexible Extent RAP/TRAP-MUSIC",
        category="Subspace Methods",
        description=(
            "Recursive MUSIC variant with flexible extent estimation. Iteratively "
            "selects sources (RAP/TRAP-MUSIC) while allowing spatially extended "
            "patches via flexible-order smoothing."
        ),
        references=[
            "Hecker, L., Tebartz van Elst, L., & Kornmeier, J. (2023). Source localization using recursively applied and projected MUSIC with flexible extent estimation. Frontiers in Neuroscience, 17, 1170862.",
            "Mosher, J. C., & Leahy, R. M. (1999). Source localization using recursively applied and projected (RAP) MUSIC. IEEE Transactions on Signal Processing, 47(2), 332–340.",
            "Aydin, Ü., Rampp, S., Wollbrink, A., Kugel, H., Cho, J.-H., Knösche, T. R., Grova, C., & Wolters, C. H. (2018). Zoomed MRI-guided TRAP-MUSIC (ZM-TRAP-MUSIC) for MEG/EEG source analysis in presurgical evaluation of epilepsy patients. NeuroImage: Clinical, 17, 566–575.",
            "Schmidt, R. O. (1986). Multiple emitter location and signal parameter estimation. IEEE Transactions on Antennas and Propagation, 34(3), 276–280.",
        ],
    )

    def __init__(self, name="FLEX-MUSIC", scale_leadfield=False, **kwargs):
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
        truncate=False,
        alpha="auto",
        n="enhanced",
        k="auto",
        stop_crit=0.95,
        refine_solution=False,
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
        n_orders : int
            Controls the maximum smoothness to pursue.
        truncate : bool
            If True: Truncate SVD's eigenvectors (like TRAP-MUSIC), otherwise
            don't (like RAP-MUSIC).
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
        self.truncate = truncate

        data = self.unpack_data_obj(mne_obj)

        if not self.is_prepared:
            self.prepare_flex()

        inverse_operator = self.make_flex(
            data,
            n,
            k,
            stop_crit,
            truncate,
            refine_solution=refine_solution,
            max_iter=max_iter,
            depth_weights=depth_weights,
        )

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name),
        ]
        return self

    def make_flex(
        self,
        y,
        n,
        k,
        stop_crit,
        truncate,
        refine_solution=False,
        max_iter=1000,
        depth_weights=None,
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
        truncate : bool
            If True: Truncate SVD's eigenvectors (like TRAP-MUSIC), otherwise
            don't (like RAP-MUSIC).
        depth_weights : numpy.ndarray
            The depth weights to use for depth weighting the leadfields. If None, no depth weighting is applied.

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
        # Compute Data Covariance
        C = y @ y.T

        I = np.identity(n_chans)
        Q = np.identity(n_chans)
        U, D, _ = np.linalg.svd(C, full_matrices=False)

        if not isinstance(n, int):
            n_comp = self.estimate_n_sources(y, method=n)
        else:
            n_comp = n

        # print("n_comp: ", n_comp)
        Us = U[:, :n_comp]

        C_initial = Us @ Us.T
        source_covariance = np.zeros(n_dipoles)
        S_AP = []
        for i in range(k):
            Ps = Us @ Us.T
            PsQ = Ps @ Q

            mu = np.zeros((n_orders, n_dipoles))
            for nn in range(n_orders):
                norm_1 = np.linalg.norm(PsQ @ leadfields[nn], axis=0)
                norm_2 = np.linalg.norm(Q @ leadfields[nn], axis=0)
                # norm_1 = np.diag(leadfields[nn].T @ PsQ @ leadfields[nn])
                # norm_2 = np.diag(leadfields[nn].T @ Q @ leadfields[nn])
                mu[nn, :] = norm_1 / norm_2

            self.mu = mu
            # Find the dipole/ patch with highest correlation with the residual
            best_order, best_dipole = np.unravel_index(np.argmax(mu), mu.shape)

            if i > 0 and np.max(mu) < stop_crit:
                # print(f"\t break because mu is ", np.max(mu))
                break
            S_AP.append([best_order, best_dipole])

            # source_covariance += np.squeeze(self.gradients[best_order][best_dipole] * (1/np.sqrt(i+1)))
            # source_covariance += np.squeeze(self.gradients[best_order][best_dipole])

            if i == 0:
                B = leadfields[best_order][:, best_dipole][:, np.newaxis]
            else:
                B = np.hstack(
                    [B, leadfields[best_order][:, best_dipole][:, np.newaxis]]
                )

            Q = I - B @ np.linalg.pinv(B)
            C = Q @ Us
            U, D, _ = np.linalg.svd(C, full_matrices=False)
            # new_cov = Q @ C @ Q
            # U, D, _= np.linalg.svd(new_cov, full_matrices=False)

            # Truncate eigenvectors
            if truncate:
                Us = U[:, : n_comp - i]
            else:
                Us = U[:, :n_comp]
        # Phase 2: refinement
        C = C_initial
        S_AP_2 = deepcopy(S_AP)
        # print(S_AP)

        if len(S_AP) > 1 and refine_solution:
            # best_vals = np.zeros(n_comp)
            for iter in range(max_iter):
                S_AP_2_Prev = deepcopy(S_AP_2)
                for q in range(len(S_AP)):
                    S_AP_TMP = S_AP_2.copy()
                    S_AP_TMP.pop(q)

                    B = np.stack(
                        [leadfields[order][:, dipole] for order, dipole in S_AP_TMP],
                        axis=1,
                    )

                    # Q = I - B @ np.linalg.pinv(B)
                    # Ps = C_initial

                    P_A = B @ np.linalg.pinv(B.T @ B) @ B.T
                    Q = np.identity(P_A.shape[0]) - P_A

                    ap_val2 = np.zeros((n_orders, n_dipoles))
                    for nn in range(n_orders):
                        L = leadfields[nn]
                        upper = np.diag(L.T @ Q @ C @ Q @ L)
                        lower = np.diag(L.T @ Q @ L)
                        # upper = np.linalg.norm(Ps @ Q @ L, axis=0)
                        # lower = np.linalg.norm(Q @ L, axis=0)
                        ap_val2[nn] = upper / lower

                    best_order, best_dipole = np.unravel_index(
                        np.argmax(ap_val2), ap_val2.shape
                    )
                    # best_val = ap_val2.max()
                    S_AP_2[q] = [best_order, best_dipole]
                    # print(f"refinement: adding new value {best_val} at idx {best_dipole}, best_order {best_order}")
                    # best_vals[q] = best_val

                if iter > 0 and S_AP_2 == S_AP_2_Prev:
                    break

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

        # Version 7: Standard Minimum Norm Estimate with Source Covariance (MNE PDF p.122)
        # alpha = np.trace(L.T @ L) / np.trace(L @ source_covariance @ L.T) #
        # source_covariance *= alpha
        # inverse_operator[nonzero, :] = source_covariance @ L.T @ np.linalg.pinv(L @ source_covariance @ L.T)

        # Version 8: Lower rank MNE
        n = len(S_AP_2)
        source_covariance = np.identity(n)
        L = np.stack([leadfields[order][:, dipole] for order, dipole in S_AP_2], axis=1)
        gradients = np.stack(
            [self.gradients[order][dipole].toarray() for order, dipole in S_AP_2],
            axis=1,
        )[0]
        inverse_operator = (
            gradients.T
            @ source_covariance
            @ L.T
            @ np.linalg.pinv(L @ source_covariance @ L.T)
        )

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
