import logging
from copy import deepcopy
from typing import Optional

import mne
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian

from ...util import find_corner
from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverGeneralizedIterative(BaseSolver):
    """Class that generalizes iterative solutions like RAP-MUSIC, AP and SSM
        inverse solution with flexible extent estimation (FLEX-AP).

    References
    ---------
    [1] Wax, M., & Adler, A. (2021). Direction of arrival estimation in the
    presence of model errors by signal subspace matching. Signal Processing,
    181, 107900. [2] TBD.

    """

    meta = SolverMeta(
        acronym="GI",
        full_name="Generalized Iterative Subspace Solver",
        category="Subspace Methods",
        description=(
            "Generalized iterative framework covering RAP-MUSIC, alternating "
            "projections (AP), and signal subspace matching (SSM), with optional "
            "flexible-extent patch estimation."
        ),
        references=[
            "Mosher, J. C., & Leahy, R. M. (1999). Source localization using recursively applied and projected (RAP) MUSIC. IEEE Transactions on Signal Processing, 47(2), 332–340.",
            "Adler, A., Wax, M., & Pantazis, D. (2022). Brain Source Localization by Alternating Projection. In 2022 IEEE 19th International Symposium on Biomedical Imaging (ISBI) (pp. 1–5). IEEE.",
            "Wax, M., & Adler, A. (2021). Direction of arrival estimation in the presence of model errors by signal subspace matching. Signal Processing, 181, 107900.",
            "Hecker, L., Tebartz van Elst, L., & Kornmeier, J. (2023). Source localization using recursively applied and projected MUSIC with flexible extent estimation. Frontiers in Neuroscience, 17, 1170862.",
        ],
    )

    def __init__(
        self, name="Flexible Signal Subspace Matching", scale_leadfield=False, **kwargs
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
        inverse_type="SSM",
        n_orders=3,
        alpha="auto",
        n="enhanced",
        k="auto",
        refine_solution=True,
        max_iter=5,
        diffusion_smoothing=True,
        diffusion_parameter=0.1,
        adjacency_type="spatial",
        adjacency_distance=3e-3,
        lambda_reg1=0.001,
        lambda_reg2=0.0001,
        lambda_reg3=0.0,
        **kwargs,
    ):
        """Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        inverse_type : str
            The type of inverse solution to use. "SSM" -> Signal Subspace Matching.
            "AP" -> Alternating Projections. "RAP" -> Recursively Applied and Projected MUSIC.
        n_orders : int
            Controls the maximum smoothness to pursue.
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
        self.inverse_type = inverse_type
        data = self.unpack_data_obj(mne_obj)

        if not self.is_prepared:
            self.prepare_flex()

        if inverse_type == "SSM":
            self.get_source = self.get_source_ssm
            self.get_covariance = self.get_covariance_ssm
        elif inverse_type == "AP":
            self.get_source = self.get_source_ap
            self.get_covariance = self.get_covariance_ap
        elif inverse_type == "RAP":
            self.get_source = self.get_source_rap
            self.get_covariance = self.get_covariance_rap
        else:
            self.get_source = None
            raise AttributeError(f"Unknown inverse_type: {inverse_type}")

        inverse_operator = self.make_iterative_solution(
            data,
            n,
            k,
            max_iter=max_iter,
            refine_solution=refine_solution,
            lambda_reg1=lambda_reg1,
            lambda_reg2=lambda_reg2,
            lambda_reg3=lambda_reg3,
        )
        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name),
        ]

        return self

    def make_iterative_solution(
        self,
        Y,
        n,
        k,
        refine_solution=True,
        max_iter=5,
        lambda_reg1=0.001,
        lambda_reg2=0.0001,
        lambda_reg3=0.0,
    ):
        """Create the iterative inverse solution to the EEG data.

        Parameters
        ----------
        Y : numpy.ndarray
            EEG data matrix (channels, time)
        n : int/ str
            Number of eigenvectors to use or "auto" for l-curve method.
        k : int
            Number of recursions.
        refine_solution : bool
            If True: Re-visit each selected candidate and check if there is a
            better alternative.
        lambda_reg1 : float
            Regularization parameter for the projection matrix.
        lambda_reg2 : float
            Regularization parameter for the source covariance matrix.
        lambda_reg3 : float
            Regularization parameter for the source covariance matrix.

        Return
        ------
        x_hat : numpy.ndarray
            Source data matrix (sources, time)
        """
        n_chans, n_dipoles = self.leadfield.shape
        Y.shape[1]

        leadfields = self.leadfields
        len(self.leadfields)

        if k == "auto":
            k = n_chans

        # Determine the number of sources
        if not isinstance(n, int):
            n_comp = self.estimate_n_sources(Y, method=n)
        else:
            n_comp = n

        P_Y = self.get_covariance(Y, n_comp, lambda_reg1=lambda_reg1)
        Q = np.eye(n_chans)  # np.zeros((n_chans, n_chans))

        candidates = []
        A_q = []

        # Initial source location
        expression = self.get_source(P_Y, Q, leadfields, lambda_reg=lambda_reg3)
        order, dipole = np.unravel_index(np.nanargmax(expression), expression.shape)
        candidates.append([order, dipole])
        logger.debug("%s %s", order, dipole)

        # Now, add one source at a time
        for _ in range(1, n_comp):
            order, location = candidates[-1]
            A_q.append(leadfields[order][:, location])
            Q = self.compute_projection_matrix(A_q, lambda_reg=lambda_reg2)
            expression = self.get_source(
                P_Y, Q, leadfields, candidates, lambda_reg=lambda_reg3
            )
            order, dipole = np.unravel_index(np.nanargmax(expression), expression.shape)
            candidates.append([order, dipole])

        A_q.append(leadfields[order][:, dipole])

        # Phase 2: refinement
        candidates_2 = deepcopy(candidates)
        if len(candidates_2) > 1 and refine_solution:
            candidates_2_prev = deepcopy(candidates_2)
            for _j in range(max_iter):
                A_q_j = A_q.copy()
                for qq in range(n_comp):
                    A_temp = np.delete(A_q_j, qq, axis=0)  # delete the current source
                    qq_temp = np.delete(
                        candidates_2, qq, axis=0
                    )  # delete the current source
                    Q = self.compute_projection_matrix(A_temp, lambda_reg=lambda_reg2)
                    expression = self.get_source(
                        P_Y, Q, leadfields, qq_temp, lambda_reg=lambda_reg3
                    )
                    order, dipole = np.unravel_index(
                        np.nanargmax(expression), expression.shape
                    )
                    candidates_2[qq] = [order, dipole]
                    A_q_j[qq] = leadfields[candidates_2[qq][0]][:, candidates_2[qq][1]]

                if candidates_2 == candidates_2_prev:
                    # print(f"No change after {j+1} iterations")
                    break
                # else:
                #     print(candidates_2_prev, " ==> ", candidates_2)
                candidates_2_prev = deepcopy(candidates_2)

        self.candidates = candidates_2

        # Low-rank minimum norm solution
        source_covariance = np.identity(n_comp)
        L = np.stack(
            [leadfields[order][:, dipole] for order, dipole in candidates_2], axis=1
        )
        gradients = np.stack(
            [self.gradients[order][dipole].toarray() for order, dipole in candidates_2],
            axis=1,
        )[0]
        inverse_operator = (
            gradients.T
            @ source_covariance
            @ L.T
            @ np.linalg.pinv(L @ source_covariance @ L.T)
        )

        return inverse_operator

    def estimate_comps(self, Y, n):
        """Estimate the number of sources to use.

        Parameters
        ----------
        Y : numpy.ndarray
            The data matrix.
        n : str
            The method to use. "L" -> L-curve method. "drop" -> based on eigenvalue drop-off.
            "auto" -> Combine L and drop method. "mean" -> Selects the eigenvalues that are larger than the mean of all eigs.

        Return
        ------
        n_comp : int
            The number of sources to use.
        """
        # Use the base class method for estimating number of sources
        return self.estimate_n_sources(Y, method=n)

    @staticmethod
    def get_covariance_ssm(Y, *args, lambda_reg1=0.001, **kwargs):
        """Compute the source covariance matrix for the SSM algorithm."""
        n_time = Y.shape[1]

        M_Y = Y.T @ Y
        YY = M_Y + lambda_reg1 * np.trace(M_Y) * np.eye(n_time)
        P_Y = (Y @ np.linalg.inv(YY)) @ Y.T
        return P_Y

    @staticmethod
    def get_covariance_rap(Y, n_comp, *args, lambda_reg1=0.001, **kwargs):
        """Compute the source covariance matrix for the SSM algorithm."""
        n_chans, n_time = Y.shape[1]

        C = Y @ Y.T
        U, _, _ = np.linalg.svd(C, full_matrices=False)
        Us = U[:, :n_comp]

        # MUSIC subspace-based Covariance
        P_Y = Us @ Us.T

        return P_Y

    @staticmethod
    def get_covariance_ap(Y, *args, lambda_reg1=0.001, **kwargs):
        """Compute the source covariance matrix for the SSM algorithm."""
        n_chans, n_time = Y.shape
        C = Y @ Y.T
        P_Y = C + lambda_reg1 * np.trace(C) * np.eye(n_chans)
        return P_Y

    @staticmethod
    def get_source_ssm(
        P_Y: np.ndarray,
        Q: np.ndarray,
        leadfields: list,
        q_ignore: Optional[list] = None,
        lambda_reg=0.0,
    ):
        """Compute the source with the highest AP value.
        Parameters
        ----------
        P_Y : numpy.ndarray
            The projection matrix of the data covariance.
        Q : numpy.ndarray
            The out-projection matrix of the source covariance.
        leadfields : list
            The list of leadfields.
        q_ignore : list
            List of sources to ignore (e.g., already selected sources).
        lambda_reg : float
            Regularization parameter to enhance stability.
        """
        # print(f"Check within function")
        # print(P_Y.shape, P_Y)
        # print(Q.shape, Q)
        # print(leadfields[0].shape, leadfields[0])
        # print(q_ignore, lambda_reg)

        if q_ignore is None:
            q_ignore = []
        n_dipoles = leadfields[0].shape[1]
        n_orders = len(leadfields)

        P_Y_T_P_Y = P_Y.T @ P_Y

        expression = np.zeros((n_orders, n_dipoles))
        for jj in range(n_orders):
            a_s = Q @ leadfields[jj]

            # Fast
            upper = np.einsum("ij,ij->j", a_s, P_Y_T_P_Y @ a_s)
            lower = np.einsum("ij,ij->j", a_s, a_s) + lambda_reg
            expression[jj, :] = upper / lower

        if len(q_ignore) > 0:
            for order, dipole in q_ignore:
                expression[order, dipole] = np.nan

        return expression

    @staticmethod
    def get_source_rap(
        P_Y: np.ndarray,
        Q: np.ndarray,
        leadfields: list,
        q_ignore: Optional[list] = None,
        lambda_reg=0.0,
    ):
        if q_ignore is None:
            q_ignore = []
        n_orders = len(leadfields)
        n_chans, n_dipoles = leadfields[0].shape
        P_YQ = P_Y @ Q
        expression = np.zeros((n_orders, n_dipoles))
        for jj in range(n_orders):
            L = leadfields[jj]
            norm_1 = np.linalg.norm(P_YQ @ L, axis=0)
            norm_2 = np.linalg.norm(Q @ L, axis=0)
            expression[jj, :] = norm_1 / norm_2

        return expression

    @staticmethod
    def get_source_ap(C, Q, leadfields, q_ignore: Optional[list] = None, **kwargs):
        """Compute the source with the highest AP value.
        Parameters
        ----------
        C : numpy.ndarray
            The data covariance matrix.
        Q : numpy.ndarray
            The out-projection matrix of the source covariance.
        leadfields : list
            The list of leadfields.
        q_ignore : list
            List of sources to ignore (e.g., already selected sources).

        Returns
        -------
        order : int
            The order of the selected source.
        dipole : int
            The dipole index of the selected source.

        """
        # print(Q)
        # print()
        # print(C)

        if q_ignore is None:
            q_ignore = []
        n_dipoles = leadfields[0].shape[1]
        n_orders = len(leadfields)
        QC = Q @ C

        expression = np.zeros((n_orders, n_dipoles))
        for jj in range(n_orders):
            L = leadfields[jj]
            expression[jj, :] = np.sum(L * (QC @ (Q @ L)), axis=0) / np.sum(
                L * (Q @ L), axis=0
            )  # fast and stable

        if len(q_ignore) > 0:
            for order, dipole in q_ignore:
                expression[order, dipole] = np.nan

        return expression

    @staticmethod
    def compute_projection_matrix(A_q, lambda_reg=0.0001):
        """Compute the out-projection matrix Q."""

        A_q = np.stack(A_q, axis=1)
        M_A = A_q.T @ A_q
        AA = M_A + lambda_reg * np.trace(M_A) * np.eye(M_A.shape[0])
        P_A = (A_q @ np.linalg.pinv(AA)) @ A_q.T
        Q = np.eye(P_A.shape[0]) - P_A
        return Q

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

    @staticmethod
    def get_comps_L(D):
        # L-curve method
        iters = np.arange(len(D))
        n_comp_L = find_corner(deepcopy(iters), deepcopy(D))
        return n_comp_L

    @staticmethod
    def get_comps_drop(D):
        D_ = D / D.max()
        n_comp_drop = np.where(abs(np.diff(D_)) < 0.001)[0]

        if len(n_comp_drop) > 0:
            n_comp_drop = n_comp_drop[0] + 1
        else:
            n_comp_drop = 1
        return n_comp_drop
