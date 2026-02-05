import logging
from copy import deepcopy
from typing import Optional

import mne
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverSignalSubspaceMatching(BaseSolver):
    """Class for the Signal Subspace Matching inverse solution with flexible
        extent estimation (FLEX-AP). This approach combines the SSM-approach by
        Adler et al. [1] with dipoles with flexible extents, e.g., FLEX-MUSIC
        (Hecker 2023, unpublished, [2]).

    References
    ---------
    [1] Wax, M., & Adler, A. (2021). Direction of arrival estimation in the
    presence of model errors by signal subspace matching. Signal Processing,
    181, 107900.
    [2] TBD.

    """

    meta = SolverMeta(
        acronym="SSM",
        full_name="Signal Subspace Matching",
        category="Subspace Methods",
        description=(
            "Signal subspace matching (SSM) for robust source localization under "
            "model mismatch, extended with flexible-extent patch estimation."
        ),
        references=[
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
        n_orders=3,
        alpha="auto",
        n="enhanced",
        refine_solution=True,
        max_iter=5,
        diffusion_smoothing=True,
        diffusion_parameter=0.1,
        adjacency_type="spatial",
        adjacency_distance=3e-3,
        lambda_reg1=0.001,
        lambda_reg2=0.0001,
        lambda_reg3=0.0,
        adaptive_reg=False,
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
        inverse_operator = self.make_ssm(
            data,
            n,
            max_iter=max_iter,
            refine_solution=refine_solution,
            lambda_reg1=lambda_reg1,
            lambda_reg2=lambda_reg2,
            lambda_reg3=lambda_reg3,
            adaptive_reg=adaptive_reg,
        )
        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name),
        ]
        return self

    def make_ssm(
        self,
        Y,
        n,
        refine_solution=True,
        max_iter=5,
        lambda_reg1=0.001,
        lambda_reg2=0.0001,
        lambda_reg3=0.0,
        adaptive_reg=False,
    ):
        """Create the FLEX-SSM inverse solution to the EEG data.

        Parameters
        ----------
        Y : numpy.ndarray
            EEG data matrix (channels, time)
        n : int/ str
            Number of eigenvectors to use or "auto" for l-curve method.
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
        n_time = Y.shape[1]

        leadfields = self.leadfields
        len(self.leadfields)

        # Determine the number of sources
        if isinstance(n, str):
            # Use base class method for estimating number of sources
            n_comp = self.estimate_n_sources(Y, method=n)
        else:
            n_comp = deepcopy(n)
        # print("n_comp: ", n_comp)

        channel_types = self.forward["info"].get_channel_types()
        for ch_type in set(channel_types):
            selection = np.where(np.array(channel_types) == ch_type)[0]
            C = Y[selection, :] @ Y[selection, :].T
            scaler = np.sqrt(np.trace(C)) / C.shape[0]
            # scaler = np.std(Y[selection, :])
            Y[selection, :] /= scaler

        # Old (erroneous for correlated noise)
        M_Y = Y.T @ Y
        if adaptive_reg:
            YY = M_Y + lambda_reg1 * (50 / n_time) * np.trace(M_Y) * np.eye(n_time)
        else:
            YY = M_Y + lambda_reg1 * np.trace(M_Y) * np.eye(n_time)
            # print("reg_trace = ", np.trace(M_Y) )
        P_Y = (Y @ np.linalg.inv(YY)) @ Y.T
        C = P_Y.T @ P_Y

        P_A = np.zeros((n_chans, n_chans))

        S_SSM = []
        A_q = []
        self.maps = []
        # Initial source location
        S_SSM.append(self.get_source_ssm(C, P_A, leadfields, lambda_reg=lambda_reg3))
        # Now, add one source at a time
        for _ in range(1, n_comp):
            order, location = S_SSM[-1]
            A_q.append(leadfields[order][:, location])
            P_A = self.compute_projection_matrix(A_q, lambda_reg=lambda_reg2)
            S_SSM.append(
                self.get_source_ssm(C, P_A, leadfields, S_SSM, lambda_reg=lambda_reg3)
            )

        A_q.append(leadfields[S_SSM[-1][0]][:, S_SSM[-1][1]])

        # Phase 2: refinement
        S_SSM_2 = deepcopy(S_SSM)
        if len(S_SSM_2) > 1 and refine_solution:
            S_SSM_prev = deepcopy(S_SSM_2)
            for j in range(max_iter):
                A_q_j = A_q.copy()
                for qq in range(n_comp):
                    A_temp = np.delete(A_q_j, qq, axis=0)  # delete the current source
                    qq_temp = np.delete(
                        S_SSM_2, qq, axis=0
                    )  # delete the current source
                    P_A = self.compute_projection_matrix(A_temp, lambda_reg=lambda_reg2)
                    S_SSM_2[qq] = self.get_source_ssm(
                        C, P_A, leadfields, qq_temp, lambda_reg=lambda_reg3
                    )
                    A_q_j[qq] = leadfields[S_SSM_2[qq][0]][:, S_SSM_2[qq][1]]

                if S_SSM_2 == S_SSM_prev:
                    logger.debug(f"No change after {j + 1} iterations")
                    break

                S_SSM_prev = deepcopy(S_SSM_2)
        self.candidates = S_SSM_2

        # Low-rank minimum norm solution
        source_covariance = np.identity(n_comp)
        L = np.stack(
            [leadfields[order][:, dipole] for order, dipole in S_SSM_2], axis=1
        )
        gradients = np.stack(
            [self.gradients[order][dipole].toarray() for order, dipole in S_SSM_2],
            axis=1,
        )[0]
        inverse_operator = (
            gradients.T
            @ source_covariance
            @ L.T
            @ np.linalg.pinv(L @ source_covariance @ L.T)
        )

        return inverse_operator

    # @staticmethod
    def get_source_ssm(
        self,
        C: np.ndarray,
        P_A: np.ndarray,
        leadfields: list,
        q_ignore: Optional[list] = None,
        lambda_reg=0.0,
    ):
        """Compute the source with the highest AP value.
        Parameters
        ----------
        P_Y : numpy.ndarray
            The projection matrix of the data covariance.
        P_A : numpy.ndarray
            The projection matrix of the source covariance.
        leadfields : list
            The list of leadfields.
        q_ignore : list
            List of sources to ignore (e.g., already selected sources).
        lambda_reg : float
            Regularization parameter to enhance stability. This is not within the original SSM algorithm.
        """
        if q_ignore is None:
            q_ignore = []
        n_dipoles = leadfields[0].shape[1]
        n_orders = len(leadfields)

        # C = P_Y.T @ P_Y
        R = np.eye(P_A.shape[0]) - P_A

        expression = np.zeros((n_orders, n_dipoles))
        for jj in range(n_orders):
            a_s = R @ leadfields[jj]

            # Fast
            upper = np.einsum("ij,ij->j", a_s, C @ a_s)
            lower = np.einsum("ij,ij->j", a_s, a_s) + lambda_reg
            expression[jj, :] = upper / lower

            # # Slow
            # upper = a_s.T @ C @ a_s
            # lower = a_s.T @ a_s + np.eye(n_dipoles) * lambda_reg
            # # print(upper.shape, lower.shape)
            # expression[jj, :] = np.diag(upper @ np.linalg.inv(lower))

        self.maps.append(expression)
        if len(q_ignore) > 0:
            for order, dipole in q_ignore:
                expression[order, dipole] = np.nan
        order, dipole = np.unravel_index(np.nanargmax(expression), expression.shape)
        return order, dipole

    @staticmethod
    def compute_projection_matrix(A_q, lambda_reg=0.0001):
        """Compute the projection matrix for the SSM algorithm."""
        A_q = np.stack(A_q, axis=1)
        M_A = A_q.T @ A_q
        AA = M_A + lambda_reg * np.trace(M_A) * np.eye(M_A.shape[0])
        # or adjusted
        # AA = M_A + lambda_reg * np.mean(np.diag(M_A)) * np.eye(M_A.shape[0])

        P_A = (A_q @ np.linalg.inv(AA)) @ A_q.T
        return P_A

    # @staticmethod
    # def compute_projection_matrix(A_q, lambda_reg=0.0001):
    #     ''' Compute the projection matrix for the SSM algorithm with adaptive regularization.'''
    #     print("new")
    #     A_q = np.stack(A_q, axis=1)
    #     M_A = A_q.T @ A_q
    #     eigenvalues, _ = np.linalg.eigh(M_A)
    #     lambda_adaptive = lambda_reg * np.mean(eigenvalues)
    #     AA = M_A + lambda_adaptive * np.eye(M_A.shape[0])
    #     P_A = (A_q @ np.linalg.inv(AA)) @ A_q.T
    #     return P_A

    # @staticmethod
    # def compute_projection_matrix(A_q, lambda_reg=0.0001):
    #     ''' Compute the projection matrix for the SSM algorithm with scaled regularization.'''
    #     A_q = np.stack(A_q, axis=1)
    #     M_A = A_q.T @ A_q
    #     frobenius_norm = np.linalg.norm(M_A, 'fro')
    #     lambda_scaled = lambda_reg * frobenius_norm
    #     AA = M_A + lambda_scaled * np.eye(M_A.shape[0])
    #     P_A = (A_q @ np.linalg.inv(AA)) @ A_q.T
    #     return P_A

    # @staticmethod
    # def compute_projection_matrix(A_q, lambda_reg=0.0001):
    #     ''' Compute the projection matrix for the SSM algorithm with statistical property-based regularization.'''
    #     A_q = np.stack(A_q, axis=1)
    #     M_A = A_q.T @ A_q
    #     std_deviation = np.std(A_q)
    #     lambda_stat = lambda_reg * std_deviation
    #     AA = M_A + lambda_stat * np.eye(M_A.shape[0])
    #     P_A = (A_q @ np.linalg.inv(AA)) @ A_q.T
    #     return P_A

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
