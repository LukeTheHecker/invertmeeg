from copy import deepcopy

import mne
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

from ...util import find_corner, pos_from_forward
from ..base import BaseSolver, InverseOperator, SolverMeta


class SolverFLEXMUSIC_2(BaseSolver):
    """Class for the RAP Multiple Signal Classification with flexible extent
        estimation (FLEX-MUSIC).

    References
    ---------
    This method is of my own making (Lukas Hecker, 2022) and soon to be
    published.

    """

    meta = SolverMeta(
        acronym="FLEX-MUSIC2",
        full_name="FLEX-MUSIC 2",
        category="Subspace Methods",
        description=(
            "Experimental FLEX-MUSIC variant with iterative neighborhood expansion "
            "and optional distance-weighted patch growth."
        ),
        references=[
            "Lukas Hecker 2025, unpublished",
            "Mosher, J. C., & Leahy, R. M. (1999). Source localization using recursively applied and projected (RAP) MUSIC. IEEE Transactions on Signal Processing, 47(2), 332–340.",
            "Schmidt, R. O. (1986). Multiple emitter location and signal parameter estimation. IEEE Transactions on Antennas and Propagation, 34(3), 276–280.",
        ],
    )

    def __init__(self, name="FLEX-MUSIC 2", truncate=False, **kwargs):
        self.name = name
        self.truncate = truncate
        self.is_prepared = False
        return super().__init__(**kwargs)

    def make_inverse_operator(
        self,
        forward,
        mne_obj,
        *args,
        alpha="auto",
        n="enhanced",
        k="auto",
        stop_crit=0.95,
        distance_weighting=False,
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


        Return
        ------
        self : object returns itself for convenience
        """
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)

        data = self.unpack_data_obj(mne_obj)
        if not self.is_prepared:
            self.prepare_flex()

        inverse_operator = self.make_flex(
            data, n, k, stop_crit, self.truncate, distance_weighting=distance_weighting
        )

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name),
        ]
        return self

    def make_flex(self, y, n, k, stop_crit, truncate, distance_weighting=False):
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

        Return
        ------
        x_hat : numpy.ndarray
            Source data matrix (sources, time)
        """
        n_chans, n_dipoles = self.leadfield.shape
        y.shape[1]

        leadfield = self.leadfield
        # leadfield -= leadfield.mean(axis=0)

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
        self.n_comp = n_comp
        Us = U[:, :n_comp]

        source_covariance = np.zeros(n_dipoles)

        for i in range(k):
            Ps = Us @ Us.T
            selected_idc = []

            norm_1 = np.linalg.norm(Ps @ Q @ leadfield, axis=0)
            norm_2 = np.linalg.norm(Q @ leadfield, axis=0)
            mu = norm_1 / norm_2
            selected_idc.append(np.argmax(mu))
            current_max = np.max(mu)
            current_leadfield = leadfield[:, selected_idc[0]]
            component_strength = [
                1,
            ]

            while True:
                # Find all neighboring dipoles of current candidates
                neighbors = np.unique(
                    np.concatenate([self.members_ex[idx] for idx in selected_idc])
                )
                # Filter out candidates from neighbors
                for idx, n in reversed(list(enumerate(neighbors))):
                    if n in selected_idc:
                        neighbors = np.delete(neighbors, idx)

                dist = self.distances[neighbors, selected_idc[0]]

                # construct new candidate leadfields:
                if distance_weighting:
                    b = np.stack(
                        [
                            leadfield[:, n] / d + current_leadfield
                            for d, n in zip(dist, neighbors)
                        ],
                        axis=1,
                    )
                else:
                    b = np.stack(
                        [
                            leadfield[:, n] + current_leadfield
                            for d, n in zip(dist, neighbors)
                        ],
                        axis=1,
                    )

                norm_1 = np.linalg.norm(Ps @ Q @ b, axis=0)
                norm_2 = np.linalg.norm(Q @ b, axis=0)

                mu_b = norm_1 / norm_2
                max_mu_b = np.max(mu_b)

                if (max_mu_b - current_max) < 0.00:
                    # if max_mu_b / current_max < 1.0001:
                    break
                else:
                    new_idx = neighbors[np.argmax(mu_b)]
                    b_best = b[:, np.argmax(mu_b)]
                    current_max = max_mu_b
                    current_leadfield = b_best  # / np.linalg.norm(b_best)
                    if distance_weighting:
                        component_strength.append(1 / dist[np.argmax(mu_b)])
                    else:
                        component_strength.append(1)

                    selected_idc.append(new_idx)

            # Find the dipole/ patch with highest correlation with the residual

            if i > 0 and current_max < stop_crit:
                break
            selected_idc = np.array(selected_idc)
            current_cov = np.zeros(n_dipoles)
            current_cov[selected_idc] = np.array(component_strength)
            source_covariance += current_cov
            # source_covariance += np.squeeze(self.identity[selected_idc].sum(axis=0))
            current_leadfield /= np.linalg.norm(current_leadfield)
            if i == 0:
                B = current_leadfield[:, np.newaxis]
            else:
                B = np.hstack([B, current_leadfield[:, np.newaxis]])

            # B = B / np.linalg.norm(B, axis=0)
            Q = I - B @ np.linalg.pinv(B)
            # Q -= Q.mean(axis=0)
            C = Q @ Us

            U, D, _ = np.linalg.svd(C, full_matrices=False)
            # U -= U.mean(axis=0)

            # Truncate eigenvectors
            if truncate:
                Us = U[:, : n_comp - i]
            else:
                Us = U[:, :n_comp]

        # Prior-Cov based version 2: Use the selected smooth patches as source covariance priors
        # source_covariance = csr_matrix(np.diag(source_covariance))
        # L_s = self.leadfield @ source_covariance
        # L = self.leadfield
        # W = np.diag(np.linalg.norm(L, axis=0))
        # # print(source_covariance.shape, L.shape, W.shape)
        # inverse_operator = source_covariance @ np.linalg.inv(L_s.T @ L_s + W.T @ W) @ L_s.T

        Gamma = csr_matrix(np.diag(source_covariance))
        Gamma_LT = Gamma @ leadfield.T
        Sigma_y = leadfield @ Gamma_LT
        Sigma_y_inv = np.linalg.pinv(Sigma_y)
        inverse_operator = Gamma_LT @ Sigma_y_inv

        return inverse_operator

    def prepare_flex(self):
        """Create the dictionary of increasingly smooth sources unless self.n_orders==0.

        Parameters
        ----------


        """
        n_dipoles = self.leadfield.shape[1]
        self.identity = np.identity(n_dipoles)
        self.adjacency = mne.spatial_src_adjacency(
            self.forward["src"], verbose=0
        ).toarray()
        self.adjacency_ex = deepcopy(self.adjacency)
        np.fill_diagonal(self.adjacency_ex, 0)
        self.members_ex = [np.where(row)[0] for row in self.adjacency_ex]
        pos = pos_from_forward(self.forward, verbose=0)
        self.distances = cdist(pos, pos)
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
