import mne
import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta


class SolverMUSIC(BaseSolver):
    """Class for the Multiple Signal Classification (MUSIC) inverse solution
        [1].

    References
    ----------
    [1] Baillet, S., Mosher, J. C., & Leahy, R. M. (2001). Electromagnetic brain
    mapping. IEEE Signal processing magazine, 18(6), 14-30.

    """

    meta = SolverMeta(
        acronym="MUSIC",
        full_name="Multiple Signal Classification",
        category="Subspace Methods",
        description=(
            "Subspace-based dipole localization using the MUSIC pseudospectrum. "
            "Estimates a signal subspace from the data covariance and scores each "
            "candidate leadfield by its projection onto that subspace."
        ),
        references=[
            "Schmidt, R. O. (1986). Multiple emitter location and signal parameter estimation. IEEE Transactions on Antennas and Propagation, 34(3), 276–280.",
            "Mosher, J. C., Lewis, P. S., & Leahy, R. M. (1992). Multiple dipole localization and source waveform estimation using spatio-temporal MUSIC and recursive MUSIC. IEEE Transactions on Biomedical Engineering, 39(6), 541–557.",
        ],
    )

    def __init__(self, name="MUSIC", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(
        self,
        forward,
        mne_obj,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
        n="enhanced",
        stop_crit=0.95,
        verbose=0,
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
        n : int or str
            Number of eigenvectors (signal subspace dimension). Pass an int, or
            "enhanced" (default, robust for correlated sources), "auto", "L",
            "drop", or "mean" to use base class estimate_n_sources.
        stop_crit : float
            Criterion to stop recursions. The lower, the more dipoles will be
            incorporated.

        Return
        ------
        self : object returns itself for convenience
        """
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        wf = self.prepare_whitened_forward(noise_cov)
        data = self.unpack_data_obj(mne_obj)
        data = wf.sensor_transform @ data
        inverse_operator = self.make_music(data, n, stop_crit)
        self.inverse_operators = [
            InverseOperator(inverse_operator @ wf.sensor_transform, self.name),
        ]
        return self

    def make_music(self, y, n, stop_crit):
        """Apply the MUSIC inverse solution to the EEG data.

        Parameters
        ----------
        y : numpy.ndarray
            EEG data matrix (channels, time)
        n : int
            Number of eigenvectors to use.
        stop_crit : float
            Criterion at which to select candidate dipoles. The lower, the more
            dipoles will be incorporated.

        Return
        ------
        x_hat : numpy.ndarray
            Source data matrix (sources, time)
        """
        n_chans, n_dipoles = self.leadfield.shape
        y.shape[1]

        leadfield = self.leadfield
        # leadfield -= leadfield.mean(axis=0)

        # Data Covariance
        C = y @ y.T
        U, D, _ = np.linalg.svd(C, full_matrices=False)

        # # Get optimal eigenvectors
        # U, D, _ = np.linalg.svd(C, full_matrices=False)
        # if n == "auto":
        #     D_ = D/D.max()
        #     n_comp = np.where( abs(np.diff(D_)) < 0.01 )[0][0]+1
        # else:
        #     n_comp = deepcopy(n)

        if not isinstance(n, int):
            n_comp = self.estimate_n_sources(y, method=n)
        else:
            n_comp = n
        n_comp = max(1, min(n_comp, n_chans))  # avoid empty or oversized subspace

        Us = U[:, :n_comp]
        Ps = Us @ Us.T

        mu = np.zeros(n_dipoles)
        for p in range(n_dipoles):
            l = leadfield[:, p][:, np.newaxis]
            norm_1 = np.linalg.norm(Ps @ l)
            norm_2 = np.linalg.norm(l)
            mu[p] = norm_1 / norm_2 if norm_2 > 0 else 0.0
        mu[mu < stop_crit] = 0

        dipole_idc = np.where(mu != 0)[0]
        # If no dipole passes the threshold, use the dipole(s) with largest mu
        if len(dipole_idc) == 0:
            n_take = min(max(1, n_comp), n_dipoles)
            dipole_idc = np.argsort(mu)[-n_take:][::-1]
        # x_hat = np.zeros((n_dipoles, n_time))
        # x_hat[dipole_idc, :] = np.linalg.pinv(leadfield[:, dipole_idc]) @ y
        # return x_hat

        # WMNE-based
        # x_hat = np.zeros((n_dipoles, n_time))
        inverse_operator = np.zeros((n_dipoles, n_chans))

        L = self.leadfield[:, dipole_idc]
        W = np.diag(np.linalg.norm(L, axis=0))

        inverse_operator[dipole_idc, :] = np.linalg.inv(L.T @ L + W.T @ W) @ L.T

        return inverse_operator
