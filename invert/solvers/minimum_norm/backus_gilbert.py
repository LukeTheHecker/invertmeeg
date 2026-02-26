import logging

import mne
import numpy as np
from scipy.spatial.distance import cdist

from ...util import pos_from_forward
from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverBackusGilbert(BaseSolver):
    """Class for the Backus Gilbert inverse solution."""

    meta = SolverMeta(
        acronym="BG",
        full_name="Backus–Gilbert",
        category="Minimum Norm",
        description=(
            "Resolution-optimizing linear inverse method that trades off spatial "
            "resolution versus noise amplification using Backus–Gilbert theory."
        ),
        references=[
            "Backus, G., & Gilbert, F. (1968). The resolving power of gross earth data. Geophysical Journal of the Royal Astronomical Society, 16(2), 169–205.",
        ],
    )
    SUPPORTS_VECTOR_ORIENTATION: bool = True

    def __init__(self, name="Backus-Gilbert", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(
        self,
        forward,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
        **kwargs,
    ):
        """Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        alpha : float
            The regularization parameter.

        Return
        ------
        self : object returns itself for convenience
        """
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        wf = self.prepare_whitened_forward(noise_cov)
        leadfield = wf.G_white
        _, n_dipoles = leadfield.shape
        pos = pos_from_forward(forward, verbose=self.verbose)
        n_sources = len(pos)
        n_orient = n_dipoles // n_sources  # 1 for fixed, 3 for free
        dist = cdist(pos, pos)

        F = leadfield @ leadfield.T
        L = leadfield @ np.ones((n_dipoles, 1))

        # The original implementation computes identical filters for all
        # orientation components of the same source index; compute once per
        # source and replicate to dipole rows.
        T_sources = []
        for source_idx in range(n_sources):
            d = np.repeat(dist[source_idx, :], n_orient)
            E_gamma = (leadfield * d) @ leadfield.T + F
            E_gamma_pinv = np.linalg.pinv(E_gamma)
            T_gamma = (E_gamma_pinv @ L) / (L.T @ E_gamma_pinv @ L)
            T_sources.append(T_gamma[:, 0])
        T = np.repeat(np.asarray(T_sources, dtype=float), n_orient, axis=0)

        inverse_operators = [
            T @ wf.sensor_transform,
        ]

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]
        return self
