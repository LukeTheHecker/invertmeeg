from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverECD(BaseSolver):
    """Equivalent Current Dipole (ECD) fitting on a discretized source grid.

    This solver performs a grid-search over the fixed-orientation source space
    provided by the forward model and selects the location with the highest
    goodness-of-fit (least-squares projection of the data).

    Notes
    -----
    - The forward model is converted to fixed orientation by BaseSolver.
    - The returned estimate is sparse (one non-zero vertex).
    """

    meta = SolverMeta(
        acronym="ECD",
        full_name="Equivalent Current Dipole",
        category="Dipole Fitting",
        description=(
            "Grid-search ECD fitting on the forward-model source grid. Selects the "
            "single best-fitting dipole and estimates its moment time course."
        ),
        references=[
            "Mosher, J. C., Lewis, P. S., & Leahy, R. M. (1992). Multiple dipole modeling and localization from spatiotemporal MEG data. IEEE Transactions on Biomedical Engineering, 39(6), 541–557.",
        ],
    )

    def __init__(self, name: str = "Equivalent Current Dipole", **kwargs: Any) -> None:
        self.name = name
        self.dipole_index: int | None = None
        self.gof: float | None = None
        super().__init__(**kwargs)

    def make_inverse_operator(  # type: ignore[override]
        self,
        forward,
        mne_obj,
        *args: Any,
        alpha: str | float = "auto",
        tmin: float | None = None,
        tmax: float | None = None,
        **kwargs: Any,
    ):
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)

        data = self.unpack_data_obj(mne_obj)
        sfreq = float(self.obj_info["sfreq"])
        if tmin is not None or tmax is not None:
            start = 0 if tmin is None else int(round((tmin - self.tmin) * sfreq))
            stop = data.shape[1] if tmax is None else int(round((tmax - self.tmin) * sfreq))
            start = int(np.clip(start, 0, data.shape[1]))
            stop = int(np.clip(stop, start + 1, data.shape[1]))
            data = data[:, start:stop]

        leadfield = self.leadfield
        n_chans, n_dipoles = leadfield.shape

        norms = np.sum(leadfield * leadfield, axis=0)
        norms = np.where(norms <= 0, 1e-15, norms)

        proj = leadfield.T @ data  # (n_dipoles, n_times)
        explained = np.sum(proj * proj, axis=1) / norms
        total_energy = float(np.sum(data * data))
        total_energy = max(total_energy, 1e-15)

        gof = explained / total_energy
        best = int(np.argmax(gof))

        kernel = np.zeros((n_dipoles, n_chans), dtype=np.float64)
        kernel[best, :] = leadfield[:, best] / norms[best]

        self.dipole_index = best
        self.gof = float(gof[best])

        self.inverse_operators = [InverseOperator(kernel, self.name)]
        return self
