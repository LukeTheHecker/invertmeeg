from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd


@dataclasses.dataclass
class SimulationBatch:
    """Structured container for simulation output.

    Replaces the ad-hoc ``(x, y, info, y_patches)`` tuple that generators
    previously yielded.  Supports iteration/unpacking for backward
    compatibility::

        batch = next(gen.generate())
        x, y, info, y_patches = batch          # still works
        x, y, info = batch.x, batch.y, batch.info  # preferred
    """

    x: np.ndarray
    """Sensor data, shape ``(batch_size, n_channels, n_timepoints)``."""

    y: np.ndarray
    """Source activity (including background if mixture mode),
    shape ``(batch_size, n_dipoles, n_timepoints)``."""

    info: pd.DataFrame
    """Per-sample simulation metadata."""

    y_patches: np.ndarray | None = None
    """Patch-only source activity (mixture mode), or ``None``."""

    # ------------------------------------------------------------------
    # Backward-compatible iteration  (``x, y, info, yp = batch``)
    # ------------------------------------------------------------------
    def __iter__(self):
        return iter((self.x, self.y, self.info, self.y_patches))

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return (self.x, self.y, self.info, self.y_patches)[idx]
