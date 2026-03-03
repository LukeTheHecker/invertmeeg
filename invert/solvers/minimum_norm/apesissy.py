from __future__ import annotations

from typing import Any

import mne
import numpy as np

from ..base import SolverMeta
from .sissy import SolverSISSY


class SolverAPESISSY(SolverSISSY):
    """Adaptive Parameter Estimation SISSY (APESISSY).

    Same structured-sparsity model as SISSY, but with Morozov discrepancy-based
    adaptive updates of the data-fidelity penalty ``lambda``.
    """

    meta = SolverMeta(
        acronym="APESISSY",
        full_name="Adaptive Parameter Estimation SISSY",
        category="Structured Sparsity",
        description=(
            "SISSY variant with ADMM and Morozov discrepancy principle for "
            "automatic lambda estimation."
        ),
        references=[
            "Aud'hui, M., Kachenoura, A., Albera, L., Merlet, I., Yochum, M., Van Bogaert, P., ... & Benquet, P. (2025). Fully Automated EEG Source Imaging Using Structured Sparsity for Single and Multiple Synchronous Epileptic Activities. bioRxiv, 2025-11.",
        ],
    )

    def __init__(
        self,
        name: str = "APESISSY",
        default_discrepancy_scale: float = 1.0,
        default_discrepancy_c: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.default_discrepancy_scale = float(default_discrepancy_scale)
        self.default_discrepancy_c = (
            None if default_discrepancy_c is None else float(default_discrepancy_c)
        )
        super().__init__(name=name, **kwargs)

    def apply_inverse_operator(
        self,
        mne_obj,
        *,
        lambda_init: float | None = None,
        alpha_l1: float | None = None,
        rho1: float | None = None,
        rho2: float | None = None,
        rho3: float | None = None,
        max_iter: int | None = None,
        abs_tol: float | None = None,
        rel_tol: float | None = None,
        discrepancy_c: float | None = None,
        discrepancy_scale: float | None = None,
        lambda_floor: float | None = None,
        lambda_ceil: float | None = None,
    ) -> mne.SourceEstimate:
        data = self.unpack_data_obj(mne_obj)
        self.validate_operator_data_compatibility(data)
        x = self._sensor_transform @ data

        if discrepancy_c is None:
            if self.default_discrepancy_c is not None:
                discrepancy_c = float(self.default_discrepancy_c)
            else:
                scale = (
                    self.default_discrepancy_scale
                    if discrepancy_scale is None
                    else float(discrepancy_scale)
                )
                discrepancy_c = max(scale, 1e-12) * np.sqrt(float(x.size))

        s, info = self._run_sissy_admm(
            np.asarray(x, dtype=float),
            lambda_init=self.default_lambda if lambda_init is None else float(lambda_init),
            alpha_l1=self.default_alpha_l1 if alpha_l1 is None else float(alpha_l1),
            rho1=self.default_rho1 if rho1 is None else float(rho1),
            rho2=self.default_rho2 if rho2 is None else float(rho2),
            rho3=self.default_rho3 if rho3 is None else float(rho3),
            max_iter=self.default_max_iter if max_iter is None else int(max_iter),
            abs_tol=self.default_abs_tol if abs_tol is None else float(abs_tol),
            rel_tol=self.default_rel_tol if rel_tol is None else float(rel_tol),
            adaptive_lambda=True,
            discrepancy_c=float(discrepancy_c),
            lambda_floor=(
                self.default_lambda_floor if lambda_floor is None else float(lambda_floor)
            ),
            lambda_ceil=(
                self.default_lambda_ceil if lambda_ceil is None else float(lambda_ceil)
            ),
        )
        info["discrepancy_c"] = float(discrepancy_c)
        self.last_solver_info = info
        return self.source_to_object(s)
