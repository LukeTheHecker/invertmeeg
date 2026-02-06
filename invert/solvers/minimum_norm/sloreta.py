from __future__ import annotations

import logging

import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta

logger = logging.getLogger(__name__)


class SolverSLORETA(BaseSolver):
    """Class for the standardized Low Resolution Tomography (sLORETA) inverse
        solution [1].

    When ``alpha="auto"``, regularization selection (L-curve, GCV, product) is
    performed on the underlying MNE kernel.  Once the optimal regularization
    parameter has been identified, the sLORETA standardization is applied only
    to that selected kernel.  This avoids the problem of comparing differently-
    standardized operators across alpha values.

    References
    ----------
    [1] Pascual-Marqui, R. D. (2002). Standardized low-resolution brain
    electromagnetic tomography (sLORETA): technical details. Methods Find Exp
    Clin Pharmacol, 24(Suppl D), 5-12.
    """

    meta = SolverMeta(
        acronym="sLORETA",
        full_name="Standardized Low Resolution Electromagnetic Tomography",
        category="Minimum Norm",
        description=(
            "Standardized (variance-normalized) LORETA/MNE-type inverse designed "
            "to reduce localization bias by normalizing each source by its "
            "estimated variance."
        ),
        references=[
            "Pascual-Marqui, R. D. (2002). Standardized low-resolution brain electromagnetic tomography (sLORETA): technical details. Methods and Findings in Experimental and Clinical Pharmacology, 24(Suppl D), 5–12.",
        ],
    )

    def __init__(
        self, name="Standardized Low Resolution Tomography", reduce_rank=False, **kwargs
    ):
        self.name = name
        self.reduce_rank = reduce_rank
        return super().__init__(reduce_rank=reduce_rank, **kwargs)

    def make_inverse_operator(self, forward, *args, alpha="auto", verbose=0, **kwargs):
        """Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        alpha : float | "auto"
            The regularization parameter. When set to "auto", regularization
            selection is performed on the MNE kernel and sLORETA standardization
            is applied afterwards.

        Return
        ------
        self : object returns itself for convenience
        """
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)

        leadfield = self.leadfield
        n_chans = leadfield.shape[0]

        LLT = leadfield @ leadfield.T

        I = np.identity(n_chans)

        mne_operators = []
        sloreta_operators = []
        eps = 1e-12
        for alpha in self.alphas:
            K_MNE = leadfield.T @ np.linalg.pinv(LLT + alpha * I)
            resolution_diag = np.maximum(np.diag(K_MNE @ leadfield), eps)
            W_diag = np.sqrt(resolution_diag)
            W_slor = (K_MNE.T / W_diag).T

            mne_operators.append(K_MNE)
            sloreta_operators.append(W_slor)

        # Store MNE operators for regularization selection and sLORETA
        # operators for the final result.
        self.inverse_operators = [
            InverseOperator(op, self.name) for op in mne_operators
        ]
        self._sloreta_operators = [
            InverseOperator(op, self.name) for op in sloreta_operators
        ]
        return self

    def apply_inverse_operator(self, mne_obj):
        """Apply the inverse operator.

        Regularization selection is performed using the MNE kernels stored in
        ``self.inverse_operators``.  The returned source estimate is computed
        with the sLORETA-standardized kernel at the selected regularization
        index.

        Parameters
        ----------
        mne_obj : mne.Evoked | mne.Epochs | mne.io.Raw
            The MNE data object.

        Return
        ------
        stc : mne.SourceEstimate
            The source estimate.
        """
        data = self.unpack_data_obj(mne_obj)

        if self.use_last_alpha and self.last_reg_idx is not None:
            idx = self.last_reg_idx
        else:
            if self.regularisation_method.lower() == "l":
                _, idx = self.regularise_lcurve(data, plot=self.plot_reg)
            elif self.regularisation_method.lower() in {"gcv", "mgcv"}:
                gamma = (
                    self.gcv_gamma
                    if self.regularisation_method.lower() == "gcv"
                    else self.mgcv_gamma
                )
                _, idx = self.regularise_gcv(data, plot=self.plot_reg, gamma=gamma)
            elif self.regularisation_method.lower() == "product":
                _, idx = self.regularise_product(data, plot=self.plot_reg)
            else:
                msg = f"{self.regularisation_method} is no valid regularisation method."
                raise AttributeError(msg)
            self.last_reg_idx = idx

        # Apply the sLORETA operator at the selected regularization index
        source_mat = self._sloreta_operators[idx].apply(data)
        stc = self.source_to_object(source_mat)
        return stc
