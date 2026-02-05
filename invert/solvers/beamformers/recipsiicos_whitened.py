"""Whitened ReciPSIICOS Beamformer."""

from __future__ import annotations

import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta
from .utils import (
    _cov,
    _lcmv_inverse_operator,
    _project_covariance_whitened,
    _psd_spectral_flip,
    _virtual_sensors,
    _whiten,
)


class SolverReciPSIICOSWhitened(BaseSolver):
    """
    Whitened ReciPSIICOS Beamformer for M/EEG inverse solution (recommended).

    Extends plain ReciPSIICOS by additionally removing correlation subspace
    components in a whitened representation, providing better suppression of
    correlated source cross-talk.

    References
    ----------
    [1] Kuznetsova, A., Nurislamova, Y., & Ossadtchi, A. (2021). Modified
        covariance beamformer for solving MEG inverse problem in the environment
        with correlated sources. Neuroimage, 228, 117677.
    """

    meta = SolverMeta(
        slug="recipsiicos_whitened",
        full_name="ReciPSIICOS (whitened)",
        category="Beamformers",
        description=(
            "Whitened ReciPSIICOS variant that additionally removes correlation "
            "subspace components in a whitened representation."
        ),
        references=[
            "Kuznetsova, A., Nurislamova, Y., & Ossadtchi, A. (2021). Modified "
            "covariance beamformer for solving MEG inverse problem in the environment "
            "with correlated sources. NeuroImage, 228, 117677.",
        ],
    )

    def __init__(
        self, name="ReciPSIICOS-Whitened", reduce_rank=True, rank="auto", **kwargs
    ):
        self.name = name
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(
        self,
        forward,
        mne_obj,
        *args,
        alpha="auto",
        noise_cov=None,
        virtual_sensor_energy=0.99,
        pwr_energy=0.9,
        pwr_rank=None,
        cor_rank=None,
        max_pairs=None,
        seed=0,
        spectral_flip=True,
        verbose=0,
        **kwargs,
    ):
        """
        Calculate inverse operator using whitened ReciPSIICOS method.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        alpha : float or 'auto'
            The regularization parameter for LCMV backend.
        noise_cov : array, optional
            Noise covariance matrix for whitening.
        virtual_sensor_energy : float
            Fraction of leadfield energy to retain (default: 0.99)
        pwr_energy : float
            Fraction of power subspace energy to retain (default: 0.995)
        pwr_rank : int, optional
            Explicit power subspace rank (overrides pwr_energy)
        cor_rank : int, optional
            Number of correlation modes to remove (default: auto)
        max_pairs : int, optional
            Max source pairs to sample (default: all)
        seed : int
            Random seed for pair sampling (default: 0)
        spectral_flip : bool
            Apply spectral flip to ensure PSD (default: True)

        Returns
        -------
        self : object
            Returns itself for convenience
        """
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)

        L = self.leadfield
        Y = data
        n_chans, n_dipoles = L.shape

        # Normalize leadfield columns
        L = L / np.linalg.norm(L, axis=0)

        # 1) Optional noise whitening
        Yw, Lw = _whiten(Y, L, noise_cov)

        # 2) Virtual sensors via leadfield SVD
        UrT, Lr = _virtual_sensors(Lw, keep_energy=virtual_sensor_energy)
        Yr = UrT @ Yw

        # 3) Data covariance in reduced space
        Cr = _cov(Yr)

        # 4) Whitened ReciPSIICOS projection
        Ct = _project_covariance_whitened(
            Cr,
            Lr,
            pwr_energy=pwr_energy,
            pwr_rank=pwr_rank,
            cor_rank=cor_rank,
            max_pairs=max_pairs,
            seed=seed,
        )

        # 5) Spectral flip
        if spectral_flip:
            Ct = _psd_spectral_flip(Ct)

        # 6) Build inverse operators for different regularization values
        # ReciPSIICOS uses trace-shrinkage regularization: C + lam*trace(C)/m*I,
        # where lam is dimensionless. Do NOT eigenvalue-scale it.
        if alpha == "auto":
            reg_values = np.logspace(-6, 1, self.n_reg_params)
        else:
            reg_values = np.asarray([float(alpha)], dtype=float)
        self.alphas = list(np.asarray(reg_values, dtype=float))

        inverse_operators = []
        for reg in self.alphas:
            W_reduced = _lcmv_inverse_operator(Lr, Ct, reg=reg)
            W_full = W_reduced @ UrT
            inverse_operator = W_full
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [
            InverseOperator(inv_op, self.name) for inv_op in inverse_operators
        ]
        return self
