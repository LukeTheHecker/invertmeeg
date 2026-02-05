from __future__ import annotations

import logging
from copy import deepcopy

import mne
import numpy as np
_TORCH_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    import torch  # type: ignore[import-not-found]
    import torch.nn as nn  # type: ignore[import-not-found]
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]

    class _NNModule:  # pragma: no cover
        pass

    class _NN:  # pragma: no cover
        Module = _NNModule

    nn = _NN()  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc
from sklearn.covariance import OAS

from ...simulate import SimulationGenerator
from ..base import BaseSolver, InverseOperator, SolverMeta
from .torch_utils import (
    activation_from_name,
    count_trainable_parameters,
    get_torch_device,
    loss_from_name,
)

logger = logging.getLogger(__name__)


class _CovCNNNet(nn.Module):
    def __init__(
        self,
        leadfield: np.ndarray,
        *,
        n_outputs: int | None = None,
        n_dense_layers: int,
        n_dense_units: int,
        activation_function: str,
        output_activation: str,
    ) -> None:
        super().__init__()
        n_channels, n_dipoles = leadfield.shape
        if n_outputs is None:
            n_outputs = int(n_dipoles)
        n_outputs = int(n_outputs)

        leadfield_processed = leadfield.astype(np.float32, copy=True)
        leadfield_processed -= leadfield_processed.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(leadfield_processed, axis=0, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        leadfield_processed /= norms

        kernel = torch.from_numpy(leadfield_processed.T).unsqueeze(1).unsqueeze(2)
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=n_dipoles,
            kernel_size=(1, n_channels),
            bias=False,
        )
        with torch.no_grad():
            self.conv.weight.copy_(kernel)
        self.conv.weight.requires_grad = False

        mlp_layers: list[nn.Module] = []
        in_features = int(n_dipoles * n_channels)
        hidden_activation = activation_from_name(activation_function)
        for _ in range(int(n_dense_layers)):
            mlp_layers.append(nn.Linear(in_features, int(n_dense_units)))
            mlp_layers.append(hidden_activation)
            in_features = int(n_dense_units)

        self.mlp = nn.Sequential(*mlp_layers)
        self.out = nn.Linear(in_features, int(n_outputs))
        self.out_activation = activation_from_name(output_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, n_channels, n_channels)
        x = self.conv(x)  # (batch, n_dipoles, n_channels, 1)
        x = x.flatten(start_dim=1)  # (batch, n_dipoles*n_channels)
        x = self.mlp(x)
        x = self.out(x)
        return self.out_activation(x)


class SolverCovCNN(BaseSolver):
    """Class for the Covariance-based Convolutional Neural Network (CovCNN) for EEG inverse solutions.

"""

    meta = SolverMeta(
        acronym="CovCNN",
        full_name="Covariance-based Convolutional Neural Network",
        category="Neural Networks",
        description=(
            "Supervised CNN that operates on sensor covariance features (optionally "
            "with shrinkage) to predict source activity on a cortical grid."
        ),
        references=["Lukas Hecker 2025, unpublished"],
    )

    def __init__(self, name="Cov-CNN", reduce_rank=False, use_shrinkage=True, **kwargs):
        self.name = name
        self.use_shrinkage = use_shrinkage
        self.Uk = None
        self.model = None
        self.optimizer = None
        self.device = None
        return super().__init__(reduce_rank=reduce_rank, **kwargs)

    def make_inverse_operator(
        self,
        forward,
        simulation_config,
        *args,
        parcellator=None,
        n_filters="auto",
        n_dense_units=300,
        n_dense_layers=2,
        activation_function="tanh",
        output_activation="sigmoid",
        epochs=300,
        learning_rate=1e-3,
        loss="cosine_similarity",
        size_validation_set=256,
        epsilon=0.0,
        patience=100,
        cov_type="basic",
        alpha="auto",
        **kwargs,
    ):
        """Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        simulation_config : SimulationConfig
            A SimulationConfig object for data generation containing all
            simulation parameters (batch_size, n_sources, n_orders, snr_range, etc.).
        parcellator : Parcellator
            The Parcellator object. Default None.
        n_filters : int
            Number of filters in the convolution layer. Default "auto" sets to n_channels.
        n_dense_units : int
            Number of units in dense layers. Default 300.
        n_dense_layers : int
            Number of dense layers. Default 2.
        activation_function : str
            The activation function of the hidden layers. Default "tanh".
        output_activation : str
            The activation function of the output layer. Default "sigmoid".
        epochs : int
            The number of epochs to train. Default 300.
        learning_rate : float
            The learning rate of the optimizer. Default 1e-3.
        loss : str
            The loss function of the neural network. Default "cosine_similarity".
        size_validation_set : int
            The size of validation data set. Default 256.
        epsilon : float
            The threshold at which to select sources as "active". Default 0.
        patience : int
            Early stopping patience. Default 100.
        cov_type : str
            The type of covariance matrix to compute.
            "basic" -> Y @ Y.T
            "SSM" -> Signal Subspace Matching covariance
            "riemann" -> Riemannian covariance
            Default "basic".
        alpha : float
            The regularization parameter. Default "auto".

        Return
        ------
        self : object returns itself for convenience
        """
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        n_channels, n_dipoles = self.leadfield.shape

        if n_filters == "auto":
            n_filters = n_channels
        self.forward = forward

        # Store simulation config
        self.simulation_config = simulation_config

        # Store Parameters
        # Architecture
        self.parcellator = parcellator
        self.n_filters = n_filters
        self.activation_function = activation_function
        self.output_activation = output_activation
        self.n_dense_layers = n_dense_layers
        self.n_dense_units = n_dense_units
        # Training
        self.batch_size = self.simulation_config.batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss = loss
        self.size_validation_set = size_validation_set
        self.patience = patience
        # Training Data (from simulation_config)
        self.n_timepoints = self.simulation_config.n_timepoints
        self.n_sources = self.simulation_config.n_sources
        self.n_orders = self.simulation_config.n_orders
        self.batch_repetitions = self.simulation_config.batch_repetitions
        self.snr_range = self.simulation_config.snr_range
        self.add_forward_error = self.simulation_config.add_forward_error
        self.forward_error = self.simulation_config.forward_error
        self.inter_source_correlation = self.simulation_config.inter_source_correlation
        self.cov_type = cov_type
        self.correlation_mode = self.simulation_config.correlation_mode
        self.noise_color_coeff = self.simulation_config.noise_color_coeff
        # Inference
        self.epsilon = epsilon
        logger.info("Create Generator:..")
        self.create_generator()
        logger.info("Build Model:..")
        self.build_model()
        logger.info("Train Model:..")
        self.train_model()

        self.inverse_operators = []
        return self

    def apply_inverse_operator(self, mne_obj, prior=None) -> mne.SourceEstimate:
        """Apply the inverse operator.

        Parameters
        ----------
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        prior : numpy.ndarray
            The prior gamma vector. Default None.
        Return
        ------
        stc : mne.SourceEstimate
            The mne Source Estimate object.
        """
        data = self.unpack_data_obj(mne_obj)

        source_mat = self.apply_model(data, prior=prior)
        stc = self.source_to_object(source_mat)

        return stc

    def compute_covariance(self, Y: np.ndarray) -> np.ndarray:
        """Compute the covariance matrix of the data.

        Parameters
        ----------
        Y : numpy.ndarray
            The data matrix.

        Return
        ------
        C : numpy.ndarray
            The covariance matrix.
        """
        if self.cov_type == "basic":
            C = Y @ Y.T
        elif self.cov_type == "log":
            C = self.log_covariance(Y)
        elif self.cov_type == "recipsiicos":
            C = self.recipsiicos_covariance(Y)
        elif self.cov_type == "riemann":
            C = self.riemannian_covariance(Y)
        elif self.cov_type == "SSM":
            n_time = Y.shape[1]
            M_Y = Y.T @ Y
            YY = M_Y + 0.001 * (50 / n_time) * np.trace(M_Y) * np.eye(n_time)
            P_Y = (Y @ np.linalg.inv(YY)) @ Y.T
            C = P_Y.T @ P_Y
            # print("yes its new")
        else:
            msg = f"Covariance type '{self.cov_type}' not recognized. Use 'basic', 'SSM' or 'riemann' or provide a custom covariance matrix."
            raise ValueError(msg)

        if self.use_shrinkage:
            C = self._shrinkage_covariance(Y)

        return C

    def _shrinkage_covariance(self, Y: np.ndarray) -> np.ndarray:
        """Compute shrinkage covariance using OAS estimator."""
        lw = OAS(assume_centered=False)
        C = lw.fit(Y.T).covariance_
        return C

    def log_covariance(
        self,
        Y,
        eps=1e-6,  # jitter & eigenvalue floor
        demean=True,  # remove channel means before covariance
    ):
        """
        Compute a single log–covariance matrix from EEG data Y (m x T).

        Steps:
        1) (Optional) Demean channels over time
        3) Shrinkage: C_shrunk = (1-alpha)*C + alpha*(tr(C)/m) * I  [or Ledoit–Wolf]
        4) (Optional) Whitening: Cw = W^{-1/2} C_shrunk W^{-1/2}
        5) Matrix log via eigendecomposition: C_log = U log(Λ) U^T

        Args:
            Y: np.ndarray, shape (m, T)
            noise_cov: np.ndarray (m, m) SPD baseline noise covariance for whitening; if None, no whitening
            eps: float, numerical floor for SPD safety and eigenvalues
            demean: bool, subtract per-channel temporal mean before covariance

        Returns:
            C_log: np.ndarray, shape (m, m) — the log–covariance matrix
        """
        Y = np.asarray(Y)
        m, T = Y.shape
        assert m <= T, "For stable covariance estimation, prefer T >= m."

        # 1) Demean over time (recommended)
        X = Y - Y.mean(axis=1, keepdims=True) if demean else Y

        C = X @ X.T / T

        # Symmetrize (guard against tiny FP asymmetries)
        C = 0.5 * (C + C.T)

        C += eps * np.eye(m)

        # 5) Matrix log via eigen-decomposition (log-Euclidean map)
        C_log = self.compute_log_cov(C, eps=eps)

        return C_log

    def compute_log_cov(self, C: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        evals, evecs = np.linalg.eigh(C)
        evals = np.clip(evals, eps, None)
        log_evals = np.log(evals)
        C_log = (evecs * log_evals) @ evecs.T  # evecs @ diag(log_evals) @ evecs.T
        C_log = 0.5 * (C_log + C_log.T)  # re-symmetrize
        return C_log

    def recipsiicos_covariance(
        self, Y: np.ndarray, pwr_energy: float = 0.95
    ) -> np.ndarray:
        """Compute the ReciPSIICOS covariance matrix of the data."""
        m, n = self.leadfield.shape
        # Center Y over time
        Y = Y - Y.mean(axis=1, keepdims=True)

        if self.Uk is None:  # type: ignore[has-type]
            Qp = self._Q_power(self.leadfield)  # (m^2, n)

            U, s, _ = np.linalg.svd(Qp, full_matrices=False)
            k = self._svd_rank_from_energy(s, pwr_energy)
            self.Uk = U[:, :k]  # (m^2, k)

        v = np.kron(Y, Y)  # (m^2,)
        v_tilde = self.Uk @ (self.Uk.T @ v)  # project
        Ct = np.reshape(v_tilde, (m, m))  # (m, m)

        # Spectral flip
        w, V = np.linalg.eigh(Ct)
        Ct = (V * np.abs(w)) @ V.T

        # Log-Euclidean map
        Ct_log = self.compute_log_cov(Ct, eps=1e-6)
        return Ct_log

    @staticmethod
    def _svd_rank_from_energy(s: np.ndarray, keep_energy: float) -> int:
        s2 = s**2
        cum = np.cumsum(s2) / np.sum(s2) if np.sum(s2) > 0 else np.ones_like(s2)
        return int(np.searchsorted(cum, keep_energy) + 1)

    @staticmethod
    def _Q_power(Lr: np.ndarray) -> np.ndarray:
        """
        Build power subspace columns: q_i = vec(g_i g_i^T) = kron(g_i, g_i).
        Returns Q_pwr: (r^2, n)
        """
        r, n = Lr.shape
        Q = np.empty((r * r, n), dtype=Lr.dtype)
        for i in range(n):
            gi = Lr[:, i]
            Q[:, i] = np.kron(gi, gi)
        return Q

    def riemannian_covariance(self, X, eps=1e-12):
        """
        Compute the Riemannian (log-mapped) covariance of EEG data.

        Parameters
        ----------
        X : ndarray of shape (n_channels, n_times)
            EEG data matrix.
        eps : float
            Small value for numerical stability.

        Returns
        -------
        Z : ndarray of shape (n_channels, n_channels)
            Tangent-space (Riemannian) covariance matrix.
        """
        # 1) Empirical covariance
        C = (X @ X.T) / X.shape[1]
        C = 0.5 * (C + C.T)  # enforce symmetry

        # 2) Reference covariance (for simplicity use C itself here)
        C0 = C.copy()

        # 3) Compute inverse sqrt and sqrt of C0
        eigvals, eigvecs = np.linalg.eigh(C0)
        eigvals = np.clip(eigvals, eps, None)
        C0_sqrt = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
        C0_invsqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T

        # 4) Log map: Z = C0^{1/2} log( C0^{-1/2} C C0^{-1/2} ) C0^{1/2}
        A = C0_invsqrt @ C @ C0_invsqrt
        vals, vecs = np.linalg.eigh(0.5 * (A + A.T))
        vals = np.clip(vals, eps, None)
        logA = (vecs * np.log(vals)) @ vecs.T
        Z = C0_sqrt @ logA @ C0_sqrt
        Z = 0.5 * (Z + Z.T)

        return Z

    def apply_model(self, data: np.ndarray, prior=None) -> np.ndarray:
        """Compute the inverse solution of the M/EEG data.

        Parameters
        ----------
        data : numpy.ndarray
            The M/EEG data matrix.
        prior : numpy.ndarray
            The prior gamma vector. Default None.
        Return
        ------
        x_hat : numpy.ndarray
            The source esimate.

        """
        if self.parcellator is not None:
            leadfield = self.parcellator.forward_parcellated["sol"]["data"]
        else:
            leadfield = self.leadfield

        y = deepcopy(data)
        y = y - y.mean(axis=1, keepdims=True)  # Temporal mean centering

        n_channels, n_times = y.shape

        # Compute Data Covariance Matrix
        C = self.compute_covariance(y)
        # Scale
        C /= abs(C).max()
        # C /= y.shape[1]

        # Add empty batch and (color-) channel dimension
        C = C[np.newaxis, np.newaxis, :, :].astype(np.float32, copy=False)

        # Get prior source covariance from model
        assert self.model is not None
        self.model.eval()
        device = self.device or get_torch_device()
        with torch.no_grad():
            gammas = (
                self.model(torch.as_tensor(C, dtype=torch.float32, device=device))
                .detach()
                .cpu()
                .numpy()[0]
            )
        # print(f"Gammas shape: {gammas.shape}")
        # gammas = np.maximum(gammas, 0)
        max_gamma = float(np.max(gammas))
        if max_gamma > 0:
            gammas /= max_gamma
        # gammas = gammas**2
        gammas = gammas ** (1)
        # gammas[gammas<self.epsilon] = 0

        if prior is not None:
            prior_max = float(np.max(prior))
            if prior_max > 0:
                gammas = gammas * (prior / prior_max)

        self.gammas = deepcopy(gammas)

        source_covariance = np.diag(gammas)

        # Minimum norm
        Sigma_y = leadfield @ source_covariance @ leadfield.T
        if self.alpha == "auto":
            r_grid = np.asarray(self.r_values, dtype=float)
        else:
            r_grid = np.asarray([float(self.alpha)], dtype=float)
        self.alphas = list(r_grid)

        inverse_operators = []
        for r in r_grid:
            reg_term = float(r) * np.trace(Sigma_y) / n_channels
            inverse_operator = (
                source_covariance
                @ leadfield.T
                @ np.linalg.inv(Sigma_y + reg_term * np.eye(n_channels))
            )
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [
            InverseOperator(inverse_operator, self.name)
            for inverse_operator in inverse_operators
        ]

        if self.parcellator is not None:
            leadfield_parcellated = self.parcellator.forward_parcellated["sol"]["data"]
            leadfield_original = self.leadfield
            self.leadfield = leadfield_parcellated
            x_hat, _ = self.regularise_gcv(y)
            self.leadfield = leadfield_original
            return self.parcellator.decompress_data(x_hat)
        else:
            x_hat, _ = self.regularise_gcv(y)

        return x_hat

    def train_model(
        self,
    ):
        """Train the neural network model."""
        import time

        if self.model is None:
            raise RuntimeError("Model not initialized. Call build_model() first.")
        if self.optimizer is None:
            raise RuntimeError("Optimizer not initialized. Call build_model() first.")

        start_t = time.perf_counter()

        self.model.train()
        loss_fn = loss_from_name(self.loss)

        # Re-initialize generator (matches previous TF behaviour).
        self.create_generator()

        # Get validation data from generator (consume one repeated batch).
        for _ in range(self.batch_repetitions):
            x_val, y_val = next(self.generator)

        device = self.device or get_torch_device()
        x_val_t = torch.as_tensor(x_val, dtype=torch.float32, device=device)
        y_val_t = torch.as_tensor(y_val, dtype=torch.float32, device=device)

        history: dict[str, list[float]] = {"loss": [], "val_loss": []}
        best_val = float("inf")
        best_state = None
        patience_left = int(self.patience)

        total_epochs = int(self.epochs)
        log_every = 10
        logger.info(
            "Training %s on simulated data (epochs=%d, batch_size=%d, batch_repetitions=%d, lr=%s, loss=%s, patience=%d, device=%s)",
            self.__class__.__name__,
            total_epochs,
            int(getattr(self, "batch_size", -1)),
            int(getattr(self, "batch_repetitions", 1)),
            str(getattr(self, "learning_rate", "?")),
            str(getattr(self, "loss", "?")),
            int(getattr(self, "patience", 0)),
            str(device),
        )

        epochs_ran = 0
        for epoch in range(total_epochs):
            self.model.train()
            running = 0.0
            for _step in range(int(self.batch_repetitions)):
                x_batch, y_batch = next(self.generator)
                x_t = torch.as_tensor(x_batch, dtype=torch.float32, device=device)
                y_t = torch.as_tensor(y_batch, dtype=torch.float32, device=device)

                self.optimizer.zero_grad(set_to_none=True)
                y_pred = self.model(x_t)
                loss = loss_fn(y_pred, y_t)
                loss.backward()
                self.optimizer.step()
                running += float(loss.detach().cpu().item())

            train_loss = running / float(self.batch_repetitions)

            self.model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.model(x_val_t), y_val_t).cpu().item())

            history["loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            epochs_ran = epoch + 1

            if val_loss < best_val:
                best_val = val_loss
                best_state = deepcopy(self.model.state_dict())
                patience_left = int(self.patience)
                logger.info(
                    "Epoch %d/%d - loss=%.6f val_loss=%.6f (new best)",
                    epoch + 1,
                    total_epochs,
                    train_loss,
                    val_loss,
                )
            else:
                patience_left -= 1
                if (epoch == 0) or ((epoch + 1) % log_every == 0):
                    logger.info(
                        "Epoch %d/%d - loss=%.6f val_loss=%.6f (patience_left=%d)",
                        epoch + 1,
                        total_epochs,
                        train_loss,
                        val_loss,
                        patience_left,
                    )
                if patience_left <= 0:
                    logger.info(
                        "Early stopping at epoch %d/%d (best_val=%.6f)",
                        epoch + 1,
                        total_epochs,
                        best_val,
                    )
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        self.history = history
        elapsed_s = time.perf_counter() - start_t
        logger.info(
            "Training finished (epochs_ran=%d, best_val=%.6f, elapsed=%.2fs)",
            epochs_ran,
            best_val,
            elapsed_s,
        )

    def build_model(
        self,
    ):
        """Build the neural network model."""
        if self.parcellator is None:
            leadfield = self.leadfield
            n_channels, n_dipoles = leadfield.shape
        else:
            leadfield = self.parcellator.forward_parcellated["sol"]["data"]
            n_channels, n_dipoles = leadfield.shape

        self.device = get_torch_device()
        model = _CovCNNNet(
            leadfield,
            n_dense_layers=int(self.n_dense_layers),
            n_dense_units=int(self.n_dense_units),
            activation_function=str(self.activation_function),
            output_activation=str(self.output_activation),
        ).to(self.device)

        self.model = model
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(self.learning_rate)
        )

        logger.info(
            "Total number of trainable parameters: %d",
            count_trainable_parameters(self.model),
        )

    def create_generator(
        self,
    ):
        """Create the data generator used for the simulations."""

        # Create SimulationGenerator using the config
        sim_gen = SimulationGenerator(self.forward, config=self.simulation_config)

        # Wrap the generator to compute covariance and return (x_cov, y_mask)
        def wrapped_generator():
            for x, y, _info in sim_gen.generate():
                x_cov = np.stack([self.compute_covariance(xx) for xx in x], axis=0)
                max_abs = np.abs(x_cov).max(axis=(1, 2), keepdims=True)
                max_abs = np.where(max_abs == 0, 1.0, max_abs)
                x_cov = (x_cov / max_abs).astype(np.float32, copy=False)

                # Torch expects NCHW: (batch, 1, n_channels, n_channels)
                x_cov = x_cov[:, np.newaxis, :, :]

                # Use continuous values as target
                y_abs_mean = np.abs(y).mean(axis=2)
                y_scale = y_abs_mean.max(axis=1, keepdims=True)
                y_scale = np.where(y_scale == 0, 1.0, y_scale)
                y_continuous = (y_abs_mean / y_scale).astype(np.float32, copy=False)

                if self.parcellator is not None:
                    y_continuous = self.parcellator.compress_data(y_continuous.T).T

                yield x_cov, y_continuous

        self.generator = wrapped_generator()
