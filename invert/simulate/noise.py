import numpy as np


def powerlaw_noise(beta, n_timepoints, n_signals=1, rng=None):
    """Generate 1/f^beta colored noise via FFT spectral shaping.

    Parameters
    ----------
    beta : float or array-like
        Power-law exponent(s). 0=white, 1=pink, 2=brown.
        If array, must have length n_signals.
    n_timepoints : int
        Number of time samples.
    n_signals : int
        Number of independent signals to generate.
    rng : numpy.random.Generator or None
        Random number generator.

    Returns
    -------
    signals : ndarray, shape (n_signals, n_timepoints)
        Colored noise signals.
    """
    if rng is None:
        rng = np.random.default_rng()

    beta = np.atleast_1d(np.asarray(beta, dtype=float))
    if beta.shape[0] == 1:
        beta = np.broadcast_to(beta, (n_signals,))

    # Generate white noise and FFT
    white = rng.standard_normal((n_signals, n_timepoints))
    spectrum = np.fft.rfft(white)

    # Build frequency-domain filter: 1/f^(beta/2) (power spectrum goes as 1/f^beta)
    freqs = np.fft.rfftfreq(n_timepoints)
    # Skip DC (index 0) to avoid division by zero
    freq_filter = np.ones((n_signals, len(freqs)))
    freq_filter[:, 1:] = freqs[1:][np.newaxis, :] ** (-beta[:, np.newaxis] / 2.0)

    spectrum *= freq_filter
    signals = np.fft.irfft(spectrum, n=n_timepoints)

    return signals


def add_white_noise(
    X_clean, snr, rng, channel_types, noise_color_coeff=0.5, correlation_mode=None
):
    """
    Parameters
    ----------
    X_clean : numpy.ndarray
        The clean EEG data.
    snr : float
        The signal to noise ratio in dB.
    correlation_mode : None/str
        None implies no correlation between the noise in different channels.
        'banded' : Colored banded noise, where channels closer to each other will be more correlated.
        'diagonal' : Some channels have varying degrees of noise.
        'cholesky' : A set correlation coefficient between each pair of channels
    noise_color_coeff : float
        The magnitude of spatial coloring of the noise (not the magnitude of noise overall!).
    """
    n_chans, n_time = X_clean.shape
    X_noise = rng.standard_normal((n_chans, n_time))
    snr_linear = 10 ** (snr / 10)

    if isinstance(channel_types, list):
        channel_types = np.array(channel_types)
    # Ensure the channel_types array is correct length
    assert len(channel_types) == n_chans, (
        "Length of channel_types must match the number of channels in X_clean"
    )

    unique_types = np.unique(channel_types)
    X_full = np.zeros_like(X_clean)

    for ch_type in unique_types:
        type_indices = np.where(channel_types == ch_type)[0]
        X_clean_type = X_clean[type_indices, :]
        X_noise_type = X_noise[type_indices, :]
        if isinstance(noise_color_coeff, str) and isinstance(
            correlation_mode, np.ndarray
        ):
            # Real Noise Covariance
            X_noise_type = (
                np.linalg.cholesky(correlation_mode[type_indices][:, type_indices])
                @ X_noise_type
            )
        elif correlation_mode == "cholesky":
            covariance_matrix = np.full(
                (len(type_indices), len(type_indices)), noise_color_coeff
            )
            np.fill_diagonal(covariance_matrix, 1)  # Set diagonal to 1 for variance

            # Cholesky decomposition
            X_noise_type = np.linalg.cholesky(covariance_matrix) @ X_noise_type
        elif correlation_mode == "banded":
            num_sensors = X_noise_type.shape[0]
            Y = np.zeros_like(X_noise_type)
            for i in range(num_sensors):
                Y[i, :] = X_noise_type[i, :]
                for j in range(num_sensors):
                    if abs(i - j) % num_sensors == 1:
                        Y[i, :] += (noise_color_coeff / np.sqrt(2)) * X_noise_type[j, :]
            X_noise_type = Y
        elif correlation_mode == "diagonal":
            X_noise_type[1::3, :] *= 1 - noise_color_coeff
            X_noise_type[2::3, :] *= 1 + noise_color_coeff
        elif correlation_mode is None:
            pass
        else:
            msg = f"correlation_mode can be either None, cholesky, banded or diagonal, but was {correlation_mode}"
            raise AttributeError(msg)

        rms_noise = rms(X_noise_type)
        rms_signal = rms(X_clean_type)
        scaler = rms_signal / (snr_linear * rms_noise)

        X_full[type_indices] = X_clean_type + X_noise_type * scaler

    return X_full


def add_error(leadfield, forward_error, gradient, rng):
    n_chans, n_dipoles = leadfield.shape
    noise = rng.uniform(-1, 1, (n_chans, n_dipoles)) @ gradient
    leadfield_mix = leadfield / np.linalg.norm(
        leadfield
    ) + forward_error * noise / np.linalg.norm(noise)
    return leadfield_mix


def rms(x):
    return np.sqrt(np.mean(x**2))


def make_sensor_noise_covariance(
    n_chans,
    mode=None,
    noise_color_coeff=0.5,
    rng=None,
    low_rank_dim=4,
    eps=1e-12,
):
    """Create a PSD spatial covariance matrix for sensor noise.

    Parameters
    ----------
    n_chans : int
        Number of channels.
    mode : None | str
        One of {None, "cholesky", "banded", "diagonal", "low_rank"}.
    noise_color_coeff : float
        Strength of spatial correlation/coloring.
    rng : numpy.random.Generator | None
        Random number generator.
    low_rank_dim : int
        Latent rank used when ``mode="low_rank"``.
    eps : float
        Diagonal jitter for numerical safety.
    """
    if rng is None:
        rng = np.random.default_rng()

    coeff = float(np.clip(noise_color_coeff, 0.0, 0.999))

    if mode is None:
        cov = np.eye(n_chans)
    elif mode == "cholesky":
        cov = np.full((n_chans, n_chans), coeff)
        np.fill_diagonal(cov, 1.0)
    elif mode == "banded":
        idx = np.arange(n_chans)
        dist = np.abs(idx[:, None] - idx[None, :])
        cov = coeff**dist
        np.fill_diagonal(cov, 1.0)
    elif mode == "diagonal":
        variances = np.ones(n_chans)
        variances[1::3] *= max(1.0 - coeff, eps)
        variances[2::3] *= 1.0 + coeff
        cov = np.diag(variances)
    elif mode == "low_rank":
        latent_rank = int(np.clip(low_rank_dim, 1, max(1, n_chans - 1)))
        basis = rng.standard_normal((n_chans, latent_rank))
        low_rank = (basis @ basis.T) / max(latent_rank, 1)
        low_rank = low_rank / max(np.trace(low_rank) / n_chans, eps)
        cov = (1.0 - coeff) * np.eye(n_chans) + coeff * low_rank
    else:
        msg = (
            "mode must be one of None, 'cholesky', 'banded', 'diagonal', "
            f"'low_rank', but got {mode!r}"
        )
        raise ValueError(msg)

    cov = 0.5 * (cov + cov.T)
    # Normalize to unit average variance to keep coefficient effects interpretable.
    cov = cov / max(np.trace(cov) / n_chans, eps)
    cov = cov + eps * np.eye(n_chans)
    return cov


def make_rank_projector(n_chans, rank_deficiency=0, rng=None):
    """Create an orthogonal projector P = I - U U^T with controlled rank loss."""
    if rng is None:
        rng = np.random.default_rng()

    k = int(np.clip(rank_deficiency, 0, n_chans - 1))
    if k == 0:
        return np.eye(n_chans), np.zeros((n_chans, 0))

    basis = rng.standard_normal((n_chans, k))
    q, _ = np.linalg.qr(basis)
    U = q[:, :k]
    P = np.eye(n_chans) - U @ U.T
    P = 0.5 * (P + P.T)
    return P, U


def sample_sensor_noise(
    covariance,
    n_timepoints,
    rng=None,
    temporal_beta=0.0,
    eps=1e-12,
):
    """Sample sensor noise with desired spatial covariance and temporal color."""
    if rng is None:
        rng = np.random.default_rng()

    covariance = np.asarray(covariance, dtype=float)
    n_chans = covariance.shape[0]

    if np.isscalar(temporal_beta):
        beta = np.full(n_chans, float(temporal_beta))
    else:
        beta = np.asarray(temporal_beta, dtype=float)
        if beta.shape[0] != n_chans:
            msg = f"temporal_beta length {beta.shape[0]} must match n_chans={n_chans}"
            raise ValueError(msg)

    if np.allclose(beta, 0.0):
        base = rng.standard_normal((n_chans, n_timepoints))
    else:
        base = powerlaw_noise(beta, n_timepoints, n_signals=n_chans, rng=rng)

    # Normalize each channel before spatial mixing.
    std = np.std(base, axis=1, keepdims=True)
    base = base / np.maximum(std, eps)

    # Numerically safe covariance factorization.
    covariance = 0.5 * (covariance + covariance.T)
    evals, evecs = np.linalg.eigh(covariance)
    evals = np.maximum(evals, eps)
    sqrt_cov = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
    noise = sqrt_cov @ base
    return noise


def empirical_covariance(noise, shrinkage=0.0, eps=1e-12):
    """Estimate covariance from noise samples with optional Ledoit-style shrinkage."""
    noise = np.asarray(noise, dtype=float)
    n_chans, n_time = noise.shape
    demeaned = noise - noise.mean(axis=1, keepdims=True)
    cov = (demeaned @ demeaned.T) / max(n_time - 1, 1)
    cov = 0.5 * (cov + cov.T)

    gamma = float(np.clip(shrinkage, 0.0, 1.0))
    if gamma > 0:
        target = np.trace(cov) / n_chans
        cov = (1.0 - gamma) * cov + gamma * target * np.eye(n_chans)

    if eps > 0:
        cov = cov + eps * np.eye(n_chans)
    return cov


def scale_noise_to_snr(signal, noise, snr_db, eps=1e-12):
    """Scale noise to match target SNR (in dB) for a given signal."""
    signal_power = float(np.mean(np.asarray(signal, dtype=float) ** 2))
    noise_power = float(np.mean(np.asarray(noise, dtype=float) ** 2))
    snr_linear = float(10 ** (float(snr_db) / 10.0))

    scale = np.sqrt(signal_power / max(snr_linear * noise_power, eps))
    noise_scaled = noise * scale
    realized_snr = 10.0 * np.log10(
        signal_power / max(float(np.mean(noise_scaled**2)), eps)
    )
    return noise_scaled, scale, realized_snr
