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
