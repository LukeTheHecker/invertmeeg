# Simulation Realism Features (invert.simulate)

This document summarizes the "realism" features added to the simulation framework in:

- `/Users/lukas/projects/invert/invert-package/invert/simulate/simulate.py`
- `/Users/lukas/projects/invert/invert-package/invert/simulate/noise.py`
- `/Users/lukas/projects/invert/invert-package/invert/simulate/config.py`

The goal is to simulate conditions that stress real inverse methods (correlated noise, rank
deficiency, temporally colored noise, and availability of a noise covariance for whitening),
instead of only clean-room i.i.d. sensor noise.

## 1. Spatially Correlated Sensor Noise (Noise Covariance Modes)

**Intuition**
- Real EEG/MEG sensor noise is not independent across channels.
- We now generate sensor noise with a controllable spatial covariance `C_n` (PSD by construction),
  with several convenient families that capture common correlation structures.

**Configuration**
- `SimulationConfig.correlation_mode`: `None`, `"diagonal"`, `"banded"`, `"cholesky"`, `"low_rank"`
- `SimulationConfig.noise_color_coeff`: strength of correlation/coloring (float or range)
- `SimulationConfig.noise_low_rank_dim`: latent rank used for `"low_rank"` (int or range)

**Math**
We sample noise in sensor space as:

1. Draw a base process `Z` with shape `(m, T)` (m sensors, T time samples), with unit variance per
   channel (before mixing).
2. Apply a spatial mixing matrix `L` such that:

   `C_n = L L^T`  (PSD)

3. Form sensor noise:

   `N = L Z`

In code, `L` is built via an eigen-decomposition of `C_n`:

- `C_n = V diag(λ) V^T`
- `L = V diag(sqrt(max(λ, eps))) V^T`

This is implemented in:
- `/Users/lukas/projects/invert/invert-package/invert/simulate/noise.py:139` (`make_sensor_noise_covariance`)
- `/Users/lukas/projects/invert/invert-package/invert/simulate/noise.py:221` (`sample_sensor_noise`)

**Mode details**
- `None`: `C_n = I`
- `"diagonal"`: channel-dependent variances (heteroscedastic noise)
- `"banded"`: Toeplitz-like correlations, roughly `C_ij ~ coeff^{|i-j|}`
- `"cholesky"`: constant off-diagonal correlation (all pairs share the same correlation coefficient)
- `"low_rank"`: `C_n = (1-coeff) I + coeff * (A A^T)` (normalized), capturing structured low-rank noise

## 2. Temporally Colored Sensor Noise (1/f^beta)

**Intuition**
- Real sensor noise is typically not temporally white; it often has 1/f structure.
- We can color the noise time series per sensor using a power-law exponent `beta`.

**Configuration**
- `SimulationConfig.noise_temporal_beta`: float or `(min, max)` range.

**Math**
We use FFT-based spectral shaping so that the power spectrum scales like:

`S(f) ∝ 1 / f^beta`

Implementation sketch:
- Generate white noise in time.
- Transform to frequency domain.
- Multiply by `1 / f^(beta/2)` (amplitude shaping).
- Inverse FFT back to time domain.

This is implemented by reusing:
- `/Users/lukas/projects/invert/invert-package/invert/simulate/noise.py:4` (`powerlaw_noise`)
- `/Users/lukas/projects/invert/invert-package/invert/simulate/noise.py:221` (`sample_sensor_noise`)

## 3. Rank Deficiency via Sensor-Space Projectors

**Intuition**
- In practice, preprocessing steps (SSP/SSS, CAR, artifact subspace removal) reduce data rank.
- Whitening and inverse solvers must respect this rank deficiency; otherwise projected-out
  subspaces can be accidentally over-weighted.
- We now simulate rank loss by applying an orthogonal projector.

**Configuration**
- `SimulationConfig.noise_rank_deficiency`: integer or `(min, max)` range
- `SimulationConfig.apply_sensor_projector`: apply `P` to both the signal and the noise

**Math**
We construct an orthogonal projector:

- Draw an orthonormal basis `U ∈ R^{m×k}` (`U^T U = I`)
- Define:

  `P = I - U U^T`

Properties:
- `P^T = P` (symmetric)
- `P^2 = P` (idempotent)
- `rank(P) = m - k` (rank deficiency `k`)

We then apply `P` to the simulated sensor data:

`X = P (X_clean + N)`

Implementation:
- `/Users/lukas/projects/invert/invert-package/invert/simulate/noise.py:204` (`make_rank_projector`)
- `/Users/lukas/projects/invert/invert-package/invert/simulate/simulate.py:277` (applies `P`)

## 4. Explicit SNR Calibration (Per Sample)

**Intuition**
- Many simulators add “some noise” and label it SNR, but realized SNR varies widely.
- Here we scale the realized noise per sample so the achieved SNR matches the configured target.

**Configuration**
- `SimulationConfig.snr_range`: (min_dB, max_dB) sampled per simulation.

**Math**
Given a clean (possibly projected) signal `S` and raw noise `N`, define:

- `snr_linear = 10^(snr_db/10)`
- Signal power: `P_s = mean(S^2)`
- Noise power:  `P_n = mean(N^2)`

Choose a scale `a` so that:

`P_s / mean((aN)^2) = snr_linear`

So:

`a = sqrt(P_s / (snr_linear * P_n))`

This is implemented in:
- `/Users/lukas/projects/invert/invert-package/invert/simulate/noise.py:279` (`scale_noise_to_snr`)
- `/Users/lukas/projects/invert/invert-package/invert/simulate/simulate.py:348` (per-sample scaling)

The simulator records both:
- `snr` (target)
- `snr_realized` (post-scaling)

## 5. True vs Estimated Noise Covariance (for Whitening Pipelines)

**Intuition**
- Many inverse methods (including MNE-style pipelines) need a noise covariance `C_n` for whitening.
- In real data you don’t know the true `C_n`; you estimate it from baseline segments.
- The simulator now produces (optionally) both:
  - a “true” covariance derived from the realized noise of the sample
  - an “estimated” covariance from an independent baseline noise segment

**Configuration**
- `SimulationConfig.return_noise_cov`: store covariance matrices in metadata (can be heavy)
- `SimulationConfig.estimate_noise_cov`: whether to estimate `noise_cov_est`
- `SimulationConfig.noise_cov_n_baseline`: baseline length `T0` for estimation
- `SimulationConfig.noise_cov_shrinkage`: shrinkage factor `γ`

**Math (empirical covariance)**
Given baseline noise `N0 ∈ R^{m×T0}`, centered per channel:

- `N0c = N0 - mean_t(N0)`
- Sample covariance:

  `S = (N0c N0c^T) / (T0 - 1)`

**Math (shrinkage)**
We apply a simple diagonal shrinkage toward an isotropic target:

- `μ = tr(S) / m`
- `C_hat = (1-γ) S + γ μ I`

Implementation:
- `/Users/lukas/projects/invert/invert-package/invert/simulate/noise.py:261` (`empirical_covariance`)
- `/Users/lukas/projects/invert/invert-package/invert/simulate/simulate.py:357` (baseline + shrinkage)

**Important note about rank**
- The *true* realized covariance is computed with `eps=0.0` so rank deficiency remains visible.
- The *estimated* covariance includes shrinkage and a tiny diagonal jitter (via `eps` in the estimator),
  so it can become full-rank even if the projected process is rank-deficient.

This mirrors a real-world situation: estimators often regularize covariances for numerical stability.

## 6. Metadata Added to `SimulationGenerator.generate()`

The generator still yields `(x, y, info)` where `info` is a `pandas.DataFrame`.
New columns include (depending on configuration):

- `snr` / `snr_realized`
- `correlation_mode`, `noise_color_coeff`
- `noise_temporal_beta`
- `noise_rank_deficiency`
- `projector_rank`
- `noise_cov_rank_true`, `noise_cov_rank_est`
- Optional heavy objects when `return_noise_cov=True`:
  - `projector` (the projector matrix `P`)
  - `noise_cov_true` (empirical cov from realized noise)
  - `noise_cov_est` (baseline-estimated cov)

These are produced in:
- `/Users/lukas/projects/invert/invert-package/invert/simulate/simulate.py:400` (`_build_metadata`)

## 7. How to Use This in Solver Benchmarks

**Intuition**
- Solvers that support whitening / noise covariance can now be tested in a realistic way by feeding
  them `noise_cov_est` (estimated from baseline), rather than assuming `C_n = I`.

**Example**
For a single sample:

1. Generate data with `return_noise_cov=True` and `estimate_noise_cov=True`.
2. Extract the estimated noise covariance:

   `noise_cov = info.iloc[i]["noise_cov_est"]`

3. Pass it into solvers that accept `noise_cov`:

   `solver.make_inverse_operator(forward, alpha="auto", noise_cov=noise_cov)`

This is the mechanism used to compare `dSPM` vs `dSPM-MNE` under realistic correlated noise.

