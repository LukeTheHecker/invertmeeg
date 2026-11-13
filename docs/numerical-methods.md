# Projection, rank, regularization and depth

These corrections were introduced in invertmeeg 1.4.0.

## SSP and sensor rank

Pass the data object when constructing an inverse if its SSP projections differ
from those stored in the forward model:

```python
solver = Solver("mne")
solver.make_inverse_operator(forward, evoked, noise_cov=noise_cov)
stc = solver.apply_inverse_operator(evoked)
```

The shared sensor pipeline uses the data's projections, falling back to the
forward's projections. It aligns projector channels with the retained forward
channels and applies SSP even when no noise covariance is supplied. Invalid
projection metadata raises an error instead of silently disabling projection.
The shared signal-rank reduction removes SSP components before estimating or
truncating the signal subspace. GCV and trace normalization use the retained
sensor rank, including when the projection matrix remains square.

This matters for an average EEG reference or artifact SSP: removed components
must not consume the retained signal rank or contribute to residual degrees of
freedom. Noise whitening still uses the projected noise covariance. These are
consistent with [MNE's rank and projection conventions](https://mne.tools/stable/generated/mne.beamformer.make_lcmv.html).

## ESMV

ESMV variants honor the requested integer signal rank. Automatic rank uses
scale-normalized spectra and a determinant-based corner calculation. The
covariance-weighted subspace is dimensionless; iterative ESMV computes its
inverse in normalized covariance units. Changing the units of the data should
scale source amplitudes by the same factor without changing the selected
subspace or iterative weights. The `rank` setting controls signal-subspace
truncation; it is distinct from the rank of the noise whitener.

## Regularization

Existing solver defaults and regularization grids are retained. Proposed global
switches to RGCV with gamma 0.3 or condition-number loading with target 10 showed
scenario-dependent losses in paired benchmarks.

For minimum-norm solvers, RGCV and R1GCV remain explicit options:

```python
solver = Solver("sloreta", regularisation_method="RGCV", rgcv_gamma=0.3)
```

RGCV uses the trace of the squared influence operator. With colored noise,
that operator need not be symmetric in the original sensor coordinates, so its
squared Frobenius norm is not a valid substitute. sLORETA now routes these
methods through the shared selector while applying its standardization only
after selecting the underlying MNE kernel.

## MNE depth prior

```python
solver = Solver("mne", depth=0.8, depth_limit=10.0)
```

`depth=None` preserves the existing default. Explicit `depth=0` selects an
identity source prior; values between 0 and 1 increase compensation for low
sensitivity sources. `depth_limit` must be at least 1. The prior is computed
after channel selection, SSP and noise whitening. For free orientation, each
three-column source block shares a prior based on its largest squared singular
value, preserving invariance to rotation of that source's orientation basis.

With effective gain `G`, block sensitivity `s`, exponent `d` and limit `l`, the
prior is `R = max(s / max(s), 1 / l**2)**(-d)`. The kernel is
`R G.T (G R G.T + alpha I)^-1`, mapped back through the sensor transform. This
also restores physical source amplitudes when trace normalization is used.
Explicit `depth` replaces the legacy `use_depth_weighting` leadfield scaling
for MNE. It does not combine the two weighting mechanisms.

The depth prior is optional because its accuracy tradeoffs depend on source
configuration and noise. The paired validation report accompanies the change
in the repository's `results/deferred_numerics/2026-11-14/` directory.
