# Changelog

## 1.3.0 — Mixed-norm solvers and numerical corrections

- Add irMxNE (`irmxne`) for reweighted sparse estimation and TF-MxNE
  (`tf-mxne`) for time-frequency sparse estimation. Both are opt-in through
  `Solver` and support fixed, PCA, and free orientations.
- Build SSP projectors with the current MNE API so EEG average reference and
  other configured projections are applied in the inverse pipeline.
- Keep the anatomical subject from the forward model in MNE source estimates
  and convert free-orientation vector output to Cartesian components.
- Use consistent posterior statistics for Champagne's diagonal MM noise update
  and correct the full-noise geodesic covariance update.
- Keep the cMEM objective and gradient consistent at high energies and align
  VB-SBL data to the channels selected by whitening.
- Normalize covariance eigenweights in four ESMV variants to preserve output
  scaling and keep SESAME's best particle aligned with its pre-resampling residual.
- Update the locked Tornado version used by the optional notebook dependencies.

## 1.2.0 — Python 3.14 maintenance update

- Require Python 3.14 or newer and refresh the complete dependency lockfile.
- Remove the obsolete 2024 requirements snapshot; use `uv sync --locked`.
- Separate development tools from optional neural-network, GUI, and notebook dependencies.
- Validate tests, lint, types, documentation, package builds, and dependency advisories in CI.
- Group weekly dependency updates to reduce maintenance noise.


# 0.1.0 11.10.2024
* Package underwent a complete makeover with new solvers and improved/ fixed mathematical formulations

# May 2024
The package was taken offline due to a makeover

# 0.0.7 27.04.2024
* Updated to python 3.10.11
* Many changes such as
  * added new solvers
  * added more sophisticated simulations


# 0.0.6 03.01.2022
* fixed an error in RAP- and TRAP-MUSIC resulting in a potentially erroneous extra source found
* standardized the final estimation of the current source density with all recursive MUSIC approaches
* speeding up calculations of the singular values on multiple occasions
* Added the FUN and HS Champagne inverse solutions

## 0.0.5 31.12.2022
* removed automated regularisation from sloreta due to unexpected behaviors
* refactored FLEX-MUSIC with minor speed improvements
* fixed an error in the final estimation of the current source density with FLEX-MUSIC
* added new Champagne solvers: "MacKay Champagne" and "Convexity Champagne"

## 0.0.4 30.12.2022
* Change in FLEX-MUSIC: use selected dipoles/ smooth patches as source covariance during WMNE inversion
  * before: indices were selected as full flat priors for WMNE inversion
* Fixed error with the sLORETA formula

## 0.0.3 28.12.2022
* Added rank-reduction option to invert.BaseSolver class
* Set rank_reduction as default for all Beamforming approaches for better results

## 0.0.2 20.12.2022
* Added PyQt5 to the package dependencies

## 0.0.1 20.12.2022
* Release of the invert package.
