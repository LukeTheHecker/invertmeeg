# Changelog


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
