# Benchmark Datasets

The benchmark suite in `invert` is not a claim that one simulation prior is the truth. It is a set of stress tests. Each dataset favors some solver assumptions and penalizes others. That is exactly why the suite should contain multiple regimes.

This page explains what each dataset is designed to test, what it does not test, and why the `Mixed Cortical Activity` dataset matters for researchers who care about realistic M/EEG source imaging rather than only clean toy recovery.

## Why dataset choice matters

Inverse solvers are prior-driven estimators. Minimum-norm methods favor distributed low-energy solutions. Sparse Bayesian and dipole-style methods favor a small number of dominant sources. Beamformers favor situations where covariance structure is informative and sources are sufficiently separable.

A benchmark is therefore never neutral. If the simulated sources are all focal, sparse methods get an advantage. If the simulated sources are all diffuse, L2-style methods get an advantage. A useful benchmark suite should not pretend to remove this dependence. It should make the dependence visible.

## Focal Source

**What it tests**

- Recovery of a single compact source at moderate SNR.
- Basic localization accuracy and peak finding.
- Whether a solver over-smooths simple focal activity.

**What it does not test**

- Interactions between multiple simultaneously active sources.
- Extended cortical patches.
- Ongoing distributed background activity.
- Realistic mesoscale temporal dynamics.

This is a necessary baseline, but it is also the most forgiving dataset for methods that assume compact activity.

## Multi Source

**What it tests**

- Separation of multiple focal sources.
- Cross-talk between nearby active regions.
- Whether a solver merges or splits active generators.

**What it does not test**

- Extended patches with internal structure.
- Dense background activity.
- Strongly nonstationary or biologically structured temporal dynamics.

This dataset is useful for assessing source interference, but it still leans toward sparse source assumptions.

## Extended Source

**What it tests**

- Recovery of contiguous patch-like activity rather than isolated dipoles.
- Sensitivity to spatial extent and smoothness.
- Whether a solver produces plausible extended support without spreading everywhere.

**What it does not test**

- Dense background activity outside the main patches.
- Mesoscopic oscillatory or neural-mass dynamics.
- Broad heterogeneity in patch amplitudes.

This is a better test for cortical sheet activity than focal-source datasets, but it is still a relatively clean scene.

## Noisy

**What it tests**

- Robustness under low SNR.
- Whether solver ranking changes when the inverse problem becomes noise-limited.
- Sensitivity to covariance estimation and regularization choices.

**What it does not test**

- Rich spatial realism by itself.
- Biologically motivated temporal structure.
- Realistic coexistence of salient sources and ongoing cortical background.

This dataset is about failure behavior under adverse conditions, not about realism in a biological sense.

## Neural Mass Model

**What it tests**

- Whether solvers can recover patch activity with more structured oscillatory or spike-like temporal dynamics.
- Sensitivity to temporal patterns that are less arbitrary than colored noise.
- Performance under the same realistic sensor-noise pipeline used elsewhere in the benchmark.

**What it does not test**

- Full whole-brain realism.
- Anatomically grounded large-scale network coupling.
- Truly empirical sensor noise or subject-specific background activity.

The current neural mass model dataset is more realistic in time than the purely synthetic patch datasets, but it is still a stylized source generator. It should be interpreted as one temporal regime, not as a final proxy for real human brain activity.

## Mixed Cortical Activity

**What it tests**

- Recovery of salient contiguous cortical patches when they are embedded in diffuse ongoing activity.
- Robustness to the fact that real brains are rarely silent outside the task-relevant generators.
- Whether a solver can represent both focal salience and distributed background without collapsing to either extreme.
- Performance when patch amplitudes are heterogeneous rather than artificially equal.
- Recovery under temporally colored and spatially structured sensor noise.

**What it does not test**

- Subject-specific anatomy, conductivity errors, or preprocessing artifacts beyond the current simulator.
- Empirical resting-state network structure.
- Biophysical microscale realism at the neuron or laminar-circuit level.
- Clinical event classes such as spikes, seizures, or evoked responses with known latencies.

### Why this dataset is more realistic

Real M/EEG data rarely looks like a few clean sources on a zero background. Even in well-controlled experiments, salient task-related generators sit on top of ongoing cortical dynamics, residual spontaneous rhythms, and structured sensor noise. A benchmark that only simulates isolated sources rewards solvers for solving an artificially easy version of the problem.

`Mixed Cortical Activity` matters because it removes that simplification. It combines:

- contiguous cortical patches rather than isolated dipoles,
- low-rank patch dynamics rather than completely independent vertices,
- diffuse 1/f-like background activity,
- heavy-tailed source amplitudes, and
- realistic correlated sensor noise.

That combination better matches the practical question researchers face: can a solver recover the meaningful part of the activity without pretending the rest of the brain is off?

### Why researchers should care

Researchers often need a solver that is not just optimal under one prior, but stable under plausible departures from that prior. If a method performs well only when the ground truth is sparse and clean, that is useful to know. If it remains accurate when patches are embedded in structured background activity, that is usually more relevant for real experiments.

For that reason, `Mixed Cortical Activity` is meant to be a benchmark of practical robustness. It does not declare one theory of brain activity to be correct. It asks whether a solver still behaves sensibly when the world is less sparse, less clean, and closer to what M/EEG users actually record.

## How to interpret the leaderboard

No single dataset should decide the "best" solver. A solver that wins `Focal Source` may be too sparse for realistic distributed activity. A solver that wins `Extended Source` may blur focal events. A solver that wins `Mixed Cortical Activity` is not automatically the best everywhere, but it has passed a harder realism test than methods evaluated only on clean isolated sources.

The right reading of the leaderboard is therefore:

- use focal datasets to understand localization sharpness,
- use noisy datasets to understand robustness,
- use neural-mass datasets to understand performance under structured temporal dynamics,
- use mixed cortical activity to understand behavior in a more realistic ongoing-brain regime.
