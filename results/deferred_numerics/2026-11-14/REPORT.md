# Deferred numerical corrections — 2026-11-14

Base: v1.3.0, release tree at 97bb91c0b002067ac85429d691faa20e8cd3bc06.
The preserved baseline checkout is the identical-tree release candidate 4f3edf76dbd8fcafc42a85c887565a58bca303a3.

## Decisions

- Ship the shared SSP API correction, projection before rank selection, effective-rank GCV, and SSP in previously bypassed no-covariance paths. Invalid projector construction now raises by default.
- Ship stable automatic rank calculation and dimensionless ESMV covariance weighting. IR variants normalize the covariance before inversion so absolute numerical floors cannot change weights with sensor units.
- Ship corrected RGCV influence trace and sLORETA dispatch. Retain existing regularization defaults and grids.
- Provide MNE depth as an explicit source prior computed after channel selection and whitening. Keep depth=None by default; depth=0 is identity. Free-orientation weights are shared within each three-component source block. Preserve physical amplitudes through trace normalization.
- Reject the proposed global RGCV gamma=0.3, grid=20 and condition-target=10 policies. Retain OAS defaults. These settings improve some cases but deteriorate others (tables below).
- Reject importing the research LOOCV prototype: its global-mean rank-one downdate does not recompute the training-fold mean and unbiased denominator. Its optional combined regularization pipeline also invalidates the assumed sample-covariance downdate. This is not an exposed release API; no new LOOCV policy is shipped.
- Keep unrelated research checkout changes out of this branch. No version tag, merge or PyPI publication is part of this preparation.

## Evidence and interpretation

The public-API harness runs every solver's make_inverse_operator and apply_inverse_operator per sample. It has no implicit depth policy and no matrix-only shortcut. It supplies each sample's own noise covariance. This matters because the existing BenchmarkRunner uses a matrix shortcut for non-recompute solvers, bypassing sLORETA's final standardization; its depth recommendations also depend on constructor signatures.

There are 3,960 paired baseline/candidate evaluations: 11 solvers x 4 scenarios x 30 samples x 3 conditions. All recorded metrics are finite. The scenarios are focal, multi-source, extended and noisy, with their SNR ranges and full sample metrics preserved in JSON. The geometry is HCP100307, ico2, 324 fixed sources and 32 EEG sensors. Seed 20261114 is shared between versions. Optional policy comparisons add 840 candidate evaluations.

With a noise covariance, all mean localization deltas are zero in the plain and SSP conditions. Only tiny ESMV differences appear in other metrics. Without a noise covariance, corrected effective-rank GCV materially improves minimum-norm localization and precision, while some spatial-dispersion scores increase. Tiny ESMV tradeoffs remain; all scenario metrics are shown below, rather than treating localization alone as a universal gate.

These are matched synthetic EEG results on one geometry, not evidence of universal superiority on real EEG/MEG recordings. Tests separately cover scale invariance, invalid ranks, rejected artifacts, colored covariance, channel subsets, fixed/free orientations, rotation invariance, dense posterior agreement and physical amplitude normalization.

### Discarded exploratory comparison

An initial SSP benchmark only attached projector metadata to unreferenced simulated data. It compared a baseline that accidentally ignored SSP (retaining the common-reference signal) against a candidate that removed it. That is not a same-information accuracy comparison. The final SSP harness projects the simulation forward, all samples and covariance consistently for both versions, and passes metadata to construction. The initial large ESMV losses therefore do not establish a solver regression. The corrected public artifact-invariance tests also exposed and fixed projection-after-rank ordering.

Earlier BenchmarkRunner-based default-policy results are exploratory only, superseded by this public-API harness. The final evidence below still rejects all proposed global default switches.

### Reproduction

Use dependencies from uv.lock and the HCP cached forward/info files from the prior math audit. The harness accepts --assets to override their location; it does not download or regenerate that geometry.

```bash
python benchmark.py --source /path/to/baseline --assets /path/to/audit-assets --out baseline.json --ssp
python benchmark.py --source /path/to/candidate --assets /path/to/audit-assets --out candidate.json --ssp
python compare.py baseline.json candidate.json --out comparison.json
```

Omit --ssp for the plain case. Add --no-noise-cov for projected data without noise whitening. Policy files contain the exact init__/make__ settings. Raw artifacts include software versions, per-solver source SHA256 hashes and dataset configurations. The subsequent source edits only document SSP behavior and validate NumPy integer ranks; default numerical paths are unchanged.

## Paired scenario means

Values are candidate minus baseline. Lower MLE/EMD/spatial dispersion is better; higher average precision/correlation is better. MLE and EMD are in millimetres. Full sample pairs and paired standard errors are in the compressed comparison JSON files. Raw JSON artifacts use deterministic gzip compression; compare.py reads either .json or .json.gz.

### Plain, with covariance

| Solver | Scenario | MLE | EMD | Dispersion | AP | Correlation |
|---|---|---:|---:|---:|---:|---:|
| mne | focal_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| dspm | focal_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| sloreta | focal_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| eloreta | focal_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| lcmv | focal_source | +0.000000 | -0.000000 | +0.000000 | +0.000000 | +0.000000 |
| lcmv-mvpure | focal_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| esmv | focal_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | -0.000000 |
| ir-esmv | focal_source | +0.000000 | -0.000000 | +0.000000 | +0.000000 | +0.000000 |
| ssp-esmv | focal_source | +0.000000 | -0.000000 | +0.000000 | +0.000000 | +0.000000 |
| ssp-ir-esmv | focal_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | -0.000000 |
| esmv-mvpure | focal_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | -0.000000 |
| mne | multi_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| dspm | multi_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| sloreta | multi_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| eloreta | multi_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| lcmv | multi_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| lcmv-mvpure | multi_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| esmv | multi_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| ir-esmv | multi_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | -0.000000 |
| ssp-esmv | multi_source | +0.000000 | -0.000000 | +0.000000 | +0.000000 | +0.000000 |
| ssp-ir-esmv | multi_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| esmv-mvpure | multi_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| mne | extended_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| dspm | extended_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| sloreta | extended_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| eloreta | extended_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| lcmv | extended_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | -0.000000 |
| lcmv-mvpure | extended_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| esmv | extended_source | +0.000000 | -0.008024 | +0.000000 | -0.000096 | +0.000238 |
| ir-esmv | extended_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | -0.000000 |
| ssp-esmv | extended_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| ssp-ir-esmv | extended_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | -0.000000 |
| esmv-mvpure | extended_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| mne | noisy | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| dspm | noisy | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| sloreta | noisy | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| eloreta | noisy | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| lcmv | noisy | +0.000000 | +0.000000 | +0.000000 | +0.000000 | -0.000000 |
| lcmv-mvpure | noisy | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| esmv | noisy | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| ir-esmv | noisy | +0.000000 | -0.000000 | +0.000000 | +0.000000 | +0.000000 |
| ssp-esmv | noisy | +0.000000 | -0.000000 | +0.000000 | +0.000000 | +0.000000 |
| ssp-ir-esmv | noisy | +0.000000 | -0.000000 | +0.000000 | +0.000000 | +0.000000 |
| esmv-mvpure | noisy | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |

### SSP, with covariance

| Solver | Scenario | MLE | EMD | Dispersion | AP | Correlation |
|---|---|---:|---:|---:|---:|---:|
| mne | focal_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| dspm | focal_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | -0.000000 |
| sloreta | focal_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| eloreta | focal_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | -0.000000 |
| lcmv | focal_source | +0.000000 | -0.000000 | +0.000000 | +0.000000 | +0.000000 |
| lcmv-mvpure | focal_source | +0.000000 | -0.000000 | +0.000000 | +0.000000 | +0.000000 |
| esmv | focal_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| ir-esmv | focal_source | +0.000000 | -0.000000 | +0.000000 | +0.000000 | -0.000000 |
| ssp-esmv | focal_source | +0.000000 | -0.000000 | +0.000000 | +0.000000 | -0.000000 |
| ssp-ir-esmv | focal_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | -0.000000 |
| esmv-mvpure | focal_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| mne | multi_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | -0.000000 |
| dspm | multi_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | -0.000000 |
| sloreta | multi_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| eloreta | multi_source | +0.000000 | -0.000000 | +0.000000 | +0.000000 | +0.000000 |
| lcmv | multi_source | +0.000000 | -0.000000 | +0.000000 | +0.000000 | +0.000000 |
| lcmv-mvpure | multi_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| esmv | multi_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| ir-esmv | multi_source | +0.000000 | -0.000000 | +0.000000 | +0.000000 | +0.000000 |
| ssp-esmv | multi_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| ssp-ir-esmv | multi_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| esmv-mvpure | multi_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| mne | extended_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| dspm | extended_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| sloreta | extended_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| eloreta | extended_source | +0.000000 | -0.000000 | +0.000000 | +0.000000 | -0.000000 |
| lcmv | extended_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | -0.000000 |
| lcmv-mvpure | extended_source | +0.000000 | -0.000000 | +0.000000 | +0.000000 | +0.000000 |
| esmv | extended_source | +0.000000 | -0.026777 | -0.005527 | +0.000042 | -0.000137 |
| ir-esmv | extended_source | +0.000000 | -0.000000 | +0.000000 | +0.000000 | +0.000000 |
| ssp-esmv | extended_source | +0.000000 | -0.000000 | +0.000000 | +0.000000 | +0.000000 |
| ssp-ir-esmv | extended_source | +0.000000 | -0.000000 | +0.000000 | +0.000000 | -0.000000 |
| esmv-mvpure | extended_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| mne | noisy | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| dspm | noisy | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| sloreta | noisy | +0.000000 | +0.000000 | +0.000000 | +0.000000 | -0.000000 |
| eloreta | noisy | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| lcmv | noisy | +0.000000 | -0.000000 | +0.000000 | +0.000000 | +0.000000 |
| lcmv-mvpure | noisy | +0.000000 | +0.000000 | +0.000000 | +0.000000 | -0.000000 |
| esmv | noisy | +0.000000 | -0.000000 | +0.000000 | +0.000000 | +0.000000 |
| ir-esmv | noisy | +0.000000 | +0.000000 | +0.000000 | +0.000000 | -0.000000 |
| ssp-esmv | noisy | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| ssp-ir-esmv | noisy | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| esmv-mvpure | noisy | +0.000000 | -0.000000 | +0.000000 | +0.000000 | -0.000000 |

### SSP, without covariance

| Solver | Scenario | MLE | EMD | Dispersion | AP | Correlation |
|---|---|---:|---:|---:|---:|---:|
| mne | focal_source | -18.938780 | -13.113242 | +13.967583 | +0.191684 | +0.081849 |
| dspm | focal_source | -12.732123 | -10.621484 | -240.950791 | +0.199295 | +0.080430 |
| sloreta | focal_source | -47.932783 | -9.574737 | +50.395313 | +0.980669 | +0.091608 |
| eloreta | focal_source | -49.284038 | -12.057841 | +49.842412 | +0.984705 | +0.100353 |
| lcmv | focal_source | +0.000000 | -0.000000 | +0.000000 | +0.000000 | +0.000000 |
| lcmv-mvpure | focal_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | -0.000000 |
| esmv | focal_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| ir-esmv | focal_source | +0.000000 | -0.000000 | +0.000000 | +0.000000 | +0.000000 |
| ssp-esmv | focal_source | +0.000000 | -0.000000 | +0.000000 | +0.000000 | -0.000000 |
| ssp-ir-esmv | focal_source | +0.000000 | -0.000000 | +0.000000 | +0.000000 | +0.000000 |
| esmv-mvpure | focal_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | -0.000000 |
| mne | multi_source | -9.844444 | -7.522925 | +4.962677 | +0.111906 | +0.079962 |
| dspm | multi_source | -7.519595 | -3.609684 | -80.250439 | +0.028437 | +0.080483 |
| sloreta | multi_source | -17.918530 | -3.962243 | +29.029219 | +0.296883 | +0.086323 |
| eloreta | multi_source | -21.447287 | -8.077302 | +26.829363 | +0.287484 | +0.094847 |
| lcmv | multi_source | +0.000000 | -0.000000 | +0.000000 | +0.000000 | -0.000000 |
| lcmv-mvpure | multi_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | -0.000000 |
| esmv | multi_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| ir-esmv | multi_source | +0.000000 | -0.000000 | +0.000000 | +0.000000 | -0.000000 |
| ssp-esmv | multi_source | +0.000000 | -0.000000 | +0.000000 | +0.000000 | -0.000000 |
| ssp-ir-esmv | multi_source | +0.000000 | -0.000000 | +0.000000 | +0.000000 | -0.000000 |
| esmv-mvpure | multi_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | -0.000000 |
| mne | extended_source | -6.802548 | -13.551904 | +2.468955 | +0.066618 | +0.115807 |
| dspm | extended_source | -2.412277 | -3.725828 | -29.249377 | +0.025465 | +0.099612 |
| sloreta | extended_source | -14.065786 | -5.406419 | +12.931722 | +0.113761 | +0.109974 |
| eloreta | extended_source | -17.041504 | -11.385862 | +12.449765 | +0.123104 | +0.117869 |
| lcmv | extended_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | -0.000000 |
| lcmv-mvpure | extended_source | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| esmv | extended_source | +0.000000 | -0.026777 | -0.005527 | +0.000042 | -0.000137 |
| ir-esmv | extended_source | -0.189163 | -0.001111 | +0.000000 | -0.000029 | -0.000125 |
| ssp-esmv | extended_source | +0.000000 | +0.110867 | +0.000000 | -0.000038 | -0.000824 |
| ssp-ir-esmv | extended_source | +0.000000 | +0.030157 | +0.000000 | -0.000258 | -0.001025 |
| esmv-mvpure | extended_source | +0.000000 | -0.000241 | +0.000000 | +0.000000 | -0.000064 |
| mne | noisy | -10.983131 | -7.949638 | +11.829902 | +0.118625 | +0.097253 |
| dspm | noisy | -11.857172 | -3.088732 | -113.003761 | +0.092770 | +0.090079 |
| sloreta | noisy | -20.802860 | -3.497083 | +58.263935 | +0.310915 | +0.097130 |
| eloreta | noisy | -21.553617 | -3.851468 | +58.824015 | +0.320597 | +0.101847 |
| lcmv | noisy | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| lcmv-mvpure | noisy | +0.000000 | +0.000000 | +0.000000 | +0.000000 | -0.000000 |
| esmv | noisy | +0.000000 | -0.000000 | +0.000000 | +0.000000 | +0.000000 |
| ir-esmv | noisy | +0.000000 | -0.000000 | +0.000000 | +0.000000 | -0.000000 |
| ssp-esmv | noisy | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| ssp-ir-esmv | noisy | +0.000000 | -0.000000 | +0.000000 | +0.000000 | +0.000000 |
| esmv-mvpure | noisy | +0.000000 | +0.000000 | +0.000000 | +0.000000 | -0.000000 |

### Proposed RGCV defaults

| Solver | Scenario | MLE | EMD | Dispersion | AP | Correlation |
|---|---|---:|---:|---:|---:|---:|
| mne | focal_source | -5.366583 | -5.225377 | +1.912511 | +0.022465 | +0.015505 |
| dspm | focal_source | +0.096997 | -0.815465 | +15.714290 | +0.039850 | +0.006823 |
| sloreta | focal_source | +0.020498 | -2.572526 | +4.600373 | +0.000000 | +0.008119 |
| eloreta | focal_source | +0.382344 | -2.274466 | +12.319253 | +0.000000 | +0.007131 |
| mne | multi_source | -3.412841 | -2.572956 | -0.111207 | +0.003048 | +0.013412 |
| dspm | multi_source | -1.569190 | -0.813836 | +10.639908 | +0.004171 | +0.003868 |
| sloreta | multi_source | +0.040992 | -1.115694 | +4.508271 | -0.004251 | +0.005428 |
| eloreta | multi_source | +0.140990 | -1.267589 | +4.941116 | -0.002442 | +0.004259 |
| mne | extended_source | -2.000268 | -2.401686 | -0.211071 | +0.019309 | +0.012158 |
| dspm | extended_source | +0.792862 | -0.871195 | +4.100345 | -0.019079 | +0.005063 |
| sloreta | extended_source | +0.306708 | -0.610147 | +1.548967 | -0.016176 | +0.006066 |
| eloreta | extended_source | +0.700731 | -0.128578 | +1.356137 | -0.006738 | +0.003335 |
| mne | noisy | +0.561507 | -0.201551 | +1.137540 | -0.009803 | +0.001243 |
| dspm | noisy | -0.375165 | +0.371906 | +13.788244 | +0.047374 | -0.000359 |
| sloreta | noisy | +2.067768 | +0.194763 | +5.682742 | -0.018580 | -0.001703 |
| eloreta | noisy | +2.171989 | +0.416099 | +7.501327 | -0.033691 | -0.000590 |

### Optional MNE depth 0.8

| Solver | Scenario | MLE | EMD | Dispersion | AP | Correlation |
|---|---|---:|---:|---:|---:|---:|
| mne | focal_source | -8.666831 | -2.263986 | +1.801521 | +0.007666 | +0.008953 |
| mne | multi_source | -2.917607 | -5.295589 | +1.865598 | -0.034067 | +0.005251 |
| mne | extended_source | -0.282183 | -5.274586 | +0.618659 | +0.002809 | -0.014379 |
| mne | noisy | +5.794450 | +2.307924 | +6.216798 | -0.079694 | -0.010825 |

### Proposed condition target 10

| Solver | Scenario | MLE | EMD | Dispersion | AP | Correlation |
|---|---|---:|---:|---:|---:|---:|
| lcmv | focal_source | -4.083424 | -4.984225 | +0.604818 | +0.000000 | +0.067211 |
| esmv | focal_source | +9.118894 | +37.274320 | +3.746904 | +0.000000 | -0.564855 |
| lcmv | multi_source | +4.810642 | -1.255485 | -6.929293 | -0.078302 | +0.007667 |
| esmv | multi_source | +4.172278 | +4.504162 | +3.181416 | -0.123517 | -0.179571 |
| lcmv | extended_source | +1.945413 | -0.954416 | -4.329894 | +0.003402 | +0.024860 |
| esmv | extended_source | +0.644094 | +2.008220 | +1.996602 | -0.017452 | -0.036652 |
| lcmv | noisy | -1.699011 | -1.246018 | -19.891533 | +0.007031 | +0.022335 |
| esmv | noisy | +3.609770 | +10.075735 | +5.414153 | -0.036023 | -0.104740 |

## Validation

- Focused mathematical/API checks: 102 passed, including free-orientation basis rotation.
- Final full suite: 852 passed, one skipped, 15 existing warnings (tests-852.txt). The earlier 846-test receipt is retained.
- Ruff lint and formatting pass; mypy reports no issues in 185 source files.
- Pre-commit hooks pass. The mypy hook was skipped locally to avoid synchronizing the shared environment; the identical mypy command ran separately and passed.
- Strict MkDocs build passes.
- Wheel and sdist build; twine metadata checks pass. The wheel imports directly and its SSP/depth APIs pass smoke checks.
- Locked-dependency audit: no known vulnerabilities found. No dependency versions changed.
- The package version is intentionally still 1.3.0 during preparation; these local verification artifacts were never uploaded.
