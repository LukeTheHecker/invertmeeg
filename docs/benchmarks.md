# Leaderboard

The default run compares development solvers on the same simulated EEG recordings across six scenarios. Each solver uses its public API and the noise covariance for that recording. Distances are displayed in millimetres when the artifact records its units.

Global rank averages four equally weighted metric ranks across scenarios. The four metrics are mean localization error, earth mover’s distance, average precision with tied predictions grouped, and unsigned waveform correlation at active vertices. A rank requires valid scores for every contributing case. Spatial dispersion and two-hop waveform correlation are available as diagnostics. The mixed cortical activity scenario scores recovery of the patches within diffuse background activity.

Each scenario uses 50 recordings on the fixed-orientation BioSemi32/ico3 forward model. MWE and ReMWE use a separate grouped benchmark. The development run includes methods and parameter settings outside the released package. Earlier runs remain available in the selector with their original scores.

[Download the corrected per-recording results (compressed JSON)](https://github.com/LukeTheHecker/invertmeeg/raw/refs/heads/main/results/release/leaderboard-protocol-v2.json.gz)

## Choosing a solver

- **One compact source.** Select *Focal Source* and compare mean localization error with MNE as a baseline. Check average precision too if the extent of the estimated activity matters.
- **A patch of active cortex.** Select *Extended Source* and compare earth mover’s distance with a distributed baseline such as eLORETA. This metric penalizes placing estimated activity far from its true location even when the peak is correct.
- **Weak signals or few electrodes.** Start with the *Noisy* scenario (−5 to 0 dB) and compare against a baseline on the same recordings. The published run uses 32 channels throughout. For a sparse montage, repeat the comparison with your electrode positions and noise level before choosing a method.

## Dashboard

<div id="benchmarks-app" class="benchmarks-app">
  <div class="benchmarks-controls">
    <label>
      Run
      <select id="benchmarks-run"></select>
    </label>
    <label>
      Dataset
      <select id="benchmarks-dataset"></select>
    </label>
    <label>
      Category
      <select id="benchmarks-category"></select>
    </label>
    <label>
      Metric
      <div id="benchmarks-metric-toggle" class="benchmarks-metric-toggle"></div>
    </label>
  </div>

  <div id="benchmarks-summary" class="benchmarks-summary"></div>

  <div class="benchmarks-table-wrap">
    <table class="benchmarks-table">
      <thead><tr id="benchmarks-thead"></tr></thead>
      <tbody id="benchmarks-tbody"></tbody>
    </table>
  </div>

  <noscript>
    This dashboard requires JavaScript.
  </noscript>
</div>
