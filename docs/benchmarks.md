# Leaderboard

This page renders an interactive dashboard from benchmark artifacts generated at
docs build time. It is designed to work as a **fully static** site (e.g. GitHub
Pages).

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
