(() => {
  const METRIC_LABELS = {
    mean_localization_error: "MLE (m)",
    emd: "EMD",
    spatial_dispersion: "Spatial Disp.",
    average_precision: "Avg Precision",
    correlation: "Correlation",
  };

  const HIGHER_BETTER = new Set(["average_precision", "correlation"]);

  const RANK_VIEW = "__rank__";
  const RANK_LABEL = "Rank";

  const CATEGORY_COLORS = {
    bayesian: "#3b82f6",
    beamformer: "#f97316",
    minimum_norm: "#22c55e",
    loreta: "#ef4444",
    music: "#a855f7",
    matching_pursuit: "#14b8a6",
    neural_networks: "#eab308",
    other: "#64748b",
  };

  function $(id) {
    return document.getElementById(id);
  }

  function escapeHtml(value) {
    if (value === null || value === undefined) return "";
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatNumber(value) {
    if (value === null || value === undefined || Number.isNaN(value)) return "–";
    if (typeof value !== "number") return "–";
    const fixed = value.toFixed(2);
    return fixed.replace(/\.?0+$/, "");
  }

  function formatTimestamp(isoString) {
    if (!isoString || typeof isoString !== "string") return "–";
    const d = new Date(isoString);
    if (Number.isNaN(d.getTime())) return "–";
    const months = [
      "Jan", "Feb", "Mar", "Apr", "May", "Jun",
      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    return `${months[d.getMonth()]} ${d.getDate()}, ${d.getFullYear()}`;
  }

  function toLabel(metric) {
    return METRIC_LABELS[metric] || metric.replace(/_/g, " ");
  }

  function toCategoryLabel(category) {
    const c = category || "other";
    const label = c.replace(/_/g, " ");
    return label.replace(/\b\w/g, (m) => m.toUpperCase());
  }

  function hashHue(value) {
    let hash = 0;
    const s = String(value || "");
    for (let i = 0; i < s.length; i += 1) {
      hash = (hash * 31 + s.charCodeAt(i)) >>> 0;
    }
    return hash % 360;
  }

  function categoryColor(category) {
    const key = category || "other";
    if (CATEGORY_COLORS[key]) return CATEGORY_COLORS[key];
    return `hsl(${hashHue(key)}, 70%, 50%)`;
  }

  function pageBaseUrl() {
    return new URL(".", window.location.href);
  }

  function siteRootUrl() {
    return new URL("..", pageBaseUrl());
  }

  function assetsUrl(relativeFromSiteRoot) {
    return new URL(relativeFromSiteRoot, pageBaseUrl());
  }

  function docsUrl(relativeFromSiteRoot) {
    return new URL(relativeFromSiteRoot, siteRootUrl());
  }

  async function fetchJson(url) {
    const res = await fetch(url, { cache: "no-cache" });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status} loading ${url}`);
    }
    return await res.json();
  }

  function initBenchmarksPage() {
    document.body.classList.remove("benchmarks-page");
    const app = $("benchmarks-app");
    if (!app) return;
    document.body.classList.add("benchmarks-page");

    const runSel = $("benchmarks-run");
    const datasetSel = $("benchmarks-dataset");
    const categorySel = $("benchmarks-category");
    const metricToggle = $("benchmarks-metric-toggle");
    const summaryEl = $("benchmarks-summary");
    const thead = $("benchmarks-thead");
    const tbody = $("benchmarks-tbody");

    if (
      !runSel ||
      !datasetSel ||
      !categorySel ||
      !metricToggle ||
      !summaryEl ||
      !thead ||
      !tbody
    ) {
      return;
    }

    const state = {
      runId: null,
      dataset: null,
      category: "all",
      metric: "mean_localization_error",
      metricView: RANK_VIEW,
      sortKey: "rank",
      sortMode: "best",
    };

    let manifest = null;
    let runDataCache = new Map();
    let solverPages = null;

    function setSummaryText(text) {
      summaryEl.textContent = text;
    }

    function setSummaryHtml(html) {
      summaryEl.innerHTML = html;
    }

    function formatRange(value) {
      if (typeof value === "number") return String(value);
      if (Array.isArray(value) && value.length === 2) {
        const a = value[0];
        const b = value[1];
        if (typeof a === "number" && typeof b === "number") return `${a}\u2013${b}`;
      }
      return "–";
    }

    function asInt(value) {
      if (typeof value === "number" && Number.isFinite(value)) return Math.trunc(value);
      if (typeof value === "string" && value.trim() && !Number.isNaN(Number(value))) {
        return Math.trunc(Number(value));
      }
      return null;
    }

    function datasetLabel(key, cfg) {
      if (key === "all") return "All (aggregate)";
      const name = cfg && typeof cfg.name === "string" ? cfg.name : null;
      if (name && name.toLowerCase() !== String(key).toLowerCase()) return `${name} (${key})`;
      return String(key);
    }

    function metaItem(label, value) {
      return `<div class="benchmarks-meta-item"><div class="benchmarks-meta-k">${escapeHtml(
        label
      )}</div><div class="benchmarks-meta-v">${escapeHtml(value)}</div></div>`;
    }

    function renderMetaWidget({
      runEntry,
      runData,
      datasetKey,
      metricLabel,
      sortLabel,
      sortArrow,
      solverCount,
    }) {
      const md = runData && typeof runData.metadata === "object" ? runData.metadata : {};
      const datasets = runData && typeof runData.datasets === "object" ? runData.datasets : {};
      const dsCfg =
        datasetKey && datasets && Object.prototype.hasOwnProperty.call(datasets, datasetKey)
          ? datasets[datasetKey]
          : null;

      const runName =
        (runEntry && runEntry.name) ||
        (md && typeof md.name === "string" ? md.name : null) ||
        (runEntry && runEntry.id ? String(runEntry.id) : "–");
      const timestamp = runEntry && runEntry.timestamp ? formatTimestamp(runEntry.timestamp) : "–";
      const nSamples =
        runEntry && runEntry.n_samples !== undefined && runEntry.n_samples !== null
          ? String(runEntry.n_samples)
          : md && md.n_samples !== undefined && md.n_samples !== null
            ? String(md.n_samples)
            : "–";
      const seed = md && md.random_seed !== undefined && md.random_seed !== null ? String(md.random_seed) : "–";

      const m = asInt(md && (md.m_electrodes ?? md.m));
      const n = asInt(md && (md.n_sources ?? md.n));
      const nCols = asInt(md && md.n_leadfield_columns);
      const nOrient = asInt(md && md.n_orient);

      const dsName = dsCfg && typeof dsCfg.name === "string" ? dsCfg.name : null;
      const dsDesc = dsCfg && typeof dsCfg.description === "string" ? dsCfg.description : null;
      const nActive = dsCfg ? formatRange(dsCfg.n_sources) : "–";
      const nOrders = dsCfg ? formatRange(dsCfg.n_orders) : "–";
      const snr = dsCfg ? formatRange(dsCfg.snr_range) : "–";
      const nTime = dsCfg && asInt(dsCfg.n_timepoints) !== null ? String(asInt(dsCfg.n_timepoints)) : "–";

      const datasetTitle = datasetKey ? datasetLabel(datasetKey, dsCfg) : "–";
      const showing = `${solverCount} solvers`;

      const simItems = [
        metaItem("Run", runName),
        metaItem("Timestamp", timestamp),
        metaItem("Samples", nSamples),
        metaItem("Seed", seed),
        metaItem("m (electrodes)", m === null ? "–" : String(m)),
        metaItem("n (sources)", n === null ? "–" : String(n)),
      ];

      if (nCols !== null) simItems.push(metaItem("Leadfield cols", String(nCols)));
      if (nOrient !== null) simItems.push(metaItem("Orient", String(nOrient)));

      const dsItems =
        datasetKey === "all"
          ? [metaItem("Dataset", datasetTitle)]
          : [
              metaItem("Dataset", datasetTitle),
              metaItem("Active sources", nActive),
              metaItem("Patch order", nOrders),
              metaItem("SNR range", snr),
              metaItem("Timepoints", nTime),
            ];

      const dsHeadline = dsName ? `<div class="benchmarks-meta-headline">${escapeHtml(dsName)}</div>` : "";
      const dsDescription = dsDesc
        ? `<div class="benchmarks-meta-desc">${escapeHtml(dsDesc)}</div>`
        : datasetKey === "all"
          ? `<div class="benchmarks-meta-desc">Aggregate across datasets; parameters vary.</div>`
          : "";

      const footer = `<div class="benchmarks-meta-footer">Metric: ${escapeHtml(
        metricLabel
      )} &middot; Sort: ${escapeHtml(sortLabel)} ${escapeHtml(sortArrow)} &middot; Showing ${escapeHtml(showing)}</div>`;

      return `
        <div class="benchmarks-meta">
          <details class="benchmarks-meta-section benchmarks-meta-collapsible">
            <summary class="benchmarks-meta-title">Simulation</summary>
            <div class="benchmarks-meta-grid">${simItems.join("")}</div>
          </details>
          <div class="benchmarks-meta-section">
            <div class="benchmarks-meta-title">Dataset</div>
            ${dsHeadline}
            ${dsDescription}
            <div class="benchmarks-meta-grid">${dsItems.join("")}</div>
          </div>
          ${footer}
        </div>
      `;
    }

    function clearTable() {
      thead.innerHTML = "";
      tbody.innerHTML = "";
    }

    async function loadSolverPages() {
      if (solverPages) return solverPages;
      try {
        const url = assetsUrl("../assets/benchmarks/solver_pages.json");
        solverPages = await fetchJson(url);
      } catch (_e) {
        solverPages = {};
      }
      return solverPages;
    }

    function getSolverHref(solverName) {
      if (!solverPages) return null;
      const rel = solverPages[solverName];
      if (!rel || typeof rel !== "string") return null;
      return docsUrl(rel).toString();
    }

    function renderMetricButtons(metrics) {
      metricToggle.innerHTML = "";

      const ordered = [...metrics].sort((a, b) => toLabel(a).localeCompare(toLabel(b)));
      const view = state.metricView;

      const buttons = [
        { key: RANK_VIEW, label: RANK_LABEL },
        ...ordered.map((m) => ({ key: m, label: toLabel(m) })),
      ];

      for (const { key, label } of buttons) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = label;
        btn.dataset.metric = key;
        if (key === RANK_VIEW) btn.title = "Average rank across metrics (lower is better).";
        if (key === view) btn.classList.add("active");
        btn.addEventListener("click", () => {
          if (key === RANK_VIEW) {
            state.metricView = RANK_VIEW;
            state.sortKey = "rank";
            state.sortMode = "best";
          } else {
            state.metric = key;
            state.metricView = key;
            state.sortKey = "median";
            state.sortMode = "best";
          }
          for (const b of metricToggle.querySelectorAll("button")) {
            b.classList.toggle("active", b.dataset.metric === key);
          }
          render();
        });
        metricToggle.appendChild(btn);
      }
    }

    function populateSelect(select, options, { includeAll = false, allLabel = "All" } = {}) {
      select.innerHTML = "";
      if (includeAll) {
        const opt = document.createElement("option");
        opt.value = "all";
        opt.textContent = allLabel;
        select.appendChild(opt);
      }
      for (const o of options) {
        const opt = document.createElement("option");
        opt.value = o.value;
        opt.textContent = o.label;
        select.appendChild(opt);
      }
    }

    function getSelectedRunEntry() {
      if (!manifest) return null;
      return (manifest.runs || []).find((r) => r.id === state.runId) || null;
    }

    async function loadRunData(runId) {
      if (runDataCache.has(runId)) return runDataCache.get(runId);
      const url = assetsUrl(`../assets/benchmarks/${runId}.json`);
      const data = await fetchJson(url);
      runDataCache.set(runId, data);
      return data;
    }

    function getMetricStat(row, field) {
      const metrics = row.metrics || {};
      const stat = metrics[state.metric] || {};
      const value = stat[field];
      if (value === null || value === undefined) return null;
      if (typeof value !== "number" || Number.isNaN(value)) return null;
      return value;
    }

    function compareText(a, b, sortMode) {
      const av = a === null || a === undefined ? "" : String(a);
      const bv = b === null || b === undefined ? "" : String(b);
      const cmp = av.localeCompare(bv, undefined, { numeric: true, sensitivity: "base" });
      return sortMode === "worst" ? -cmp : cmp;
    }

    function normalizeNumberForDir(value, dir) {
      if (value === null || value === undefined) return dir === 1 ? Infinity : -Infinity;
      if (typeof value !== "number" || Number.isNaN(value)) return dir === 1 ? Infinity : -Infinity;
      return value;
    }

    function compareNumber(a, b, dir) {
      const av = normalizeNumberForDir(a, dir);
      const bv = normalizeNumberForDir(b, dir);
      return dir === 1 ? av - bv : bv - av;
    }

    function isMetricValueKey(key) {
      return key === "mean" || key === "median" || key === "worst_10_pct";
    }

    function numericDirForKey(key, hb, sortMode) {
      let dir = 1;
      if (isMetricValueKey(key)) dir = hb ? -1 : 1;
      else dir = 1;
      if (sortMode === "worst") dir *= -1;
      return dir;
    }

    function effectiveSortDir(key, hb, sortMode) {
      if (key === "solver" || key === "category") {
        return sortMode === "worst" ? "desc" : "asc";
      }
      const dir = numericDirForKey(key, hb, sortMode);
      return dir === 1 ? "asc" : "desc";
    }

    function renderTableHeader(columns, hb) {
      thead.innerHTML = "";
      for (const col of columns) {
        const th = document.createElement("th");
        th.scope = "col";
        th.textContent = col.label;
        if (col.type === "number") th.classList.add("num");

        if (col.sortable) {
          th.classList.add("sortable");
          const indicator = document.createElement("span");
          indicator.className = "benchmarks-sort-indicator";
          th.appendChild(indicator);

          if (col.key === state.sortKey) {
            th.classList.add("sorted");
            th.dataset.sortDir = effectiveSortDir(col.key, hb, state.sortMode);
            th.setAttribute(
              "aria-sort",
              th.dataset.sortDir === "desc" ? "descending" : "ascending"
            );
          } else {
            th.setAttribute("aria-sort", "none");
          }

          th.addEventListener("click", () => {
            if (state.sortKey === col.key) {
              state.sortMode = state.sortMode === "best" ? "worst" : "best";
            } else {
              state.sortKey = col.key;
              state.sortMode = "best";
            }
            render();
          });
        } else {
          th.setAttribute("aria-sort", "none");
        }

        thead.appendChild(th);
      }
    }

    function render() {
      const runEntry = getSelectedRunEntry();
      if (!runEntry) return;

      loadRunData(runEntry.id)
        .then((data) => {
          const results = Array.isArray(data.results) ? data.results : [];
          const perDatasetRanks = data.ranks && typeof data.ranks === "object" ? data.ranks : {};
          const globalRanks =
            data.global_ranks && typeof data.global_ranks === "object" ? data.global_ranks : {};

          const filteredByDataset = results.filter(
            (r) => r && r.dataset_name === state.dataset
          );
          const filtered =
            state.category === "all"
              ? filteredByDataset
              : filteredByDataset.filter((r) => (r.category || "other") === state.category);

          const view = state.metricView;
          const showRank = view === RANK_VIEW;
          const hb = HIGHER_BETTER.has(state.metric);

          const metricValue = (r, field) => getMetricStat(r, field);

          // Rank accessor: global_ranks when dataset="all", per-dataset ranks otherwise
          const rankFor = (r) => {
            if (state.dataset === "all") {
              const v = globalRanks[r.solver_name];
              return typeof v === "number" && !Number.isNaN(v) ? v : null;
            }
            const ds = perDatasetRanks[state.dataset] || {};
            const v = ds[r.solver_name];
            return typeof v === "number" && !Number.isNaN(v) ? v : null;
          };

          const columns = showRank
            ? [
                { key: "#", label: "#", sortable: false },
                { key: "solver", label: "Solver", sortable: true, type: "text" },
                { key: "category", label: "Category", sortable: true, type: "text" },
                { key: "rank", label: "Rank", sortable: true, type: "number" },
              ]
            : [
                { key: "#", label: "#", sortable: false },
                { key: "solver", label: "Solver", sortable: true, type: "text" },
                { key: "category", label: "Category", sortable: true, type: "text" },
                { key: "median", label: "Median", sortable: true, type: "number" },
                { key: "mean", label: "Mean", sortable: true, type: "number" },
                { key: "worst_10_pct", label: "Worst 10%", sortable: true, type: "number" },
              ];

          const sortableKeys = new Set(columns.filter((c) => c.sortable).map((c) => c.key));
          if (!sortableKeys.has(state.sortKey)) {
            state.sortKey = showRank ? "rank" : "median";
            state.sortMode = "best";
          }

          const sortKey = state.sortKey;
          const sortMode = state.sortMode;

          filtered.sort((a, b) => {
            if (sortKey === "solver") {
              const cmp = compareText(a.solver_name, b.solver_name, sortMode);
              if (cmp !== 0) return cmp;
              return compareText(a.solver_name, b.solver_name, "best");
            }
            if (sortKey === "category") {
              const ca = a.category || "other";
              const cb = b.category || "other";
              const cmp = compareText(ca, cb, sortMode);
              if (cmp !== 0) return cmp;
              return compareText(a.solver_name, b.solver_name, "best");
            }
            if (sortKey === "rank") {
              const dir = numericDirForKey(sortKey, false, sortMode);
              const ar = rankFor(a);
              const br = rankFor(b);
              const cmp = compareNumber(ar, br, dir);
              if (cmp !== 0) return cmp;
              return compareText(a.solver_name, b.solver_name, "best");
            }
            if (isMetricValueKey(sortKey)) {
              const dir = numericDirForKey(sortKey, hb, sortMode);
              const cmp = compareNumber(metricValue(a, sortKey), metricValue(b, sortKey), dir);
              if (cmp !== 0) return cmp;
              return compareText(a.solver_name, b.solver_name, "best");
            }
            return compareText(a.solver_name, b.solver_name, "best");
          });

          const sortLabel = columns.find((c) => c.key === sortKey)?.label || sortKey;
          const sortArrow = effectiveSortDir(sortKey, hb, sortMode) === "desc" ? "↓" : "↑";

          const metricLabel = showRank ? RANK_LABEL : toLabel(state.metric);

          setSummaryHtml(
            renderMetaWidget({
              runEntry,
              runData: data,
              datasetKey: state.dataset,
              metricLabel,
              sortLabel,
              sortArrow,
              solverCount: filtered.length,
            })
          );

          renderTableHeader(columns, hb);

          tbody.innerHTML = "";
          filtered.forEach((r, idx) => {
            const tr = document.createElement("tr");

            const category = r.category || "other";

            const badge = document.createElement("span");
            badge.className = "benchmarks-badge";
            badge.style.setProperty("--benchmarks-category-color", categoryColor(category));

            const dot = document.createElement("span");
            dot.className = "benchmarks-dot";
            dot.setAttribute("aria-hidden", "true");
            badge.appendChild(dot);
            badge.appendChild(document.createTextNode(toCategoryLabel(category)));

            const solverHref = getSolverHref(r.solver_name);
            let solverNode = null;
            if (solverHref) {
              const a = document.createElement("a");
              a.className = "benchmarks-solver-link";
              a.href = solverHref;
              a.target = "_blank";
              a.rel = "noopener";
              a.title = "Open solver documentation";
              a.textContent = r.solver_name;
              const icon = document.createElement("span");
              icon.className = "benchmarks-link-icon";
              icon.setAttribute("aria-hidden", "true");
              a.appendChild(icon);
              solverNode = a;
            } else {
              solverNode = document.createTextNode(r.solver_name);
            }

            const cells = showRank
              ? [
                  { text: String(idx + 1), cls: "" },
                  { node: solverNode, cls: "" },
                  { node: badge, cls: "" },
                  { text: formatNumber(rankFor(r)), cls: "num" },
                ]
              : [
                  { text: String(idx + 1), cls: "" },
                  { node: solverNode, cls: "" },
                  { node: badge, cls: "" },
                  { text: formatNumber(metricValue(r, "median")), cls: "num" },
                  { text: formatNumber(metricValue(r, "mean")), cls: "num" },
                  { text: formatNumber(metricValue(r, "worst_10_pct")), cls: "num" },
                ];

            for (const c of cells) {
              const td = document.createElement("td");
              if (c.cls) td.className = c.cls;
              if (c.node) td.appendChild(c.node);
              else td.textContent = c.text;
              tr.appendChild(td);
            }

            tbody.appendChild(tr);
          });
        })
        .catch((e) => {
          clearTable();
          setSummaryText(`Error: ${e.message}`);
          // eslint-disable-next-line no-console
          console.error(e);
        });
    }

    async function boot() {
      setSummaryText("Loading leaderboard manifest…");
      clearTable();

      try {
        const url = assetsUrl("../assets/benchmarks/manifest.json");
        manifest = await fetchJson(url);
      } catch (e) {
        setSummaryText(`Error: ${e.message}`);
        // eslint-disable-next-line no-console
        console.error(e);
        return;
      }

      const runs = Array.isArray(manifest.runs) ? manifest.runs : [];
      if (!runs.length) {
        setSummaryText("No leaderboard runs found.");
        return;
      }

      const defaultRun = manifest.default_run || runs[0].id;
      state.runId = runs.some((r) => r.id === defaultRun) ? defaultRun : runs[0].id;

      populateSelect(
        runSel,
        runs.map((r) => ({
          value: r.id,
          label: r.name ? r.name : r.timestamp ? `${r.id} (${r.timestamp})` : r.id,
        }))
      );
      runSel.value = state.runId;

      runSel.addEventListener("change", () => {
        state.runId = runSel.value;
        state.category = "all";
        renderControlsForRun();
      });

      datasetSel.addEventListener("change", () => {
        state.dataset = datasetSel.value;
        render();
      });

      categorySel.addEventListener("change", () => {
        state.category = categorySel.value;
        render();
      });

      renderControlsForRun();
    }

    async function renderControlsForRun() {
      const runEntry = getSelectedRunEntry();
      if (!runEntry) return;

      const datasets = Array.isArray(runEntry.datasets) ? runEntry.datasets : [];
      const metrics = Array.isArray(runEntry.metrics) ? runEntry.metrics : [];

      state.dataset = datasets.includes("all") ? "all" : datasets[0];
      if (metrics.length && !metrics.includes(state.metric)) {
        state.metric = metrics[0];
      }
      if (
        state.metricView !== RANK_VIEW &&
        !metrics.includes(state.metricView)
      ) {
        state.metricView = RANK_VIEW;
      }

      // Seed the dataset selector with keys, then refine labels after run data loads.
      populateSelect(datasetSel, datasets.map((d) => ({ value: d, label: String(d) })));
      datasetSel.value = state.dataset || (datasets[0] || "all");

      renderMetricButtons(metrics);
      await loadSolverPages();

      // Categories depend on the selected run's contents.
      setSummaryText("Loading leaderboard data…");
      clearTable();
      try {
        const data = await loadRunData(runEntry.id);
        const results = Array.isArray(data.results) ? data.results : [];

        // Now that we have the run payload, use dataset configs (if present) for labels.
        const dsCfgs = data && typeof data.datasets === "object" ? data.datasets : {};
        populateSelect(
          datasetSel,
          datasets.map((d) => ({ value: d, label: datasetLabel(d, dsCfgs ? dsCfgs[d] : null) })),
          { includeAll: false }
        );
        if (!datasets.includes(state.dataset)) state.dataset = datasets.includes("all") ? "all" : datasets[0];
        datasetSel.value = state.dataset;

        const cats = Array.from(
          new Set(
            results
              .filter((r) => r && typeof r.category === "string")
              .map((r) => r.category)
          )
        )
          .filter(Boolean)
          .sort();
        populateSelect(
          categorySel,
          cats.map((c) => ({ value: c, label: toCategoryLabel(c) })),
          { includeAll: true, allLabel: "All" }
        );
        categorySel.value = "all";
      } catch (e) {
        // fall back to only "All" category
        populateSelect(categorySel, [], { includeAll: true, allLabel: "All" });
        categorySel.value = "all";
      }

      render();
    }

    boot();
  }

  // MkDocs Material supports SPA-like navigation; re-init on page change if present.
  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(() => initBenchmarksPage());
  } else {
    document.addEventListener("DOMContentLoaded", () => initBenchmarksPage());
  }
})();
