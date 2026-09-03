(() => {
  "use strict";

  const initial = JSON.parse(document.getElementById("initial-state").textContent);
  const state = {
    overview: initial,
    selected: null,
    tab: "attacker",
    contract: "all",
    recordId: null,
    attackerPayload: null,
    evaluatorPayload: null,
    requestToken: 0,
  };
  const byId = (id) => document.getElementById(id);

  const text = (id, value, fallback = "—") => {
    byId(id).textContent = value === null || value === undefined || value === "" ? fallback : String(value);
  };

  const formatValue = (value) => {
    if (value === null || value === undefined) return "N/A";
    if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
    return String(value);
  };

  const humanize = (value) => String(value || "unspecified").replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());

  const metricPresentation = (name, value, forcedUnit = "") => {
    let label = String(name || "unspecified");
    let unit = forcedUnit;
    if (!unit && label.endsWith("_percent")) {
      label = label.slice(0, -8);
      unit = "%";
    } else if (!unit && label.endsWith("_ms")) {
      label = label.slice(0, -3);
      unit = "ms";
    } else if (!unit && label.endsWith("_m")) {
      label = label.slice(0, -2);
      unit = "m";
    } else if (unit === "ms" && label.endsWith("_ms")) {
      label = label.slice(0, -3);
    }

    let formatted;
    if (typeof value === "boolean") {
      formatted = value ? "Có" : "Không";
    } else if (Array.isArray(value)) {
      formatted = value.map((item) => formatValue(item)).join(", ");
    } else if (value && typeof value === "object") {
      formatted = JSON.stringify(value);
    } else {
      formatted = formatValue(value);
    }
    if (unit && typeof value === "number" && Number.isFinite(value)) formatted = `${formatted} ${unit}`;
    return { label: humanize(label), value: formatted };
  };

  function renderMetricGrid(gridId, values, emptyMessage, forcedUnit = "") {
    const grid = byId(gridId);
    grid.replaceChildren();
    const entries = values && typeof values === "object" && !Array.isArray(values)
      ? Object.entries(values)
      : [];
    entries.forEach(([name, value]) => {
      const card = document.createElement("div");
      card.className = "metric";
      const presentation = metricPresentation(name, value, forcedUnit);
      const label = document.createElement("span");
      label.textContent = presentation.label;
      const number = document.createElement("strong");
      number.textContent = presentation.value;
      card.append(label, number);
      grid.appendChild(card);
    });
    if (!entries.length) {
      const empty = document.createElement("p");
      empty.className = "metric-empty";
      empty.textContent = emptyMessage;
      grid.appendChild(empty);
    }
  }

  function showUnavailable(message) {
    byId("dashboard").classList.add("hidden");
    byId("empty-state").classList.remove("hidden");
    text("empty-message", message, "Artifact không tồn tại hoặc không hợp lệ.");
    const status = byId("artifact-status");
    status.textContent = "Artifact unavailable";
    status.className = "status-pill status-error";
  }

  function renderOverview() {
    const overview = state.overview;
    if (!overview || !overview.available) {
      showUnavailable(overview && overview.error);
      return;
    }
    byId("empty-state").classList.add("hidden");
    byId("dashboard").classList.remove("hidden");

    const status = byId("artifact-status");
    status.textContent = overview.status || "Artifact loaded";
    status.className = `status-pill ${overview.status === "READY" ? "status-ready" : "status-warning"}`;
    text("notice-status", overview.status || "Artifact");
    text("notice-copy", overview.disclaimer, "Luôn kiểm tra provenance trước khi diễn giải kết quả.");
    text("mobility-source", overview.provenance.mobility_source || overview.provenance.dataset);
    text("mobility-label", overview.provenance.mobility_label);
    text("mechanism-count", overview.mechanisms.length);
    text("record-count", overview.record_count);
    text("point-count", overview.provenance.point_count_label);
    const commit = overview.provenance.source_commit;
    text("source-commit", commit ? commit.slice(0, 10) : null);
    text("source-clean", overview.provenance.source_dirty_before_run === false ? "clean source at run time" : "kiểm tra dirty-state");
    text("artifact-schema", overview.schema);

    const contractSelect = byId("contract-select");
    overview.contracts.forEach((contract) => {
      const option = document.createElement("option");
      option.value = contract.id;
      option.textContent = `${contract.label} (${contract.mechanisms})`;
      contractSelect.appendChild(option);
    });
    renderMechanisms();

    const visuals = overview.artifacts;
    if (visuals.preview.available && visuals.preview.integrity_verified) {
      byId("preview-image").src = visuals.preview.url;
    } else {
      text("preview-missing", visuals.preview.error, "Preview chưa có hoặc không qua kiểm tra SHA-256.");
      byId("preview-missing").classList.remove("hidden");
    }
    if (!visuals.map.available || !visuals.map.integrity_verified) {
      text("map-missing", visuals.map.error, "Bản đồ chưa có hoặc không qua kiểm tra SHA-256.");
      byId("map-missing").classList.remove("hidden");
    }

    if (overview.mechanisms.length) selectMechanism(overview.mechanisms[0].id);
  }

  function renderMechanisms() {
    const list = byId("mechanism-list");
    list.replaceChildren();
    state.overview.mechanisms
      .filter((item) => state.contract === "all" || item.output_kind === state.contract)
      .forEach((item) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = `mechanism-button${state.selected === item.id ? " active" : ""}`;
        button.dataset.mechanism = item.id;
        const name = document.createElement("strong");
        name.textContent = item.label;
        const contract = document.createElement("small");
        contract.textContent = humanize(item.output_kind);
        button.append(name, contract);
        button.addEventListener("click", () => selectMechanism(item.id));
        list.appendChild(button);
      });
  }

  async function getJson(url) {
    const response = await fetch(url, { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }

  async function selectMechanism(mechanismId) {
    const mechanism = state.overview.mechanisms.find((item) => item.id === mechanismId);
    if (!mechanism) return;
    state.selected = mechanismId;
    state.recordId = null;
    state.attackerPayload = null;
    state.evaluatorPayload = null;
    const token = ++state.requestToken;
    renderMechanisms();
    text("selected-name", mechanism.label);
    text("selected-contract", humanize(mechanism.output_kind));
    text("selected-source", mechanism.source_method);
    text("record-badge", `${mechanism.records || 0} record${mechanism.records === 1 ? "" : "s"}`);
    text("attacker-contract", `contract: ${mechanism.output_kind}`);
    text("implementation-level", mechanism.implementation_level);
    text(
      "reproduction-status",
      mechanism.reportable_as_reproduced_sota
        ? "Đủ điều kiện báo cáo là SOTA đã tái hiện"
        : "Không được báo cáo là SOTA đã tái hiện",
    );
    text("adaptation-summary", mechanism.adaptation_summary);
    const missing = byId("missing-components");
    missing.replaceChildren();
    (mechanism.missing_components || []).forEach((component) => {
      const item = document.createElement("li");
      item.textContent = component;
      missing.appendChild(item);
    });
    byId("missing-details").classList.toggle("hidden", !missing.children.length);
    const source = mechanism.source || {};
    text(
      "source-revision",
      source.repository_revision ? `upstream revision: ${source.repository_revision}` : "upstream revision: chưa có mã nguồn công khai",
    );
    byId("event-rows").innerHTML = '<tr><td colspan="3">Đang đọc attacker view…</td></tr>';
    byId("metric-grid").textContent = "";
    byId("paper-metric-grid").textContent = "";
    byId("runtime-grid").textContent = "";
    byId("truth-summary").textContent = "";
    const recordSelect = byId("record-select");
    recordSelect.disabled = true;
    recordSelect.replaceChildren(new Option("Đang tải…", ""));

    try {
      const attacker = await getJson(`/api/mechanisms/${encodeURIComponent(mechanismId)}/attacker-view`);
      if (state.selected !== mechanismId || token !== state.requestToken) return;
      state.attackerPayload = attacker;
      populateRecords(attacker.runs || []);
      renderAttacker();
      if (state.tab === "evaluator") await loadEvaluator();
    } catch (error) {
      byId("event-rows").innerHTML = `<tr><td colspan="3">Không thể đọc mechanism: ${error.message}</td></tr>`;
    }
  }

  function populateRecords(runs) {
    const selector = byId("record-select");
    selector.replaceChildren();
    runs.forEach((run, index) => {
      const recordId = String(run.record_id || `record-${index + 1}`);
      selector.appendChild(new Option(recordId, recordId));
    });
    state.recordId = runs.length ? String(runs[0].record_id || "record-1") : null;
    selector.value = state.recordId || "";
    selector.disabled = runs.length < 2;
  }

  function selectedRun(payload) {
    const runs = (payload && payload.runs) || [];
    return runs.find((run) => String(run.record_id) === state.recordId) || runs[0] || {};
  }

  function renderAttacker() {
    const rows = byId("event-rows");
    rows.replaceChildren();
    const run = selectedRun(state.attackerPayload);
    const events = (run.attacker_view && run.attacker_view.events) || [];
    events.forEach((event) => {
      const tr = document.createElement("tr");
      const eventCell = document.createElement("td");
      eventCell.textContent = event.event_id || "—";
      const timeCell = document.createElement("td");
      timeCell.textContent = event.timestamp_s === undefined ? "—" : `${formatValue(event.timestamp_s)} s`;
      const candidateCell = document.createElement("td");
      (event.candidates || []).forEach((candidate) => {
        const chip = document.createElement("span");
        chip.className = "candidate-chip";
        chip.textContent = `${candidate.candidate_id}: ${formatValue(candidate.lat)}, ${formatValue(candidate.lon)}`;
        candidateCell.appendChild(chip);
      });
      tr.append(eventCell, timeCell, candidateCell);
      rows.appendChild(tr);
    });
    if (!rows.children.length) rows.innerHTML = '<tr><td colspan="3">Không có event công khai.</td></tr>';
  }

  async function loadEvaluator() {
    if (state.evaluatorPayload || !state.selected) {
      renderEvaluator();
      return;
    }
    const mechanismId = state.selected;
    const token = state.requestToken;
    byId("metric-grid").textContent = "Đang đọc dữ liệu evaluator-only…";
    byId("paper-metric-grid").textContent = "";
    byId("runtime-grid").textContent = "";
    try {
      const payload = await getJson(`/api/mechanisms/${encodeURIComponent(mechanismId)}/evaluation`);
      if (state.selected !== mechanismId || token !== state.requestToken) return;
      state.evaluatorPayload = payload;
      renderEvaluator();
    } catch (error) {
      byId("metric-grid").textContent = `Evaluator view không khả dụng: ${error.message}`;
      byId("paper-metric-grid").textContent = "";
      byId("runtime-grid").textContent = "";
    }
  }

  function renderEvaluator() {
    const run = selectedRun(state.evaluatorPayload);
    renderMetricGrid(
      "metric-grid",
      run.metrics,
      "Artifact không cung cấp chỉ số chẩn đoán chung cho bản ghi này.",
    );
    renderMetricGrid(
      "paper-metric-grid",
      run.paper_metrics,
      "Artifact không cung cấp chỉ số theo paper cho cơ chế này.",
    );
    const runtimeBreakdown = run.runtime_breakdown_ms && typeof run.runtime_breakdown_ms === "object"
      && !Array.isArray(run.runtime_breakdown_ms)
      ? { ...run.runtime_breakdown_ms }
      : {};
    if (runtimeBreakdown.end_to_end_runtime_ms === undefined && run.runtime_ms !== null && run.runtime_ms !== undefined) {
      runtimeBreakdown.end_to_end_runtime_ms = run.runtime_ms;
    }
    renderMetricGrid(
      "runtime-grid",
      runtimeBreakdown,
      "Artifact không cung cấp phân rã thời gian thực thi cho bản ghi này.",
      "ms",
    );
    const truth = run.evaluator_truth || {};
    const trajectory = Array.isArray(truth.real_trajectory) ? truth.real_trajectory : [];
    const realIds = Array.isArray(truth.real_candidate_ids) ? truth.real_candidate_ids : [];
    byId("truth-summary").innerHTML = [
      `<p><strong>${trajectory.length}</strong> ground-truth samples được giữ riêng cho evaluator.</p>`,
      `<p><strong>${realIds.length}</strong> real-candidate labels; các nhãn này không xuất hiện trong attacker-view endpoint.</p>`,
    ].join("");
  }

  function switchTab(tab) {
    state.tab = tab;
    document.querySelectorAll(".tab").forEach((button) => button.classList.toggle("active", button.dataset.tab === tab));
    byId("attacker-panel").classList.toggle("hidden", tab !== "attacker");
    byId("evaluator-panel").classList.toggle("hidden", tab !== "evaluator");
    if (tab === "evaluator") loadEvaluator();
  }

  document.querySelectorAll(".tab").forEach((button) => button.addEventListener("click", () => switchTab(button.dataset.tab)));
  byId("contract-select").addEventListener("change", (event) => {
    state.contract = event.target.value;
    renderMechanisms();
    const visible = state.overview.mechanisms.filter((item) => state.contract === "all" || item.output_kind === state.contract);
    if (visible.length && !visible.some((item) => item.id === state.selected)) selectMechanism(visible[0].id);
  });
  byId("record-select").addEventListener("change", (event) => {
    state.recordId = event.target.value;
    renderAttacker();
    if (state.evaluatorPayload) renderEvaluator();
  });
  byId("show-preview").addEventListener("click", () => {
    byId("show-preview").classList.add("active");
    byId("show-map").classList.remove("active");
    byId("preview-container").classList.remove("hidden");
    byId("map-container").classList.add("hidden");
  });
  byId("show-map").addEventListener("click", () => {
    byId("show-map").classList.add("active");
    byId("show-preview").classList.remove("active");
    byId("preview-container").classList.add("hidden");
    byId("map-container").classList.remove("hidden");
    if (!byId("benchmark-map").src && state.overview.artifacts.map.available) {
      byId("benchmark-map").src = state.overview.artifacts.map.url;
    }
  });

  renderOverview();
})();
