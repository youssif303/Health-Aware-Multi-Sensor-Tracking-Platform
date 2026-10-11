const $ = (selector) => document.querySelector(selector);
const colors = ["#a8f06a", "#6ed5df", "#f6b85d", "#c2a0f5", "#f17878"];
const rampFaults = new Set(["camera_ramp", "radar_ramp"]);
let currentRunId = "";

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
}

function percent(value, digits = 1) {
  return Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(digits)}%` : "—";
}

function seconds(value, digits = 1) {
  return value === null || value === undefined || !Number.isFinite(Number(value))
    ? "Not detected" : `${Number(value).toFixed(digits)} s`;
}

function setStatus(message, kind = "") {
  const node = $("#form-status");
  node.textContent = message;
  node.className = `form-status ${kind}`.trim();
}

function updateRampFields() {
  $("#ramp-fields").hidden = !rampFaults.has($("#fault").value);
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
  return body;
}

async function checkApi() {
  const indicator = $("#api-indicator");
  try {
    const health = await requestJson("/health");
    const ready = health.tracker_available;
    indicator.className = `api-indicator ${ready ? "online" : "offline"}`;
    indicator.lastElementChild.textContent = ready ? "API READY" : "TRACKER NOT BUILT";
  } catch {
    indicator.className = "api-indicator offline";
    indicator.lastElementChild.textContent = "API UNAVAILABLE";
  }
}

async function loadLatestSweep() {
  const caption = $("#sweep-caption");
  const rows = $("#sweep-rows");
  try {
    const result = await requestJson("/sweeps/latest");
    if (!result.available) {
      caption.textContent = "No saved sweep";
      return;
    }
    const summary = result.summary;
    caption.textContent = `${summary.runs_per_fault} runs/profile · ${summary.steps} steps · ${result.output_directory}`;
    rows.innerHTML = Object.entries(summary.faults).map(([fault, metrics]) => {
      const baseline = metrics.baseline_position_rmse;
      const adaptive = metrics.adaptive_position_rmse;
      const improvement = metrics.rmse_improvement_percent;
      const value = Number(improvement.mean);
      const direction = Number.isFinite(value) ? `${value >= 0 ? "↓ " : "↑ "}${Math.abs(value).toFixed(1)}%` : "—";
      const ciLow = improvement.ci95_low;
      const ciHigh = improvement.ci95_high;
      const confidenceInterval = ciLow !== null && ciLow !== undefined && ciHigh !== null && ciHigh !== undefined
        && Number.isFinite(Number(ciLow)) && Number.isFinite(Number(ciHigh))
        ? `${Number(ciLow).toFixed(1)}–${Number(ciHigh).toFixed(1)}%` : "—";
      const delay = metrics.fault_detection_delay_s.mean;
      const rul = metrics.rul_prediction_mae_seconds.mean;
      const delayText = delay !== null && delay !== undefined && Number.isFinite(Number(delay)) ? `${Number(delay).toFixed(1)} s` : "—";
      const rulText = rul !== null && rul !== undefined && Number.isFinite(Number(rul)) ? `${Number(rul).toFixed(2)} s` : "—";
      return `<tr><td>${escapeHtml(fault.replaceAll("_", " "))}</td><td>${escapeHtml(improvement.count)}</td><td>${Number.isFinite(Number(baseline.mean)) ? `${Number(baseline.mean).toFixed(3)} m` : "—"}</td><td>${Number.isFinite(Number(adaptive.mean)) ? `${Number(adaptive.mean).toFixed(3)} m` : "—"}</td><td class="${value >= 0 ? "positive" : "negative"}">${direction}</td><td>${confidenceInterval}</td><td>${delayText}</td><td>${rulText}</td></tr>`;
    }).join("");
  } catch (error) {
    caption.textContent = "Sweep summary unavailable";
    rows.innerHTML = `<tr><td colspan="6" class="empty-state">${escapeHtml(error.message)}</td></tr>`;
  }
}

async function refreshRuns(selectRun = "") {
  const selector = $("#run-select");
  try {
    const { runs } = await requestJson("/experiments?limit=12");
    selector.innerHTML = runs.length
      ? runs.map(({ run_id, metrics }) => `<option value="${escapeHtml(run_id)}">${escapeHtml(metrics.fault)} · seed ${escapeHtml(metrics.seed)} · ${escapeHtml(run_id.slice(0, 8))}</option>`).join("")
      : '<option value="">No runs yet</option>';
    const next = selectRun || currentRunId;
    if (next && runs.some((run) => run.run_id === next)) selector.value = next;
    else if (runs.length) selector.value = runs[0].run_id;
    if (selector.value) await loadRun(selector.value);
  } catch (error) {
    setStatus(`Could not load recent runs: ${error.message}`, "error");
  }
}

function renderMetrics(metrics) {
  const improvement = Number(metrics.rmse_improvement_percent);
  $("#metric-rmse").textContent = `${Number(metrics.health_adaptive.position_rmse).toFixed(3)} m`;
  $("#metric-rmse-foot").textContent = `Baseline: ${Number(metrics.baseline.position_rmse).toFixed(3)} m`;
  const improvementNode = $("#metric-improvement");
  improvementNode.textContent = Number.isFinite(improvement)
    ? `${Math.abs(improvement).toFixed(1)}% ${improvement >= 0 ? "lower" : "higher"}` : "—";
  improvementNode.style.color = improvement >= 0 ? "var(--accent)" : "var(--orange)";
  $("#metric-delay").textContent = seconds(metrics.fault_detection_delay_s);
  $("#metric-false-alarms").textContent = percent(metrics.false_alarm_rate, 2);
  $("#summary-list").innerHTML = [
    ["Scenario", metrics.fault],
    ["Random seed", metrics.seed],
    ["Track availability", percent(metrics.health_adaptive.track_availability)],
    ["RUL prediction MAE", metrics.rul_prediction_mae_seconds == null
      ? "—" : `${Number(metrics.rul_prediction_mae_seconds).toFixed(2)} s (${metrics.rul_prediction_samples} samples)`],
  ].map(([label, value]) => `<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("");
}

function renderSensors(healthRows, rulRows) {
  const entries = Object.entries(healthRows || {}).sort(([a], [b]) => a.localeCompare(b));
  if (!entries.length) {
    $("#sensor-list").innerHTML = '<div class="empty-state">No sensor health data in this run.</div>';
    return;
  }
  $("#sensor-list").innerHTML = entries.map(([sensor, row], index) => {
    const score = Math.max(0, Math.min(1, Number(row.degradation_score)));
    const delivery = Number(row.delivery_rate);
    const multiplier = Number(row.covariance_multiplier);
    const healthy = row.reason === "healthy";
    const className = healthy ? "" : (score > 0.7 ? "faulted" : "degraded");
    const rul = rulRows?.[sensor];
    const rulHtml = rul
      ? `<div class="rul-line">RUL estimate <strong>${seconds(rul.predicted_rul_seconds)}</strong> <span>± ${seconds(rul.neighbor_spread_seconds)}</span></div>` : "";
    return `<article class="sensor-card ${className}">
      <div class="sensor-top"><div class="sensor-name"><i class="sensor-dot"></i>${escapeHtml(sensor)}</div><span class="health-badge">${healthy ? "HEALTHY" : escapeHtml(row.reason.replaceAll("_", " ").toUpperCase())}</span></div>
      <div class="sensor-metrics"><div class="sensor-metric"><span>HEALTH SCORE</span><strong>${percent(1 - score)}</strong></div><div class="sensor-metric"><span>DELIVERY</span><strong>${percent(delivery)}</strong></div><div class="sensor-metric"><span>R NOISE SCALE</span><strong>${Number.isFinite(multiplier) ? `${multiplier.toFixed(1)}×` : "—"}</strong></div></div>
      <div class="health-bar"><i style="width:${(1 - score) * 100}%"></i></div>${rulHtml}
    </article>`;
  }).join("");
}

function renderEvents(events = []) {
  $("#event-count").textContent = `${events.length} ${events.length === 1 ? "event" : "events"}`;
  if (!events.length) {
    $("#event-list").innerHTML = '<div class="empty-state">No sensor degradation alerts in this run.</div>';
    return;
  }
  $("#event-list").innerHTML = events.slice(-30).reverse().map((event) => {
    const span = event.first_step === event.last_step ? `step ${event.first_step}` : `steps ${event.first_step}–${event.last_step}`;
    return `<div class="event-row"><i class="event-dot"></i><div><strong>${escapeHtml(event.sensor_id)} · ${escapeHtml(event.reason.replaceAll("_", " "))}</strong><small>${escapeHtml(span)}</small></div><span class="event-time">${((event.last_step - event.first_step + 1) / 10).toFixed(1)} s</span></div>`;
  }).join("");
}

function renderTracks(trackData) {
  const svg = $("#track-plot");
  const targets = Object.keys(trackData.estimates || {}).sort();
  const all = [...Object.values(trackData.estimates || {}).flat(), ...Object.values(trackData.truth || {}).flat()];
  if (!all.length) {
    svg.innerHTML = '<text class="plot-empty" x="450" y="230" text-anchor="middle">No track estimates in this run</text>';
    $("#plot-legend").innerHTML = "";
    $("#plot-time").textContent = "—";
    return;
  }
  const width = 900, height = 460, left = 58, right = 18, top = 22, bottom = 43;
  let minX = Math.min(...all.map((p) => p.x)), maxX = Math.max(...all.map((p) => p.x));
  let minY = Math.min(...all.map((p) => p.y)), maxY = Math.max(...all.map((p) => p.y));
  const padX = Math.max((maxX - minX) * 0.09, 3), padY = Math.max((maxY - minY) * 0.09, 3);
  minX -= padX; maxX += padX; minY -= padY; maxY += padY;
  const px = (x) => left + (x - minX) / (maxX - minX) * (width - left - right);
  const py = (y) => top + (maxY - y) / (maxY - minY) * (height - top - bottom);
  const parts = [];
  for (let i = 0; i <= 4; i += 1) {
    const y = top + (height - top - bottom) * i / 4;
    const x = left + (width - left - right) * i / 4;
    const yValue = maxY - (maxY - minY) * i / 4;
    const xValue = minX + (maxX - minX) * i / 4;
    parts.push(`<line class="plot-grid" x1="${left}" y1="${y}" x2="${width - right}" y2="${y}"/><text class="plot-tick" x="${left - 9}" y="${y + 3}" text-anchor="end">${yValue.toFixed(0)}</text>`);
    parts.push(`<line class="plot-grid" x1="${x}" y1="${top}" x2="${x}" y2="${height - bottom}"/><text class="plot-tick" x="${x}" y="${height - 19}" text-anchor="middle">${xValue.toFixed(0)}</text>`);
  }
  parts.push(`<line class="plot-axis" x1="${left}" y1="${height - bottom}" x2="${width - right}" y2="${height - bottom}"/><line class="plot-axis" x1="${left}" y1="${top}" x2="${left}" y2="${height - bottom}"/>`);
  targets.forEach((target, index) => {
    const color = colors[index % colors.length];
    const estimated = trackData.estimates[target] || [];
    const truth = trackData.truth[target] || [];
    const polyline = (points) => points.map((p) => `${px(p.x).toFixed(1)},${py(p.y).toFixed(1)}`).join(" ");
    if (truth.length) parts.push(`<polyline class="track-truth" points="${polyline(truth)}" stroke="${color}"/>`);
    if (estimated.length) {
      parts.push(`<polyline class="track-estimate" points="${polyline(estimated)}" stroke="${color}"/>`);
      const last = estimated[estimated.length - 1];
      parts.push(`<circle class="track-head" cx="${px(last.x)}" cy="${py(last.y)}" r="4.5" fill="${color}"/>`);
      parts.push(`<text class="plot-tick" x="${px(last.x) + 7}" y="${py(last.y) - 7}">D${escapeHtml(target)}</text>`);
    }
  });
  svg.innerHTML = parts.join("");
  $("#plot-legend").innerHTML = '<span class="legend-item"><i class="legend-line"></i>ESTIMATE</span><span class="legend-item"><i class="legend-line truth"></i>GROUND TRUTH</span>';
  const lastStep = Math.max(...all.map((p) => p.step));
  $("#plot-time").textContent = `LAST STEP ${lastStep} · ${(lastStep * 0.1).toFixed(1)} S`;
}

async function loadRun(runId) {
  if (!runId) return;
  try {
    const [detail, tracks] = await Promise.all([
      requestJson(`/experiments/${encodeURIComponent(runId)}`),
      requestJson(`/experiments/${encodeURIComponent(runId)}/tracks?limit=160`),
    ]);
    currentRunId = runId;
    $("#run-id").textContent = `RUN ${runId}`;
    renderMetrics(detail.metrics);
    renderSensors(detail.latest_sensor_health, detail.latest_rul_predictions);
    renderEvents(detail.fault_events);
    renderTracks(tracks);
  } catch (error) {
    setStatus(`Could not load run: ${error.message}`, "error");
  }
}

$("#fault").addEventListener("change", updateRampFields);
$("#run-select").addEventListener("change", (event) => loadRun(event.target.value));
$("#experiment-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = $("#run-button");
  const fault = $("#fault").value;
  const payload = { fault, seed: Number($("#seed").value), steps: Number($("#steps").value) };
  if (rampFaults.has(fault)) {
    payload.onset_step = Number($("#onset-step").value);
    payload.failure_step = Number($("#failure-step").value);
  }
  button.disabled = true;
  button.lastElementChild.textContent = "Running…";
  setStatus("Simulating the flight, replaying adaptive fusion, and scoring the run…");
  try {
    const response = await requestJson("/experiments", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    setStatus(`Run complete · ${response.run_id}`, "success");
    await refreshRuns(response.run_id);
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    button.disabled = false;
    button.lastElementChild.textContent = "Run experiment";
  }
});

updateRampFields();
checkApi();
refreshRuns();
loadLatestSweep();
