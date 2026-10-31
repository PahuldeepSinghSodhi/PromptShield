"use strict";

const $ = (sel) => document.querySelector(sel);
const charts = {};

// ---------------------------------------------------------------- api
function apiKey() {
  try { return localStorage.getItem("ps_api_key") || "dev-key"; } catch { return "dev-key"; }
}

async function api(path, opts = {}) {
  const headers = { "X-API-Key": apiKey(), ...(opts.headers || {}) };
  if (opts.json !== undefined) {
    headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(opts.json);
  }
  const res = await fetch(path, { ...opts, headers });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail ? JSON.stringify(body.detail) : `HTTP ${res.status}`);
  return body;
}

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 2500);
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function badge(el, value) {
  el.className = `badge ${value || ""}`;
  el.textContent = value || "";
}

// ---------------------------------------------------------------- tabs
document.querySelectorAll("nav button").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("nav button").forEach((b) => b.setAttribute("aria-selected", b === btn));
    document.querySelectorAll(".tab").forEach((s) => s.classList.toggle("active", s.id === btn.dataset.tab));
    if (btn.dataset.tab === "review") loadReview();
    if (btn.dataset.tab === "monitoring") loadMonitoring();
  });
});

const keyInput = $("#apiKey");
keyInput.value = apiKey();
keyInput.addEventListener("change", () => { try { localStorage.setItem("ps_api_key", keyInput.value); } catch {} });

// ---------------------------------------------------------------- playground
function randomFrom(alphabet, n) {
  let s = "";
  const buf = new Uint32Array(n);
  crypto.getRandomValues(buf);
  for (const x of buf) s += alphabet[x % alphabet.length];
  return s;
}

const EXAMPLES = {
  // fake key generated in the browser, so no key-shaped literal lives in the repo
  secret: () => `Our S3 upload fails with AccessDenied. Config:\nAWS_ACCESS_KEY_ID=${"AKIA" + randomFrom("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567", 16)}\nregion=ap-south-1\nWhat is wrong?`,
  kyc: () => "Please verify KYC for Priya Sharma: PAN ABCPS1234K, phone +91 91234 56789, address 12 MG Road, Bengaluru 560001. Draft an approval note.",
  clean: () => "Summarise the top five complaint themes from last quarter and suggest one fix for each.",
};
document.querySelectorAll("[data-example]").forEach((b) =>
  b.addEventListener("click", () => { $("#prompt").value = EXAMPLES[b.dataset.example](); }));

function highlight(text, entities) {
  let html = "", cursor = 0;
  for (const e of [...entities].sort((a, b) => a.start - b.start)) {
    if (e.start < cursor) continue;
    html += escapeHtml(text.slice(cursor, e.start));
    html += `<mark class="${e.label}" title="${e.label} · ${(e.score * 100).toFixed(0)}% · ${e.source}">`
      + `${escapeHtml(text.slice(e.start, e.end))}<sup>${e.label}</sup></mark>`;
    cursor = e.end;
  }
  return html + escapeHtml(text.slice(cursor));
}

async function run(sendToLlm) {
  const text = $("#prompt").value.trim();
  if (!text) return;
  const team = $("#team").value || null;
  const buttons = [$("#btnScan"), $("#btnSend")];
  buttons.forEach((b) => (b.disabled = true));
  try {
    if (!sendToLlm) {
      const r = await api("/v1/redact", { method: "POST", json: { text, team } });
      $("#highlighted").innerHTML = r.entities.length ? highlight(text, r.entities) : "No sensitive data detected.";
      $("#sent").textContent = r.action === "block" ? "(request would be blocked)" : r.redacted_text;
      $("#reply").textContent = "-";
      badge($("#actionBadge"), r.action);
      $("#meta").textContent = `${r.latency_ms} ms · ${r.model_version} · request ${r.request_id}`;
    } else {
      const r = await api("/v1/proxy/chat", { method: "POST", json: { messages: [{ role: "user", content: text }], team } });
      $("#highlighted").innerHTML = r.entities.length ? highlight(text, r.entities) : "No sensitive data detected.";
      badge($("#actionBadge"), r.action);
      if (r.action === "block") {
        $("#sent").textContent = "Nothing was sent: the request was blocked.";
        $("#reply").textContent = `Blocked because it contains: ${r.blocked_types.join(", ")}`;
      } else {
        $("#sent").textContent = r.sent_to_llm.map((m) => m.content).join("\n");
        $("#reply").textContent = r.reply;
      }
      const llm = r.llm_latency_ms != null ? ` + LLM ${r.llm_latency_ms} ms` : "";
      $("#meta").textContent = `scan ${r.latency_ms} ms${llm} · request ${r.request_id}`;
    }
  } catch (err) {
    toast(`Error: ${err.message}`);
  } finally {
    buttons.forEach((b) => (b.disabled = false));
  }
}
$("#btnScan").addEventListener("click", () => run(false));
$("#btnSend").addEventListener("click", () => run(true));

// ---------------------------------------------------------------- batch
$("#btnUpload").addEventListener("click", async () => {
  const f = $("#file").files[0];
  if (!f) return toast("Choose a file first");
  const form = new FormData();
  form.append("file", f);
  $("#jobStatus").textContent = "uploading…";
  try {
    const res = await fetch("/v1/scan/batch", { method: "POST", headers: { "X-API-Key": apiKey() }, body: form });
    const job = await res.json();
    if (!res.ok) throw new Error(job.detail || res.status);
    pollJob(job.id);
  } catch (err) {
    $("#jobStatus").textContent = "";
    toast(`Error: ${err.message}`);
  }
});

async function pollJob(id) {
  const job = await api(`/v1/jobs/${id}`);
  $("#jobStatus").textContent = job.status;
  if (job.status === "queued" || job.status === "running") return setTimeout(() => pollJob(id), 1000);
  if (job.status === "failed") return toast(`Scan failed: ${job.error}`);
  renderBatch(job.report);
}

function renderBatch(r) {
  const cols = Object.entries(r.per_column)
    .map(([c, counts]) => `<tr><td>${escapeHtml(c)}</td><td>${Object.entries(counts).map(([k, v]) => `${k}: ${v}`).join(", ")}</td></tr>`)
    .join("");
  const totals = Object.entries(r.entity_totals).map(([k, v]) => `<tr><td>${k}</td><td class="num">${v}</td></tr>`).join("");
  $("#batchReport").innerHTML = `
    <div class="card"><h2>Summary ${r.contains_secrets ? '<span class="badge block">secrets found</span>' : ""}</h2>
      <p><b>${r.rows_with_sensitive_data}</b> of ${r.rows_scanned} rows (${r.pct_rows_with_sensitive_data}%) contain sensitive data.</p>
      <div class="tablewrap"><table><thead><tr><th>Entity</th><th>Count</th></tr></thead><tbody>${totals || "<tr><td colspan=2>None</td></tr>"}</tbody></table></div></div>
    <div class="card"><h2>Riskiest columns</h2>
      <div class="tablewrap"><table><thead><tr><th>Column</th><th>Entities</th></tr></thead><tbody>${cols || "<tr><td colspan=2>None</td></tr>"}</tbody></table></div></div>`;
}

// ---------------------------------------------------------------- review
async function loadReview() {
  const tbody = $("#reviewTable tbody");
  try {
    const items = await api("/v1/review-queue?limit=50");
    tbody.innerHTML = items.length ? "" : '<tr><td colspan="7" class="muted">Nothing to review.</td></tr>';
    for (const it of items) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${new Date(it.created_at).toLocaleString()}</td><td>${escapeHtml(it.team || "-")}</td>
        <td><span class="badge ${it.action}">${it.action}</span></td>
        <td>${Object.entries(it.entity_counts).map(([k, v]) => `${k}×${v}`).join(", ")}</td>
        <td class="num">${it.min_confidence != null ? (it.min_confidence * 100).toFixed(0) + "%" : "-"}</td>
        <td class="mono">${escapeHtml(it.masked_preview || "")}</td>
        <td><button class="chip" data-ok>Correct</button> <button class="chip" data-fp>False positive</button></td>`;
      tr.querySelector("[data-ok]").addEventListener("click", () => sendFeedback(it, "wrong_label", tr, true));
      tr.querySelector("[data-fp]").addEventListener("click", () => sendFeedback(it, "false_positive", tr, false));
      tbody.appendChild(tr);
    }
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="7">Error: ${escapeHtml(err.message)}</td></tr>`;
  }
}

async function sendFeedback(item, kind, row, confirmed) {
  try {
    await api("/v1/feedback", { method: "POST", json: {
      request_id: item.request_id, kind: confirmed ? "wrong_label" : kind,
      comment: confirmed ? "reviewer confirmed detection" : "reviewer marked as false positive",
    } });
    row.remove();
    toast("Feedback recorded");
  } catch (err) {
    toast(`Error: ${err.message}`);
  }
}

// ---------------------------------------------------------------- monitoring
function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function drawChart(id, config) {
  if (!window.Chart) return;
  charts[id]?.destroy();
  Chart.defaults.color = cssVar("--muted");
  Chart.defaults.borderColor = cssVar("--border");
  charts[id] = new Chart(document.getElementById(id), config);
}

async function loadMonitoring() {
  try {
    const [m, d] = await Promise.all([api("/v1/metrics?hours=24"), api("/v1/drift")]);
    const a = m.actions || {};
    const kpis = [
      ["Requests (24 h)", m.requests || 0],
      ["Blocked", a.block || 0],
      ["Redacted", a.redact || 0],
      ["p95 scan latency", m.latency_ms ? `${m.latency_ms.p95} ms` : "-"],
    ];
    $("#kpis").innerHTML = kpis.map(([l, v]) => `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");

    const tl = m.timeline || [];
    const colors = { allow: cssVar("--ok"), redact: cssVar("--warn"), block: cssVar("--bad") };
    drawChart("chartTimeline", {
      type: "bar",
      data: {
        labels: tl.map((r) => new Date(r.hour).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })),
        datasets: ["allow", "redact", "block"].map((k) => ({ label: k, data: tl.map((r) => r[k] || 0), backgroundColor: colors[k] })),
      },
      options: { maintainAspectRatio: false, scales: { x: { stacked: true }, y: { stacked: true, beginAtZero: true } } },
    });
    const ents = Object.entries(m.entities || {});
    drawChart("chartEntities", {
      type: "bar",
      data: { labels: ents.map((e) => e[0]), datasets: [{ label: "detections", data: ents.map((e) => e[1]), backgroundColor: cssVar("--accent") }] },
      options: { indexAxis: "y", maintainAspectRatio: false, plugins: { legend: { display: false } } },
    });

    const tbody = $("#driftTable tbody");
    if (d.status !== "ok") {
      badge($("#driftBadge"), "");
      tbody.innerHTML = `<tr><td colspan="4" class="muted">${d.status.replace("_", " ")}${d.min_events ? ` (need ${d.min_events} events, have ${d.events})` : ""}</td></tr>`;
    } else {
      badge($("#driftBadge"), d.overall);
      tbody.innerHTML = Object.entries(d.checks).map(([k, c]) =>
        `<tr><td>${k.replaceAll("_", " ")}</td><td class="num">${c.psi}</td><td class="num">${c.ks ?? "-"}</td><td><span class="badge ${c.level}">${c.level}</span></td></tr>`).join("");
    }
  } catch (err) {
    toast(`Error: ${err.message}`);
  }
}
