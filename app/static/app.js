"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pretty = (s) => String(s ?? "unknown").replace(/_/g, " ");
const SVGNS = "http://www.w3.org/2000/svg";

const state = { file: null, corners: [], vw: 0, vh: 0, frame: null, job: null };

/* ---------- status ---------- */
fetch("/api/status").then((r) => r.json()).then((s) => {
  $("status").textContent = `${s.device} · ball: ${s.ball_model} · strokes: ${s.stroke_model}`;
  $("use-llm").checked = s.llm;
  $("use-llm").disabled = !s.llm;
  $("llm-note").textContent = s.llm ? `(${s.llm_model})` : "(set OPENAI_API_KEY on the server to enable)";
}).catch(() => {});

/* ---------- file selection ---------- */
const drop = $("drop");
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault(); drop.classList.remove("over");
  if (e.dataTransfer.files.length) pickFile(e.dataTransfer.files[0]);
});
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") $("file").click(); });
$("file").addEventListener("change", (e) => e.target.files.length && pickFile(e.target.files[0]));
$("change").addEventListener("click", resetUpload);
$("again").addEventListener("click", () => {
  $("results").hidden = true; $("upload").hidden = false; resetUpload();
  history.replaceState(null, "", location.pathname);
});
const fromHash = location.hash.match(/job=([0-9a-f]+)/);
if (fromHash) { state.job = fromHash[1]; showResults(); }

function resetUpload() {
  state.file = null; state.corners = [];
  $("file").value = "";
  $("drop").hidden = false; $("setup").hidden = true; $("progress").hidden = true; $("error").hidden = true;
}

function pickFile(file) {
  state.file = file; state.corners = []; state.frame = null;
  $("drop").hidden = true; $("setup").hidden = false; $("error").hidden = true;
  const v = document.createElement("video");
  v.muted = true; v.preload = "auto"; v.src = URL.createObjectURL(file);
  v.addEventListener("loadedmetadata", () => { v.currentTime = Math.min(1, (v.duration || 2) / 2); });
  v.addEventListener("seeked", () => {
    state.vw = v.videoWidth; state.vh = v.videoHeight;
    const c = $("preview"); c.width = state.vw; c.height = state.vh;
    c.getContext("2d").drawImage(v, 0, 0);
    state.frame = c.getContext("2d").getImageData(0, 0, c.width, c.height);
    drawCorners();
  }, { once: true });
  v.addEventListener("error", () => {
    $("corner-help").textContent = "This browser can't preview the file (it will still be analysed). The table will be auto-detected.";
    const c = $("preview"); c.width = 640; c.height = 60;
  });
}

/* ---------- table corners ---------- */
$("preview").addEventListener("click", (e) => {
  if (!state.frame || state.corners.length >= 4) return;
  const r = e.target.getBoundingClientRect();
  state.corners.push([(e.clientX - r.left) * state.vw / r.width, (e.clientY - r.top) * state.vh / r.height]);
  drawCorners();
});
$("reset").addEventListener("click", () => { state.corners = []; drawCorners(); });

function drawCorners() {
  const c = $("preview"), ctx = c.getContext("2d");
  if (!state.frame) return;
  ctx.putImageData(state.frame, 0, 0);
  const s = state.vw / 960;
  ctx.lineWidth = 3 * s; ctx.strokeStyle = "#f0cf78"; ctx.fillStyle = "#f0cf78";
  if (state.corners.length > 1) {
    ctx.beginPath();
    state.corners.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    if (state.corners.length === 4) ctx.closePath();
    ctx.stroke();
  }
  state.corners.forEach(([x, y]) => { ctx.beginPath(); ctx.arc(x, y, 7 * s, 0, 7); ctx.fill(); });
  const left = 4 - state.corners.length;
  $("corner-help").textContent = left === 4
    ? "Click the 4 corners of the table top for more accurate speed and placement, or skip to auto-detect."
    : left ? `${left} more corner${left > 1 ? "s" : ""} to click…` : "Table marked. Ready to analyse.";
}

/* ---------- upload + polling ---------- */
$("go").addEventListener("click", () => {
  if (!state.file) return;
  if (state.corners.length && state.corners.length !== 4) {
    return showError("Click all 4 table corners, or press 'Reset corners' to auto-detect.");
  }
  const fd = new FormData();
  fd.append("file", state.file);
  fd.append("side", "right");
  fd.append("corners", state.corners.length === 4 ? JSON.stringify(state.corners) : "");
  fd.append("use_llm", $("use-llm").checked ? "true" : "false");
  fd.append("racket", $("racket").value);

  $("setup").hidden = true; $("progress").hidden = false; $("error").hidden = true;
  setProgress("uploading", 0, state.file.name);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/analyze");
  xhr.upload.onprogress = (e) => e.lengthComputable && setProgress("uploading", e.loaded / e.total, state.file.name);
  xhr.onload = () => {
    if (xhr.status !== 200) return showError(parseErr(xhr.responseText));
    state.job = JSON.parse(xhr.responseText).id;
    poll();
  };
  xhr.onerror = () => showError("Upload failed; is the server running?");
  xhr.send(fd);
});

function parseErr(t) { try { return JSON.parse(t).detail; } catch { return t; } }

function setProgress(stage, p, note = "") {
  $("stage").textContent = stage;
  $("bar").style.width = `${Math.round(p * 100)}%`;
  $("progress-note").textContent = note;
}

function showError(msg) {
  $("error").textContent = msg; $("error").hidden = false;
  $("progress").hidden = true;
  if (state.file) $("setup").hidden = false;
}

async function poll() {
  const r = await fetch(`/api/jobs/${state.job}`).then((r) => r.json());
  if (r.status === "error") return showError(r.error);
  if (r.status === "done") return showResults();
  setProgress(r.stage, r.progress, r.status === "queued" ? "waiting for the previous job…" : "");
  setTimeout(poll, 1000);
}

/* ---------- results ---------- */
async function showResults() {
  const res = await fetch(`/api/jobs/${state.job}/report`);
  if (!res.ok) { history.replaceState(null, "", location.pathname); return; }
  const rep = await res.json();
  history.replaceState(null, "", `#job=${state.job}`);  // results survive a page reload
  $("upload").hidden = true; $("results").hidden = false;
  const v = $("overlay");
  v.src = `/api/jobs/${state.job}/overlay.mp4`;
  renderTiles(rep);
  renderRallies(rep);
  renderPlacement(rep);
  renderBars($("tech-chart"), rep.technique_counts);
  renderBars($("lean-chart"), rep.lean_counts);
  renderBars($("feet-chart"), rep.feet_counts);
  renderCoach(rep);
  renderShots(rep);
  const racket = rep.racket_hand ? ` · racket hand: ${rep.racket_hand} (${rep.racket_source === "player" ? "set by you" : "estimated"})` : "";
  $("models").textContent = `ball: ${rep.models.ball} · pose: ${rep.models.pose} · strokes: ${rep.models.stroke}${racket} · ${rep.processing_s}s`;
  window.scrollTo({ top: 0, behavior: "smooth" });
}

const seek = (t) => { const v = $("overlay"); v.currentTime = Math.max(0, t - 0.4); v.play().catch(() => {}); };

function renderTiles(rep) {
  const t = rep.totals, sp = rep.speed;
  $("t-strokes").textContent = t.strokes;
  $("t-strokes-sub").innerHTML = `${t.rallies} rallies · points won ${t.points_won} / lost ${t.points_lost}<br>ball tracked in ${t.ball_visible_pct}% of frames`;
  const known = t.forehand + t.backhand;
  $("t-fhbh").textContent = known ? `${t.forehand} / ${t.backhand}` : "– / –";
  $("t-fhbh-sub").textContent = known ? `${Math.round(100 * t.forehand / known)}% forehand` : "needs the trained stroke model";
  $("t-speed").textContent = sp.avg_kmh != null ? Math.round(sp.avg_kmh) : "–";
  const f = (x) => (x != null ? `${Math.round(x)} km/h` : "–");
  $("t-speed-sub").innerHTML = `peak: ${f(sp.peak_kmh)}<br>forehand avg: ${f(sp.forehand_avg_kmh)}<br>backhand avg: ${f(sp.backhand_avg_kmh)}` +
    (rep.scale_source === "player-height" ? "<br><span class='muted'>(table not found: rough scale)</span>" : "");
}

/* tooltip */
const tip = $("tooltip");
function showTip(e, html) {
  tip.innerHTML = html; tip.hidden = false;
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = e.clientX + pad, y = e.clientY - h - pad;
  if (x + w > innerWidth - 8) x = e.clientX - w - pad;
  if (y < 8) y = e.clientY + pad;
  tip.style.left = `${x}px`; tip.style.top = `${y}px`;
}
const hideTip = () => { tip.hidden = true; };

function el(name, attrs = {}, parent) {
  const n = document.createElementNS(SVGNS, name);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}

function niceMax(m) {
  if (m <= 4) return 4;
  const step = m <= 10 ? 2 : m <= 20 ? 5 : 10;
  return Math.ceil(m / step) * step;
}

function renderRallies(rep) {
  const box = $("rally-chart"); box.innerHTML = "";
  const rallies = rep.rallies;
  $("rally-legend").innerHTML = `<span><i style="background:var(--won)"></i>won</span><span><i style="background:var(--lost)"></i>lost</span>`;
  if (!rallies.length) { box.innerHTML = `<div class="empty">no rallies detected</div>`; return; }
  const W = 520, H = 250, m = { l: 26, r: 4, t: 8, b: 20 };
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const ymax = niceMax(Math.max(...rallies.map((r) => r.n_strokes)));
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "strokes per rally" }, box);
  const ticks = [0, ymax / 2, ymax];
  for (const t of ticks) {
    const y = m.t + ph - (t / ymax) * ph;
    el("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, class: "gridline" }, svg);
    el("text", { x: m.l - 6, y: y + 3, "text-anchor": "end", class: "axis" }, svg).textContent = t;
  }
  const slot = pw / rallies.length, bw = Math.max(2, Math.min(26, slot - 2));
  const marks = [];
  rallies.forEach((r, i) => {
    const x = m.l + i * slot + (slot - bw) / 2;
    const h = Math.max(2, (r.n_strokes / ymax) * ph), y = m.t + ph - h;
    const rad = Math.min(4, bw / 2, h);
    const color = r.won === true ? "var(--won)" : r.won === false ? "var(--lost)" : "var(--neutral)";
    const d = `M${x},${m.t + ph} V${y + rad} Q${x},${y} ${x + rad},${y} H${x + bw - rad} Q${x + bw},${y} ${x + bw},${y + rad} V${m.t + ph} Z`;
    marks.push(el("path", { d, fill: color, class: "mark" }, svg));
    if (rallies.length <= 24) el("text", { x: x + bw / 2, y: H - 5, "text-anchor": "middle", class: "axis" }, svg).textContent = i + 1;
    const hit = el("rect", { x: m.l + i * slot, y: m.t, width: slot, height: ph, class: "hit" }, svg);
    hit.addEventListener("mousemove", (e) => {
      marks.forEach((mk, j) => mk.classList.toggle("dim", j !== i));
      const res = r.won === true ? "won" : r.won === false ? "lost" : "unknown";
      showTip(e, `rally ${i + 1} · ${r.start_s.toFixed(1)}s<br>${r.n_strokes} strokes (${r.player_strokes} yours)<br>${res} · ${pretty(r.outcome)} by ${r.last_hitter}`);
    });
    hit.addEventListener("mouseleave", () => { marks.forEach((mk) => mk.classList.remove("dim")); hideTip(); });
    hit.addEventListener("click", () => seek(r.start_s));
  });
}

function renderPlacement(rep) {
  const box = $("placement"); box.innerHTML = "";
  const shots = rep.shots.filter((s) => s.landing);
  $("place-note").textContent = rep.table ? `${shots.length} of ${rep.shots.length} landings located` : "table not detected";
  const L = 2.74, Wd = 1.525, S = 180, pad = 16;
  const W = L * S + 2 * pad, H = Wd * S + 2 * pad + 16;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "shot landing positions" }, box);
  const X = (u) => pad + Math.max(-0.1, Math.min(L + 0.1, u)) * S;
  const Y = (v) => pad + Math.max(-0.1, Math.min(Wd + 0.1, v)) * S;
  el("rect", { x: pad, y: pad, width: L * S, height: Wd * S, fill: "var(--table)", rx: 2 }, svg);
  el("rect", { x: pad + 3, y: pad + 3, width: L * S - 6, height: Wd * S - 6, fill: "none", stroke: "var(--table-line)", "stroke-width": 2 }, svg);
  el("line", { x1: pad, x2: pad + L * S, y1: pad + Wd * S / 2, y2: pad + Wd * S / 2, stroke: "var(--table-line)", "stroke-width": 1, opacity: 0.8 }, svg);
  el("line", { x1: pad + L * S / 2, x2: pad + L * S / 2, y1: pad - 6, y2: pad + Wd * S + 6, stroke: "#ddd", "stroke-width": 4 }, svg);
  const mine = rep.player_side === "right";
  el("text", { x: pad + L * S * 0.25, y: H - 4, "text-anchor": "middle", class: "axis" }, svg).textContent = mine ? "opponent" : "you";
  el("text", { x: pad + L * S * 0.75, y: H - 4, "text-anchor": "middle", class: "axis" }, svg).textContent = mine ? "you" : "opponent";
  if (!shots.length) return;
  const last = shots[shots.length - 1];
  const groups = [];
  shots.forEach((s) => {
    const g = el("g", { class: "mark", style: "cursor:pointer" }, svg);
    const cx = X(s.landing.u), cy = Y(s.landing.v);
    const isLast = s === last;
    el("circle", { cx, cy, r: 13, fill: "var(--table)" }, g); // 2px surface ring against overlaps
    el("circle", { cx, cy, r: 11, fill: isLast ? "var(--won)" : "#f7f3ea", stroke: "#1f2a22", "stroke-width": 1.5 }, g);
    el("text", { x: cx, y: cy + 4, "text-anchor": "middle", style: `font:700 11px var(--mono);fill:${isLast ? "#fff" : "#1f2a22"}` }, g).textContent = s.n;
    groups.push(g);
    g.addEventListener("mousemove", (e) => {
      groups.forEach((o) => o.classList.toggle("dim", o !== g));
      showTip(e, `shot #${s.n} · ${s.time_s.toFixed(1)}s<br>${pretty(s.hand)} ${pretty(s.technique)}<br>${s.speed_kmh ? s.speed_kmh + " km/h · " : ""}depth ${Math.round(s.landing.depth * 100)}%`);
    });
    g.addEventListener("mouseleave", () => { groups.forEach((o) => o.classList.remove("dim")); hideTip(); });
    g.addEventListener("click", () => seek(s.time_s));
  });
}

function renderBars(box, counts) {
  box.innerHTML = "";
  const entries = Object.entries(counts || {});
  if (!entries.length) { box.innerHTML = `<div class="empty">no strokes</div>`; return; }
  const max = Math.max(...entries.map(([, n]) => n));
  for (const [k, n] of entries) {
    box.insertAdjacentHTML("beforeend",
      `<div class="name">${esc(pretty(k))}</div><div class="track"><div class="fill" style="width:${(100 * n) / max}%;${k === "unknown" ? "background:var(--neutral)" : ""}"></div></div><div class="val">${n}</div>`);
  }
}

function md(text) {
  const out = []; let list = false;
  const inline = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/\*(.+?)\*/g, "<i>$1</i>");
  for (const raw of String(text).split("\n")) {
    const line = raw.trim();
    const li = line.match(/^([-*]|\d+\.)\s+(.*)/);
    if (li) { if (!list) { out.push("<ul>"); list = true; } out.push(`<li>${inline(li[2])}</li>`); continue; }
    if (list) { out.push("</ul>"); list = false; }
    if (/^#{1,4}\s/.test(line)) out.push(`<h3>${inline(line.replace(/^#+\s*/, ""))}</h3>`);
    else if (line) out.push(`<p>${inline(line)}</p>`);
  }
  if (list) out.push("</ul>");
  return out.join("");
}

function renderCoach(rep) {
  $("tips").innerHTML = rep.tips.map((t) =>
    `<div class="tip"><b>${esc(t.title)}</b>${esc(t.detail)}<div class="ev">${esc(t.evidence)}</div></div>`).join("");
  const llm = $("llm");
  if (rep.coach) { llm.innerHTML = md(rep.coach); llm.hidden = false; $("coach-model").textContent = rep.models.coach || ""; }
  else if (rep.coach_error) { llm.innerHTML = `<p class="muted">AI coach unavailable: ${esc(rep.coach_error)}</p>`; llm.hidden = false; }
  else { llm.hidden = true; $("coach-model").textContent = "rule-based"; }
}

function renderShots(rep) {
  const tb = $("shots").querySelector("tbody");
  tb.innerHTML = rep.shots.map((s) =>
    `<tr data-t="${s.time_s}"><td>${s.n}</td><td>${s.time_s.toFixed(1)}s</td><td>${esc(pretty(s.hand))}</td><td>${esc(pretty(s.technique))}</td>` +
    `<td>${esc(pretty(s.lean))}</td><td>${esc(pretty(s.feet))}</td><td class="num">${s.speed_kmh ?? "–"}</td><td>${s.rally != null ? s.rally + 1 : "–"}</td></tr>`).join("");
  tb.querySelectorAll("tr").forEach((tr) => tr.addEventListener("click", () => seek(+tr.dataset.t)));
}
