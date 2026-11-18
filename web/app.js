// Optional API token (MUSE_API_TOKEN): pass ?token=... once; it is kept in sessionStorage and sent on every call.
const TOKEN = new URLSearchParams(location.search).get("token") || sessionStorage.getItem("museToken") || "";
if (TOKEN) {
  sessionStorage.setItem("museToken", TOKEN);
  if (location.search.includes("token=")) history.replaceState(null, "", location.pathname);  // keep it out of history
  const rawFetch = window.fetch.bind(window);
  window.fetch = (url, opts = {}) => rawFetch(url, { ...opts, headers: { ...(opts.headers || {}), Authorization: `Bearer ${TOKEN}` } });
}
// SSE over fetch so the token travels in the Authorization header, never in a URL (EventSource cannot set headers).
function openStream(id, onEvent) {
  const ctl = new AbortController(); const handle = { readyState: 1, close() { handle.readyState = 2; ctl.abort(); } };
  (async () => {
    try {
      const res = await fetch(`/api/runs/${id}/stream`, { signal: ctl.signal });
      const reader = res.body.getReader(); const dec = new TextDecoder(); let buf = "";
      for (;;) {
        const { value, done } = await reader.read(); if (done) break;
        buf += dec.decode(value, { stream: true });
        let i; while ((i = buf.indexOf("\n\n")) >= 0) {
          const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
          if (chunk.startsWith("data: ")) onEvent(JSON.parse(chunk.slice(6)));
        }
      }
    } catch (err) { /* aborted or network drop */ }
    handle.readyState = 2;
  })();
  return handle;
}
const SESSION = localStorage.getItem("museSession") || (() => {
  const id = "s-" + Math.random().toString(36).slice(2, 10); localStorage.setItem("museSession", id); return id; })();
/* Meta Muse Studio front-end: zero-build vanilla JS. Streams LangGraph events over SSE. */
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const AGENT_COLORS = { sentinel: "#f04438", planner: "#7a5af8", research: "#06aed4", creative: "#ee46bc", code: "#12b76a",
  action: "#f79009", critic: "#0a66ff", composer: "#6941c6", memory: "#667085", system: "#0a66ff" };
const TOOL_LABELS = { web_search: "Searched the web", python_sandbox: "Ran the code and tests", search_flights: "Searched flights",
  hold_hotel: "Held a hotel", book_ride: "Booked a ride", find_meeting_slots: "Checked calendars", create_calendar_event: "Added to calendar",
  search_inbox: "Checked your inbox", draft_email: "Drafted an email", send_email: "Sent an email", send_message: "Sent a message",
  create_reminder: "Set a reminder", reserve_restaurant: "Reserved a table", search_products: "Searched products", add_to_cart: "Added to cart",
  checkout_cart: "Placed the order", wallet_balance: "Checked your balance", charge_card: "Made a payment", split_payment: "Requested a bill split",
  refund_payment: "Issued a refund" };
const AGENT_ICONS = { sentinel: "⛨", planner: "✦", research: "⌕", creative: "✎", code: "</>", action: "⚡", critic: "★", composer: "❖", memory: "◎", system: "●" };
const WF_ICONS = { rocket: "➶", code: "⌨", plane: "✈", bag: "◍", sparkles: "✦", shield: "◈" };
const DOMAIN_ICONS = { travel: "✈", calendar: "▦", email: "✉", messaging: "✆", productivity: "✓", dining: "☕", shopping: "◍", payments: "$" };
const STRATEGY_HINTS = { balanced: "Per-agent best fit: Claude plans & writes, GPT codes, Gemini judges, Groq triages.",
  quality: "Highest-quality model first for every agent.", cost: "Cheapest capable model first.",
  speed: "Lowest-latency model first (Groq).", offline: "Local only: Ollama, then the built-in offline model." };

const state = { workflows: [], workflow: null, strategy: "balanced", run: null, es: null, timer: null, started: 0,
  metrics: { llm: 0, tools: 0, tokens: 0, cost: 0 }, usage: {}, events: [] };

/* ---------------------------------------------------------------- graph layout */
const MAIN = { recall: [200, 28], sentinel: [200, 82], blocked: [338, 82], planner: [200, 136], critic: [200, 578],
  revise: [58, 578], composer: [200, 632], remember: [200, 684] };
const CLUSTERS = {
  research: { x: 6, y: 178, w: 190, h: 176, nodes: ["plan_queries", "search", "synthesize", "verify", "finalize"] },
  creative: { x: 204, y: 178, w: 190, h: 176, nodes: ["ideate", "select", "draft", "critique", "revise", "finalize"] },
  code: { x: 6, y: 366, w: 190, h: 176, nodes: ["specify", "implement", "run_tests", "debug", "finalize"] },
  action: { x: 204, y: 366, w: 190, h: 176, nodes: ["plan_actions", "discover", "apply_policy", "human_approval", "execute", "finalize"] },
};
const EDGES = [["recall", "sentinel"], ["sentinel", "planner"], ["sentinel", "blocked"], ["research", "critic"], ["creative", "critic"],
  ["code", "critic"], ["action", "critic"], ["planner", "research"], ["planner", "creative"], ["planner", "code"], ["planner", "action"],
  ["critic", "composer"], ["critic", "revise"], ["revise", "planner"], ["composer", "remember"]];

function anchor(id, side) {
  if (MAIN[id]) { const [x, y] = MAIN[id]; return [x, side === "top" ? y - 15 : y + 15]; }
  const c = CLUSTERS[id]; return [c.x + c.w / 2, side === "top" ? c.y : c.y + c.h];
}
function drawGraph() {
  const svg = $("#graph");
  let h = `<defs><filter id="glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    <linearGradient id="edgeGrad" x1="0" x2="1"><stop stop-color="#7c5cff"/><stop offset="1" stop-color="#00d4ff"/></linearGradient>
    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0L10,5L0,10z" fill="rgba(160,170,255,.45)"/></marker></defs>`;
  for (const [a, b] of EDGES) {
    let d;
    if (a === "revise" && b === "planner") d = `M${MAIN.revise[0]},${MAIN.revise[1] - 15} C 0,420 0,180 ${MAIN.planner[0] - 55},${MAIN.planner[1]}`;
    else if (b === "blocked") d = `M255,82 L283,82`;
    else if (b === "revise") d = `M145,578 L113,578`;
    else { const [x1, y1] = anchor(a, "bottom"); const [x2, y2] = anchor(b, "top"); const my = (y1 + y2) / 2; d = `M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}`; }
    h += `<path class="g-edge" data-from="${a}" data-to="${b}" d="${d}" marker-end="url(#arr)"/>`;
  }
  for (const [name, c] of Object.entries(CLUSTERS)) {
    const col = AGENT_COLORS[name];
    h += `<g class="g-cluster" id="n-main-${name}"><rect class="box" x="${c.x}" y="${c.y}" width="${c.w}" height="${c.h}" rx="14" stroke="${col}"/>
      <text class="ctitle" x="${c.x + 12}" y="${c.y + 18}" fill="${col}">${AGENT_ICONS[name]} ${name} subgraph</text>`;
    c.nodes.forEach((n, i) => {
      const y = c.y + 28 + i * 24;
      h += `<g class="sub" id="n-${name}-${n}"><rect x="${c.x + 14}" y="${y}" width="${c.w - 28}" height="19" rx="6"/><text x="${c.x + 24}" y="${y + 13}">${n}</text></g>`;
    });
    h += `</g>`;
  }
  for (const [name, [x, y]] of Object.entries(MAIN)) {
    const w = name === "blocked" || name === "revise" ? 90 : 110;
    const col = AGENT_COLORS[{ recall: "memory", remember: "memory", blocked: "sentinel", revise: "critic" }[name] || name] || "#fff";
    h += `<g class="g-node" id="n-main-${name}"><rect x="${x - w / 2}" y="${y - 15}" width="${w}" height="30" rx="10"/>
      <circle cx="${x - w / 2 + 13}" cy="${y}" r="4" fill="${col}"/><text x="${x - w / 2 + 24}" y="${y + 4}">${name}</text></g>`;
  }
  svg.innerHTML = h;
}
function setNode(graph, node, cls) {
  const el = document.getElementById(`n-${graph}-${node}`); if (!el) return;
  el.classList.remove("active", "done", "error", "waiting"); if (cls) el.classList.add(cls);
  if (graph === "main") document.querySelectorAll(`.g-edge[data-to="${node}"]`).forEach((e) => e.classList.toggle("live", cls === "active"));
}

/* ---------------------------------------------------------------- markdown */
function inline(s) {
  return esc(s).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/(^|[^*])\*([^*]+)\*/g, "$1<i>$2</i>").replace(/\[([^\]]+)\]\((https?:(?:[^()\s]|\([^()\s]*\))+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
}
function md(src) {
  const lines = String(src || "").split("\n"); let out = "", i = 0;
  while (i < lines.length) {
    const l = lines[i];
    if (l.startsWith("```")) { let code = []; i++; while (i < lines.length && !lines[i].startsWith("```")) code.push(lines[i++]); i++; out += `<pre><code>${esc(code.join("\n"))}</code></pre>`; continue; }
    if (/^<details><summary>(.*)<\/summary>$/.test(l)) { out += `<details><summary>${esc(l.match(/<summary>(.*)<\/summary>/)[1])}</summary>`; i++; continue; }
    if (l.trim() === "</details>") { out += "</details>"; i++; continue; }
    const hm = l.match(/^(#{1,4})\s+(.*)/); if (hm) { out += `<h${hm[1].length}>${inline(hm[2])}</h${hm[1].length}>`; i++; continue; }
    if (l.startsWith("|")) { const rows = []; while (i < lines.length && lines[i].startsWith("|")) rows.push(lines[i++]);
      const cells = (r) => r.replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      out += "<table><thead><tr>" + cells(rows[0]).map((c) => `<th>${inline(c)}</th>`).join("") + "</tr></thead><tbody>" +
        rows.slice(2).map((r) => "<tr>" + cells(r).map((c) => `<td>${inline(c)}</td>`).join("") + "</tr>").join("") + "</tbody></table>"; continue; }
    if (/^\s*[-*]\s+/.test(l)) { out += "<ul>"; while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) out += `<li>${inline(lines[i++].replace(/^\s*[-*]\s+/, ""))}</li>`; out += "</ul>"; continue; }
    if (/^\s*\d+\.\s+/.test(l)) { out += "<ol>"; while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) out += `<li>${inline(lines[i++].replace(/^\s*\d+\.\s+/, ""))}</li>`; out += "</ol>"; continue; }
    if (l.startsWith(">")) { out += `<blockquote>${inline(l.replace(/^>\s?/, ""))}</blockquote>`; i++; continue; }
    if (l.trim()) out += `<p>${inline(l)}</p>`; i++;
  }
  return out;
}

/* ---------------------------------------------------------------- sidebar */
async function boot() {
  drawGraph();
  const [wfs, prov, conns] = await Promise.all([fetch("/api/workflows").then((r) => r.json()),
    fetch("/api/providers").then((r) => r.json()), fetch("/api/connectors").then((r) => r.json())]);
  renderConnectors(conns);
  state.workflows = wfs; state.strategy = prov.default_strategy || "balanced";
  $("#workflows").innerHTML = wfs.map((w) => `<div class="wf${w.id === "auto" ? " dev-only" : ""}" data-id="${w.id}"><div class="wf-icon">${WF_ICONS[w.icon] || "✦"}</div>
    <div><div class="wf-name">${esc(w.name)}</div><div class="wf-desc">${esc(w.description)}</div></div></div>`).join("");
  document.querySelectorAll(".wf").forEach((el) => el.onclick = () => selectWorkflow(el.dataset.id));
  $("#strategies").innerHTML = prov.strategies.map((s) => `<button data-s="${s}">${s}</button>`).join("");
  document.querySelectorAll("#strategies button").forEach((b) => b.onclick = () => selectStrategy(b.dataset.s));
  selectStrategy(state.strategy); renderProviders(prov.providers); selectWorkflow("launch");
  loadHistory(); loadArch();
  document.querySelectorAll(".tab").forEach((t) => t.onclick = () => showTab(t.dataset.tab));
  $("#runBtn").onclick = launch;
  const setDev = (on) => { document.body.classList.toggle("dev", on); localStorage.setItem("museDev", on ? "1" : "");
    $("#devToggle").textContent = on ? "Product view" : "Developer view"; };
  // Developer view is opt-in via ?dev=1 (remembered until ?dev=0 or the "Product view" button).
  setDev(new URLSearchParams(location.search).get("dev") === "1" || (localStorage.getItem("museDev") === "1" && !location.search.includes("dev=0")));
  if (!document.body.classList.contains("dev")) { $("#goal").value = ""; document.querySelectorAll(".wf").forEach((w) => w.classList.remove("active")); }
  $("#devToggle").onclick = () => { setDev(!document.body.classList.contains("dev")); showTab("timeline"); };
  $("#goal").addEventListener("keydown", (ev) => { if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); launch(); } });
  $("#approveBtn").onclick = () => sendApproval("approve");
  $("#rejectBtn").onclick = () => sendApproval("reject");
}
function renderProviders(list) {
  const live = list.filter((p) => p.available && p.name !== "muse-local");
  $("#liveCount").textContent = `${live.length} live · ${list.length} integrated`;
  $("#pillProviders").innerHTML = `<i class="dot ${live.length ? "ok" : "warn"}"></i>${live.length} live / ${list.length} providers`;
  $("#providers").innerHTML = list.map((p) => {
    const tag = p.name === "muse-local" ? `<span class="prov-tag live">offline</span>` : p.available ? `<span class="prov-tag live">live</span>` : `<span class="prov-tag">add key</span>`;
    return `<div class="prov" title="${esc(p.strengths.join(", "))}"><i class="dot ${p.available ? "ok" : "off"}"></i>
      <div><div class="prov-name">${esc(p.label)}</div><div class="prov-model">${esc(p.model)} · q${p.quality} · ⚡${p.speed}</div></div>${tag}</div>`;
  }).join("");
}
function renderConnectors(list) {
  const groups = {}; list.forEach((c) => (groups[c.domain] = groups[c.domain] || []).push(c));
  $("#connCount").textContent = `${list.length} actions`;
  $("#connectors").innerHTML = Object.entries(groups).map(([d, cs]) => `<div class="conn-group">
    <div class="conn-domain">${DOMAIN_ICONS[d] || "•"} ${d}<span>${esc([...new Set(cs.map((c) => c.provider))].filter((x) => x !== "sandbox").join(" · ") || "built in")}</span><em class="conn-state ${cs.some((c) => c.mode === "live") ? "on" : ""}">${cs.some((c) => c.mode === "live") ? "Connected" : "Sandbox"}</em></div>
    <div class="conn-tools">${cs.map((c) => `<span class="conn ${c.risk} ${c.mode}" title="${esc(c.description)} · ${c.risk} · ${c.mode}"><span class="dev-only">${c.name}</span><span class="consumer-only">${c.name.replace(/_/g, " ")}</span></span>`).join("")}</div></div>`).join("");
}
function selectWorkflow(id) {
  const w = state.workflows.find((x) => x.id === id); state.workflow = w;
  document.querySelectorAll(".wf").forEach((el) => el.classList.toggle("active", el.dataset.id === id));
  $("#wfBadge").textContent = w.name; $("#wfDesc").textContent = w.description; $("#goal").value = w.example;
  $("#agentChips").innerHTML = w.agents.map((a) => `<span class="chip"><i style="background:${AGENT_COLORS[a]}"></i>${a}</span>`).join("");
}
function selectStrategy(s) {
  state.strategy = s; document.querySelectorAll("#strategies button").forEach((b) => b.classList.toggle("active", b.dataset.s === s));
  $("#strategyHint").textContent = STRATEGY_HINTS[s] || "";
}
function showTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".tab-body").forEach((b) => b.classList.toggle("hidden", b.id !== `tab-${name}`));
  if (name === "memory") loadMemory();
  if (name === "security") loadSecurity();
}
const LAYER_COLORS = ["#0a66ff", "#06aed4", "#7a5af8", "#12b76a", "#f79009", "#ee46bc"];
async function loadMemory() {
  const q = `session_id=${SESSION}` + (state.run ? `&run_id=${state.run}` : "");
  const data = await fetch(`/api/memory?${q}`).then((r) => r.json());
  $("#memory").innerHTML = `<div class="mem-head"><b><span class="consumer-only">What I remember</span><span class="dev-only">Six-layer memory</span></b><span class="muted small"><span class="consumer-only">Saved on this device so I can pick up where we left off</span><span class="dev-only">session ${esc(data.session_id)} · every layer is persisted in SQLite and survives restarts</span></span></div>` +
    data.layers.map((l, i) => `<div class="mem-layer" style="--c:${LAYER_COLORS[i]}">
      <div class="mem-top"><span class="mem-n">${l.layer}</span><div><div class="mem-name">${esc(l.name)}</div>
        <div class="muted small">${esc(l.description)} · scope: ${esc(l.scope)}</div></div><span class="mem-count">${l.count}</span></div>
      <div class="mem-items">${l.items.map((it) => `<div class="mem-item"><span>${esc(it.text)}</span>${it.meta ? `<em>${esc(it.meta)}</em>` : ""}</div>`).join("") || '<div class="muted small">Empty until a mission completes.</div>'}</div></div>`).join("");
}
async function loadSecurity() {
  const [sec, audit] = await Promise.all([fetch("/api/security").then((r) => r.json()), fetch("/api/audit?limit=12").then((r) => r.json())]);
  $("#security").innerHTML = `<div class="sec-banner ${sec.audit.valid ? "ok" : "bad"}"><b>${sec.audit.valid ? "✓ Activity log verified" : "✗ Activity log was altered"}</b>
      <span>${sec.audit.entries} tamper-evident entries · <span class="dev-only">API token ${sec.api_token_required ? "required" : "off"} · </span>limits $${sec.limits.per_payment_usd}/payment, $${sec.limits.per_mission_usd.toLocaleString()}/mission</span></div>
    <div class="sec-grid">${sec.controls.map((c) => `<div class="sec-card"><div class="sec-name"><i class="dot ok"></i>${esc(c.name)}</div><div class="muted small">${esc(c.detail)}</div></div>`).join("")}</div>
    <div class="dev-only"><div class="card-title" style="margin-top:16px">Least-privilege tool allowlists</div>
    ${Object.entries(sec.agent_tools).map(([a, t]) => `<div class="allow"><b>${a}</b><span>${t.length ? t.map((x) => `<code>${x}</code>`).join(" ") : '<em class="muted">no tools</em>'}</span></div>`).join("")}
    </div><div class="card-title" style="margin-top:16px">Recent activity</div>
    <table class="audit"><tbody>${audit.map((a) => `<tr><td>${new Date(a.ts * 1000).toLocaleTimeString()}</td><td><b>${esc(a.kind)}</b></td><td>${esc(JSON.stringify(a.payload)).slice(0, 110)}</td><td><code>${a.hash}</code></td></tr>`).join("") || '<tr><td class="muted">No entries yet.</td></tr>'}</tbody></table>`;
}
async function loadHistory() {
  const runs = await fetch("/api/runs").then((r) => r.json());
  if (!runs.length) return;
  $("#history").innerHTML = runs.map((r) => `<div class="hist" data-id="${r.id}"><b>${esc((state.workflows.find((w) => w.id === r.workflow) || {}).name || r.workflow)}</b> · ${esc({ completed: "done", blocked: "refused", awaiting_approval: "waiting for you" }[r.status] || r.status)}
<div class="muted">${esc(r.goal.slice(0, 70))}</div></div>`).join("");
  document.querySelectorAll(".hist").forEach((el) => el.onclick = () => replay(el.dataset.id));
}
async function loadArch() {
  const mermaid = await fetch("/api/graph/mermaid").then((r) => r.text());
  $("#arch").innerHTML = md(`# Runtime architecture\n\nThe supervisor graph runs **memory recall → Sentinel → planner**, fans out with LangGraph \`Send\` into four compiled **subgraphs** that run in parallel, then a cross-model **critic** (LLM-as-judge) either approves or loops back through **revise → planner**. The **composer** writes the brief and **remember** stores it in long-term memory. All state is checkpointed to SQLite, so runs paused on \`interrupt()\` resume exactly where they stopped.\n\n| Layer | Implementation |\n|---|---|\n| Orchestration | LangGraph StateGraph + 4 compiled subgraphs, Send fan-out, conditional loops |\n| Providers | OpenAI, Anthropic, Gemini, Groq, OpenRouter, Ollama, offline muse-local |\n| Routing | Strategy policies, fallback chains, retries with backoff, circuit breakers |\n| Safety | Sentinel rules + classifier, PII redaction, sandbox policy, risk-tiered approvals |\n| Durability | AsyncSqliteSaver checkpoints, durable run registry, persistent idempotency ledger |\n| Memory | 6 layers: working, session, episodic, semantic, procedural, profile |\n| Security | Sentinel, untrusted-content firewall, tool allowlists, secret redaction, hash-chained audit |\n| Streaming | astream(subgraphs=True, stream_mode=[tasks, custom]) → SSE |\n\n### Graph (generated live from the compiled graph, xray=1)\n\`\`\`\n${mermaid}\n\`\`\``);
}

/* ---------------------------------------------------------------- run lifecycle */
function resetRunUI() {
  state.metrics = { llm: 0, tools: 0, tokens: 0, cost: 0 }; state.usage = {}; state.events = [];
  $("#timeline").innerHTML = ""; $("#brief").innerHTML = `<p class="muted">The composed brief appears here when the run completes.</p>`;
  $("#agents").innerHTML = ""; $("#trace").textContent = ""; $("#usage").innerHTML = `<div class="muted small">No calls yet.</div>`;
  setScore(null); drawGraph(); updateStats();
}
async function launch() {
  const goal = $("#goal").value.trim(); if (!goal) return;
  $("#runBtn").disabled = true; resetRunUI(); showTab("timeline");
  const res = await fetch("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ goal, workflow: state.workflow.id, strategy: state.strategy, session_id: SESSION }) });
  if (!res.ok) { toast("Failed to start run"); $("#runBtn").disabled = false; return; }
  addEv("user", "system", esc(goal), "", "");
  const run = await res.json(); state.run = run.id; state.started = Date.now(); startTimer(); subscribe(run.id);
}
function subscribe(id) {
  if (state.es) state.es.close();
  state.es = openStream(id, (e) => handle(e, true));
}
async function replay(id) {
  const run = await fetch(`/api/runs/${id}`).then((r) => r.json());
  resetRunUI(); state.run = id; stopTimer();
  run.events.forEach((e) => handle(e, false));
  $("#sTime").textContent = `${run.metrics.duration_s}s`;
  if (run.status === "awaiting_approval") { subscribe(id); }
}
function startTimer() { stopTimer(); state.timer = setInterval(() => $("#sTime").textContent = `${((Date.now() - state.started) / 1000).toFixed(1)}s`, 100); }
function stopTimer() { if (state.timer) clearInterval(state.timer); state.timer = null; }
function updateStats() {
  const m = state.metrics; $("#sLlm").textContent = m.llm; $("#sTools").textContent = m.tools;
  $("#sTokens").textContent = m.tokens.toLocaleString(); $("#sCost").textContent = `$${m.cost.toFixed(4)}`;
  const total = Object.values(state.usage).reduce((a, b) => a + b, 0);
  if (total) $("#usage").innerHTML = Object.entries(state.usage).sort((a, b) => b[1] - a[1]).map(([p, n]) =>
    `<div class="ubar"><span>${esc(p)}</span><div class="track"><div class="fill" style="width:${(n / total) * 100}%"></div></div><b>${n}</b></div>`).join("");
}
function addEv(cls, agent, title, body, meta) {
  const col = AGENT_COLORS[agent] || "#0a66ff";
  const el = document.createElement("div"); el.className = `ev ${cls}`;
  el.innerHTML = `<div class="ev-icon" style="background:${col}">${AGENT_ICONS[agent] || "●"}</div>
    <div><div class="ev-title">${title}</div>${body ? `<div class="ev-body">${body}</div>` : ""}</div><div class="ev-meta">${meta || ""}</div>`;
  const tl = $("#timeline"); tl.appendChild(el); tl.parentElement.scrollTop = tl.parentElement.scrollHeight;
}
function handle(e, live) {
  state.events.push(e); $("#trace").textContent = JSON.stringify(state.events, null, 1);
  const t = new Date((e.ts || 0) * 1000).toLocaleTimeString([], { hour12: false });
  switch (e.type) {
    case "status": return onStatus(e, live, t);
    case "node": {
      setNode(e.graph, e.node, e.status === "start" ? "active" : e.status === "error" ? "error" : "done");
      if (e.graph === "main" && CLUSTERS[e.node]) {
        if (e.status === "start") addEv("tech", e.node, `Entered <span class="agent">${e.node}</span> subgraph`, "", t);
        else addEv("tech", e.node, `<span class="agent">${e.node}</span> subgraph finished`, "", t);
      }
      return;
    }
    case "llm": {
      state.metrics.llm++; state.metrics.tokens += (e.input_tokens || 0) + (e.output_tokens || 0); state.metrics.cost += e.cost_usd || 0;
      state.usage[e.provider] = (state.usage[e.provider] || 0) + 1; updateStats();
      const r = e.route || {}; const fb = r.fallback ? `<div class="fb">↳ routed: ${esc(r.preferred)} → ${esc(r.used)} (${esc((r.skipped || []).map((s) => `${s.provider}: ${s.reason}`).slice(0, 3).join("; "))})</div>` : "";
      const prev = esc(String(e.preview || "").replace(/[#*`>{}\[\]"]/g, " ").replace(/\s+/g, " ").slice(0, 150));
      addEv("tech", e.agent, `<span class="agent">${e.agent}</span> · ${esc(e.task)} <span class="prov-chip">${esc(e.provider)}/${esc(e.model)}</span>`,
        `${prev}…${fb}`, `${t}<br>${Math.round(e.latency_ms)}ms · ${(e.input_tokens || 0) + (e.output_tokens || 0)} tok`);
      return;
    }
    case "tool": {
      if (e.status !== "done") return;
      state.metrics.tools++; updateStats();
      let out = e.output; if (Array.isArray(out)) out = out.join(" · "); else if (out && typeof out === "object") out = JSON.stringify(out).slice(0, 160);
      addEv("tool", e.agent, `<span class="consumer-only">${esc(TOOL_LABELS[e.tool] || e.tool.replace(/_/g, " "))}</span><span class="dev-only">Tool <code>${esc(e.tool)}</code>${e.backend ? ` <span class="prov-chip">${esc(e.backend)}</span>` : ""}</span>`,
        `${esc(e.input || "")} → ${esc(out || "")}`, t);
      return;
    }
    case "plan":
      addEv("plan", "planner", `<span class="consumer-only">Here's my plan</span><span class="dev-only">Planner decomposed the mission into ${e.tasks.length} parallel task(s)</span>`,
        `${esc(e.rationale)}<div class="task-list">${e.tasks.map((x) => `<div><b style="color:${AGENT_COLORS[x.agent]}">${x.agent}</b> · ${esc(x.objective)}</div>`).join("")}</div>`, t);
      return;
    case "security":
      addEv(e.blocked ? "blocked" : "security tech", "sentinel", e.blocked ? "I can't help with that request" : "Sentinel cleared the request",
        `Risk ${Number(e.risk).toFixed(2)} · ${esc([...(e.reasons || []), ...(e.findings || []).map((f) => f.message)].slice(0, 4).join(" · "))}${e.redacted ? " · PII redacted before any model call" : ""}`, t);
      return;
    case "policy":
      addEv("tech", "action", `Risk policy applied to ${e.actions.length} actions · committed spend $${Number(e.committed_usd || 0).toFixed(2)}`,
        e.actions.map((a) => `<span class="risk ${a.risk}">${a.risk}</span> ${esc(a.title)} <span class="muted">(${esc(a.policy)})</span>`).join("<br>"), t);
      return;
    case "approval_required":
      setNode("action", "human_approval", "waiting");
      addEv("approval", "action", `<span class="consumer-only">I need your OK before I continue</span><span class="dev-only">Paused for human approval (LangGraph interrupt)</span>`, `${e.pending.length} action(s) awaiting your decision`, t);
      if (live) openApproval(e);
      return;
    case "approval_decision":
      addEv("approval", "action", e.decision === "approve" ? `You approved ${(e.approved || []).length} action(s)` : "You declined the actions", `<span class="dev-only">Approved: ${esc((e.approved || []).join(", ") || "none")}</span>`, t);
      return;
    case "critique":
      setScore(e.muse_score, e);
      addEv("critique tech", "critic", `Critic verdict: ${e.pass ? "PASS" : "REVISE"} · ${Number(e.overall).toFixed(2)}/10`,
        `${esc(e.feedback || "")} ${Object.entries(e.scores || {}).map(([k, v]) => `<span class="prov-chip">${k} ${v}</span>`).join("")}`, t);
      return;
    case "memory":
      addEv("tech", "memory", e.action === "recall" ? `Recalled ${e.count} items across memory layers` : "Mission consolidated into memory",
        esc(e.summary || "") + " " + Object.entries(e.layers || {}).map(([k, v]) => `<span class="prov-chip">${k} ${v}</span>`).join(""), t);
      return;
    case "security_finding":
      addEv("blocked", "sentinel", `<span class="consumer-only">I ignored a suspicious instruction hidden in ${esc((TOOL_LABELS[e.source] || e.source).toLowerCase())} results</span><span class="dev-only">Untrusted content neutralised (${esc(e.source)})</span>`,
        `${esc((e.findings || []).join(", "))} — ${esc(e.action || "")}`, t);
      return;
    case "recovered":
      addEv("plan", "system", "Run recovered after restart", `Resumed from ${esc(e.from)} · next: ${esc((e.next || []).join(", "))}`, t);
      return;
    case "log":
      addEv("", e.agent || "system", esc(e.message), "", t);
      return;
  }
}
function onStatus(e, live, t) {
  $("#sStatus").textContent = e.status.replace("_", " ");
  if (e.status === "running") { const again = state.events.filter((x) => x.type === "status" && x.status === "running").length > 1;
    addEv(again ? "" : "tech", "system", again ? "Picking up where I left off" : "Mission launched", again ? "" : esc(e.goal || ""), t); return; }
  if (e.status === "awaiting_approval") return;
  stopTimer(); $("#runBtn").disabled = false; if (e.metrics) $("#sTime").textContent = `${e.metrics.duration_s}s`;
  if (e.status === "failed") { addEv("blocked", "system", "Run failed", esc(e.error), t); return; }
  if (e.muse_score) setScore(e.muse_score, e.critique);
  $("#brief").innerHTML = md(e.final);
  $("#agents").innerHTML = (e.results || []).map((r) => `<div class="agent-card"><div class="ac-head"><b style="color:${AGENT_COLORS[r.agent]}">${AGENT_ICONS[r.agent]} ${esc(r.title)}</b>
    <span class="muted small">${Object.entries(r.signals || {}).map(([k, v]) => `${k}: ${v}`).join(" · ")}</span></div><div class="md">${md(r.content)}</div></div>`).join("") || `<p class="muted">No specialist outputs.</p>`;
  if (e.status === "blocked") { setNode("main", "blocked", "error"); addEv("blocked", "sentinel", "Nothing was run", "This request looked like an attempt to misuse the agent, so no tools or apps were touched.", t); }
  else addEv("final", "composer", "All done", "Open the Result tab to see everything I did.", t);
  document.querySelectorAll(".g-cluster.active, .g-node.active, .g-cluster .sub.active").forEach((el) => {
    el.classList.remove("active"); el.classList.add("done"); });
  document.querySelectorAll(".g-edge.live").forEach((el) => el.classList.remove("live"));
  if (live) { setTimeout(() => showTab("brief"), 900); loadHistory(); refreshProviders(); }
}
async function refreshProviders() { const p = await fetch("/api/providers").then((r) => r.json()); renderProviders(p.providers); }
function setScore(s, critique) {
  if (!$("#scoreTotal")) return;  // the score is internal telemetry; it is not shown in the product UI
  const arc = $("#gaugeArc"), bon = $("#gaugeBonus");
  if (!s) { arc.style.strokeDashoffset = 314.16; bon.style.strokeDashoffset = 257.6; $("#scoreTotal").textContent = "--";
    $("#scoreQuality").textContent = "--"; $("#scoreBonus").textContent = "--"; $("#scoreVerdict").textContent = "--"; $("#scoreChecks").innerHTML = ""; return; }
  arc.style.strokeDashoffset = 314.16 * (1 - Math.min(100, s.quality) / 100);
  bon.style.strokeDashoffset = 257.6 * (1 - (s.bonus || 0) / 10);
  animateNumber($("#scoreTotal"), s.total);
  $("#scoreQuality").textContent = `${s.quality} / 100`; $("#scoreBonus").textContent = `+${s.bonus} / 10`;
  if (critique) $("#scoreVerdict").textContent = critique.pass ? "PASS ✓" : "REVISE";
  $("#scoreChecks").innerHTML = (s.checks || []).map((c) => `<li class="${c.ok ? "ok" : "no"}"><span class="ck">${c.ok ? "✓" : "✕"}</span>${esc(c.label)}</li>`).join("");
}
function animateNumber(el, target) {
  const start = performance.now(), from = 0;
  const step = (now) => { const p = Math.min(1, (now - start) / 1400); el.textContent = (from + (target - from) * (1 - Math.pow(1 - p, 3))).toFixed(1); if (p < 1) requestAnimationFrame(step); };
  requestAnimationFrame(step);
}
const RISK_LABELS = { read: "Look up", write: "Change", payment: "Payment" };
function policyText(a) {
  const p = String(a.policy);
  if (p.startsWith("blocked")) return "Not allowed: " + p.replace(/^blocked: ?/, "");
  if (p.startsWith("auto-approved")) return "Done automatically (only reads information)";
  return p.replace("needs human approval", "Needs your OK").replace(/spend \$/, "pays $").replace("external recipient", "outside your organisation");
}
function openApproval(e) {
  const n = e.actions.filter((a) => a.requires_approval && !String(a.policy).startsWith("blocked")).length;
  $("#approvalSummary").innerHTML = `<span class="consumer-only">${n} step${n === 1 ? "" : "s"} need${n === 1 ? "s" : ""} your OK. Anything that only looks things up is done for you.</span><span class="dev-only">${esc(e.summary || "")}</span>`;
  $("#approvalList").innerHTML = e.actions.map((a) => {
    const needs = a.requires_approval && !String(a.policy).startsWith("blocked");
    const args = Object.entries(a.args || {}).map(([k, v]) => `${k.replace(/_usd$/, "").replace(/_/g, " ")}: ${Array.isArray(v) ? v.join(", ") : v}`).join(" · ");
    return `<label class="appr"><input type="checkbox" value="${a.id}" ${needs ? "checked" : "checked disabled"} ${needs ? "" : 'data-auto="1"'}/>
      <div><b>${esc(a.title)}</b>${a.amount_usd ? `<span class="amt">$${Number(a.amount_usd).toFixed(2)}</span>` : ""}
        <div class="muted small"><span class="dev-only"><code>${esc(a.tool)}</code> · </span><span class="args">${esc(args)}</span></div>
        <div class="small"><span class="consumer-only">${esc(policyText(a))}</span><span class="dev-only">${esc(a.policy)}${(a.depends_on || []).length ? ` <span class="dep">· after ${esc(a.depends_on.join(", "))}</span>` : ""}</span></div></div>
      <span class="risk ${a.risk}"><span class="consumer-only">${RISK_LABELS[a.risk] || a.risk}</span><span class="dev-only">${a.risk}</span></span></label>`;
  }).join("");
  const spend = e.actions.filter((a) => a.risk === "payment" && !String(a.policy).startsWith("blocked")).reduce((x, a) => x + (a.amount_usd || 0), 0);
  $("#approvalSpend").innerHTML = `<span>Total to pay if you approve everything</span><b>$${spend.toFixed(2)} <span class="muted" style="font-weight:500">of your $1,000 limit</span></b>`;
  $("#approvalSpend").classList.toggle("hidden", spend === 0);
  $("#approvalModal").classList.remove("hidden");
}
async function sendApproval(decision) {
  const approved = [...document.querySelectorAll("#approvalList input:checked:not([data-auto])")].map((i) => i.value);
  $("#approvalModal").classList.add("hidden"); startTimer();
  const res = await fetch(`/api/runs/${state.run}/approval`, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ decision, approved: decision === "approve" ? approved : [] }) });
  if (!res.ok) { toast("Approval failed"); return; }
  setNode("action", "human_approval", "done");
  if (!state.es || state.es.readyState === 2) subscribeFrom(state.run);
}
function subscribeFrom(id) {
  // Re-subscribe; skip events already rendered.
  if (state.es) state.es.close();
  const seen = state.events.length; let i = 0;
  state.es = openStream(id, (e) => { if (i++ < seen) return; handle(e, true); });
}
function toast(msg) { const t = $("#toast"); t.textContent = msg; t.classList.remove("hidden"); setTimeout(() => t.classList.add("hidden"), 2500); }
boot();
