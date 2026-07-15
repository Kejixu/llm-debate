/* llm-debate app: one event stream, two renderers (chat + observe). */
mermaid.initialize({ startOnLoad: false, theme: "dark", securityLevel: "loose" });
const $ = (id) => document.getElementById(id);

/* ---------- tab bar ---------- */
function showTab(name) {
  $("view-chat").classList.toggle("hidden", name !== "chat");
  $("view-observe").classList.toggle("hidden", name !== "observe");
  $("tab-chat").classList.toggle("active", name === "chat");
  $("tab-observe").classList.toggle("active", name === "observe");
  $("tab-chat").setAttribute("aria-selected", String(name === "chat"));
  $("tab-observe").setAttribute("aria-selected", String(name === "observe"));
}
$("tab-chat").onclick = () => showTab("chat");
$("tab-observe").onclick = () => showTab("observe");

/* ---------- state ---------- */
let selectedRunId = null;
let source = null;
let runsCache = [];

/* ---------- sidebar ---------- */
async function loadRuns() {
  runsCache = await (await fetch("/api/runs")).json();
  const box = $("sidebar-runs");
  box.innerHTML = "";
  $("running-pill").classList.toggle("hidden", !runsCache.some((r) => r.status === "running"));
  for (const r of runsCache) {
    const el = document.createElement("div");
    el.className = "run-entry" + (r.id === selectedRunId ? " selected" : "");
    const matchup = `${short(r.participants?.A)} vs ${short(r.participants?.B)}`;
    // model ids are free-text user input — textContent only, never interpolate
    el.innerHTML = `<div class="task"></div>
      <div class="badges"><span class="badge matchup"></span>
      <span class="badge condition"></span>
      <span class="chip ${r.status}">${r.status === "running" ? "● running" : r.status}</span></div>`;
    el.querySelector(".badge.matchup").textContent = matchup;
    el.querySelector(".badge.condition").textContent = r.condition;
    el.querySelector(".task").textContent = r.task || r.id;
    el.onclick = () => selectRun(r.id);
    box.append(el);
  }
}
function short(name) {
  if (!name) return "?";
  return name.includes(":") ? name.split(":", 2)[1] : name.replace("-cli", "");
}

/* ---------- selection + subscription ---------- */
function newDebate() {
  selectedRunId = null;
  if (source) { source.close(); source = null; }
  resetChat();
  resetObserve();
  loadRuns(); // clears the sidebar .selected highlight
  $("chat-input").focus();
}
$("new-debate").onclick = newDebate;

function selectRun(runId) {
  selectedRunId = runId;
  if (source) source.close();
  resetChat();
  resetObserve();
  loadRuns();
  const es = new EventSource(`/api/runs/${runId}/events`);
  source = es;
  let sawTerminal = false;
  es.onerror = () => {
    es.close();
    if (es !== source || selectedRunId !== runId) return; // stale stream — never touch the live one
    if (runsCache.some((r) => r.id === runId)) {
      // transient drop while watching: resubscribe (server replays history, panes reset cleanly)
      setTimeout(() => {
        if (selectedRunId === runId && source === es) selectRun(runId);
      }, 1200);
    } else {
      loadRuns(); // run genuinely unknown (404)
    }
  };
  es.onmessage = (m) => {
    if (es !== source) return; // stale stream
    const e = JSON.parse(m.data);
    if (e.type === "stream_end") {
      es.close();
      if (!sawTerminal) showRestartBanner();
      loadRuns(); // stream_end fires after the server marks the run done — clears ● running now
      return;
    }
    if (e.type === "terminal") sawTerminal = true;
    renderChatEvent(e);
    renderObserveEvent(e);
  };
}
function showRestartBanner() {
  const note = "run didn't survive a server restart — partial rounds preserved";
  feed("sys", "server", note);
  if (pendingBubble) {
    pendingBubble.classList.remove("pending");
    pendingBubble.textContent = "⚠ " + note;
  }
}

/* ---------- chat pane ---------- */
let pendingBubble = null;
function resetChat() {
  $("chat-messages").innerHTML = "";
  pendingBubble = null;
  const run = runsCache.find((r) => r.id === selectedRunId);
  if (run) {
    bubble("user", run.task);
    pendingBubble = bubble("assistant pending", "⏳ …");
    $("compose-note").textContent = run.status === "running"
      ? "watching a live experiment — sending below launches a separate new one"
      : "viewing a past experiment — sending below starts a new one";
    $("compose-note").classList.remove("hidden");
  } else {
    const hint = document.createElement("div");
    hint.className = "chat-empty";
    hint.textContent = "New debate — ask a question below and two blind models will argue it out.";
    $("chat-messages").append(hint);
    $("compose-note").classList.add("hidden");
  }
}
function bubble(cls, text) {
  const el = document.createElement("div");
  el.className = "bubble " + cls;
  el.textContent = text;
  $("chat-messages").append(el);
  el.scrollIntoView({ block: "end" });
  return el;
}

/* model answers arrive as markdown — render it (sanitized), fall back to text */
function renderMarkdown(el, text) {
  if (window.marked && window.DOMPurify) {
    el.classList.add("md");
    el.innerHTML = DOMPurify.sanitize(marked.parse(text));
  } else {
    el.textContent = text;
  }
}
function renderChatEvent(e) {
  if (!pendingBubble) return;
  if (e.type === "state" && e.to === "debate_exchange")
    pendingBubble.textContent = `⏳ round ${e.round} — they disagree, debating…`;
  if (e.type === "call_started") pendingBubble.textContent = `⏳ round ${e.round} — ${e.label} thinking…`;
  if (e.type === "activity") pendingBubble.textContent = `⏳ round ${e.round} — ${e.label}: ${e.detail}`;
  if (e.type === "judge_started") pendingBubble.textContent = `⏳ round ${e.round} — judge deliberating…`;
  if (e.type === "judge_eval") {
    pendingBubble.dataset.answer = e.ruling.best_answer;
    pendingBubble.textContent =
      `⏳ round ${e.round}: ${e.ruling.verdict} (${e.ruling.convergence_score}/100)`;
  }
  if (e.type === "terminal") {
    pendingBubble.classList.remove("pending");
    if (pendingBubble.dataset.answer) renderMarkdown(pendingBubble, pendingBubble.dataset.answer);
    else pendingBubble.textContent = `(${e.status}: no answer)`;
    const meta = document.createElement("div");
    meta.className = "bubble-meta";
    meta.innerHTML = `${e.status} · ${e.rounds_completed} round(s) · <a href="#" id="watch-link">watch in Observe →</a>`;
    pendingBubble.after(meta);
    meta.querySelector("#watch-link").onclick = (ev) => { ev.preventDefault(); showTab("observe"); };
    addTraceLink(meta);
  }
  if (e.type === "exported" && e.url) refreshTraceLinks(e.url);
}
function addTraceLink(container) {
  const run = runsCache.find((r) => r.id === selectedRunId);
  if (run?.trace_url && /^https?:\/\//.test(run.trace_url))
    container.insertAdjacentHTML("beforeend",
      ` · <a class="trace-link" href="${run.trace_url}" target="_blank">trace ↗</a>`);
}
function refreshTraceLinks(url) {
  if (!/^https?:\/\//.test(url)) return;
  document.querySelectorAll(".bubble-meta, #observe-answer").forEach((el) => {
    if (!el.querySelector(".trace-link"))
      el.insertAdjacentHTML("beforeend",
        ` · <a class="trace-link" href="${url}" target="_blank">trace ↗</a>`);
  });
  loadRuns();
}

/* ---------- launch ---------- */
$("chat-send").onclick = async () => {
  const prompt = $("chat-input").value.trim();
  if (!prompt) return;
  $("chat-send").disabled = true;
  try {
    const res = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        cap: Number($("opt-cap").value),
        condition: $("opt-condition").value,
        model_a: $("opt-model-a").value.trim() || null,
        model_b: $("opt-model-b").value.trim() || null,
        // one judge pick encodes family + pinned model: "claude:claude-opus-4-8"
        judge: $("opt-judge").value.split(":")[0],
        judge_model: $("opt-judge").value.split(":")[1] || null,
      }),
    });
    if (!res.ok) { alertInline(await res.text()); return; }
    const { run_id } = await res.json();
    $("chat-input").value = "";
    await loadRuns();
    selectRun(run_id);
  } finally { $("chat-send").disabled = false; }
};
function alertInline(text) { feed("sys", "error", text); bubble("assistant", "⚠ " + text); }

/* ---------- observe pane ---------- */
function diagramText(active) {
  return `stateDiagram-v2
    direction LR
    init --> independent_answer
    independent_answer --> judge_eval
    debate_exchange --> judge_eval
    judge_eval --> consensus
    judge_eval --> cap_reached
    judge_eval --> budget_exceeded
    judge_eval --> debate_exchange
    independent_answer --> context_exceeded
    debate_exchange --> context_exceeded
    independent_answer --> error
    judge_eval --> error
    debate_exchange --> error
    classDef activeNode fill:#f59e0b,stroke:#fff,color:#111,font-weight:bold
    classDef resultNode fill:#0c3b2a,color:#34d399
    classDef interruptNode fill:#3a2f10,color:#fbbf24
    classDef errNode fill:#3b1618,color:#f87171
    class consensus,cap_reached resultNode
    class budget_exceeded,context_exceeded interruptNode
    class error errNode
    ${active ? `class ${active} activeNode` : ""}`;
}
let lastRuling = null;
let renderSeq = 0;
async function drawDiagram(active) {
  const seq = ++renderSeq;
  const { svg } = await mermaid.render("sm" + seq, diagramText(active));
  if (seq === renderSeq) $("observe-diagram").innerHTML = svg;
}
function resetObserve() {
  const run = runsCache.find((r) => r.id === selectedRunId);
  $("observe-context").innerHTML = "";
  if (run) {
    $("observe-context").innerHTML = `<span class="q"></span><span class="meta"></span>`;
    $("observe-context").querySelector(".q").textContent = "❝ " + (run.task || run.id) + " ❞";
    $("observe-context").querySelector(".meta").textContent =
      `${run.condition} · ${short(run.participants?.A)} vs ${short(run.participants?.B)}`;
  }
  ["A", "B", "J"].forEach((l) => setCard(l, "idle", ""));
  if (run) {
    setCardTitle("A", short(run.participants?.A));
    setCardTitle("B", short(run.participants?.B));
    setCardTitle("J", short((run.judge || "").replace("judge-", "")) || "?");
  }
  $("observe-gauge-fill").style.width = "0%";
  $("observe-gauge-label").textContent = "convergence: –";
  $("observe-feed").innerHTML = "";
  $("observe-answer").classList.add("hidden");
  $("observe-answer").innerHTML = "";
  lastRuling = null;
  drawDiagram(null);
}
function setCardTitle(label, name) {
  const heading = card(label).querySelector("h3");
  heading.firstChild.nodeValue =
    (label === "J" ? `Judge (invisible) · ${name} ` : `Participant ${label} · ${name} `);
}
function card(label) {
  return $("card-" + label);
}
function setCard(label, state, text) {
  const c = card(label);
  const badge = c.querySelector(".state");
  badge.textContent = state;
  badge.className = "state " + state + (state === "thinking" ? " pulse" : "");
  if (text !== undefined) c.querySelector("p").textContent = (text || "").slice(0, 220);
}
function feed(cls, who, text, md = false) {
  const el = document.createElement("div");
  el.className = "msg " + cls;
  const whoEl = document.createElement("span");
  whoEl.className = "who";
  whoEl.textContent = who; // carries model ids — free-text, never innerHTML
  el.append(whoEl);
  if (md) {
    const body = document.createElement("div");
    renderMarkdown(body, text);
    el.append(body);
  } else {
    el.append(text);
  }
  $("observe-feed").append(el);
}
function renderObserveEvent(e) {
  if (e.type === "state") {
    drawDiagram(e.to);
    if (["consensus","cap_reached","budget_exceeded","context_exceeded","error"].includes(e.to))
      feed("sys", "orchestrator", `terminal: ${e.to.toUpperCase()}`);
  } else if (e.type === "call_started") setCard(e.label, "thinking", "");
  else if (e.type === "activity") setCard(e.label, "thinking", e.detail);
  else if (e.type === "participant_call") {
    setCard(e.label, "answered", e.answer_raw);
    feed(e.label, `${e.label} · ${short(e.participant)} · round ${e.round}`, e.answer_raw, true);
  } else if (e.type === "participant_call_failed") {
    setCard(e.label, "failed", e.error);
    feed("sys", `participant ${e.label}`, "FAILED: " + e.error);
  } else if (e.type === "judge_started") setCard("J", "thinking", "");
  else if (e.type === "judge_eval") {
    lastRuling = e.ruling;
    const gist = (e.ruling.agreement_reasons?.[0] || e.ruling.cruxes?.[0] || "");
    setCard(
      "J", "answered",
      `${e.ruling.verdict} · ${e.ruling.convergence_score}/100 · quality ${e.ruling.best_answer_quality}` +
        (gist ? ` — ${gist}` : "")
    );
    $("observe-gauge-fill").style.width = e.ruling.convergence_score + "%";
    $("observe-gauge-label").textContent =
      `convergence: ${e.ruling.convergence_score}/100 · round ${e.round} · ${e.ruling.verdict}`;
    const cruxes = (e.ruling.cruxes || []).length ? "\ncruxes: " + e.ruling.cruxes.join(" | ") : "";
    feed("judge", `Judge · round ${e.round}`,
      `${e.ruling.verdict} (${e.ruling.convergence_score}/100), quality ${e.ruling.best_answer_quality}${cruxes}`);
  } else if (e.type === "terminal") {
    const box = $("observe-answer");
    box.classList.remove("hidden");
    box.textContent = "";
    const strong = document.createElement("b");
    strong.textContent = e.status + ": ";
    box.append(strong);
    if (e.error) {
      box.append(e.error);
    } else if (lastRuling?.best_answer) {
      const answer = document.createElement("div");
      renderMarkdown(answer, lastRuling.best_answer);
      box.append(answer);
    }
    if (lastRuling) {
      const why = document.createElement("div");
      why.className = "answer-why";
      const k = document.createElement("span");
      k.className = "k";
      k.textContent = `judge: ${lastRuling.verdict} (${lastRuling.convergence_score}/100, quality ${lastRuling.best_answer_quality}). `;
      why.append(k);
      if (lastRuling.agreement_reasons?.length)
        why.append("agreement: " + lastRuling.agreement_reasons.slice(0, 2).join(" · "));
      if (lastRuling.cruxes?.length) {
        const crux = document.createElement("div");
        crux.className = "crux";
        crux.textContent = "cruxes: " + lastRuling.cruxes.join(" · ");
        why.append(crux);
      }
      box.append(why);
    }
    addTraceLink(box);
    if (!document.querySelector("#observe-answer .trace-link")) {
      const retry = document.createElement("a");
      retry.href = "#"; retry.textContent = " · export trace";
      retry.onclick = async (ev) => {
        ev.preventDefault();
        const res = await fetch(`/api/runs/${selectedRunId}/export`, { method: "POST" });
        if (res.ok) refreshTraceLinks((await res.json()).url);
        else feed("sys", "langfuse", "export failed: " + (await res.text()));
      };
      box.append(retry);
    }
    loadRuns();
  } else if (e.type === "export_failed") {
    feed("sys", "langfuse", "auto-export failed: " + e.error + " (use 'export trace' to retry)");
  }
}

/* ---------- boot ---------- */
drawDiagram(null);
newDebate(); // land on a fresh compose; past experiments are one sidebar click away
setInterval(loadRuns, 15000);
