/* Eval tab — communicates over the shared WebSocket + REST endpoints */

"use strict";

// ── state ──────────────────────────────────────────────────────────────────
let evalRunning = false;
let evalSessions = [];
let evalTotal = 0;
let evalCompleted = 0;

// ── init ───────────────────────────────────────────────────────────────────

function evalInit() {
    fetch("/eval/status")
        .then(r => r.json())
        .then(d => {
            if (d.status === "running") {
                evalRunning = true;
                evalCompleted = d.completed || 0;
                evalTotal = d.total_jobs || 0;
                _evalSetRunning(true);
                _updateProgress();
            }
        })
        .catch(() => {});
}

// ── start ───────────────────────────────────────────────────────────────────

function evalStart() {
    if (evalRunning) return;

    const mode = document.getElementById("eval-mode").value;
    const limitVal = document.getElementById("eval-limit").value;
    const minScore = parseFloat(document.getElementById("eval-minscore").value) || 60.0;
    const videoId = (document.getElementById("eval-video-id") || {}).value || "";
    const limit = limitVal ? parseInt(limitVal, 10) : null;

    const body = { mode, min_match_score: minScore };
    if (limit) body.limit_videos = limit;
    if (videoId) body.video_ids = [videoId];

    evalSessions = [];
    evalTotal = 0;
    evalCompleted = 0;
    _evalClearResults();
    _evalSetStatus("Starting…", "info");

    fetch("/eval/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    })
        .then(r => {
            if (!r.ok) return r.json().then(d => { throw new Error(d.detail || r.statusText); });
            return r.json();
        })
        .then(d => {
            evalRunning = true;
            _evalSetRunning(true);
            _evalSetStatus("Run " + d.run_id + " started (" + mode.toUpperCase() + ")", "info");
        })
        .catch(err => {
            _evalSetStatus("Error: " + err.message, "error");
        });
}

// ── WebSocket message handler (called from app.js) ─────────────────────────

function evalHandleMessage(msg) {
    switch (msg.type) {
        case "eval_status":
            if (msg.status === "loading") {
                _evalSetStatus("Loading dataset…", "info");
            } else if (msg.status === "running") {
                evalTotal = msg.total || 0;
                _evalSetStatus("Running " + evalTotal + " sessions in " + (msg.mode || "headless").toUpperCase() + " mode", "info");
                _updateProgress();
            }
            break;

        case "eval_progress":
            evalTotal = msg.total || evalTotal;
            evalCompleted = msg.completed || evalCompleted;
            _evalSetStatus("Session " + (msg.job_idx + 1) + "/" + evalTotal + ": " + msg.job_id, "info");
            _updateProgress();
            break;

        case "eval_session_progress":
            _evalUpdateSessionProgress(msg);
            break;

        case "eval_session_done":
            evalCompleted = msg.completed || evalCompleted;
            evalTotal = msg.total || evalTotal;
            evalSessions.push(msg);
            _evalAppendSessionRow(msg);
            _updateProgress();
            break;

        case "eval_complete":
            evalRunning = false;
            evalCompleted = msg.completed || evalCompleted;
            _evalSetRunning(false);
            _evalSetStatus("Done — " + evalCompleted + " sessions complete", "ok");
            if (msg.report) _evalShowKpis(msg.report);
            break;

        case "eval_error":
            evalRunning = false;
            _evalSetRunning(false);
            _evalSetStatus("Error: " + msg.error, "error");
            break;
    }
}

// ── helpers ─────────────────────────────────────────────────────────────────

function _evalSetRunning(running) {
    const btn = document.getElementById("eval-start-btn");
    if (btn) {
        btn.disabled = running;
        btn.textContent = running ? "Running…" : "Start Eval";
    }
}

function _evalSetStatus(text, cls) {
    const el = document.getElementById("eval-status-text");
    if (!el) return;
    el.textContent = text;
    el.className = "eval-status-text eval-status-" + (cls || "info");
}

function _updateProgress() {
    const bar = document.getElementById("eval-progress-bar");
    const lbl = document.getElementById("eval-progress-label");
    if (!bar || !lbl) return;
    const pct = evalTotal > 0 ? Math.round(evalCompleted / evalTotal * 100) : 0;
    bar.style.width = pct + "%";
    lbl.textContent = evalTotal > 0 ? evalCompleted + " / " + evalTotal + "  (" + pct + "%)" : "";
}

function _evalClearResults() {
    const tbody = document.getElementById("eval-session-tbody");
    if (tbody) tbody.textContent = "";
    const report = document.getElementById("eval-report");
    if (report) report.hidden = true;
    const prog = document.getElementById("eval-session-progress");
    if (prog) prog.textContent = "";
}

function _evalUpdateSessionProgress(msg) {
    const el = document.getElementById("eval-session-progress");
    if (!el) return;
    el.textContent = msg.session_id + "  " + msg.elapsed_s + "s / " + msg.total_s + "s  (" + msg.pct + "%)";
}

function _evalAppendSessionRow(m) {
    const tbody = document.getElementById("eval-session-tbody");
    if (!tbody) return;

    const tr = document.createElement("tr");

    // Session ID
    const tdId = document.createElement("td");
    tdId.className = "eval-mono";
    const sid = m.session_id || m.video_id || "";
    tdId.textContent = sid.length > 32 ? sid.slice(0, 32) + "…" : sid;

    // Lock %
    const lockOk = (m.lock_accuracy_pct || 0) >= 80;
    const tdLock = document.createElement("td");
    tdLock.className = "eval-center " + (lockOk ? "eval-ok" : "eval-fail");
    tdLock.textContent = (m.lock_accuracy_pct ?? 0).toFixed(0) + "%";

    // TTFCL
    const tdTtfcl = document.createElement("td");
    tdTtfcl.className = "eval-center";
    tdTtfcl.textContent = m.ttfcl_s != null ? m.ttfcl_s.toFixed(1) + "s" : "—";

    // Transition detection
    const tdTrans = document.createElement("td");
    tdTrans.className = "eval-center " + ((m.detection_rate_pct || 0) >= 85 ? "eval-ok" : "eval-warn");
    tdTrans.textContent = m.gt_transitions ? (m.detection_rate_pct || 0).toFixed(0) + "%" : "—";

    // Line ±1
    const lineOk = (m.line_accuracy_pm1_pct || 0) >= 80;
    const tdLine = document.createElement("td");
    tdLine.className = "eval-center " + (lineOk ? "eval-ok" : "eval-fail");
    tdLine.textContent = (m.line_accuracy_pm1_pct ?? 0).toFixed(0) + "%";

    // % time correct
    const tdCorr = document.createElement("td");
    tdCorr.className = "eval-center " + ((m.pct_time_correct || 0) >= 70 ? "eval-ok" : "eval-fail");
    tdCorr.textContent = (m.pct_time_correct ?? 0).toFixed(0) + "%";

    tr.append(tdId, tdLock, tdTtfcl, tdTrans, tdLine, tdCorr);
    tbody.appendChild(tr);

    const wrap = tbody.closest(".eval-table-wrap");
    if (wrap) wrap.scrollTop = wrap.scrollHeight;
}

function _evalShowKpis(r) {
    const el = document.getElementById("eval-report");
    if (!el) return;
    el.hidden = false;

    // Axis A — Lock
    _set("eval-r-lock-acc",      _pct(r.median_lock_accuracy_pct));
    _set("eval-r-lock-cov",      _pct(r.median_lock_coverage_pct));
    _set("eval-r-ttfcl-p50",     _sec(r.p50_ttfcl_s));
    _set("eval-r-ttfcl-p90",     _sec(r.p90_ttfcl_s));
    _set("eval-r-wrong-first",   _pct(r.wrong_first_lock_rate_pct));
    _set("eval-r-never-locked",  _pct(r.never_locked_rate_pct));

    // Axis B — Transitions
    _set("eval-r-gt-trans",      String(r.total_gt_transitions || 0));
    _set("eval-r-trans-rate",    _pct(r.overall_detection_rate_pct));
    _set("eval-r-trans-p50",     _sec(r.p50_transition_latency_s));
    _set("eval-r-trans-p90",     _sec(r.p90_transition_latency_s));
    _set("eval-r-spurious",      (r.median_spurious_per_hr || 0).toFixed(1) + "/hr");

    // Axis C — Line tracking
    _set("eval-r-line-exact",    _pct(r.median_line_accuracy_exact_pct));
    _set("eval-r-line-pm1",      _pct(r.median_line_accuracy_pm1_pct));
    _set("eval-r-line-lag-p50",  _sec(r.p50_line_lag_s));
    _set("eval-r-line-lag-p90",  _sec(r.p90_line_lag_s));
    _set("eval-r-line-skips",    String(r.total_line_skips || 0));
    _set("eval-r-line-flickers", String(r.total_line_flickers || 0));

    // Axis D — Disruption
    _set("eval-r-ttr-p50",       _sec(r.p50_ttr_s));
    _set("eval-r-ttr-p90",       _sec(r.p90_ttr_s));
    _set("eval-r-disrupt-hr",    (r.median_disruption_per_hr || 0).toFixed(1) + "/hr");
    _set("eval-r-composite",     _pct(r.composite_pct_time_correct));

    // Colour the composite tile
    const compEl = document.getElementById("eval-r-composite");
    if (compEl) {
        const v = r.composite_pct_time_correct || 0;
        compEl.className = v >= 70 ? "eval-kpi-value eval-ok" : "eval-kpi-value eval-fail";
    }
}

function _set(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}

function _pct(v) {
    return v != null ? v.toFixed(1) + "%" : "N/A";
}

function _sec(v) {
    return v != null ? v.toFixed(2) + "s" : "N/A";
}
