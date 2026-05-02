/* Eval tab — communicates over the shared WebSocket + REST endpoints */

"use strict";

// ── state ──────────────────────────────────────────────────────────────────
let evalRunning = false;
let evalSessions = [];
let evalTotal = 0;
let evalCompleted = 0;
let _evalVideoDebounce = null;
let _evalSessionInfo = null;

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
    evalLoadPastRuns();
    evalOnModeChange();
}

// ── mode / video ID change ─────────────────────────────────────────────────

function evalOnModeChange() {
    const mode = document.getElementById("eval-mode").value;
    const playerWrap = document.getElementById("eval-player-wrap");
    if (playerWrap) playerWrap.hidden = (mode !== "mic");
    if (mode === "mic") {
        const videoId = (document.getElementById("eval-video-id") || {}).value || "";
        if (videoId.length >= 8) _evalLoadSession(videoId);
    }
}

function evalOnVideoIdChange() {
    clearTimeout(_evalVideoDebounce);
    _evalVideoDebounce = setTimeout(() => {
        const mode = document.getElementById("eval-mode").value;
        if (mode !== "mic") return;
        const videoId = document.getElementById("eval-video-id").value.trim();
        if (videoId.length >= 8) _evalLoadSession(videoId);
    }, 600);
}

function _evalLoadSession(videoId) {
    _evalSessionInfo = null;
    fetch("/eval/session/" + encodeURIComponent(videoId))
        .then(r => { if (!r.ok) throw new Error(r.statusText); return r.json(); })
        .then(info => {
            _evalSessionInfo = info;
            const titleEl = document.getElementById("eval-player-title");
            if (titleEl) titleEl.textContent = info.title || info.video_id;
            const segEl = document.getElementById("eval-player-segment");
            if (segEl) segEl.textContent =
                "Eval segment: " + _fmtTime(info.audio_t0) + " \u2013 " + _fmtTime(info.audio_t_end) +
                "  (" + Math.round(info.duration_s) + "s)";
            const audio = document.getElementById("eval-audio");
            if (audio) {
                audio.src = "/eval/audio/" + encodeURIComponent(videoId);
                audio.load();
                audio.addEventListener("loadedmetadata", () => {
                    audio.currentTime = info.audio_t0;
                }, { once: true });
            }
            const playerWrap = document.getElementById("eval-player-wrap");
            if (playerWrap) playerWrap.hidden = false;
        })
        .catch(err => {
            _evalSetStatus("Session not found: " + err.message, "error");
        });
}

function _fmtTime(s) {
    if (s == null) return "?";
    const m = Math.floor(s / 60), sec = Math.round(s % 60);
    return m + ":" + String(sec).padStart(2, "0");
}

// ── start ───────────────────────────────────────────────────────────────────

function evalStart() {
    if (evalRunning) return;

    const mode = document.getElementById("eval-mode").value;
    const limitVal = document.getElementById("eval-limit").value;
    const minScore = parseFloat(document.getElementById("eval-minscore").value) || 60.0;
    const videoId = (document.getElementById("eval-video-id") || {}).value.trim() || "";
    const limit = limitVal ? parseInt(limitVal, 10) : null;

    const body = { mode, min_match_score: minScore };
    if (limit) body.limit_videos = limit;
    if (videoId) body.video_ids = [videoId];
    if (mode === "mic") body.audio_device = 2;

    evalSessions = [];
    evalTotal = 0;
    evalCompleted = 0;
    _evalClearResults();
    _evalSetStatus("Starting\u2026", "info");

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
            // For mic mode, ensure audio is playing
            if (mode === "mic") {
                const audio = document.getElementById("eval-audio");
                if (audio && audio.paused) audio.play().catch(() => {});
            }
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
                _evalSetStatus("Loading dataset\u2026", "info");
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
            _evalSetStatus("Done \u2014 " + evalCompleted + " sessions complete", "ok");
            if (msg.report) _evalShowKpis(msg.report);
            evalLoadPastRuns();
            // Stop audio playback when eval completes
            const audioEl = document.getElementById("eval-audio");
            if (audioEl && !audioEl.paused) audioEl.pause();
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
        btn.textContent = running ? "Scoring\u2026" : "Start Scoring";
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

    const tdId = document.createElement("td");
    tdId.className = "eval-mono";
    const sid = m.session_id || m.video_id || "";
    tdId.textContent = sid.length > 32 ? sid.slice(0, 32) + "\u2026" : sid;

    const lockOk = (m.lock_accuracy_pct || 0) >= 80;
    const tdLock = document.createElement("td");
    tdLock.className = "eval-center " + (lockOk ? "eval-ok" : "eval-fail");
    tdLock.textContent = (m.lock_accuracy_pct ?? 0).toFixed(0) + "%";

    const tdTtfcl = document.createElement("td");
    tdTtfcl.className = "eval-center";
    tdTtfcl.textContent = m.ttfcl_s != null ? m.ttfcl_s.toFixed(1) + "s" : "\u2014";

    const tdTrans = document.createElement("td");
    tdTrans.className = "eval-center " + ((m.detection_rate_pct || 0) >= 85 ? "eval-ok" : "eval-warn");
    tdTrans.textContent = m.gt_transitions ? (m.detection_rate_pct || 0).toFixed(0) + "%" : "\u2014";

    const lineOk = (m.line_accuracy_pm1_pct || 0) >= 80;
    const tdLine = document.createElement("td");
    tdLine.className = "eval-center " + (lineOk ? "eval-ok" : "eval-fail");
    tdLine.textContent = (m.line_accuracy_pm1_pct ?? 0).toFixed(0) + "%";

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

    _set("eval-r-lock-acc",      _pct(r.median_lock_accuracy_pct));
    _set("eval-r-lock-cov",      _pct(r.median_lock_coverage_pct));
    _set("eval-r-ttfcl-p50",     _sec(r.p50_ttfcl_s));
    _set("eval-r-ttfcl-p90",     _sec(r.p90_ttfcl_s));
    _set("eval-r-wrong-first",   _pct(r.wrong_first_lock_rate_pct));
    _set("eval-r-never-locked",  _pct(r.never_locked_rate_pct));

    _set("eval-r-gt-trans",      String(r.total_gt_transitions || 0));
    _set("eval-r-trans-rate",    _pct(r.overall_detection_rate_pct));
    _set("eval-r-trans-p50",     _sec(r.p50_transition_latency_s));
    _set("eval-r-trans-p90",     _sec(r.p90_transition_latency_s));
    _set("eval-r-spurious",      (r.median_spurious_per_hr || 0).toFixed(1) + "/hr");

    _set("eval-r-line-exact",    _pct(r.median_line_accuracy_exact_pct));
    _set("eval-r-line-pm1",      _pct(r.median_line_accuracy_pm1_pct));
    _set("eval-r-line-lag-p50",  _sec(r.p50_line_lag_s));
    _set("eval-r-line-lag-p90",  _sec(r.p90_line_lag_s));
    _set("eval-r-line-skips",    String(r.total_line_skips || 0));
    _set("eval-r-line-flickers", String(r.total_line_flickers || 0));

    _set("eval-r-ttr-p50",       _sec(r.p50_ttr_s));
    _set("eval-r-ttr-p90",       _sec(r.p90_ttr_s));
    _set("eval-r-disrupt-hr",    (r.median_disruption_per_hr || 0).toFixed(1) + "/hr");
    _set("eval-r-composite",     _pct(r.composite_pct_time_correct));

    const compEl = document.getElementById("eval-r-composite");
    if (compEl) {
        const v = r.composite_pct_time_correct || 0;
        compEl.className = v >= 70 ? "eval-kpi-value eval-ok" : "eval-kpi-value eval-fail";
    }
}

// ── past runs ───────────────────────────────────────────────────────────────

function evalLoadPastRuns() {
    fetch("/eval/runs")
        .then(r => r.json())
        .then(d => _evalRenderPastRuns(d.runs || []))
        .catch(() => {});
}

function _evalRenderPastRuns(runs) {
    const tbody = document.getElementById("eval-past-tbody");
    if (!tbody) return;
    tbody.textContent = "";

    if (!runs.length) {
        const tr = document.createElement("tr");
        const td = document.createElement("td");
        td.colSpan = 8;
        td.className = "eval-center";
        td.style.color = "#555";
        td.style.fontStyle = "italic";
        td.textContent = "No past runs";
        tr.appendChild(td);
        tbody.appendChild(tr);
        return;
    }

    runs.forEach(run => {
        const tr = document.createElement("tr");
        tr.className = "clickable";
        tr.title = "Click to expand session detail";

        const cells = [
            { text: run.run_id, cls: "eval-mono" },
            { text: _fmtDate(run.started_at), cls: "" },
            { text: (run.mode || "headless").toUpperCase(), cls: "eval-mono eval-center" },
            { text: String(run.total_sessions || 0), cls: "eval-center" },
            { text: _pct(run.lock_accuracy_pct), cls: "eval-center " + _kpiCls(run.lock_accuracy_pct, 80) },
            { text: _pct(run.overall_detection_rate_pct), cls: "eval-center " + _kpiCls(run.overall_detection_rate_pct, 85) },
            { text: _pct(run.line_accuracy_pm1_pct), cls: "eval-center " + _kpiCls(run.line_accuracy_pm1_pct, 80) },
            { text: _pct(run.composite_pct_time_correct), cls: "eval-center " + _kpiCls(run.composite_pct_time_correct, 70) },
        ];

        cells.forEach(c => {
            const td = document.createElement("td");
            td.className = c.cls;
            td.textContent = c.text;
            tr.appendChild(td);
        });

        tr.addEventListener("click", () => _evalToggleRunDetail(tr, run));
        tbody.appendChild(tr);
    });
}

function _evalToggleRunDetail(tr, run) {
    const detail = document.getElementById("eval-past-detail");
    if (!detail) return;

    const alreadySelected = tr.classList.contains("selected");
    document.querySelectorAll("#eval-past-tbody tr.selected").forEach(r => r.classList.remove("selected"));

    if (alreadySelected) {
        detail.hidden = true;
        detail.textContent = "";
        return;
    }

    tr.classList.add("selected");
    _evalExpandRun(run, detail);
    detail.hidden = false;
}

function _evalExpandRun(run, container) {
    container.textContent = "";

    const sessions = run.sessions || [];
    if (!sessions.length) {
        container.textContent = "No session detail available.";
        return;
    }

    const table = document.createElement("table");
    table.className = "eval-table";
    table.style.fontSize = "0.78rem";

    const thead = document.createElement("thead");
    const hRow = document.createElement("tr");
    ["Session", "Lock%", "TTFCL", "Trans%", "Line\xb11", "Correct%"].forEach(label => {
        const th = document.createElement("th");
        th.textContent = label;
        hRow.appendChild(th);
    });
    thead.appendChild(hRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    sessions.forEach(s => {
        const lock = s.lock || {};
        const trans = s.transitions || {};
        const line = s.line || {};
        const disrupt = s.disruption || {};

        const lockPct = lock.lock_accuracy_pct ?? s.lock_accuracy_pct;
        const ttfcl = lock.ttfcl_s ?? s.ttfcl_s;
        const transPct = trans.detection_rate_pct ?? s.detection_rate_pct;
        const linePm1 = line.line_accuracy_pm1_pct ?? s.line_accuracy_pm1_pct;
        const correct = disrupt.pct_time_correct ?? s.pct_time_correct;

        const sid = s.session_id || s.video_id || "";
        const tr = document.createElement("tr");

        const tdId = document.createElement("td");
        tdId.className = "eval-mono";
        tdId.textContent = sid.length > 28 ? sid.slice(0, 28) + "\u2026" : sid;

        const tdLock = document.createElement("td");
        tdLock.className = "eval-center " + _kpiCls(lockPct, 80);
        tdLock.textContent = _pct(lockPct);

        const tdTtfcl = document.createElement("td");
        tdTtfcl.className = "eval-center";
        tdTtfcl.textContent = ttfcl != null ? ttfcl.toFixed(1) + "s" : "\u2014";

        const tdTrans = document.createElement("td");
        tdTrans.className = "eval-center " + _kpiCls(transPct, 85);
        tdTrans.textContent = transPct != null ? _pct(transPct) : "\u2014";

        const tdLine = document.createElement("td");
        tdLine.className = "eval-center " + _kpiCls(linePm1, 80);
        tdLine.textContent = _pct(linePm1);

        const tdCorr = document.createElement("td");
        tdCorr.className = "eval-center " + _kpiCls(correct, 70);
        tdCorr.textContent = _pct(correct);

        tr.append(tdId, tdLock, tdTtfcl, tdTrans, tdLine, tdCorr);
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    container.appendChild(table);
}

// ── utility helpers ──────────────────────────────────────────────────────────

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

function _kpiCls(v, threshold) {
    if (v == null) return "";
    return v >= threshold ? "eval-ok" : "eval-fail";
}

function _fmtDate(ts) {
    if (!ts) return "\u2014";
    const d = new Date(ts * 1000);
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric" })
        + " " + d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}
