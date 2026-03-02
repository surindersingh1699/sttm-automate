/**
 * STTM Automate Dashboard
 * Real-time WebSocket client for monitoring and controlling the pipeline.
 */

let ws = null;
let isPaused = false;
let reconnectTimer = null;
let currentVerses = [];
let currentLineIndex = -1;
let currentShabadId = null;

// --- Safe DOM Helpers ---

function clearChildren(el) {
    while (el.firstChild) {
        el.removeChild(el.firstChild);
    }
}

function createElement(tag, className, textContent) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (textContent) el.textContent = textContent;
    return el;
}

// --- WebSocket Connection ---

function connect() {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(protocol + "//" + location.host + "/ws");

    ws.onopen = function() {
        setStatus("running", "Connected");
        if (reconnectTimer) {
            clearTimeout(reconnectTimer);
            reconnectTimer = null;
        }
    };

    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        handleMessage(data);
    };

    ws.onclose = function() {
        setStatus("connecting", "Disconnected - reconnecting...");
        reconnectTimer = setTimeout(connect, 3000);
    };

    ws.onerror = function() {
        setStatus("error", "Connection error");
    };
}

function send(data) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(data));
    }
}

// --- Message Handlers ---

function handleMessage(data) {
    console.log("[WS]", data.type, data);
    switch (data.type) {
        case "transcription":
            updateTranscription(data);
            break;
        case "candidates":
            updateCandidates(data.matches);
            break;
        case "shabad_locked":
            var lockedShabadId = data.shabad_id || (data.shabad && data.shabad.shabad_id) || null;
            if (data.verses && data.verses.length > 0) {
                currentShabadId = lockedShabadId;
                currentVerses = data.verses;
                currentLineIndex = 0;
                renderPangati();
                highlightPangati(0);
            } else if (lockedShabadId) {
                // Verses missing from broadcast — fetch via REST
                fetchVerses(lockedShabadId);
            }
            highlightAutoSelected();
            break;
        case "line_aligned":
            highlightPangati(data.line_index);
            break;
        case "shabad_switched":
            clearPangati();
            break;
        case "auto_selected":
            highlightAutoSelected();
            break;
        case "state":
            updateCurrentShabad(data.current);
            updateHistory(data.history);
            updatePinStatus(data.controller_pin);
            if (data.pipeline_state === "searching") {
                if (currentVerses.length > 0) {
                    clearPangati();
                }
            } else if (data.current) {
                var shabadId = data.current.shabad_id;
                // Populate pangati if we have verses and shabad changed or verses empty
                if (data.verses && data.verses.length > 0) {
                    if (currentVerses.length === 0 || currentShabadId !== shabadId) {
                        currentShabadId = shabadId;
                        currentVerses = data.verses;
                        renderPangati();
                    }
                } else if (shabadId && (currentVerses.length === 0 || currentShabadId !== shabadId)) {
                    // No verses in broadcast — fetch via REST
                    fetchVerses(shabadId);
                }
                if (data.current.current_line !== undefined) {
                    highlightPangati(data.current.current_line);
                }
            }
            break;
        case "status":
            isPaused = data.paused;
            updatePauseButton();
            break;
        case "controller_pin_updated":
            updatePinStatus(data.controller_pin);
            break;
        case "audio_level":
            updateAudioLevel(data.rms, data.has_vocals);
            break;
        case "error":
            showError(data.message);
            break;
    }
}

// --- UI Updates ---

function updateTranscription(data) {
    var el = document.getElementById("transcription-text");
    var lettersEl = document.getElementById("first-letters");

    if (data.text) {
        el.textContent = data.text;
        el.className = "";
    } else if (data.status === "music_only") {
        el.textContent = "Music playing... (waiting for vocals)";
        el.className = "placeholder";
    } else {
        el.textContent = "Listening... (no speech detected)";
        el.className = "placeholder";
    }

    if (data.first_letters) {
        lettersEl.textContent = "First letters: " + data.first_letters;
    } else {
        lettersEl.textContent = "";
    }
}

function updateAudioLevel(rms, hasVocals) {
    var el = document.getElementById("audio-level");
    if (!el) return;
    var pct = Math.min(rms * 500, 100);
    el.style.width = pct + "%";
    if (hasVocals) {
        el.className = "audio-fill active";
    } else if (rms > 0.01) {
        el.className = "audio-fill music";
    } else {
        el.className = "audio-fill";
    }
}

function updateCandidates(matches) {
    var container = document.getElementById("candidates-list");
    clearChildren(container);

    if (!matches || matches.length === 0) {
        container.appendChild(createElement("p", "placeholder", "No matches found"));
        return;
    }

    matches.forEach(function(m) {
        var pct = Math.round(m.score * 100);
        var level = m.action === "auto" ? "high" : m.action === "suggest" ? "medium" : "low";

        var item = createElement("div", "candidate-item action-" + m.action);
        item.addEventListener("click", function() { selectShabad(m.shabad_id); });

        // Text section
        var textDiv = createElement("div", "candidate-text");
        textDiv.appendChild(createElement("div", "gurmukhi", m.unicode || m.gurmukhi || ""));
        textDiv.appendChild(createElement("div", "english", m.english || ""));
        item.appendChild(textDiv);

        // Confidence bar
        var barOuter = createElement("div", "confidence-bar");
        var barFill = createElement("div", "confidence-fill " + level);
        barFill.style.width = pct + "%";
        barOuter.appendChild(barFill);
        item.appendChild(barOuter);

        // Confidence label
        item.appendChild(createElement("div", "confidence-label", pct + "%"));

        container.appendChild(item);
    });
}

function updateCurrentShabad(current) {
    var el = document.getElementById("current-shabad");
    clearChildren(el);

    if (!current) {
        el.appendChild(createElement("p", "placeholder", "No shabad selected"));
        return;
    }

    el.appendChild(createElement("div", "gurmukhi", current.unicode || current.gurmukhi || ""));
    el.appendChild(createElement("div", "english", current.english || ""));
    el.appendChild(createElement("div", "shabad-id",
        "ID: " + current.shabad_id + " | Line: " + (current.current_line + 1) + "/" + (current.total_lines || "?")));
}

// --- Pangati (Shabad Lines) ---

function renderPangati() {
    var container = document.getElementById("pangati-list");
    clearChildren(container);

    if (!currentVerses || currentVerses.length === 0) {
        container.appendChild(createElement("p", "placeholder", "Lock a shabad to see its lines"));
        return;
    }

    currentVerses.forEach(function(v, i) {
        var item = createElement("div", "pangati-item");
        item.setAttribute("data-index", i);

        item.appendChild(createElement("span", "line-num", (i + 1) + ""));
        item.appendChild(createElement("div", "gurmukhi", v.unicode || ""));
        item.appendChild(createElement("div", "english", v.english || ""));

        container.appendChild(item);
    });
}

function highlightPangati(index) {
    if (index < 0 || !currentVerses.length) return;
    currentLineIndex = index;

    var container = document.getElementById("pangati-list");
    var items = container.querySelectorAll(".pangati-item");

    items.forEach(function(item) {
        item.classList.remove("active");
    });

    if (index < items.length) {
        items[index].classList.add("active");
        items[index].scrollIntoView({ block: "center", behavior: "smooth" });
    }
}

function clearPangati() {
    currentVerses = [];
    currentLineIndex = -1;
    currentShabadId = null;
    var container = document.getElementById("pangati-list");
    clearChildren(container);
    container.appendChild(createElement("p", "placeholder", "Lock a shabad to see its lines"));
}

function fetchVerses(shabadId) {
    // Track the most recent shabad requested so stale responses can be ignored.
    currentShabadId = shabadId;
    fetch("/api/verses/" + shabadId)
        .then(function(resp) { return resp.json(); })
        .then(function(data) {
            if (data.verses && data.verses.length > 0 && currentShabadId === shabadId) {
                currentVerses = data.verses;
                renderPangati();
            }
        })
        .catch(function(err) {
            console.error("[fetchVerses] Error:", err);
        });
}

// --- History ---

function updateHistory(history) {
    var container = document.getElementById("history-list");
    clearChildren(container);

    if (!history || history.length === 0) {
        container.appendChild(createElement("p", "placeholder", "No previous shabads"));
        return;
    }

    history.forEach(function(h) {
        var time = new Date(h.started_at).toLocaleTimeString();

        var item = createElement("div", "history-item");
        item.addEventListener("click", function() { recallShabad(h.shabad_id); });

        item.appendChild(createElement("div", "gurmukhi", h.unicode || h.gurmukhi || ""));
        item.appendChild(createElement("div", "time", time));

        container.appendChild(item);
    });
}

function highlightAutoSelected() {
    var panel = document.getElementById("panel-current");
    panel.style.borderColor = "#4ecca3";
    setTimeout(function() { panel.style.borderColor = ""; }, 1000);
}

function showError(message) {
    console.error("[Pipeline Error]", message);
}

function setStatus(state, text) {
    var dot = document.getElementById("status-indicator");
    var label = document.getElementById("status-text");
    dot.className = "status-dot status-" + state;
    label.textContent = text;
}

function updatePinStatus(pinValue) {
    var status = document.getElementById("pin-status");
    var input = document.getElementById("controller-pin");
    if (!status || !input) return;

    if (pinValue === null || pinValue === undefined || pinValue === "") {
        status.textContent = "PIN: not set";
        input.value = "";
    } else {
        status.textContent = "PIN: " + pinValue;
        input.value = pinValue;
    }
}

// --- User Actions ---

function selectShabad(shabadId) {
    send({ type: "manual_select", shabad_id: shabadId });
}

function recallShabad(shabadId) {
    send({ type: "recall", shabad_id: shabadId });
}

function navigateLine(direction) {
    send({ type: "navigate", direction: direction });
}

function togglePause() {
    isPaused = !isPaused;
    send({ type: isPaused ? "pause" : "resume" });
    updatePauseButton();
}

function setControllerPin() {
    var input = document.getElementById("controller-pin");
    if (!input) return;
    var value = input.value ? parseInt(input.value, 10) : NaN;
    if (Number.isNaN(value)) {
        updatePinStatus(null);
        return;
    }
    send({ type: "set_controller_pin", controller_pin: value });
}

function clearControllerPin() {
    send({ type: "set_controller_pin", controller_pin: null });
}

function updatePauseButton() {
    var btn = document.getElementById("btn-pause");
    if (isPaused) {
        btn.textContent = "Resume";
        btn.className = "btn btn-primary";
        setStatus("paused", "Paused");
    } else {
        btn.textContent = "Pause";
        btn.className = "btn btn-warning";
        setStatus("running", "Connected");
    }
}

// --- Initialize ---
connect();
