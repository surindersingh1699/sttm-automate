/**
 * STTM Automate Dashboard
 * Real-time WebSocket client for monitoring and controlling the pipeline.
 */

let ws = null;
let isPaused = false;
let reconnectTimer = null;

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
    switch (data.type) {
        case "transcription":
            updateTranscription(data);
            break;
        case "candidates":
            updateCandidates(data.matches);
            break;
        case "auto_selected":
            highlightAutoSelected();
            break;
        case "state":
            updateCurrentShabad(data.current);
            updateHistory(data.history);
            break;
        case "status":
            isPaused = data.paused;
            updatePauseButton();
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
    }

    if (data.first_letters) {
        lettersEl.textContent = data.first_letters;
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
        "ID: " + current.shabad_id + " | Line: " + current.current_line));
}

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
