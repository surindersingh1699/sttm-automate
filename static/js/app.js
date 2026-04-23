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
let currentShabadState = null;
let historyState = [];
let confidenceMode = "balanced";
let audioSource = "local";
let audioDevice = null; // null = auto-detect
let audioWs = null;
let audioContext = null;
let audioStream = null;
let audioProcessor = null;
let decoderToggles = {
    greedy_decode: false,
    single_temperature: true,
    allow_repetition: true,
    independent_windows: false,
    cap_decode_length: true,
    skip_slow_windows: true,
};
const DASHBOARD_STATE_KEY = "sttm_automate_dashboard_state_v1";

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

function persistDashboardState() {
    try {
        localStorage.setItem(
            DASHBOARD_STATE_KEY,
            JSON.stringify({
                currentShabadId: currentShabadId,
                currentLineIndex: currentLineIndex,
                currentVerses: currentVerses,
                currentShabadState: currentShabadState,
                historyState: historyState,
                confidenceMode: confidenceMode,
            })
        );
    } catch (err) {
        console.warn("[state] persist failed", err);
    }
}

function restoreDashboardState() {
    try {
        const raw = localStorage.getItem(DASHBOARD_STATE_KEY);
        if (!raw) return;
        const data = JSON.parse(raw);

        currentShabadId = data.currentShabadId || null;
        currentLineIndex = Number.isInteger(data.currentLineIndex) ? data.currentLineIndex : -1;
        currentVerses = Array.isArray(data.currentVerses) ? data.currentVerses : [];
        currentShabadState = data.currentShabadState || null;
        historyState = Array.isArray(data.historyState) ? data.historyState : [];
        confidenceMode = data.confidenceMode || "balanced";
        updateConfidenceMode(confidenceMode);

        updateCurrentShabad(currentShabadState);
        updateHistory(historyState);

        if (currentVerses.length > 0) {
            renderPangati();
            if (currentLineIndex >= 0) {
                highlightPangati(currentLineIndex);
            }
        }
    } catch (err) {
        console.warn("[state] restore failed", err);
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
                persistDashboardState();
            } else if (lockedShabadId) {
                // Verses missing from broadcast — fetch via REST
                fetchVerses(lockedShabadId);
            }
            highlightAutoSelected();
            break;
        case "line_aligned":
            highlightPangati(data.line_index);
            persistDashboardState();
            break;
        case "shabad_switched":
            clearPangati();
            persistDashboardState();
            break;
        case "auto_selected":
            highlightAutoSelected();
            break;
        case "state":
            updateCurrentShabad(data.current);
            updateHistory(data.history);
            if (Object.prototype.hasOwnProperty.call(data, "controller_pin")) {
                updatePinStatus(data.controller_pin);
            }
            if (Object.prototype.hasOwnProperty.call(data, "confidence_mode")) {
                updateConfidenceMode(data.confidence_mode);
            }
            if (Object.prototype.hasOwnProperty.call(data, "audio_source")) {
                updateAudioSource(data.audio_source);
            }
            if (Object.prototype.hasOwnProperty.call(data, "audio_device")) {
                updateAudioDevice(data.audio_device);
            }
            if (data.decoder_toggles) {
                updateDecoderToggles(data.decoder_toggles);
            }
            if (data.pipeline_state === "searching" || data.pipeline_state === "candidate_lock") {
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
            persistDashboardState();
            break;
        case "status":
            isPaused = data.paused;
            updatePauseButton();
            if (isPaused) markModelIdle("paused");
            break;
        case "paused":
            isPaused = true;
            updatePauseButton();
            markModelIdle("paused");
            break;
        case "controller_pin_updated":
            updatePinStatus(data.controller_pin);
            break;
        case "confidence_mode_updated":
            updateConfidenceMode(data.mode);
            persistDashboardState();
            break;
        case "audio_source_updated":
            updateAudioSource(data.source);
            break;
        case "audio_device_updated":
            updateAudioDevice(data.device);
            break;
        case "decoder_toggles_updated":
            updateDecoderToggles(data.toggles);
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

function markModelIdle(label) {
    var speedEl = document.getElementById("model-speed");
    if (!speedEl) return;
    speedEl.className = "model-speed idle";
    speedEl.textContent = label || "idle";
}

function updateTranscription(data) {
    var el = document.getElementById("transcription-text");
    var lettersEl = document.getElementById("first-letters");
    var speedEl = document.getElementById("model-speed");

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

    if (speedEl) {
        if (isPaused) {
            markModelIdle("paused");
        } else if (data.status === "music_only") {
            markModelIdle("silence");
        } else if (Number.isFinite(data.transcribe_ms)) {
            var ms = data.transcribe_ms;
            var rtf = Number.isFinite(data.rtf) ? data.rtf : null;
            var cls = "model-speed";
            if (rtf !== null) {
                if (rtf <= 0.5) cls += " fast";
                else if (rtf <= 1.0) cls += " ok";
                else cls += " slow";
            }
            speedEl.className = cls;
            speedEl.textContent = rtf !== null
                ? ms + " ms · " + rtf.toFixed(2) + "× RTF"
                : ms + " ms";
        }
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
    currentShabadState = current || null;
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
                persistDashboardState();
            }
        })
        .catch(function(err) {
            console.error("[fetchVerses] Error:", err);
        });
}

// --- History ---

function updateHistory(history) {
    historyState = Array.isArray(history) ? history : [];
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

    // Ignore missing field on incremental state updates.
    if (pinValue === undefined) return;

    if (pinValue === null || pinValue === "") {
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
        return;
    }
    send({ type: "set_controller_pin", controller_pin: value });
}

function clearControllerPin() {
    send({ type: "set_controller_pin", controller_pin: null });
}

function forceUnlock() {
    send({ type: "force_unlock" });
}

function setConfidenceMode(mode) {
    updateConfidenceMode(mode);
    send({ type: "set_confidence_mode", mode: mode });
    persistDashboardState();
}

function updateConfidenceMode(mode) {
    if (mode !== "conservative" && mode !== "balanced" && mode !== "fast") {
        mode = "balanced";
    }
    confidenceMode = mode;
    var select = document.getElementById("confidence-mode");
    if (select && select.value !== mode) {
        select.value = mode;
    }
}

// --- Audio Source (Local / Remote Mic) ---

function setAudioSource(source) {
    if (source === "remote") {
        startRemoteMic();
    } else {
        stopRemoteMic();
    }
    send({ type: "set_audio_source", source: source });
    updateAudioSource(source);
}

function updateAudioSource(source) {
    if (source !== "local" && source !== "remote") {
        source = "local";
    }
    var previousSource = audioSource;
    audioSource = source;
    var select = document.getElementById("audio-source");
    if (select && select.value !== source) {
        select.value = source;
    }
    // Auto-start remote mic capture if server says we're in remote mode
    if (source === "remote" && !audioWs) {
        startRemoteMic();
    } else if (source === "local" && previousSource === "remote") {
        stopRemoteMic();
    }
    var micStatus = document.getElementById("mic-status");
    if (micStatus && source !== "remote") {
        micStatus.textContent = "";
    }
}

function startRemoteMic() {
    if (audioWs) return; // already streaming

    // Open dedicated audio WebSocket
    var protocol = location.protocol === "https:" ? "wss:" : "ws:";
    audioWs = new WebSocket(protocol + "//" + location.host + "/ws/audio");
    audioWs.binaryType = "arraybuffer";

    audioWs.onopen = function() {
        console.log("[RemoteMic] Audio WebSocket connected");
        // Now request mic access
        navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true } })
            .then(function(stream) {
                audioStream = stream;
                audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                var source = audioContext.createMediaStreamSource(stream);

                // ScriptProcessorNode: 4096 samples buffer, 1 input, 1 output
                audioProcessor = audioContext.createScriptProcessor(4096, 1, 1);
                audioProcessor.onaudioprocess = function(e) {
                    if (!audioWs || audioWs.readyState !== WebSocket.OPEN) return;
                    var inputData = e.inputBuffer.getChannelData(0);
                    // Convert float32 [-1,1] to int16 PCM
                    var pcm16 = new Int16Array(inputData.length);
                    for (var i = 0; i < inputData.length; i++) {
                        var s = Math.max(-1, Math.min(1, inputData[i]));
                        pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                    }
                    audioWs.send(pcm16.buffer);
                };

                source.connect(audioProcessor);
                audioProcessor.connect(audioContext.destination);

                var micStatus = document.getElementById("mic-status");
                if (micStatus) micStatus.textContent = "Streaming from device mic";
                console.log("[RemoteMic] Mic capture started at " + audioContext.sampleRate + "Hz");
            })
            .catch(function(err) {
                console.error("[RemoteMic] Mic access denied:", err);
                var micStatus = document.getElementById("mic-status");
                if (micStatus) micStatus.textContent = "Mic access denied!";
                stopRemoteMic();
                // Revert to local
                send({ type: "set_audio_source", source: "local" });
                updateAudioSource("local");
            });
    };

    audioWs.onclose = function() {
        console.log("[RemoteMic] Audio WebSocket closed");
        audioWs = null;
    };

    audioWs.onerror = function() {
        console.error("[RemoteMic] Audio WebSocket error");
    };
}

function stopRemoteMic() {
    if (audioProcessor) {
        audioProcessor.disconnect();
        audioProcessor = null;
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
    if (audioStream) {
        audioStream.getTracks().forEach(function(t) { t.stop(); });
        audioStream = null;
    }
    if (audioWs) {
        audioWs.close();
        audioWs = null;
    }
    var micStatus = document.getElementById("mic-status");
    if (micStatus) micStatus.textContent = "";
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

// --- Audio Device Selection ---

function loadAudioDevices() {
    fetch("/api/devices")
        .then(function(resp) { return resp.json(); })
        .then(function(data) {
            var select = document.getElementById("audio-device");
            if (!select || !data.devices) return;
            clearChildren(select);

            // Auto option
            var autoOpt = document.createElement("option");
            autoOpt.value = "auto";
            autoOpt.textContent = "Auto (Default Mic)";
            select.appendChild(autoOpt);

            data.devices.forEach(function(dev) {
                var opt = document.createElement("option");
                opt.value = dev.index;
                opt.textContent = dev.name;
                if (dev.default) opt.textContent += " (Default)";
                select.appendChild(opt);
            });

            // Set current selection
            if (audioDevice !== null) {
                select.value = String(audioDevice);
            } else {
                select.value = "auto";
            }
        })
        .catch(function(err) {
            console.error("[loadAudioDevices] Error:", err);
        });
}

function setAudioDevice(value) {
    var device = value === "auto" ? null : parseInt(value, 10);
    audioDevice = device;
    send({ type: "set_audio_device", device: device });
}

function updateAudioDevice(device) {
    audioDevice = device;
    var select = document.getElementById("audio-device");
    if (!select) return;
    if (device !== null) {
        select.value = String(device);
    } else {
        select.value = "auto";
    }
}

// --- Decoder Toggles ---

var DECODER_TOGGLE_IDS = {
    greedy_decode: "dec-greedy",
    single_temperature: "dec-single-temp",
    allow_repetition: "dec-allow-rep",
    independent_windows: "dec-indep-win",
    cap_decode_length: "dec-cap-tokens",
    skip_slow_windows: "dec-skip-slow",
};

function updateDecoderToggles(toggles) {
    if (!toggles) return;
    Object.keys(DECODER_TOGGLE_IDS).forEach(function(key) {
        if (Object.prototype.hasOwnProperty.call(toggles, key)) {
            decoderToggles[key] = !!toggles[key];
            var el = document.getElementById(DECODER_TOGGLE_IDS[key]);
            if (el) el.checked = decoderToggles[key];
        }
    });
}

function setDecoderToggle(key, checked) {
    decoderToggles[key] = !!checked;
    var payload = {};
    payload[key] = !!checked;
    send({ type: "set_decoder_toggles", toggles: payload });
}

function toggleDecoderInfo(force) {
    var popover = document.getElementById("decoder-info-popover");
    var btn = document.getElementById("decoder-info-btn");
    if (!popover) return;
    var show = typeof force === "boolean" ? force : popover.hasAttribute("hidden");
    if (show) {
        popover.removeAttribute("hidden");
        if (btn) btn.classList.add("active");
    } else {
        popover.setAttribute("hidden", "");
        if (btn) btn.classList.remove("active");
    }
}

document.addEventListener("click", function(e) {
    var popover = document.getElementById("decoder-info-popover");
    var btn = document.getElementById("decoder-info-btn");
    if (!popover || popover.hasAttribute("hidden")) return;
    if (popover.contains(e.target)) return;
    if (btn && btn.contains(e.target)) return;
    toggleDecoderInfo(false);
});

// --- Initialize ---
restoreDashboardState();
loadAudioDevices();
connect();
