// 疲劳驾驶检测系统 - 前端控制脚本

let polling = null;
let isRunning = false;
let clockTimer = null;
let viewMode = "experiment";
let recentScenarioEvents = [];
let isYoloEnabled = false;
let displayMode = "rule";
const alertOverlayHoldMs = 3000;
let distractionOverlayUntil = 0;
let fatigueOverlayUntil = 0;
const speechCooldownMs = 10000;
const speechMaxCountPerAlert = 3;
const speechState = new Map();
let isSpeechPlaying = false;

function getDistractionAlerts(d) {
    const alerts = Array.isArray(d.dms_alerts) ? d.dms_alerts : [];
    return alerts.filter(alert =>
        alert.includes("打电话") ||
        alert.includes("抽烟") ||
        alert.includes("左顾右盼") ||
        alert.includes("玩手机") ||
        alert.includes("喝水")
    );
}

function getFatigueAlerts(d) {
    const alerts = Array.isArray(d.dms_alerts) ? d.dms_alerts : [];
    const fatigueAlerts = alerts.filter(alert =>
        alert.includes("闭眼") ||
        alert.includes("打哈欠") ||
        alert.includes("低头")
    );

    if (fatigueAlerts.length > 0) {
        return fatigueAlerts;
    }

    const fallbackAlerts = [];
    if (d.eye_closed) fallbackAlerts.push("闭眼预警");
    if (d.is_yawning) fallbackAlerts.push("打哈欠预警");
    if (d.is_head_down) fallbackAlerts.push("低头预警");
    if (d.is_fatigued && Array.isArray(d.reasons)) {
        d.reasons.forEach(reason => fallbackAlerts.push(`${reason}预警`));
    }
    return fallbackAlerts;
}

function isDistractionWarningHeld() {
    const phoneOverlay = document.getElementById("phoneAlertOverlay");
    return Date.now() < distractionOverlayUntil || Boolean(phoneOverlay && phoneOverlay.classList.contains("active"));
}

function normalizeSpeechKey(message) {
    return String(message || "").trim();
}

function canSpeakAlert(key) {
    if (!key) return false;
    const now = Date.now();
    const state = speechState.get(key) || { count: 0, lastAt: 0 };
    if (state.count >= speechMaxCountPerAlert) return false;
    if (now - state.lastAt < speechCooldownMs) return false;
    return true;
}

function markAlertSpoken(key) {
    const now = Date.now();
    const prev = speechState.get(key) || { count: 0, lastAt: 0 };
    speechState.set(key, {
        count: prev.count + 1,
        lastAt: now,
    });
}

function speakText(text, key) {
    if (!("speechSynthesis" in window)) return;
    const normalizedKey = normalizeSpeechKey(key || text);
    if (!canSpeakAlert(normalizedKey)) return;

    try {
        window.speechSynthesis.cancel();
        isSpeechPlaying = false;
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = "zh-CN";
        utterance.rate = 1;
        utterance.pitch = 1;
        utterance.volume = 1;
        utterance.onstart = () => {
            isSpeechPlaying = true;
        };
        utterance.onend = () => {
            isSpeechPlaying = false;
        };
        utterance.onerror = () => {
            isSpeechPlaying = false;
        };
        window.speechSynthesis.speak(utterance);
        markAlertSpoken(normalizedKey);
    } catch (error) {
        isSpeechPlaying = false;
        console.error("语音播报失败", error);
    }
}

function speakAlertByData(d) {
    if (!isRunning) return;
    const distractionAlerts = getDistractionAlerts(d);
    const hasHeldDistractionWarning = distractionAlerts.length === 0 && isDistractionWarningHeld();
    const fatigueAlerts = (distractionAlerts.length > 0 || hasHeldDistractionWarning) ? [] : getFatigueAlerts(d);

    for (const alertText of distractionAlerts) {
        if (alertText.includes("左顾右盼")) {
            speakText("检测到左顾右盼行为，请立即专注前方道路", "dms_look_away");
            return;
        }
        if (alertText.includes("打电话")) {
            speakText("检测到手机使用行为，请立即停止使用手机，专注驾驶", "dms_phone");
            return;
        }
        if (alertText.includes("抽烟")) {
            speakText("检测到抽烟行为，请立即停止危险行为", "dms_smoke");
            return;
        }
        if (alertText.includes("玩手机")) {
            speakText("检测到玩手机行为，请立即放下手机，专注驾驶", "dms_phone_use");
            return;
        }
        if (alertText.includes("喝水")) {
            speakText("检测到喝水行为，请注意分心驾驶风险", "dms_drinking");
            return;
        }
    }

    if (hasHeldDistractionWarning) return;

    for (const alertText of fatigueAlerts) {
        if (alertText.includes("闭眼")) {
            speakText("检测到闭眼风险，请立即保持清醒", "fatigue_eye");
            return;
        }
        if (alertText.includes("低头")) {
            speakText("检测到低头行为，请抬头注意前方", "fatigue_head_down");
            return;
        }
        if (alertText.includes("哈欠")) {
            speakText("检测到打哈欠行为，请注意疲劳风险", "fatigue_yawn");
            return;
        }
    }

    if (d.is_fatigued) {
        speakText("疲劳驾驶预警，请立即休息", "fatigue_general");
        return;
    }

    if (d.eye_closed) {
        speakText("检测到闭眼风险，请保持清醒", "rule_eye_closed");
        return;
    }

    if (d.is_yawning) {
        speakText("检测到打哈欠行为，请注意疲劳风险", "rule_yawn");
        return;
    }

    if (d.is_head_down) {
        speakText("检测到低头行为，请抬头注意前方", "rule_head_down");
    }
}

function resetSpeechAlerts() {
    speechState.clear();
    isSpeechPlaying = false;
    if ("speechSynthesis" in window) {
        window.speechSynthesis.cancel();
    }
}

function updateDistractionOverlay(d) {
    const phoneOverlay = document.getElementById("phoneAlertOverlay");
    const phoneAlertText = document.getElementById("phoneAlertText");
    if (!phoneOverlay || !phoneAlertText) return;

    const now = Date.now();
    const distractionAlerts = getDistractionAlerts(d);
    const hasDistractionAlert = distractionAlerts.length > 0;

    if (distractionAlerts.some(alert => alert.includes("打电话"))) {
        phoneAlertText.textContent = "分心驾驶警告！请停止使用手机！";
        distractionOverlayUntil = now + alertOverlayHoldMs;
        phoneOverlay.classList.add("active");
        return;
    }

    if (distractionAlerts.some(alert => alert.includes("抽烟"))) {
        phoneAlertText.textContent = "分心驾驶警告！请停止抽烟！";
        distractionOverlayUntil = now + alertOverlayHoldMs;
        phoneOverlay.classList.add("active");
        return;
    }

    if (distractionAlerts.some(alert => alert.includes("左顾右盼"))) {
        phoneAlertText.textContent = "分心驾驶警告！请立即专注前方道路！";
        distractionOverlayUntil = now + alertOverlayHoldMs;
        phoneOverlay.classList.add("active");
        return;
    }

    if (distractionAlerts.some(alert => alert.includes("玩手机"))) {
        phoneAlertText.textContent = "分心驾驶警告！请立即放下手机！";
        distractionOverlayUntil = now + alertOverlayHoldMs;
        phoneOverlay.classList.add("active");
        return;
    }

    if (distractionAlerts.some(alert => alert.includes("喝水"))) {
        phoneAlertText.textContent = "分心驾驶警告！请停止喝水并专注驾驶！";
        distractionOverlayUntil = now + alertOverlayHoldMs;
        phoneOverlay.classList.add("active");
        return;
    }

    if (!hasDistractionAlert && (now < distractionOverlayUntil || isSpeechPlaying)) {
        phoneOverlay.classList.add("active");
        return;
    }

    phoneOverlay.classList.remove("active");
}

function formatCurrentTime() {
    const now = new Date();
    return now.toLocaleTimeString("zh-CN", { hour12: false });
}

function updateHeaderClock() {
    const timeEl = document.getElementById("headerCurrentTime");
    if (timeEl) {
        timeEl.textContent = formatCurrentTime();
    }
}

updateHeaderClock();
if (!clockTimer) {
    clockTimer = setInterval(updateHeaderClock, 1000);
}

function setViewMode(mode) {
    viewMode = mode === "scenario" ? "scenario" : "experiment";

    const experimentBtn = document.getElementById("viewModeExperiment");
    const scenarioBtn = document.getElementById("viewModeScenario");
    const dashboardGrid = document.getElementById("dashboardGrid");
    const experimentSidebar = document.getElementById("experimentSidebar");
    const scenarioSidebar = document.getElementById("scenarioSidebar");
    const controlBar = document.getElementById("controlBar");
    const monitorWorkspace = document.getElementById("monitorWorkspace");
    const systemFooter = document.querySelector(".system-footer");
    const workspaceTopbar = document.querySelector(".workspace-topbar");
    const headerStrategyCard = document.getElementById("headerStrategyCard");
    const headerEngineCard = document.getElementById("headerEngineCard");

    if (experimentBtn) experimentBtn.classList.toggle("active", viewMode === "experiment");
    if (scenarioBtn) scenarioBtn.classList.toggle("active", viewMode === "scenario");

    if (dashboardGrid) {
        dashboardGrid.classList.toggle("experiment-mode", viewMode === "experiment");
        dashboardGrid.classList.toggle("scenario-mode", viewMode === "scenario");
    }

    if (experimentSidebar) {
        experimentSidebar.classList.toggle("view-hidden", viewMode !== "experiment");
    }

    if (scenarioSidebar) {
        scenarioSidebar.classList.toggle("view-hidden", viewMode !== "scenario");
    }

    if (controlBar) {
        controlBar.classList.toggle("experiment-control-card", viewMode === "experiment");
        controlBar.classList.toggle("scenario-control-card", viewMode === "scenario");
        if (viewMode === "scenario" && scenarioSidebar) {
            scenarioSidebar.prepend(controlBar);
        } else if (viewMode === "experiment" && experimentSidebar) {
            experimentSidebar.prepend(controlBar);
        }
    }

    if (systemFooter) {
        systemFooter.classList.toggle("view-hidden", viewMode === "scenario");
    }

    if (workspaceTopbar) {
        workspaceTopbar.classList.toggle("view-hidden", viewMode === "scenario");
    }

    if (headerStrategyCard) {
        headerStrategyCard.classList.toggle("view-hidden", viewMode === "scenario");
    }

    if (headerEngineCard) {
        headerEngineCard.classList.toggle("view-hidden", viewMode === "scenario");
    }
}

function deriveRiskLevel(d) {
    if (getDistractionAlerts(d).length > 0 || getFatigueAlerts(d).length > 0) {
        return "高风险";
    }
    if (d.is_fatigued || d.eye_closed || d.is_yawning || d.is_head_down) {
        return "注意";
    }
    return "低风险";
}

function deriveAlertSummary(d) {
    const distractionAlerts = getDistractionAlerts(d);
    const fatigueAlerts = getFatigueAlerts(d);
    if (distractionAlerts.length > 0) {
        return `分心警告：${distractionAlerts[0]}`;
    }
    if (fatigueAlerts.length > 0) {
        return `疲劳警告：${fatigueAlerts[0]}`;
    }
    if (d.reasons && d.reasons.length > 0) {
        return `疲劳警告：${d.reasons.join("、")}`;
    }
    return "暂无告警";
}

function deriveMonitorState() {
    return isRunning ? "监测中" : "待启动";
}

function deriveCameraState(d) {
    if (!isRunning) return "未启动";
    return d && d.face_detected ? "在线" : "在线";
}

function renderScenarioEvents() {
    const list = document.getElementById("scenarioAlertList");
    if (!list) return;

    if (recentScenarioEvents.length === 0) {
        list.innerHTML = '<div class="scenario-alert-empty">暂无关键事件</div>';
        return;
    }

    list.innerHTML = recentScenarioEvents
        .slice(-2)
        .reverse()
        .map(event => (
            `<div class="scenario-alert-item ${event.level}">[${event.time}] ${event.message}</div>`
        ))
        .join("");
}

function updateYoloToggleButton() {
    const btn = document.getElementById("btnYoloToggle");
    if (!btn) return;
    btn.textContent = "YOLO模式";
    btn.classList.toggle("enabled", isYoloEnabled);
}

function updateDisplayToggleButton() {
    const btn = document.getElementById("btnDisplayToggle");
    if (!btn) return;

    btn.classList.remove("display-rule", "display-both");

    if (displayMode === "rule") {
        btn.textContent = "显示：规则";
        btn.classList.add("display-rule");
    } else if (displayMode === "both") {
        btn.textContent = "显示：双模式";
        btn.classList.add("display-both");
    } else {
        btn.textContent = "显示：YOLO";
    }
}

function updateScenarioPanel(d, riskLevel, alertSummary) {
    const scenarioStatusMain = document.getElementById("scenarioStatusMain");
    const scenarioRiskLevel = document.getElementById("scenarioRiskLevel");
    const scenarioMonitorState = document.getElementById("scenarioMonitorState");
    const scenarioCameraState = document.getElementById("scenarioCameraState");
    const scenarioAlertSummary = document.getElementById("scenarioAlertSummary");
    const scenarioEyeState = document.getElementById("scenarioEyeState");
    const scenarioMouthState = document.getElementById("scenarioMouthState");
    const scenarioHeadState = document.getElementById("scenarioHeadState");

    if (scenarioStatusMain) {
        if (riskLevel === "高风险") {
            scenarioStatusMain.textContent = "高风险告警";
        } else if (riskLevel === "注意") {
            scenarioStatusMain.textContent = "注意风险";
        } else {
            scenarioStatusMain.textContent = "正常驾驶";
        }
    }

    if (scenarioRiskLevel) scenarioRiskLevel.textContent = riskLevel;
    if (scenarioMonitorState) scenarioMonitorState.textContent = deriveMonitorState();
    if (scenarioCameraState) scenarioCameraState.textContent = deriveCameraState(d);
    if (scenarioAlertSummary) scenarioAlertSummary.textContent = alertSummary;
    if (scenarioEyeState) scenarioEyeState.textContent = d.eye_closed ? "闭眼风险" : "正常";
    if (scenarioMouthState) scenarioMouthState.textContent = d.is_yawning ? "哈欠风险" : "正常";
    if (scenarioHeadState) scenarioHeadState.textContent = d.is_head_down ? "低头风险" : "正常";
}

// ---- 控制 ----

async function startDetection() {
    const res = await fetch("/api/start", { method: "POST" });
    const data = await res.json();
    if (data.success) {
        isRunning = true;
        document.getElementById("btnStart").disabled = true;
        document.getElementById("btnStop").disabled = false;
        document.getElementById("placeholder").classList.add("hidden");
        const img = document.getElementById("videoFeed");
        img.src = "/video_feed?" + Date.now();
        img.classList.add("active");
        document.getElementById("headerMonitorStatus").textContent = "监测中";
        document.getElementById("headerCameraStatus").textContent = "在线";
        document.getElementById("scenarioMonitorState").textContent = "监测中";
        document.getElementById("scenarioCameraState").textContent = "在线";
        resetSpeechAlerts();
        startPolling();
        startLogPoll();
    } else {
        alert(data.message);
    }
}

async function stopDetection() {
    await fetch("/api/stop", { method: "POST" });
    isRunning = false;
    document.getElementById("btnStart").disabled = false;
    document.getElementById("btnStop").disabled = true;
    document.getElementById("videoFeed").classList.remove("active");
    document.getElementById("videoFeed").src = "";
    document.getElementById("placeholder").classList.remove("hidden");
    document.getElementById("fatigueOverlay").classList.remove("active");
    document.getElementById("phoneAlertOverlay").classList.remove("active");
    distractionOverlayUntil = 0;
    fatigueOverlayUntil = 0;
    resetSpeechAlerts();
    stopPolling();
    stopLogPoll();
    resetUI();
}

async function changeMode(mode) {
    await fetch("/api/mode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode }),
    });
}

function switchMode(mode) {
    // 更新按钮样式
    document.querySelectorAll(".btn-mode").forEach(btn => btn.classList.remove("active"));
    document.getElementById("mode" + mode.charAt(0).toUpperCase() + mode.slice(1)).classList.add("active");
    // 发送模式切换请求
    changeMode(mode);
}

async function applyConfig() {
    const config = {
        ear_threshold: parseFloat(document.getElementById("cfgEar").value),
        mar_threshold: parseFloat(document.getElementById("cfgMar").value),
        pitch_threshold: parseFloat(document.getElementById("cfgPitch").value),
        eye_consec_frames: parseInt(document.getElementById("cfgEyeFrames").value),
        mouth_consec_frames: parseInt(document.getElementById("cfgMouthFrames").value),
        head_consec_frames: parseInt(document.getElementById("cfgHeadFrames").value),
    };
    await fetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
    });
    // 更新界面上的阈值显示
    document.getElementById("earThreshold").textContent = config.ear_threshold.toFixed(2);
    document.getElementById("marThreshold").textContent = config.mar_threshold.toFixed(2);
    document.getElementById("pitchThreshold").textContent = config.pitch_threshold.toFixed(1);

    // 同步规则说明卡片
    document.getElementById("ruleEar").textContent = `👁️ 闭眼：EAR < ${config.ear_threshold.toFixed(2)}（连续${config.eye_consec_frames}帧）`;
    document.getElementById("ruleMar").textContent = `👄 哈欠：MAR > ${config.mar_threshold.toFixed(2)}（连续${config.mouth_consec_frames}帧）`;
    document.getElementById("rulePitch").textContent = `🧑 低头：|Pitch| > ${config.pitch_threshold.toFixed(1)}°（连续${config.head_consec_frames}帧）`;
}

async function toggleYoloMode() {
    await toggleYolo(!isYoloEnabled);
    if (!isYoloEnabled) {
        displayMode = "rule";
        updateDisplayToggleButton();
    }
}

async function setDisplayMode(mode) {
    const resolvedMode = !isYoloEnabled && mode !== "rule" ? "rule" : mode;
    const res = await fetch("/api/display_mode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: resolvedMode }),
    });
    const data = await res.json();
    if (data.success) {
        displayMode = data.display_mode || "rule";
        updateDisplayToggleButton();
    }
}

async function cycleDisplayMode() {
    if (!isYoloEnabled) {
        displayMode = "rule";
        updateDisplayToggleButton();
        return;
    }

    const nextMode = displayMode === "rule"
        ? "yolo"
        : (displayMode === "yolo" ? "both" : "rule");
    await setDisplayMode(nextMode);
}

// ---- 数据轮询 ----

function startPolling() {
    stopPolling();
    polling = setInterval(fetchData, 150);
}

function stopPolling() {
    if (polling) {
        clearInterval(polling);
        polling = null;
    }
}

async function fetchData() {
    try {
        const res = await fetch("/api/data");
        const d = await res.json();
        updateUI(d);
    } catch (e) {
        // 忽略网络错误
    }
}

// ---- UI 更新 ----

function updateUI(d) {
    const ruleEarEl = document.getElementById("ruleEar");
    const ruleMarEl = document.getElementById("ruleMar");
    const rulePitchEl = document.getElementById("rulePitch");
    const modeNameMap = {
        rule: "规则模式",
        dl: "深度学习模式",
        hybrid: "混合模式",
    };
    // EAR
    const earEl = document.getElementById("earValue");
    earEl.textContent = d.ear.toFixed(2);
    const earBar = document.getElementById("earBar");
    const earPct = Math.min(d.ear / 0.4 * 100, 100);
    earBar.style.width = earPct + "%";
    earBar.className = "progress-fill ear-fill" + (d.eye_closed ? " danger" : (d.ear < 0.25 ? " warning" : ""));
    document.getElementById("eyeFrameCount").textContent = d.eye_frame_count;

    // MAR
    const marEl = document.getElementById("marValue");
    marEl.textContent = d.mar.toFixed(2);
    const marBar = document.getElementById("marBar");
    const marPct = Math.min(d.mar / 1.5 * 100, 100);
    marBar.style.width = marPct + "%";
    marBar.className = "progress-fill mar-fill" + (d.is_yawning ? " danger" : (d.mar > 0.5 ? " warning" : ""));
    document.getElementById("mouthFrameCount").textContent = d.mouth_frame_count;

    // 头部姿态
    document.getElementById("pitchValue").textContent = d.pitch.toFixed(1) + "°";
    document.getElementById("yawValue").textContent = d.yaw.toFixed(1) + "°";
    document.getElementById("rollValue").textContent = d.roll.toFixed(1) + "°";
    document.getElementById("headFrameCount").textContent = d.head_frame_count;

    // 规则阈值高亮（仅规则模式）
    ruleEarEl.classList.remove("active");
    ruleMarEl.classList.remove("active");
    rulePitchEl.classList.remove("active");
    if (d.mode === "rule") {
        if (d.eye_closed) ruleEarEl.classList.add("active");
        if (d.is_yawning) ruleMarEl.classList.add("active");
        if (d.is_head_down) rulePitchEl.classList.add("active");
    }

    // 状态
    const statusEl = document.getElementById("statusDisplay");
    statusEl.textContent = d.status;
    statusEl.className = "status-display";

    const currentDistractionAlerts = getDistractionAlerts(d);
    const distractionAlerts = currentDistractionAlerts.length > 0
        ? currentDistractionAlerts
        : (isDistractionWarningHeld() ? ["分心驾驶预警"] : []);
    const fatigueAlerts = distractionAlerts.length > 0 ? [] : getFatigueAlerts(d);

    if (distractionAlerts.length > 0) {
        statusEl.classList.add("danger");
        statusEl.textContent = "⚠️ 分心警告";
    } else if (fatigueAlerts.length > 0 || d.is_fatigued) {
        statusEl.classList.add("danger");
        statusEl.textContent = "⚠️ 疲劳警告";
    } else if (Array.isArray(d.dms_alerts)) {
        statusEl.classList.add("warning");
        statusEl.textContent = "🟢 YOLO-DMS监测中";
    } else if (d.eye_closed || d.is_yawning || d.is_head_down) {
        statusEl.classList.add("warning");
    }

    // 状态详情
    const detailEl = document.getElementById("statusDetail");
    const earThreshold = parseFloat(document.getElementById("earThreshold").textContent) || 0.20;
    const marThreshold = parseFloat(document.getElementById("marThreshold").textContent) || 0.75;
    const pitchThreshold = parseFloat(document.getElementById("pitchThreshold").textContent) || 25.0;

    if (distractionAlerts.length > 0) {
        detailEl.textContent = "分心警告: " + distractionAlerts.join("；");
    } else if (fatigueAlerts.length > 0) {
        detailEl.textContent = "疲劳警告: " + fatigueAlerts.join("；");
    } else if (Array.isArray(d.dms_alerts)) {
        detailEl.textContent = "未触发DMS预警，系统持续检测中";
    } else if (d.mode === "rule" && d.face_detected && (d.eye_closed || d.is_yawning || d.is_head_down)) {
        const reasons = [];
        if (d.eye_closed) {
            reasons.push(`闭眼: EAR=${d.ear.toFixed(2)} < ${earThreshold.toFixed(2)}`);
        }
        if (d.is_yawning) {
            reasons.push(`哈欠: MAR=${d.mar.toFixed(2)} > ${marThreshold.toFixed(2)}`);
        }
        if (d.is_head_down) {
            reasons.push(`低头: |Pitch|=${Math.abs(d.pitch).toFixed(1)}° > ${pitchThreshold.toFixed(1)}°`);
        }
        detailEl.textContent = "规则阈值触发: " + reasons.join("；");
    } else if (d.mode === "rule" && d.face_detected) {
        detailEl.textContent = `规则阈值: EAR < ${earThreshold.toFixed(2)}，MAR > ${marThreshold.toFixed(2)}，|Pitch| > ${pitchThreshold.toFixed(1)}°`;
    } else if (d.is_fatigued && d.reasons.length > 0) {
        detailEl.textContent = "触发原因: " + d.reasons.join(", ");
    } else if (!d.face_detected) {
        detailEl.textContent = "请确保面部在摄像头范围内";
    } else {
        detailEl.textContent = "";
    }

    const riskLevel = deriveRiskLevel({
        ...d,
        dms_alerts: distractionAlerts.length > 0 ? distractionAlerts : fatigueAlerts
    });
    const alertSummary = deriveAlertSummary({
        ...d,
        dms_alerts: distractionAlerts.length > 0 ? distractionAlerts : fatigueAlerts
    });

    document.getElementById("headerMonitorStatus").textContent = deriveMonitorState();
    document.getElementById("headerRiskLevel").textContent = riskLevel;
    document.getElementById("headerCameraStatus").textContent = deriveCameraState(d);
    document.getElementById("headerEngineInfo").textContent = d.is_dms_alert || Array.isArray(d.dms_alerts)
        ? "YOLO-DMS"
        : "YOLO-DMS";
    document.getElementById("headerStrategyInfo").textContent = "规则模式 + YOLO-DMS";
    document.getElementById("controlModeValue").textContent = modeNameMap[d.mode] || d.mode || "规则模式";
    document.getElementById("panelRiskLevel").textContent = riskLevel;
    document.getElementById("panelAlertSummary").textContent = alertSummary;
    updateScenarioPanel(d, riskLevel, alertSummary);

    // 分心驾驶预警覆盖层优先处理，避免与疲劳警告同时展示。
    updateDistractionOverlay(d);

    // 疲劳警告覆盖层
    const overlay = document.getElementById("fatigueOverlay");
    const phoneOverlay = document.getElementById("phoneAlertOverlay");
    const hasFatigueAlert = fatigueAlerts.length > 0 || d.is_fatigued;
    const hasActiveDistractionOverlay = phoneOverlay && phoneOverlay.classList.contains("active");
    const now = Date.now();
    if (distractionAlerts.length > 0 || hasActiveDistractionOverlay) {
        overlay.classList.remove("active");
        fatigueOverlayUntil = 0;
    } else if (hasFatigueAlert) {
        fatigueOverlayUntil = now + alertOverlayHoldMs;
        overlay.classList.add("active");
    } else if (now < fatigueOverlayUntil || isSpeechPlaying) {
        overlay.classList.add("active");
    } else {
        overlay.classList.remove("active");
    }

    speakAlertByData(d);
}

function resetUI() {
    document.getElementById("earValue").textContent = "0.00";
    document.getElementById("marValue").textContent = "0.00";
    document.getElementById("earBar").style.width = "0%";
    document.getElementById("marBar").style.width = "0%";
    document.getElementById("pitchValue").textContent = "0.0°";
    document.getElementById("yawValue").textContent = "0.0°";
    document.getElementById("rollValue").textContent = "0.0°";
    document.getElementById("eyeFrameCount").textContent = "0";
    document.getElementById("mouthFrameCount").textContent = "0";
    document.getElementById("headFrameCount").textContent = "0";
    document.getElementById("statusDisplay").textContent = "待启动";
    document.getElementById("statusDisplay").className = "status-display";
    document.getElementById("statusDetail").textContent = "";
    document.getElementById("headerMonitorStatus").textContent = "待启动";
    document.getElementById("headerRiskLevel").textContent = "低风险";
    document.getElementById("headerCameraStatus").textContent = "未启动";
    document.getElementById("headerEngineInfo").textContent = "YOLO-DMS";
    document.getElementById("headerStrategyInfo").textContent = "规则模式 + YOLO-DMS";
    document.getElementById("controlModeValue").textContent = "规则模式";
    document.getElementById("panelRiskLevel").textContent = "低风险";
    document.getElementById("panelAlertSummary").textContent = "暂无告警";
    document.getElementById("scenarioStatusMain").textContent = "正常驾驶";
    document.getElementById("scenarioRiskLevel").textContent = "低风险";
    document.getElementById("scenarioMonitorState").textContent = "待启动";
    document.getElementById("scenarioCameraState").textContent = "未启动";
    document.getElementById("scenarioAlertSummary").textContent = "暂无告警";
    document.getElementById("scenarioEyeState").textContent = "正常";
    document.getElementById("scenarioMouthState").textContent = "正常";
    document.getElementById("scenarioHeadState").textContent = "正常";
    document.getElementById("ruleEar").classList.remove("active");
    document.getElementById("ruleMar").classList.remove("active");
    document.getElementById("rulePitch").classList.remove("active");
    document.getElementById("fatigueOverlay").classList.remove("active");
    document.getElementById("phoneAlertOverlay").classList.remove("active");
    distractionOverlayUntil = 0;
    fatigueOverlayUntil = 0;
    resetSpeechAlerts();
    renderScenarioEvents();
    updateYoloToggleButton();
    updateDisplayToggleButton();
}

// ---- 系统日志 ----

let logSince = 0;
let logPollingTimer = null;

function updateCriticalLog(level, message) {
    if (level !== "warning" && level !== "danger") return;
    const id = level === "danger" ? "criticalDanger" : "criticalWarning";
    const target = document.getElementById(id);
    if (!target) return;
    target.textContent = message;
    target.classList.add("flash");
    setTimeout(() => target.classList.remove("flash"), 600);
}

function resetCriticalLogPanel() {
    const warningEl = document.getElementById("criticalWarning");
    const dangerEl = document.getElementById("criticalDanger");
    if (warningEl) warningEl.textContent = "暂无黄色预警";
    if (dangerEl) dangerEl.textContent = "暂无红色预警";
}

function startLogPoll() {
    stopLogPoll();
    logPollingTimer = setInterval(fetchLogs, 500);
}

function stopLogPoll() {
    if (logPollingTimer) {
        clearInterval(logPollingTimer);
        logPollingTimer = null;
    }
}

async function fetchLogs() {
    try {
        const res = await fetch(`/api/logs?since=${logSince}`);
        const data = await res.json();
        if (data.logs.length > 0) {
            const container = document.getElementById("logContainer");
            // 移除空提示
            const empty = container.querySelector(".log-empty");
            if (empty) empty.remove();

            data.logs.forEach(log => {
                const div = document.createElement("div");
                div.className = `log-entry ${log.level}`;
                div.innerHTML = `<span class="log-time">${log.time}</span><span class="log-msg">${log.message}</span>`;
                container.appendChild(div);
                updateCriticalLog(log.level, `[${log.time}] ${log.message}`);
                recentScenarioEvents.push({
                    time: log.time,
                    message: log.message,
                    level: log.level || "info",
                });
            });
            recentScenarioEvents = recentScenarioEvents.slice(-2);
            logSince = data.total;
            container.scrollTop = container.scrollHeight;
            renderScenarioEvents();
        }
    } catch (e) {}
}

function clearLogDisplay() {
    const container = document.getElementById("logContainer");
    container.innerHTML = '<div class="log-empty">日志已清空</div>';
    logSince = 0;
    recentScenarioEvents = [];
    resetCriticalLogPanel();
    renderScenarioEvents();
}

// ---- 脚本运行 ----

let currentScriptId = null;
let logPolling = null;

function promptDatasetPath(type) {
    const path = prompt(`请输入${type === 'eye' ? '眼部' : '嘴部'}数据集路径：\n例如: data/eyes 或 data/mouth`);
    if (!path) return null;
    return ["--dataset_path", path, "--output_dir", "models/trained"];
}

async function runScript(script, extraArgs) {
    if (extraArgs === null) return; // 用户取消了输入

    const args = extraArgs || [];
    const res = await fetch("/api/run_script", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ script, args }),
    });
    const data = await res.json();

    if (data.success) {
        currentScriptId = data.script_id;
        document.getElementById("scriptLogContainer").style.display = "block";
        document.getElementById("scriptLog").textContent = "启动中...\n";
        startLogPolling();
    } else {
        alert(data.message);
    }
}

function startLogPolling() {
    stopLogPolling();
    logPolling = setInterval(async () => {
        if (!currentScriptId) return;
        try {
            const res = await fetch(`/api/script_log/${currentScriptId}`);
            const data = await res.json();
            const logEl = document.getElementById("scriptLog");
            logEl.textContent = data.log || "等待输出...\n";
            logEl.scrollTop = logEl.scrollHeight;
            if (!data.is_running) {
                stopLogPolling();
            }
        } catch (e) {}
    }, 500);
}

function stopLogPolling() {
    if (logPolling) {
        clearInterval(logPolling);
        logPolling = null;
    }
}

async function toggleYolo(enabled) {
    try {
        const res = await fetch("/api/yolo", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ enabled }),
        });
        const data = await res.json();
        if (data.success) {
            isYoloEnabled = enabled;
            updateYoloToggleButton();
        }
    } catch (e) {
        console.error("YOLO 切换失败，请检查后端日志");
    }
}

function closeLog() {
    stopLogPolling();
    document.getElementById("scriptLogContainer").style.display = "none";
    currentScriptId = null;
}

setViewMode("experiment");
renderScenarioEvents();
updateYoloToggleButton();
updateDisplayToggleButton();
