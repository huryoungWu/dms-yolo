// 疲劳驾驶检测系统 - 前端控制脚本

let polling = null;
let isRunning = false;
let clockTimer = null;
let viewMode = "experiment";
let isYoloEnabled = true;
let displayMode = "both";
const alarmSpeechGapMs = 500;
const alarmSpeechFallbackMs = 2200;
const alarmOverlayClearDelayMs = 450;
const fatigueOverlayMaxMs = 3000;
let alarmIsPlaying = false;
let alarmNextAllowedAt = 0;
let alarmFallbackTimer = null;
let alarmOverlayHideTimer = null;
let alarmOverlayShownAt = 0;
let alarmOverlayKey = null;
let suppressedFatigueOverlayKey = null;
let currentAlarmUtterance = null;
let currentAlarm = null;
let alarmSpeechPrimed = false;

const alarmDefinitions = [
    {
        key: "yawn",
        type: "fatigue",
        behavior: "打哈欠",
        aliases: ["打哈欠", "哈欠", "yawn", "yawning"],
        isActive: d => Boolean(d.is_yawning),
        speech: "打哈欠，请注意休息",
    },
    {
        key: "eye_closed",
        type: "fatigue",
        behavior: "闭眼",
        aliases: ["闭眼", "closed eye", "closed_eye"],
        isActive: d => Boolean(d.eye_closed),
        speech: "闭眼，请保持清醒",
    },
    {
        key: "head_down",
        type: "fatigue",
        behavior: "低头",
        aliases: ["低头", "head down", "head_down"],
        isActive: d => Boolean(d.is_head_down),
        speech: "低头，请看前方",
    },
    {
        key: "smoking",
        type: "distraction",
        behavior: "抽烟",
        aliases: ["抽烟", "吸烟", "香烟", "烟", "电子烟", "cigarette", "cigarettes", "cigar", "smoke", "smokes", "smoking", "smoker", "tobacco", "vape", "e cigarette"],
        isActive: () => false,
        speech: "请停止抽烟",
    },
    {
        key: "phone_use",
        group: "phone",
        type: "distraction",
        behavior: "玩手机",
        aliases: ["玩手机", "手机", "手机使用", "使用手机", "phone use", "using phone", "mobile phone", "mobilephone", "mobile", "cell phone", "cellphone", "phone", "phones", "smartphone", "handphone"],
        isActive: () => false,
        speech: "玩手机，请放下手机",
    },
    {
        key: "phone_call",
        group: "phone",
        type: "distraction",
        behavior: "打电话",
        aliases: ["打电话", "电话", "phone call", "calling", "phone"],
        isActive: () => false,
        speech: "打电话，请停止打电话",
    },
    {
        key: "drinking",
        type: "distraction",
        behavior: "喝水",
        aliases: ["喝水", "水杯", "杯子", "水瓶", "water cup", "watercup", "cup", "water bottle", "bottle", "drinking", "drink"],
        isActive: () => false,
        speech: "请放下水杯",
    },
    {
        key: "look_away",
        type: "distraction",
        behavior: "左顾右盼",
        aliases: ["左顾右盼", "视线偏移", "look away", "looking_away", "distracted"],
        isActive: () => false,
        speech: "左顾右盼，请看前方",
    },
];

const alarmPriority = {
    smoking: 40,
    phone_call: 35,
    phone_use: 35,
    drinking: 30,
    look_away: 25,
    head_down: 15,
    eye_closed: 15,
    yawn: 10,
};

function normalizeAlertText(text) {
    return String(text || "").trim().toLowerCase();
}

function matchAlarmDefinition(text) {
    const normalizedText = normalizeAlertText(text);
    return alarmDefinitions.find(def =>
        def.aliases.some(alias => normalizedText.includes(String(alias).toLowerCase()))
    );
}

function uniqueAlarmCandidates(candidates) {
    const seen = new Set();
    return candidates.filter(candidate => {
        if (!candidate || seen.has(candidate.key)) return false;
        seen.add(candidate.key);
        return true;
    });
}

function getAlarmOverlayKey(alarm) {
    if (!alarm) return "";
    return `${alarm.type}:${alarm.key}`;
}

function getAlarmSpeechGroup(alarm) {
    if (!alarm) return "";
    return alarm.group || alarm.key || "";
}

function resolveAlarmCandidate(d) {
    const alerts = Array.isArray(d.dms_alerts) ? d.dms_alerts : [];
    const fromAlerts = uniqueAlarmCandidates(alerts.map(matchAlarmDefinition));
    if (fromAlerts.length > 0) {
        return fromAlerts.sort((a, b) => (alarmPriority[b.key] || 0) - (alarmPriority[a.key] || 0))[0];
    }

    const fallbackFatigue = alarmDefinitions.filter(def => def.type === "fatigue" && def.isActive(d));
    const candidates = uniqueAlarmCandidates(fallbackFatigue);
    if (candidates.length === 0) return null;
    return candidates.sort((a, b) => (alarmPriority[b.key] || 0) - (alarmPriority[a.key] || 0))[0];
}

function showAlarmOverlay(alarm) {
    const overlay = document.getElementById("alarmOverlay");
    const panel = document.getElementById("alarmPanel");
    const label = document.getElementById("alarmLabel");
    if (!overlay || !panel || !label) return;

    if (alarmOverlayHideTimer) {
        clearTimeout(alarmOverlayHideTimer);
        alarmOverlayHideTimer = null;
    }

    const nextOverlayKey = getAlarmOverlayKey(alarm);
    if (alarmOverlayKey !== nextOverlayKey || !overlay.classList.contains("active")) {
        alarmOverlayShownAt = Date.now();
        alarmOverlayKey = nextOverlayKey;
    }

    const isFatigue = alarm.type === "fatigue";
    label.textContent = isFatigue ? "疲劳警告" : "分心警告";
    panel.classList.toggle("fatigue", isFatigue);
    panel.classList.toggle("distraction", !isFatigue);
    overlay.classList.add("active");
    overlay.setAttribute("aria-hidden", "false");
}

function hideAlarmOverlay() {
    const overlay = document.getElementById("alarmOverlay");
    if (!overlay) return;
    if (alarmOverlayHideTimer) {
        clearTimeout(alarmOverlayHideTimer);
        alarmOverlayHideTimer = null;
    }
    overlay.classList.remove("active");
    overlay.setAttribute("aria-hidden", "true");
    alarmOverlayShownAt = 0;
    alarmOverlayKey = null;
}

function scheduleAlarmOverlayHide() {
    if (alarmOverlayHideTimer) return;
    alarmOverlayHideTimer = setTimeout(() => {
        alarmOverlayHideTimer = null;
        hideAlarmOverlay();
    }, alarmOverlayClearDelayMs);
}

function finishAlarmPlayback() {
    if (alarmFallbackTimer) {
        clearTimeout(alarmFallbackTimer);
        alarmFallbackTimer = null;
    }
    if (currentAlarmUtterance) {
        currentAlarmUtterance.onend = null;
        currentAlarmUtterance.onerror = null;
        currentAlarmUtterance = null;
    }
    alarmIsPlaying = false;
    alarmNextAllowedAt = Date.now() + alarmSpeechGapMs;
    currentAlarm = null;
}

function cancelAlarmPlayback() {
    if (alarmFallbackTimer) {
        clearTimeout(alarmFallbackTimer);
        alarmFallbackTimer = null;
    }
    if (currentAlarmUtterance) {
        currentAlarmUtterance.onend = null;
        currentAlarmUtterance.onerror = null;
        currentAlarmUtterance = null;
    }
    alarmIsPlaying = false;
    alarmNextAllowedAt = 0;
    currentAlarm = null;
    hideAlarmOverlay();
    if ("speechSynthesis" in window) {
        window.speechSynthesis.cancel();
    }
}

function getPreferredChineseVoice() {
    if (!("speechSynthesis" in window)) return null;
    const voices = window.speechSynthesis.getVoices();
    return voices.find(voice => /zh|chinese|mandarin/i.test(`${voice.lang} ${voice.name}`)) || null;
}

function primeAlarmSpeech(force = false) {
    if (!("speechSynthesis" in window) || typeof SpeechSynthesisUtterance === "undefined") return;
    if (alarmSpeechPrimed && !force) return;
    try {
        const synth = window.speechSynthesis;
        synth.getVoices();
        synth.resume();

        const unlockUtterance = new SpeechSynthesisUtterance("。");
        unlockUtterance.lang = "zh-CN";
        unlockUtterance.volume = 0;
        unlockUtterance.rate = 1;
        unlockUtterance.onend = () => {
            alarmSpeechPrimed = true;
        };
        unlockUtterance.onerror = () => {
            alarmSpeechPrimed = true;
        };
        synth.speak(unlockUtterance);
        alarmSpeechPrimed = true;
    } catch (error) {
        console.warn("语音播报初始化失败", error);
    }
}

function startAlarmPlayback(alarm, showOverlayPanel = true) {
    if (alarmFallbackTimer) {
        clearTimeout(alarmFallbackTimer);
        alarmFallbackTimer = null;
    }
    if (currentAlarmUtterance) {
        currentAlarmUtterance.onend = null;
        currentAlarmUtterance.onerror = null;
        currentAlarmUtterance = null;
    }
    if ("speechSynthesis" in window) {
        window.speechSynthesis.cancel();
    }
    alarmIsPlaying = true;
    currentAlarm = alarm;
    if (showOverlayPanel) {
        showAlarmOverlay(alarm);
    }

    if (!("speechSynthesis" in window) || typeof SpeechSynthesisUtterance === "undefined") {
        alarmFallbackTimer = setTimeout(finishAlarmPlayback, 1800);
        return;
    }

    try {
        const synth = window.speechSynthesis;
        synth.resume();

        const utterance = new SpeechSynthesisUtterance(alarm.speech);
        const preferredVoice = getPreferredChineseVoice();
        utterance.lang = "zh-CN";
        utterance.rate = 1.15;
        utterance.pitch = 1;
        utterance.volume = 1;
        if (preferredVoice) {
            utterance.voice = preferredVoice;
        }
        utterance.onend = finishAlarmPlayback;
        utterance.onerror = finishAlarmPlayback;
        currentAlarmUtterance = utterance;
        alarmFallbackTimer = setTimeout(finishAlarmPlayback, alarmSpeechFallbackMs);
        synth.speak(utterance);
        if (synth.paused) {
            synth.resume();
        }
    } catch (error) {
        console.error("语音播报失败", error);
        finishAlarmPlayback();
    }
}

function updateAlarmPlayback(d) {
    const candidate = isRunning ? resolveAlarmCandidate(d) : null;

    if (!candidate) {
        suppressedFatigueOverlayKey = null;
        scheduleAlarmOverlayHide();
        return;
    }

    const candidateOverlayKey = getAlarmOverlayKey(candidate);
    const fatigueOverlayExpired = (
        candidate.type === "fatigue" &&
        alarmOverlayKey === candidateOverlayKey &&
        alarmOverlayShownAt > 0 &&
        Date.now() - alarmOverlayShownAt >= fatigueOverlayMaxMs
    );

    if (fatigueOverlayExpired) {
        suppressedFatigueOverlayKey = candidateOverlayKey;
        hideAlarmOverlay();
    }

    const shouldShowOverlay = !(
        candidate.type === "fatigue" &&
        suppressedFatigueOverlayKey === candidateOverlayKey
    );

    if (shouldShowOverlay) {
        showAlarmOverlay(candidate);
    }

    if (alarmIsPlaying) {
        const currentPriority = currentAlarm ? (alarmPriority[currentAlarm.key] || 0) : 0;
        const nextPriority = alarmPriority[candidate.key] || 0;
        const sameSpeechGroup = getAlarmSpeechGroup(candidate) === getAlarmSpeechGroup(currentAlarm);
        const canInterrupt = candidate.type === "distraction" || nextPriority > currentPriority;
        if (candidate.key !== currentAlarm?.key && canInterrupt && !sameSpeechGroup) {
            startAlarmPlayback(candidate, shouldShowOverlay);
        }
        return;
    }
    if (Date.now() < alarmNextAllowedAt && candidate.type !== "distraction") return;

    startAlarmPlayback(candidate, shouldShowOverlay);
}

function getDistractionAlerts(d) {
    const alerts = Array.isArray(d.dms_alerts) ? d.dms_alerts : [];
    return alerts.filter(alert => {
        const alarm = matchAlarmDefinition(alert);
        return alarm && alarm.type === "distraction";
    });
}

function getFatigueAlerts(d) {
    const alerts = Array.isArray(d.dms_alerts) ? d.dms_alerts : [];
    const fatigueAlerts = alerts.filter(alert => {
        const alarm = matchAlarmDefinition(alert);
        return alarm && alarm.type === "fatigue";
    });

    if (fatigueAlerts.length > 0) {
        return fatigueAlerts;
    }

    const fallbackAlerts = [];
    if (d.eye_closed) fallbackAlerts.push("闭眼状态");
    if (d.is_yawning) fallbackAlerts.push("打哈欠状态");
    if (d.is_head_down) fallbackAlerts.push("低头状态");
    if (d.is_fatigued && Array.isArray(d.reasons)) {
        d.reasons.forEach(reason => fallbackAlerts.push(`${reason}状态`));
    }
    return fallbackAlerts;
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

function deriveMonitorState() {
    return isRunning ? "监测中" : "待启动";
}

function deriveCameraState(d) {
    if (!isRunning) return "未启动";
    return d && d.face_detected ? "在线" : "在线";
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

function updateScenarioPanel(d, riskLevel) {
    const scenarioStatusMain = document.getElementById("scenarioStatusMain");
    const scenarioRiskLevel = document.getElementById("scenarioRiskLevel");
    const scenarioMonitorState = document.getElementById("scenarioMonitorState");
    const scenarioCameraState = document.getElementById("scenarioCameraState");
    const scenarioEyeState = document.getElementById("scenarioEyeState");
    const scenarioMouthState = document.getElementById("scenarioMouthState");
    const scenarioHeadState = document.getElementById("scenarioHeadState");

    if (scenarioStatusMain) {
        if (riskLevel === "高风险") {
            scenarioStatusMain.textContent = "高风险状态";
        } else if (riskLevel === "注意") {
            scenarioStatusMain.textContent = "注意风险";
        } else {
            scenarioStatusMain.textContent = "正常驾驶";
        }
    }

    if (scenarioRiskLevel) scenarioRiskLevel.textContent = riskLevel;
    if (scenarioMonitorState) scenarioMonitorState.textContent = deriveMonitorState();
    if (scenarioCameraState) scenarioCameraState.textContent = deriveCameraState(d);
    if (scenarioEyeState) scenarioEyeState.textContent = d.eye_closed ? "闭眼风险" : "正常";
    if (scenarioMouthState) scenarioMouthState.textContent = d.is_yawning ? "哈欠风险" : "正常";
    if (scenarioHeadState) scenarioHeadState.textContent = d.is_head_down ? "低头风险" : "正常";
}

// ---- 控制 ----

async function startDetection() {
    cancelAlarmPlayback();
    primeAlarmSpeech(true);
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
    cancelAlarmPlayback();
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
    document.getElementById("rulePitch").textContent = `🧑 低头：Pitch > ${config.pitch_threshold.toFixed(1)}°（连续${config.head_consec_frames}帧）`;
}

async function toggleYoloMode() {
    const enabled = await toggleYolo(!isYoloEnabled);
    if (enabled) {
        await setDisplayMode("both");
    } else {
        await setDisplayMode("rule");
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
    if (typeof d.yolo_enabled === "boolean") {
        isYoloEnabled = d.yolo_enabled;
        updateYoloToggleButton();
    }
    if (d.display_mode) {
        displayMode = d.display_mode;
        updateDisplayToggleButton();
    }

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
    const distractionAlerts = currentDistractionAlerts;
    const fatigueAlerts = distractionAlerts.length > 0 ? [] : getFatigueAlerts(d);

    if (distractionAlerts.length > 0) {
        statusEl.classList.add("danger");
        statusEl.textContent = "分心警告";
    } else if (fatigueAlerts.length > 0 || d.is_fatigued) {
        statusEl.classList.add("danger");
        statusEl.textContent = "疲劳警告";
    } else if (Array.isArray(d.dms_alerts)) {
        statusEl.classList.add("warning");
        statusEl.textContent = "🟢 YOLO-DMS监测中";
    }

    // 状态详情
    const detailEl = document.getElementById("statusDetail");
    const earThreshold = parseFloat(document.getElementById("earThreshold").textContent) || 0.20;
    const marThreshold = parseFloat(document.getElementById("marThreshold").textContent) || 0.75;
    const pitchThreshold = parseFloat(document.getElementById("pitchThreshold").textContent) || 20.0;

    if (distractionAlerts.length > 0) {
        detailEl.textContent = "分心警告: " + distractionAlerts.join("；");
    } else if (fatigueAlerts.length > 0) {
        detailEl.textContent = "疲劳警告: " + fatigueAlerts.join("；");
    } else if (Array.isArray(d.dms_alerts)) {
        detailEl.textContent = "DMS持续检测中";
    } else if (d.mode === "rule" && d.face_detected && (d.eye_closed || d.is_yawning || d.is_head_down)) {
        const reasons = [];
        if (d.eye_closed) {
            reasons.push(`闭眼: EAR=${d.ear.toFixed(2)} < ${earThreshold.toFixed(2)}`);
        }
        if (d.is_yawning) {
            reasons.push(`哈欠: MAR=${d.mar.toFixed(2)} > ${marThreshold.toFixed(2)}`);
        }
        if (d.is_head_down) {
            reasons.push(`低头: Pitch=${d.pitch.toFixed(1)}° > ${pitchThreshold.toFixed(1)}°`);
        }
        detailEl.textContent = "规则阈值触发: " + reasons.join("；");
    } else if (d.mode === "rule" && d.face_detected) {
        detailEl.textContent = `规则阈值: EAR < ${earThreshold.toFixed(2)}，MAR > ${marThreshold.toFixed(2)}，Pitch > ${pitchThreshold.toFixed(1)}°`;
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

    document.getElementById("headerMonitorStatus").textContent = deriveMonitorState();
    document.getElementById("headerRiskLevel").textContent = riskLevel;
    document.getElementById("headerCameraStatus").textContent = deriveCameraState(d);
    document.getElementById("headerEngineInfo").textContent = d.is_dms_alert || Array.isArray(d.dms_alerts)
        ? "YOLO-DMS"
        : "YOLO-DMS";
    document.getElementById("headerStrategyInfo").textContent = "规则模式 + YOLO-DMS";
    document.getElementById("controlModeValue").textContent = modeNameMap[d.mode] || d.mode || "规则模式";
    document.getElementById("panelRiskLevel").textContent = riskLevel;
    updateScenarioPanel(d, riskLevel);
    updateAlarmPlayback(d);
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
    document.getElementById("scenarioStatusMain").textContent = "正常驾驶";
    document.getElementById("scenarioRiskLevel").textContent = "低风险";
    document.getElementById("scenarioMonitorState").textContent = "待启动";
    document.getElementById("scenarioCameraState").textContent = "未启动";
    document.getElementById("scenarioEyeState").textContent = "正常";
    document.getElementById("scenarioMouthState").textContent = "正常";
    document.getElementById("scenarioHeadState").textContent = "正常";
    document.getElementById("ruleEar").classList.remove("active");
    document.getElementById("ruleMar").classList.remove("active");
    document.getElementById("rulePitch").classList.remove("active");
    cancelAlarmPlayback();
    updateYoloToggleButton();
    updateDisplayToggleButton();
}

// ---- 系统日志 ----

let logSince = 0;
let logPollingTimer = null;

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
            });
            logSince = data.total;
            container.scrollTop = container.scrollHeight;
        }
    } catch (e) {}
}

function clearLogDisplay() {
    const container = document.getElementById("logContainer");
    container.innerHTML = '<div class="log-empty">日志已清空</div>';
    logSince = 0;
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
            isYoloEnabled = Boolean(data.enabled);
            updateYoloToggleButton();
            return isYoloEnabled;
        }
    } catch (e) {
        console.error("YOLO 切换失败，请检查后端日志");
    }
    updateYoloToggleButton();
    return isYoloEnabled;
}

function closeLog() {
    stopLogPolling();
    document.getElementById("scriptLogContainer").style.display = "none";
    currentScriptId = null;
}

setViewMode("experiment");
updateYoloToggleButton();
updateDisplayToggleButton();
