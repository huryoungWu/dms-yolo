// 疲劳驾驶检测系统 - 前端控制脚本

let polling = null;
let isRunning = false;

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

    if (d.is_dms_alert && Array.isArray(d.dms_alerts) && d.dms_alerts.length > 0) {
        statusEl.classList.add("danger");
        statusEl.textContent = "⚠️ DMS预警";
    } else if (Array.isArray(d.dms_alerts)) {
        statusEl.classList.add("warning");
        statusEl.textContent = "🟢 YOLO-DMS监测中";
    } else if (d.is_fatigued) {
        statusEl.classList.add("danger");
        statusEl.textContent = "⚠️ 疲劳驾驶";
    } else if (d.eye_closed || d.is_yawning || d.is_head_down) {
        statusEl.classList.add("warning");
    }

    // 状态详情
    const detailEl = document.getElementById("statusDetail");
    const earThreshold = parseFloat(document.getElementById("earThreshold").textContent) || 0.20;
    const marThreshold = parseFloat(document.getElementById("marThreshold").textContent) || 0.75;
    const pitchThreshold = parseFloat(document.getElementById("pitchThreshold").textContent) || 25.0;

    if (d.is_dms_alert && Array.isArray(d.dms_alerts) && d.dms_alerts.length > 0) {
        detailEl.textContent = "DMS告警: " + d.dms_alerts.join("；");
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

    // 疲劳警告覆盖层
    const overlay = document.getElementById("fatigueOverlay");
    if (d.is_fatigued) {
        overlay.classList.add("active");
    } else {
        overlay.classList.remove("active");
    }
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
    document.getElementById("ruleEar").classList.remove("active");
    document.getElementById("ruleMar").classList.remove("active");
    document.getElementById("rulePitch").classList.remove("active");
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
    resetCriticalLogPanel();
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
        alert(data.message || (data.success ? "操作成功" : "操作失败"));
    } catch (e) {
        alert("YOLO 切换失败，请检查后端日志");
    }
}

function closeLog() {
    stopLogPolling();
    document.getElementById("scriptLogContainer").style.display = "none";
    currentScriptId = null;
}
