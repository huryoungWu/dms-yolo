let polling = null;

async function start() {
    const res = await fetch("/api/start", { method: "POST" });
    const data = await res.json();
    
    if (data.success) {
        document.getElementById("btnStart").disabled = true;
        document.getElementById("btnStop").disabled = false;
        document.getElementById("placeholder").classList.add("hidden");
        
        const img = document.getElementById("videoFeed");
        img.src = "/video_feed?" + Date.now();
        img.classList.add("active");
        
        startPolling();
    } else {
        alert("无法打开摄像头");
    }
}

async function stop() {
    await fetch("/api/stop", { method: "POST" });
    
    document.getElementById("btnStart").disabled = false;
    document.getElementById("btnStop").disabled = true;
    document.getElementById("videoFeed").classList.remove("active");
    document.getElementById("videoFeed").src = "";
    document.getElementById("placeholder").classList.remove("hidden");
    document.getElementById("alertOverlay").classList.remove("active");
    
    stopPolling();
    resetUI();
}

function startPolling() {
    stopPolling();
    polling = setInterval(fetchData, 200);
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
    } catch (e) {}
}

function updateUI(d) {
    // EAR
    document.getElementById("earValue").textContent = d.ear.toFixed(2);
    const earBar = document.getElementById("earBar");
    const earPct = Math.min(d.ear / 0.4 * 100, 100);
    earBar.style.width = earPct + "%";
    earBar.className = "bar-fill" + (d.ear < 0.15 ? " danger" : (d.ear < 0.22 ? " warning" : ""));
    document.getElementById("eyeFrames").textContent = d.eye_frames;

    // MAR
    document.getElementById("marValue").textContent = d.mar.toFixed(2);
    const marBar = document.getElementById("marBar");
    const marPct = Math.min(d.mar / 1.5 * 100, 100);
    marBar.style.width = marPct + "%";
    marBar.className = "bar-fill" + (d.mar > 0.9 ? " danger" : (d.mar > 0.6 ? " warning" : ""));
    document.getElementById("yawnFrames").textContent = d.yawn_frames;

    // 状态
    const statusEl = document.getElementById("statusValue");
    statusEl.textContent = d.status;
    statusEl.className = "status-value";
    
    if (d.is_fatigued) {
        statusEl.classList.add("danger");
        document.getElementById("alertOverlay").classList.add("active");
    } else {
        document.getElementById("alertOverlay").classList.remove("active");
        if (!d.face_detected) {
            statusEl.classList.add("warning");
        }
    }
}

function resetUI() {
    document.getElementById("earValue").textContent = "0.00";
    document.getElementById("marValue").textContent = "0.00";
    document.getElementById("earBar").style.width = "0%";
    document.getElementById("marBar").style.width = "0%";
    document.getElementById("eyeFrames").textContent = "0";
    document.getElementById("yawnFrames").textContent = "0";
    document.getElementById("statusValue").textContent = "待启动";
    document.getElementById("statusValue").className = "status-value";
}
