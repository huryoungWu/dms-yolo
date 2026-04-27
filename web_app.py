"""Flask Web 前端 - 疲劳驾驶检测系统"""

import json
import os
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from detectors.face_detector import FaceDetector
from detectors.eye_analyzer import EyeAnalyzer
from detectors.mouth_analyzer import MouthAnalyzer
from detectors.head_pose_analyzer import HeadPoseAnalyzer
from evaluators.fatigue_evaluator import FatigueEvaluator
from display.renderer import DisplayRenderer
from models.data_models import EyeResult, MouthResult, PoseResult

app = Flask(__name__, template_folder="web/templates", static_folder="web/static")

# 默认阈值
_DEFAULTS = {
    "ear_threshold": 0.2,
    "mar_threshold": 0.75,
    "pitch_threshold": 25.0,
    "eye_consec_frames": 48,
    "mouth_consec_frames": 25,
    "head_consec_frames": 50,
}


class WebDetectionSystem:
    """Web 版检测系统，支持 MJPEG 视频流推送和实时数据 API。"""

    MAX_LOG_ENTRIES = 200

    def __init__(self):
        self._cap = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_data = {
            "ear": 0.0, "mar": 0.0,
            "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
            "status": "正常", "is_fatigued": False,
            "reasons": [], "mode": "rule",
            "face_detected": False,
        }
        self.mode = "rule"
        self._logs = []
        self._log_lock = threading.Lock()
        self._prev_state = {"eye_closed": False, "is_yawning": False, "is_head_down": False, "is_fatigued": False, "face_detected": True}
        self._init_modules(_DEFAULTS)

    def _init_modules(self, config):
        self.face_detector = FaceDetector()
        self.eye_analyzer = EyeAnalyzer(
            ear_threshold=config["ear_threshold"],
            consec_frames=config["eye_consec_frames"],
        )
        self.mouth_analyzer = MouthAnalyzer(
            mar_threshold=config["mar_threshold"],
            consec_frames=config["mouth_consec_frames"],
        )
        self.head_pose_analyzer = HeadPoseAnalyzer(
            pitch_threshold=config["pitch_threshold"],
            consec_frames=config["head_consec_frames"],
        )
        self.fatigue_evaluator = FatigueEvaluator()
        self.renderer = DisplayRenderer()

    def start(self):
        """启动摄像头和处理线程。"""
        if self._running:
            return True
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            self._add_log("danger", "无法打开摄像头")
            return False
        self._running = True
        self._add_log("info", "系统启动，摄像头已开启")
        mode_names = {"rule": "规则模式", "dl": "深度学习模式", "hybrid": "混合模式"}
        self._add_log("info", f"当前模式: {mode_names.get(self.mode, self.mode)}")
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        """停止检测。"""
        self._running = False
        time.sleep(0.3)
        if self._cap and self._cap.isOpened():
            self._cap.release()
        self._cap = None
        self.eye_analyzer.reset()
        self.mouth_analyzer.reset()
        self.head_pose_analyzer.reset()
        self._add_log("info", "系统已停止")

    def _process_loop(self):
        """后台处理循环。"""
        status_map = {
            "normal": "正常", "eye_closed": "闭眼",
            "yawning": "打哈欠", "head_down": "低头",
        }
        while self._running:
            if not self._cap or not self._cap.isOpened():
                break
            ret, frame = self._cap.read()
            if not ret:
                continue

            landmarks = self.face_detector.detect(frame)

            if landmarks is not None:
                eye_result = self.eye_analyzer.analyze(landmarks.left_eye, landmarks.right_eye)
                mouth_result = self.mouth_analyzer.analyze(landmarks.mouth)
                pose_result = self.head_pose_analyzer.estimate_pose(
                    landmarks.head_pose_points, frame.shape
                )
                dl_result = None
                fatigue_status = self.fatigue_evaluator.evaluate(
                    eye_result, mouth_result, pose_result,
                    dl_result=dl_result, mode=self.mode,
                )
                status_key = DisplayRenderer._determine_status(eye_result, mouth_result, pose_result)
                rendered = self.renderer.render(
                    frame, landmarks, eye_result, mouth_result,
                    pose_result, fatigue_status, dl_result=dl_result,
                )
                with self._lock:
                    self._latest_data = {
                        "ear": round(eye_result.ear, 4),
                        "mar": round(mouth_result.mar, 4),
                        "pitch": round(pose_result.pitch, 2),
                        "yaw": round(pose_result.yaw, 2),
                        "roll": round(pose_result.roll, 2),
                        "status": status_map.get(status_key, "正常"),
                        "is_fatigued": fatigue_status.is_fatigued,
                        "reasons": fatigue_status.reasons,
                        "mode": self.mode,
                        "face_detected": True,
                        "eye_closed": eye_result.is_closed,
                        "is_yawning": mouth_result.is_yawning,
                        "is_head_down": pose_result.is_head_down,
                        "eye_frame_count": eye_result.frame_count,
                        "mouth_frame_count": mouth_result.frame_count,
                        "head_frame_count": pose_result.frame_count,
                    }
            else:
                eye_result = EyeResult(ear=0.0, is_closed=False, is_fatigued=False, frame_count=0)
                mouth_result = MouthResult(mar=0.0, is_yawning=False, is_fatigued=False, frame_count=0)
                pose_result = PoseResult(pitch=0.0, yaw=0.0, roll=0.0, is_head_down=False, is_fatigued=False, frame_count=0)
                fatigue_status = self.fatigue_evaluator.evaluate(
                    eye_result, mouth_result, pose_result, mode=self.mode,
                )
                rendered = self.renderer.render(
                    frame, None, eye_result, mouth_result,
                    pose_result, fatigue_status,
                )
                with self._lock:
                    self._latest_data = {
                        "ear": 0.0, "mar": 0.0,
                        "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
                        "status": "未检测到人脸", "is_fatigued": False,
                        "reasons": [], "mode": self.mode,
                        "face_detected": False,
                        "eye_closed": False, "is_yawning": False,
                        "is_head_down": False,
                        "eye_frame_count": 0, "mouth_frame_count": 0,
                        "head_frame_count": 0,
                    }

            _, jpeg = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with self._lock:
                self._latest_frame = jpeg.tobytes()

            # 检测状态变化并记录日志
            with self._lock:
                current_data = dict(self._latest_data)
            self._check_state_changes(current_data)

    def _add_log(self, level, message):
        """添加一条系统日志。level: info / warning / danger"""
        import datetime
        entry = {
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "level": level,
            "message": message,
        }
        with self._log_lock:
            self._logs.append(entry)
            if len(self._logs) > self.MAX_LOG_ENTRIES:
                self._logs = self._logs[-self.MAX_LOG_ENTRIES:]

    def _check_state_changes(self, data):
        """检测状态变化并记录日志。"""
        prev = self._prev_state

        if data.get("face_detected") and not prev.get("face_detected"):
            self._add_log("info", "检测到人脸")
        elif not data.get("face_detected") and prev.get("face_detected"):
            self._add_log("warning", "人脸丢失")

        if data.get("eye_closed") and not prev.get("eye_closed"):
            self._add_log("warning", f"闭眼检测中 (EAR={data.get('ear', 0):.2f})")
        elif not data.get("eye_closed") and prev.get("eye_closed"):
            self._add_log("info", "睁眼恢复")

        if data.get("is_yawning") and not prev.get("is_yawning"):
            self._add_log("warning", f"打哈欠检测中 (MAR={data.get('mar', 0):.2f})")
        elif not data.get("is_yawning") and prev.get("is_yawning"):
            self._add_log("info", "哈欠结束")

        if data.get("is_head_down") and not prev.get("is_head_down"):
            self._add_log("warning", f"低头检测中 (俯仰角={data.get('pitch', 0):.1f}°)")
        elif not data.get("is_head_down") and prev.get("is_head_down"):
            self._add_log("info", "抬头恢复")

        if data.get("is_fatigued") and not prev.get("is_fatigued"):
            reasons = ", ".join(data.get("reasons", []))
            self._add_log("danger", f"⚠️ 疲劳驾驶警告！原因: {reasons}")
        elif not data.get("is_fatigued") and prev.get("is_fatigued"):
            self._add_log("info", "疲劳状态解除")

        self._prev_state = {
            "eye_closed": data.get("eye_closed", False),
            "is_yawning": data.get("is_yawning", False),
            "is_head_down": data.get("is_head_down", False),
            "is_fatigued": data.get("is_fatigued", False),
            "face_detected": data.get("face_detected", True),
        }

    def get_logs(self, since=0):
        """获取日志，since 为起始索引。"""
        with self._log_lock:
            return self._logs[since:], len(self._logs)

    def get_frame(self):
        with self._lock:
            return self._latest_frame

    def get_data(self):
        with self._lock:
            return dict(self._latest_data)

    def update_config(self, config):
        """动态更新阈值配置。"""
        self.eye_analyzer.ear_threshold = config.get("ear_threshold", self.eye_analyzer.ear_threshold)
        self.eye_analyzer.consec_frames = config.get("eye_consec_frames", self.eye_analyzer.consec_frames)
        self.mouth_analyzer.mar_threshold = config.get("mar_threshold", self.mouth_analyzer.mar_threshold)
        self.mouth_analyzer.consec_frames = config.get("mouth_consec_frames", self.mouth_analyzer.consec_frames)
        self.head_pose_analyzer.pitch_threshold = config.get("pitch_threshold", self.head_pose_analyzer.pitch_threshold)
        self.head_pose_analyzer.consec_frames = config.get("head_consec_frames", self.head_pose_analyzer.consec_frames)
        self.eye_analyzer.reset()
        self.mouth_analyzer.reset()
        self.head_pose_analyzer.reset()


# 全局检测系统实例
system = WebDetectionSystem()


# ---- Flask 路由 ----

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/start", methods=["POST"])
def api_start():
    ok = system.start()
    return jsonify({"success": ok, "message": "摄像头启动成功" if ok else "无法打开摄像头"})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    system.stop()
    return jsonify({"success": True, "message": "检测已停止"})


@app.route("/api/data")
def api_data():
    return jsonify(system.get_data())


@app.route("/api/config", methods=["POST"])
def api_config():
    data = request.get_json(force=True)
    system.update_config(data)
    return jsonify({"success": True, "message": "配置已更新"})


@app.route("/api/mode", methods=["POST"])
def api_mode():
    data = request.get_json(force=True)
    new_mode = data.get("mode", "rule")
    mode_names = {"rule": "规则模式", "dl": "深度学习模式", "hybrid": "混合模式"}
    system._add_log("info", f"切换到{mode_names.get(new_mode, new_mode)}")
    system.mode = new_mode
    return jsonify({"success": True, "mode": system.mode})


@app.route("/api/logs")
def api_logs():
    since = request.args.get("since", 0, type=int)
    logs, total = system.get_logs(since)
    return jsonify({"logs": logs, "total": total})


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            frame = system.get_frame()
            if frame is not None:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            else:
                time.sleep(0.03)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ---- 脚本执行 API ----

# 存储正在运行的脚本进程
_running_scripts = {}
_script_logs = {}
_script_lock = threading.Lock()


@app.route("/api/run_script", methods=["POST"])
def api_run_script():
    """在后台运行 Python 脚本。"""
    data = request.get_json(force=True)
    script = data.get("script", "")
    args = data.get("args", [])

    # 安全检查：只允许运行项目内的 .py 文件
    allowed_scripts = {
        "calibrate": ["python", "-m", "calibration.threshold_calibrator"],
        "train_eye": [sys.executable, "training/train_cnn.py", "--dataset_type", "eye"],
        "train_mouth": [sys.executable, "training/train_cnn.py", "--dataset_type", "mouth"],
        "test": [sys.executable, "-m", "pytest", "tests/", "-v"],
    }

    if script not in allowed_scripts:
        return jsonify({"success": False, "message": f"不允许运行的脚本: {script}"})

    cmd = allowed_scripts[script] + args
    script_id = f"{script}_{int(time.time())}"

    def run_in_bg():
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            with _script_lock:
                _running_scripts[script_id] = proc
                _script_logs[script_id] = ""

            for line in proc.stdout:
                with _script_lock:
                    _script_logs[script_id] += line

            proc.wait()
            with _script_lock:
                _script_logs[script_id] += f"\n--- 进程结束，退出码: {proc.returncode} ---\n"
                _running_scripts.pop(script_id, None)
        except Exception as e:
            with _script_lock:
                _script_logs[script_id] = f"执行错误: {e}"
                _running_scripts.pop(script_id, None)

    threading.Thread(target=run_in_bg, daemon=True).start()
    return jsonify({"success": True, "script_id": script_id, "message": "脚本已启动"})


@app.route("/api/script_log/<script_id>")
def api_script_log(script_id):
    """获取脚本运行日志。"""
    with _script_lock:
        log = _script_logs.get(script_id, "")
        is_running = script_id in _running_scripts
    return jsonify({"log": log, "is_running": is_running})


@app.route("/api/stop_script/<script_id>", methods=["POST"])
def api_stop_script(script_id):
    """停止正在运行的脚本。"""
    with _script_lock:
        proc = _running_scripts.get(script_id)
        if proc:
            proc.terminate()
            _running_scripts.pop(script_id, None)
            return jsonify({"success": True, "message": "脚本已停止"})
    return jsonify({"success": False, "message": "脚本未在运行"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
