"""疲劳驾驶检测系统 - 简化 Demo 版本"""

import time
import threading
import cv2
import mediapipe as mp
from flask import Flask, Response, jsonify, render_template

app = Flask(__name__, template_folder="templates", static_folder="static")

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


class DemoDetector:
    """简化版检测器"""

    def __init__(self):
        self._cap = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_data = {
            "ear": 0.0, "mar": 0.0, "status": "待启动",
            "is_fatigued": False, "face_detected": False,
        }
        self.eye_closed_frames = 0
        self.yawn_frames = 0

    def start(self):
        if self._running:
            return True
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            return False
        self._running = True
        threading.Thread(target=self._process_loop, daemon=True).start()
        return True

    def stop(self):
        self._running = False
        time.sleep(0.2)
        if self._cap:
            self._cap.release()
        self._cap = None
        self.eye_closed_frames = 0
        self.yawn_frames = 0

    def _calculate_ear(self, eye_points):
        """计算眼睛纵横比"""
        import numpy as np
        p = np.array(eye_points)
        v1 = np.linalg.norm(p[1] - p[5])
        v2 = np.linalg.norm(p[2] - p[4])
        h = np.linalg.norm(p[0] - p[3])
        return (v1 + v2) / (2.0 * h + 1e-6)

    def _calculate_mar(self, mouth_points):
        """计算嘴巴纵横比"""
        import numpy as np
        p = np.array(mouth_points)
        v1 = np.linalg.norm(p[1] - p[7])
        v2 = np.linalg.norm(p[2] - p[6])
        v3 = np.linalg.norm(p[3] - p[5])
        h = np.linalg.norm(p[0] - p[4])
        return (v1 + v2 + v3) / (3.0 * h + 1e-6)

    def _process_loop(self):
        while self._running:
            if not self._cap or not self._cap.isOpened():
                break
            ret, frame = self._cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                h, w = frame.shape[:2]

                # 提取眼睛和嘴巴关键点
                left_eye = [[landmarks[i].x * w, landmarks[i].y * h] for i in [33, 160, 158, 133, 153, 144]]
                right_eye = [[landmarks[i].x * w, landmarks[i].y * h] for i in [362, 385, 387, 263, 373, 380]]
                mouth = [[landmarks[i].x * w, landmarks[i].y * h] for i in [61, 39, 0, 269, 291, 405, 314, 17]]

                # 计算 EAR 和 MAR
                ear_left = self._calculate_ear(left_eye)
                ear_right = self._calculate_ear(right_eye)
                ear = (ear_left + ear_right) / 2.0
                mar = self._calculate_mar(mouth)

                # 判断状态
                is_eye_closed = ear < 0.2
                is_yawning = mar > 0.75

                if is_eye_closed:
                    self.eye_closed_frames += 1
                else:
                    self.eye_closed_frames = 0

                if is_yawning:
                    self.yawn_frames += 1
                else:
                    self.yawn_frames = 0

                is_fatigued = self.eye_closed_frames > 30 or self.yawn_frames > 20

                # 绘制关键点
                for pt in left_eye + right_eye:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
                for pt in mouth:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)

                # 显示状态
                status = "正常"
                color = (0, 255, 0)
                if is_fatigued:
                    status = "⚠️ 疲劳驾驶"
                    color = (0, 0, 255)
                elif is_eye_closed:
                    status = "闭眼"
                    color = (0, 165, 255)
                elif is_yawning:
                    status = "打哈欠"
                    color = (0, 165, 255)

                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"MAR: {mar:.2f}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                with self._lock:
                    self._latest_data = {
                        "ear": round(ear, 2), "mar": round(mar, 2),
                        "status": status, "is_fatigued": is_fatigued,
                        "face_detected": True,
                        "eye_frames": self.eye_closed_frames,
                        "yawn_frames": self.yawn_frames,
                    }
            else:
                cv2.putText(frame, "未检测到人脸", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                with self._lock:
                    self._latest_data = {
                        "ear": 0.0, "mar": 0.0, "status": "未检测到人脸",
                        "is_fatigued": False, "face_detected": False,
                        "eye_frames": 0, "yawn_frames": 0,
                    }

            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            with self._lock:
                self._latest_frame = jpeg.tobytes()

    def get_frame(self):
        with self._lock:
            return self._latest_frame

    def get_data(self):
        with self._lock:
            return dict(self._latest_data)


detector = DemoDetector()


@app.route("/")
def index():
    return render_template("demo.html")


@app.route("/api/start", methods=["POST"])
def api_start():
    ok = detector.start()
    return jsonify({"success": ok})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    detector.stop()
    return jsonify({"success": True})


@app.route("/api/data")
def api_data():
    return jsonify(detector.get_data())


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            frame = detector.get_frame()
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            else:
                time.sleep(0.03)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    print("=" * 50)
    print("  疲劳驾驶检测 Demo - 启动中...")
    print("  访问: http://localhost:5001")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
