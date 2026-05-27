"""Flask Web 前端 - 疲劳驾驶检测系统"""

import json
import os
import subprocess
import sys
import threading
import time
import math

# macOS: 避免 OpenCV 在非主线程触发相机授权弹窗导致初始化失败
os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")

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

PHONE_CENTER = (0.0, 0.0)
CIGARETTE_CENTER = (0.0, 0.0)
DRINKING_CENTER = (0.0,0.0)
EYES_CENTER = (0.0, 0.0)
MOUTH_CENTER = (0.0, 0.0)
# 默认阈值
_DEFAULTS = {
    "ear_threshold": 0.05,
    "mar_threshold": 0.4,
    "pitch_threshold": 25.0,
    "eye_consec_frames": 25,
    "mouth_consec_frames": 25,
    "head_consec_frames": 25,
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
            "is_dms_alert": False,
            "dms_alerts": [],
        }
        self.mode = "rule"
        self._logs = []
        self._log_lock = threading.Lock()
        self._last_camera_error = ""
        self._last_yolo_error = ""
        self._yolo_model = None
        self._yolo_enabled = False
        self._display_mode = "yolo"
        self._yolo_model_path = os.path.join(
            "dms-driver-monitoring-system", "workdir", "final_model.pt"
        )

        # DMS 告警阈值（秒）
        self._dms_thresholds = {
            "eye_l1": 0.8,
            "eye_l2": 1.5,
            "head_down": 1.0,
            "yawn": 0.8,
            "look_away": 0.2,
            "phone": 0.5,       # 打电话持续时间
            "smoke": 0.5,        # 抽烟持续时间
            "no_driver": 1.0,
            "occlusion": 0.6,
            "yaw_abs": 5.0,
            "drinking": 0.2,  
            "phone_eye_dist": 150.0,   # 手机-面部关键点最大有效距离
            "cig_mouth_dist": 200.0,   # 香烟-嘴巴最大有效距离
            "drink_mouth_dist": 200.0, # 喝水-嘴巴最大有效距离
        }
        self._occlusion_cfg = {
            "dark_mean": 35.0,
            "low_var_std": 12.0,
        }
        self._dms_state = {
            "eye_closed_since": None,
            "head_down_since": None,
            "yawn_since": None,
            "look_away_since": None,
            "phone_since": None,
            "smoke_since": None,
            "no_driver_since": None,
            "occlusion_since": None,
            "last_alert_signature": "",
        }

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

    def _load_yolo_model(self):
        """懒加载 YOLOv8 模型。"""
        if self._yolo_model is not None:
            return self._yolo_model

        model_path = self._yolo_model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO 模型不存在: {model_path}")

        from ultralytics import YOLO

        self._yolo_model = YOLO(model_path)
        return self._yolo_model

    def set_yolo_enabled(self, enabled: bool):
        """启用/禁用 YOLOv8 叠加检测。"""
        if enabled:
            try:
                self._load_yolo_model()
                self._yolo_enabled = True
                self._last_yolo_error = ""
                self._add_log("info", f"YOLOv8 已启用: {self._yolo_model_path}")
                return True, "YOLOv8 已启用"
            except Exception as e:
                self._yolo_enabled = False
                self._last_yolo_error = str(e)
                self._add_log("danger", f"YOLOv8 启用失败: {e}")
                return False, f"YOLOv8 启用失败: {e}"

        self._yolo_enabled = False
        self._add_log("info", "YOLOv8 已禁用")
        return True, "YOLOv8 已禁用"

    def set_display_mode(self, mode: str):
        allowed = {"yolo", "rule", "both"}
        self._display_mode = mode if mode in allowed else "yolo"
        display_names = {
            "yolo": "YOLO显示",
            "rule": "规则显示",
            "both": "双模式显示",
        }
        self._add_log("info", f"切换到{display_names.get(self._display_mode, self._display_mode)}")
        return self._display_mode

    @staticmethod
    def _class_present(label_set, *candidates):
        return any(c in label_set for c in candidates)
    
    @staticmethod
    def _calc_dist(p1, p2):
        """计算两点欧几里得距离"""
        return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

    def _update_timer(self, key: str, active: bool, now_ts: float):
        since_key = f"{key}_since"
        if active:
            if self._dms_state[since_key] is None:
                self._dms_state[since_key] = now_ts
        else:
            self._dms_state[since_key] = None

    def _elapsed(self, key: str, now_ts: float) -> float:
        since_key = f"{key}_since"
        since = self._dms_state.get(since_key)
        if since is None:
            return 0.0
        return max(0.0, now_ts - since)

    def _build_dms_alerts(self, yolo_labels, eye_result, mouth_result, pose_result, frame=None):
        """按业务规则生成 DMS 预警信息。"""
        now_ts = time.time()
        label_set = {str(x).strip().lower() for x in yolo_labels}
        print(f'label_set:{label_set}')
        # 适配当前模型类别: open eye / closed eye / cigarette / phone / seatbelt
        phone_detected = self._class_present(label_set, "phone")
        smoke_detected = self._class_present(label_set, "cigarette")
        closed_eye_on = eye_result.is_closed or self._class_present(label_set, "closed eye", "closed_eye")
        open_eye_on = self._class_present(label_set, "open eye", "open_eye")
        seatbelt_on = self._class_present(label_set, "seatbelt", "seat belt")
        drinking_detected = self._class_present(label_set, "drinking", "drinking")
        # 低头/哈欠由现有分析器保证，YOLO 有对应类时可叠加
        yawn_on = mouth_result.is_yawning or self._class_present(label_set, "yawn", "yawning")
        head_down_on = pose_result.is_head_down or self._class_present(label_set, "head down", "head_down")

        # 左顾右盼: 优先 YOLO 类；若无则用 yaw 近似
        look_away_on = self._class_present(label_set, "look away", "looking_away", "distracted")
        if not look_away_on:
            look_away_on = abs(float(getattr(pose_result, "yaw", 0.0))) >= self._dms_thresholds["yaw_abs"]
            
        # 驾驶座无人: 当前模型无该类，使用“既无开眼/闭眼也无安全带”作为保守近似
        no_driver_on = self._class_present(label_set, "no driver", "no_driver", "empty seat", "empty_seat")
        if not no_driver_on:
            no_driver_on = (not open_eye_on) and (not closed_eye_on) and (not seatbelt_on)

        # 遮挡镜头: 若无YOLO类，使用黑屏统计近似
        occlusion_on = self._class_present(label_set, "occlusion", "covered", "blocked", "black_screen")
        if (not occlusion_on) and frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_v = float(np.mean(gray))
            std_v = float(np.std(gray))
            occlusion_on = (mean_v <= self._occlusion_cfg["dark_mean"]) and (std_v <= self._occlusion_cfg["low_var_std"])

        # ====================== 核心修改 ======================
        # 打电话：检测到手机 + 手机靠近面部关键点（眼睛/嘴巴） + 人脸存在
        phone_on = False
        if phone_detected and PHONE_CENTER != (0.0, 0.0):
            phone_dist_candidates = []
            if EYES_CENTER != (0.0, 0.0):
                phone_dist_candidates.append(self._calc_dist(PHONE_CENTER, EYES_CENTER))
            if MOUTH_CENTER != (0.0, 0.0):
                phone_dist_candidates.append(self._calc_dist(PHONE_CENTER, MOUTH_CENTER))

            if phone_dist_candidates:
                dist = min(phone_dist_candidates)
                # print(f'phone_dist:{dist}')
                if dist < self._dms_thresholds["phone_eye_dist"]:
                    phone_on = True
            else:
                # 若手机稳定出现但人脸关键点暂时不稳，保守视为打电话候选
                phone_on = True

        # 抽烟：检测到香烟 + 香烟与嘴巴距离 < 阈值 + 人脸存在
        smoke_on = False
        if smoke_detected and MOUTH_CENTER != (0.0, 0.0) and CIGARETTE_CENTER != (0.0, 0.0):
            dist = self._calc_dist(CIGARETTE_CENTER, MOUTH_CENTER)
            # print(f'smoke_dist:{dist}')
            if dist < self._dms_thresholds["cig_mouth_dist"]:
                smoke_on = True
        # ======================================================
    
        drinking_on = False
        if drinking_detected and MOUTH_CENTER != (0.0, 0.0) and DRINKING_CENTER != (0.0, 0.0):
            dist = self._calc_dist(DRINKING_CENTER, MOUTH_CENTER)
            print(f'drinking_dist:{dist}')
            if dist < self._dms_thresholds["drink_mouth_dist"]:
                drinking_on = True

        self._update_timer("eye_closed", closed_eye_on, now_ts)
        self._update_timer("head_down", head_down_on, now_ts)
        self._update_timer("yawn", yawn_on, now_ts)
        self._update_timer("look_away", look_away_on, now_ts)
        self._update_timer("phone", phone_on, now_ts)
        self._update_timer("smoke", smoke_on, now_ts)
        self._update_timer("no_driver", no_driver_on, now_ts)
        self._update_timer("occlusion", occlusion_on, now_ts)
        self._update_timer("drinking", drinking_on, now_ts)
        alerts = []

        eye_elapsed = self._elapsed("eye_closed", now_ts)
        if eye_elapsed >= self._dms_thresholds["eye_l2"]:
            alerts.append("闭眼2级预警（闭眼≥1.5秒）")
        # elif eye_elapsed >= self._dms_thresholds["eye_l1"]:
        #     alerts.append("闭眼1级预警（闭眼≥0.8秒）")

        if self._elapsed("head_down", now_ts) >= self._dms_thresholds["head_down"]:
            alerts.append("低头预警（低头≥1.0秒）")

        if self._elapsed("yawn", now_ts) >= self._dms_thresholds["yawn"]:
            alerts.append("打哈欠预警（打哈欠≥0.8秒）")

        if self._elapsed("phone", now_ts) >= self._dms_thresholds["phone"]:
            alerts.append("打电话预警")

        if self._elapsed("smoke", now_ts) >= self._dms_thresholds["smoke"]:
            alerts.append("抽烟预警")

        if self._elapsed("look_away", now_ts) >= self._dms_thresholds["look_away"]:
            alerts.append("左顾右盼预警（视线偏移≥1.0秒）")

        if self._elapsed("occlusion", now_ts) >= self._dms_thresholds["occlusion"]:
            alerts.append("遮挡镜头预警")

        if self._elapsed("no_driver", now_ts) >= self._dms_thresholds["no_driver"]:
            alerts.append("驾驶座无人预警")
        
        if self._elapsed("drinking", now_ts) >= self._dms_thresholds["drinking"]:
            alerts.append("喝水预警")

        signature = "|".join(alerts)
        if signature and signature != self._dms_state.get("last_alert_signature", ""):
            self._add_log("warning", f"DMS预警: {'; '.join(alerts)}")
        self._dms_state["last_alert_signature"] = signature

        return alerts

    def start(self):
        """启动摄像头和处理线程。"""
        if self._running:
            return True

        self._last_camera_error = ""
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            self._last_camera_error = (
                "无法打开摄像头。macOS 请检查：系统设置 -> 隐私与安全性 -> 相机，"
                "允许当前终端/IDE（如 Terminal、iTerm、PyCharm、VS Code）。"
            )
            self._add_log("danger", self._last_camera_error)
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

            raw_frame = frame.copy()
            rule_rendered = frame.copy()
            dl_result = None
            landmarks = self.face_detector.detect(frame)
            # 计算眼睛中心点坐标
            left_eye_center = (sum([p[0] for p in landmarks.left_eye]) / len(landmarks.left_eye), 
                            sum([p[1] for p in landmarks.left_eye]) / len(landmarks.left_eye)) if landmarks and landmarks.left_eye else (0,0)
            right_eye_center = (sum([p[0] for p in landmarks.right_eye]) / len(landmarks.right_eye), 
                            sum([p[1] for p in landmarks.right_eye]) / len(landmarks.right_eye)) if landmarks and landmarks.right_eye else (0,0)
            eyes_center = ((left_eye_center[0] + right_eye_center[0]) / 2, (left_eye_center[1] + right_eye_center[1]) / 2)
            global EYES_CENTER
            EYES_CENTER = eyes_center
            
            # 计算嘴巴中心点坐标
            mouth_center = (0,0)
            if landmarks and landmarks.mouth:
                mouth_points = list(landmarks.mouth.values())
                mouth_center = (sum([p[0] for p in mouth_points]) / len(mouth_points), 
                            sum([p[1] for p in mouth_points]) / len(mouth_points))
            global MOUTH_CENTER
            MOUTH_CENTER = mouth_center
            
            if landmarks is not None:
                eye_result = self.eye_analyzer.analyze(landmarks.left_eye, landmarks.right_eye)
                mouth_result = self.mouth_analyzer.analyze(landmarks.mouth)
                pose_result = self.head_pose_analyzer.estimate_pose(
                    landmarks.head_pose_points, frame.shape
                )
                fatigue_status = self.fatigue_evaluator.evaluate(
                    eye_result, mouth_result, pose_result,
                    dl_result=dl_result, mode=self.mode,
                )
                status_key = DisplayRenderer._determine_status(eye_result, mouth_result, pose_result)
                rule_rendered = self.renderer.render(
                    rule_rendered, landmarks, eye_result, mouth_result,
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
                        "is_dms_alert": False,
                        "dms_alerts": [],
                    }
            else:
                eye_result = EyeResult(ear=0.0, is_closed=False, is_fatigued=False, frame_count=0)
                mouth_result = MouthResult(mar=0.0, is_yawning=False, is_fatigued=False, frame_count=0)
                pose_result = PoseResult(pitch=0.0, yaw=0.0, roll=0.0, is_head_down=False, is_fatigued=False, frame_count=0)
                fatigue_status = self.fatigue_evaluator.evaluate(
                    eye_result, mouth_result, pose_result, mode=self.mode,
                )
                rule_rendered = self.renderer.render(
                    rule_rendered, None, eye_result, mouth_result,
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
                        "is_dms_alert": False,
                        "dms_alerts": [],
                    }

            rendered = rule_rendered

            if self._yolo_enabled:
                try:
                    yolo_model = self._load_yolo_model()
                    yolo_results = yolo_model.predict(raw_frame, conf=0.35, verbose=False)
                    yolo_labels = []
                    yolo_rendered = raw_frame.copy()
                    if yolo_results:
                        result0 = yolo_results[0]
                        names_map = getattr(result0, "names", {}) or {}
                        boxes = getattr(result0, "boxes", None)
                        
                        if boxes is not None and getattr(boxes, "cls", None) is not None:
                            # 获取边界框坐标
                            boxes_xyxy = boxes.xyxy.tolist()
                            class_ids = boxes.cls.tolist()
                            
                            # 重置坐标
                            global PHONE_CENTER, CIGARETTE_CENTER, DRINKING_CENTER
                            PHONE_CENTER = (0.0, 0.0)
                            CIGARETTE_CENTER = (0.0, 0.0)
                            DRINKING_CENTER = (0.0, 0.0)
                            
                            for i, cls_id in enumerate(class_ids):
                                cls_name = names_map.get(int(cls_id), str(int(cls_id)))
                                yolo_labels.append(str(cls_name))

                                # 计算中心点坐标
                                box = boxes_xyxy[i]
                                center_x = (box[0] + box[2]) / 2
                                center_y = (box[1] + box[3]) / 2
                                center_point = (int(center_x), int(center_y))

                                if cls_name.lower() == 'phone':
                                    PHONE_CENTER = center_point
                                elif cls_name.lower() == 'cigarette':
                                    CIGARETTE_CENTER = center_point
                                elif cls_name.lower() == 'drinking':
                                    DRINKING_CENTER = center_point
                        yolo_rendered = result0.plot()

                    dms_alerts = self._build_dms_alerts(
                        yolo_labels, eye_result, mouth_result, pose_result, frame=raw_frame
                    )
                    with self._lock:
                        self._latest_data["dms_alerts"] = dms_alerts
                        self._latest_data["is_dms_alert"] = len(dms_alerts) > 0

                    if self._display_mode == "rule":
                        rendered = rule_rendered
                    elif self._display_mode == "both":
                        rendered = self.renderer.render(
                            yolo_rendered, landmarks, eye_result, mouth_result,
                            pose_result, fatigue_status, dl_result=dl_result,
                        )
                    else:
                        rendered = yolo_rendered
                except Exception as e:
                    self._yolo_enabled = False
                    self._last_yolo_error = str(e)
                    self._add_log("danger", f"YOLO 推理失败，已自动禁用: {e}")
                    with self._lock:
                        self._latest_data["dms_alerts"] = []
                        self._latest_data["is_dms_alert"] = False
            else:
                with self._lock:
                    self._latest_data["dms_alerts"] = []
                    self._latest_data["is_dms_alert"] = False
                rendered = rule_rendered

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
    if ok:
        return jsonify({"success": True, "message": "摄像头启动成功"})
    return jsonify({
        "success": False,
        "message": system._last_camera_error or "无法打开摄像头",
    })


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


@app.route("/api/yolo", methods=["POST"])
def api_yolo_toggle():
    data = request.get_json(force=True)
    enabled = bool(data.get("enabled", False))
    ok, message = system.set_yolo_enabled(enabled)
    return jsonify({"success": ok, "enabled": system._yolo_enabled, "message": message})


@app.route("/api/display_mode", methods=["POST"])
def api_display_mode():
    data = request.get_json(force=True)
    mode = data.get("mode", "yolo")
    display_mode = system.set_display_mode(mode)
    return jsonify({"success": True, "display_mode": display_mode})


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
