"""疲劳驾驶检测系统入口文件"""

import argparse
import json
import sys

import cv2
import numpy as np

from detectors.face_detector import FaceDetector
from detectors.eye_analyzer import EyeAnalyzer
from detectors.mouth_analyzer import MouthAnalyzer
from detectors.head_pose_analyzer import HeadPoseAnalyzer
from evaluators.fatigue_evaluator import FatigueEvaluator
from display.renderer import DisplayRenderer
from models.data_models import EyeResult, MouthResult, PoseResult, DLResult

# DL 模型默认路径
_EYE_MODEL_PATH = "models/trained/eye_model.h5"
_MOUTH_MODEL_PATH = "models/trained/mouth_model.h5"

# 默认阈值
_DEFAULTS = {
    "ear_threshold": 0.2,
    "mar_threshold": 0.75,
    "pitch_threshold": 25.0,
    "eye_consec_frames": 48,
    "mouth_consec_frames": 25,
    "head_consec_frames": 50,
}


class DetectionSystem:
    """疲劳驾驶检测系统主程序，协调各检测模块并管理视频流主循环。"""

    def __init__(self, mode="rule", config_path=None):
        self.mode = mode
        self._cap = None

        # 加载配置
        config = self._load_config(config_path)

        # 初始化各模块
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

        # DL 分类器（仅 dl/hybrid 模式需要）
        self.dl_classifier = None
        if mode in ("dl", "hybrid"):
            self.dl_classifier = self._try_load_dl_classifier()
            if self.dl_classifier is None:
                print("警告: DL 模型加载失败，回退到 rule 模式")
                self.mode = "rule"

    @staticmethod
    def _load_config(config_path):
        """从 JSON 配置文件加载阈值参数，缺失字段使用默认值。"""
        config = dict(_DEFAULTS)

        if config_path is None:
            return config

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"警告: 配置文件不存在 {config_path}，使用默认阈值")
            return config
        except json.JSONDecodeError:
            print(f"警告: 配置文件格式错误 {config_path}，使用默认阈值")
            return config

        # 用配置文件中的值覆盖默认值
        for key in _DEFAULTS:
            if key in data and data[key] is not None:
                config[key] = data[key]

        return config

    @staticmethod
    def _try_load_dl_classifier():
        """尝试加载 DL 分类器，失败时返回 None。"""
        try:
            from classifiers.dl_classifier import DLClassifier
            return DLClassifier(_EYE_MODEL_PATH, _MOUTH_MODEL_PATH)
        except (FileNotFoundError, ImportError) as e:
            print(f"警告: 无法加载 DL 模型 - {e}")
            return None

    def run(self):
        """启动主检测循环。"""
        self._cap = cv2.VideoCapture(0)

        if not self._cap.isOpened():
            print("无法打开摄像头")
            sys.exit(1)

        try:
            self._main_loop()
        finally:
            self.stop()

    def _main_loop(self):
        """视频流处理主循环。"""
        while True:
            ret, frame = self._cap.read()
            if not ret:
                continue

            # 人脸关键点检测
            landmarks = self.face_detector.detect(frame)

            if landmarks is not None:
                # 各分析器处理
                eye_result = self.eye_analyzer.analyze(
                    landmarks.left_eye, landmarks.right_eye
                )
                mouth_result = self.mouth_analyzer.analyze(landmarks.mouth)
                pose_result = self.head_pose_analyzer.estimate_pose(
                    landmarks.head_pose_points, frame.shape
                )

                # DL 分类（dl/hybrid 模式）
                dl_result = None
                if self.mode in ("dl", "hybrid") and self.dl_classifier is not None:
                    dl_result = self._classify_dl(frame, landmarks)

                # 综合疲劳判断
                fatigue_status = self.fatigue_evaluator.evaluate(
                    eye_result, mouth_result, pose_result,
                    dl_result=dl_result, mode=self.mode,
                )
            else:
                # 未检测到人脸：使用默认值
                eye_result = EyeResult(ear=0.0, is_closed=False, is_fatigued=False, frame_count=0)
                mouth_result = MouthResult(mar=0.0, is_yawning=False, is_fatigued=False, frame_count=0)
                pose_result = PoseResult(pitch=0.0, yaw=0.0, roll=0.0, is_head_down=False, is_fatigued=False, frame_count=0)
                dl_result = None
                fatigue_status = self.fatigue_evaluator.evaluate(
                    eye_result, mouth_result, pose_result, mode=self.mode,
                )

            # 渲染
            rendered = self.renderer.render(
                frame, landmarks, eye_result, mouth_result,
                pose_result, fatigue_status, dl_result=dl_result,
            )

            cv2.imshow("疲劳驾驶检测系统", rendered)

            # 按 q 退出
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    def _classify_dl(self, frame, landmarks):
        """使用 DL 分类器对眼部和嘴部区域进行分类。"""
        try:
            eye_img = self._crop_eye_region(frame, landmarks)
            mouth_img = self._crop_mouth_region(frame, landmarks)
            return self.dl_classifier.classify(eye_img, mouth_img)
        except Exception:
            return None

    @staticmethod
    def _crop_eye_region(frame, landmarks):
        """从帧中裁剪眼部区域（基于左右眼关键点的包围盒 + padding）。"""
        all_eye_points = landmarks.left_eye + landmarks.right_eye
        xs = [p[0] for p in all_eye_points]
        ys = [p[1] for p in all_eye_points]

        h, w = frame.shape[:2]
        pad_x = int((max(xs) - min(xs)) * 0.3)
        pad_y = int((max(ys) - min(ys)) * 0.5)

        x1 = max(0, int(min(xs)) - pad_x)
        y1 = max(0, int(min(ys)) - pad_y)
        x2 = min(w, int(max(xs)) + pad_x)
        y2 = min(h, int(max(ys)) + pad_y)

        return frame[y1:y2, x1:x2]

    @staticmethod
    def _crop_mouth_region(frame, landmarks):
        """从帧中裁剪嘴部区域（基于嘴巴关键点的包围盒 + padding）。"""
        mouth_points = list(landmarks.mouth.values())
        xs = [p[0] for p in mouth_points]
        ys = [p[1] for p in mouth_points]

        h, w = frame.shape[:2]
        pad_x = int((max(xs) - min(xs)) * 0.3)
        pad_y = int((max(ys) - min(ys)) * 0.3)

        x1 = max(0, int(min(xs)) - pad_x)
        y1 = max(0, int(min(ys)) - pad_y)
        x2 = min(w, int(max(xs)) + pad_x)
        y2 = min(h, int(max(ys)) + pad_y)

        return frame[y1:y2, x1:x2]

    def stop(self):
        """释放摄像头资源、关闭所有窗口、关闭人脸检测器。"""
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
        cv2.destroyAllWindows()
        self.face_detector.close()


def main():
    parser = argparse.ArgumentParser(description="疲劳驾驶检测系统")
    parser.add_argument(
        "--mode",
        choices=["rule", "dl", "hybrid"],
        default="rule",
        help="检测模式: rule(规则), dl(深度学习), hybrid(混合)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON 阈值配置文件路径",
    )
    args = parser.parse_args()

    system = DetectionSystem(mode=args.mode, config_path=args.config)
    system.run()


if __name__ == "__main__":
    main()
