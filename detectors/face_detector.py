"""人脸关键点检测模块，基于 MediaPipe FaceMesh"""

from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from models.data_models import FaceLandmarks

# 关键点索引常量
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

MOUTH_INDICES = {
    "upper": 13,
    "lower": 14,
    "left": 78,
    "right": 308,
    "upper_inner": 82,
    "lower_inner": 312,
}

HEAD_POSE_INDICES = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_corner": 33,
    "right_eye_corner": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}


class FaceDetector:
    """使用 MediaPipe FaceMesh 检测人脸关键点"""

    def __init__(
        self,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
    ):
        """初始化 MediaPipe FaceMesh"""
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
            refine_landmarks=False,
        )

    def detect(self, frame: np.ndarray) -> Optional[FaceLandmarks]:
        """
        检测单帧图像中的人脸关键点。

        Args:
            frame: BGR 格式的 OpenCV 图像帧

        Returns:
            FaceLandmarks 对象，包含各区域关键点；未检测到人脸时返回 None
        """
        h, w = frame.shape[:2]

        # BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        results = self._face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        face = results.multi_face_landmarks[0]

        # 将归一化坐标转换为像素坐标
        all_landmarks = [
            (lm.x * w, lm.y * h) for lm in face.landmark
        ]

        # 提取眼睛关键点
        left_eye = [all_landmarks[i] for i in LEFT_EYE_INDICES]
        right_eye = [all_landmarks[i] for i in RIGHT_EYE_INDICES]

        # 提取嘴巴关键点
        mouth = {
            key: all_landmarks[idx] for key, idx in MOUTH_INDICES.items()
        }

        # 提取头部姿态关键点（按固定顺序）
        head_pose_points = [
            all_landmarks[HEAD_POSE_INDICES["nose_tip"]],
            all_landmarks[HEAD_POSE_INDICES["chin"]],
            all_landmarks[HEAD_POSE_INDICES["left_eye_corner"]],
            all_landmarks[HEAD_POSE_INDICES["right_eye_corner"]],
            all_landmarks[HEAD_POSE_INDICES["left_mouth"]],
            all_landmarks[HEAD_POSE_INDICES["right_mouth"]],
        ]

        return FaceLandmarks(
            left_eye=left_eye,
            right_eye=right_eye,
            mouth=mouth,
            head_pose_points=head_pose_points,
            all_landmarks=all_landmarks,
        )

    def close(self):
        """释放 MediaPipe 资源"""
        self._face_mesh.close()
