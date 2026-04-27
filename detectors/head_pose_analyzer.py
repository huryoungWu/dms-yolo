"""头部姿态分析模块，使用 solvePnP 计算头部欧拉角并判断低头疲劳状态"""

import math
from typing import List, Tuple

import cv2
import numpy as np

from models.data_models import PoseResult


# 标准 3D 人脸模型点
_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # 鼻尖
    (0.0, -330.0, -65.0),     # 下巴
    (-225.0, 170.0, -135.0),  # 左眼角
    (225.0, 170.0, -135.0),   # 右眼角
    (-150.0, -150.0, -125.0), # 左嘴角
    (150.0, -150.0, -125.0),  # 右嘴角
], dtype=np.float64)


class HeadPoseAnalyzer:
    """使用 solvePnP 计算头部俯仰角，维护低头帧计数器，输出低头疲劳信号"""

    def __init__(self, pitch_threshold: float = 25.0, consec_frames: int = 50):
        """初始化阈值和帧计数器"""
        self.pitch_threshold = pitch_threshold
        self.consec_frames = consec_frames
        self._frame_counter = 0

    def estimate_pose(self, face_points_2d: List[Tuple], frame_shape: Tuple) -> PoseResult:
        """
        估计头部姿态。

        Args:
            face_points_2d: 6 个 2D 关键点坐标 [(x, y), ...]
            frame_shape: 图像尺寸 (h, w, c) 或 (h, w)

        Returns:
            PoseResult(pitch, yaw, roll, is_head_down, is_fatigued, frame_count)
        """
        h, w = frame_shape[0], frame_shape[1]

        # 构建相机内参矩阵
        focal_length = max(h, w)
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        image_points = np.array(face_points_2d, dtype=np.float64)

        success, rotation_vector, translation_vector = cv2.solvePnP(
            _MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return PoseResult(
                pitch=0.0, yaw=0.0, roll=0.0,
                is_head_down=False, is_fatigued=False,
                frame_count=self._frame_counter,
            )

        # 旋转向量 → 旋转矩阵 → 欧拉角
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pitch, yaw, roll = self._rotation_matrix_to_euler(rotation_matrix)

        # 低头判定：|pitch| > threshold
        is_head_down = abs(pitch) > self.pitch_threshold

        if is_head_down:
            self._frame_counter += 1
        else:
            self._frame_counter = 0

        is_fatigued = self._frame_counter >= self.consec_frames

        return PoseResult(
            pitch=pitch, yaw=yaw, roll=roll,
            is_head_down=is_head_down, is_fatigued=is_fatigued,
            frame_count=self._frame_counter,
        )

    def reset(self):
        """重置帧计数器"""
        self._frame_counter = 0

    @staticmethod
    def _rotation_matrix_to_euler(rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
        """
        从旋转矩阵提取欧拉角 (pitch, yaw, roll)，单位为度。

        使用 ZYX 顺序分解。
        """
        sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)

        if sy > 1e-6:
            pitch = math.atan2(-rotation_matrix[2, 0], sy)
            yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            roll = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        else:
            pitch = math.atan2(-rotation_matrix[2, 0], sy)
            yaw = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            roll = 0.0

        # 弧度转角度
        pitch = math.degrees(pitch)
        yaw = math.degrees(yaw)
        roll = math.degrees(roll)

        return pitch, yaw, roll
