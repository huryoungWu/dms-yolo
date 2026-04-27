"""嘴巴状态分析模块，负责计算 MAR 值并判断哈欠疲劳状态"""

import math

from models.data_models import MouthResult


class MouthAnalyzer:
    """计算 MAR 值，维护哈欠帧计数器，输出哈欠疲劳信号"""

    def __init__(self, mar_threshold: float = 0.75, consec_frames: int = 25):
        """初始化阈值和帧计数器"""
        self.mar_threshold = mar_threshold
        self.consec_frames = consec_frames
        self._frame_counter = 0

    def calculate_mar(self, mouth_points: dict) -> float:
        """
        计算 MAR 值。

        公式: MAR = (|upper_inner-lower_inner| + |upper-lower|) / (2 * |left-right|)

        Args:
            mouth_points: 嘴巴关键点字典，包含 upper, lower, left, right,
                          upper_inner, lower_inner，每个值为 (x, y) 元组

        Returns:
            MAR 值，分母为零时返回 0.0
        """
        horizontal = math.dist(mouth_points["left"], mouth_points["right"])

        if horizontal == 0.0:
            return 0.0

        vertical_inner = math.dist(mouth_points["upper_inner"], mouth_points["lower_inner"])
        vertical_outer = math.dist(mouth_points["upper"], mouth_points["lower"])

        return (vertical_inner + vertical_outer) / (2.0 * horizontal)

    def analyze(self, mouth_points: dict) -> MouthResult:
        """
        分析嘴巴状态，返回 MAR 值和疲劳信号。

        Args:
            mouth_points: 嘴巴关键点字典

        Returns:
            MouthResult(mar, is_yawning, is_fatigued, frame_count)
        """
        mar = self.calculate_mar(mouth_points)

        is_yawning = mar > self.mar_threshold

        if is_yawning:
            self._frame_counter += 1
        else:
            self._frame_counter = 0

        is_fatigued = self._frame_counter >= self.consec_frames

        return MouthResult(
            mar=mar,
            is_yawning=is_yawning,
            is_fatigued=is_fatigued,
            frame_count=self._frame_counter,
        )

    def reset(self):
        """重置帧计数器"""
        self._frame_counter = 0
