"""眼睛状态分析模块，负责计算 EAR 值并判断闭眼疲劳状态"""

import math
from typing import List, Tuple

from models.data_models import EyeResult


class EyeAnalyzer:
    """计算 EAR 值，维护闭眼帧计数器，输出闭眼疲劳信号"""

    def __init__(self, ear_threshold: float = 0.2, consec_frames: int = 48):
        """初始化阈值和帧计数器"""
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self._frame_counter = 0

    def calculate_ear(self, eye_points: List[Tuple[float, float]]) -> float:
        """
        计算单只眼睛的 EAR 值。

        公式: EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

        Args:
            eye_points: 6 个眼睛轮廓关键点 [(x,y), ...]

        Returns:
            EAR 值，分母为零时返回 0.0
        """
        p1, p2, p3, p4, p5, p6 = eye_points

        vertical_1 = math.dist(p2, p6)
        vertical_2 = math.dist(p3, p5)
        horizontal = math.dist(p1, p4)

        if horizontal == 0.0:
            return 0.0

        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def analyze(self, left_eye: List[Tuple[float, float]], right_eye: List[Tuple[float, float]]) -> EyeResult:
        """
        分析双眼状态，返回 EAR 值和疲劳信号。

        Args:
            left_eye: 左眼 6 个关键点
            right_eye: 右眼 6 个关键点

        Returns:
            EyeResult(ear, is_closed, is_fatigued, frame_count)
        """
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        is_closed = avg_ear < self.ear_threshold

        if is_closed:
            self._frame_counter += 1
        else:
            self._frame_counter = 0

        is_fatigued = self._frame_counter >= self.consec_frames

        return EyeResult(
            ear=avg_ear,
            is_closed=is_closed,
            is_fatigued=is_fatigued,
            frame_count=self._frame_counter,
        )

    def reset(self):
        """重置帧计数器"""
        self._frame_counter = 0
