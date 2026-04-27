"""核心数据模型定义"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class FaceLandmarks:
    """人脸关键点检测结果"""
    left_eye: List[Tuple[float, float]]
    right_eye: List[Tuple[float, float]]
    mouth: dict
    head_pose_points: List[Tuple[float, float]]
    all_landmarks: List[Tuple[float, float]]


@dataclass
class EyeResult:
    """眼睛分析结果"""
    ear: float
    is_closed: bool
    is_fatigued: bool
    frame_count: int


@dataclass
class MouthResult:
    """嘴巴分析结果"""
    mar: float
    is_yawning: bool
    is_fatigued: bool
    frame_count: int


@dataclass
class PoseResult:
    """头部姿态分析结果"""
    pitch: float
    yaw: float
    roll: float
    is_head_down: bool
    is_fatigued: bool
    frame_count: int


@dataclass
class DLResult:
    """深度学习分类结果"""
    eye_label: str
    eye_confidence: float
    mouth_label: str
    mouth_confidence: float


@dataclass
class FatigueStatus:
    """综合疲劳状态"""
    is_fatigued: bool
    reasons: List[str]
    mode: str


@dataclass
class CalibrationResult:
    """阈值校准结果"""
    optimal_ear_threshold: float
    optimal_mar_threshold: float
    ear_accuracy: float
    ear_recall: float
    mar_accuracy: float
    mar_recall: float
    ear_distribution: dict
    mar_distribution: dict
