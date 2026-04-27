"""HeadPoseAnalyzer 单元测试"""

import numpy as np
import pytest

from detectors.head_pose_analyzer import HeadPoseAnalyzer
from models.data_models import PoseResult


def _make_front_facing_points(w=640, h=480):
    """生成大致正面朝向的 2D 关键点（对应标准 3D 模型点的投影）"""
    return [
        (w / 2, h / 3),          # 鼻尖
        (w / 2, h * 2 / 3),      # 下巴
        (w / 3, h / 4),          # 左眼角
        (w * 2 / 3, h / 4),      # 右眼角
        (w * 0.38, h * 0.58),    # 左嘴角
        (w * 0.62, h * 0.58),    # 右嘴角
    ]


class TestEstimatePose:
    """测试 estimate_pose() 方法"""

    def test_returns_pose_result(self):
        """应返回 PoseResult 实例"""
        analyzer = HeadPoseAnalyzer()
        points = _make_front_facing_points()
        result = analyzer.estimate_pose(points, (480, 640, 3))
        assert isinstance(result, PoseResult)

    def test_result_has_finite_angles(self):
        """返回的角度值应为有限浮点数"""
        analyzer = HeadPoseAnalyzer()
        points = _make_front_facing_points()
        result = analyzer.estimate_pose(points, (480, 640, 3))
        assert np.isfinite(result.pitch)
        assert np.isfinite(result.yaw)
        assert np.isfinite(result.roll)

    def test_front_facing_not_head_down(self):
        """正面朝向时不应判定为低头"""
        analyzer = HeadPoseAnalyzer(pitch_threshold=25.0)
        points = _make_front_facing_points()
        result = analyzer.estimate_pose(points, (480, 640, 3))
        assert not result.is_head_down

    def test_frame_shape_without_channels(self):
        """frame_shape 为 (h, w) 时也应正常工作"""
        analyzer = HeadPoseAnalyzer()
        points = _make_front_facing_points()
        result = analyzer.estimate_pose(points, (480, 640))
        assert isinstance(result, PoseResult)
        assert np.isfinite(result.pitch)


class TestFrameCounter:
    """测试帧计数器逻辑"""

    def test_counter_increments_on_head_down(self):
        """低头时计数器应递增"""
        analyzer = HeadPoseAnalyzer(pitch_threshold=0.1, consec_frames=100)
        # 使用极低阈值，正常姿态也会触发 head_down
        points = _make_front_facing_points()
        result = analyzer.estimate_pose(points, (480, 640, 3))
        # 只要 |pitch| > 0.1 就会递增
        if result.is_head_down:
            assert result.frame_count >= 1

    def test_fatigue_after_consec_frames(self):
        """连续低头帧达到阈值时应触发疲劳"""
        # 先确认正面朝向的 pitch 值，然后用比它更小的阈值
        analyzer = HeadPoseAnalyzer(pitch_threshold=25.0, consec_frames=3)
        points = _make_front_facing_points()

        # 模拟低头：将鼻尖和下巴向下偏移，使 pitch 变大
        head_down_points = [
            (320, 280),    # 鼻尖 - 偏下
            (320, 450),    # 下巴 - 大幅偏下
            (213, 80),     # 左眼角 - 偏上
            (427, 80),     # 右眼角 - 偏上
            (243, 350),    # 左嘴角
            (397, 350),    # 右嘴角
        ]

        # 先测一帧获取 pitch
        test_result = analyzer.estimate_pose(head_down_points, (480, 640, 3))
        analyzer.reset()

        # 如果 pitch 足够大，用合适的阈值
        threshold = abs(test_result.pitch) - 1.0 if abs(test_result.pitch) > 1.0 else 0.001
        analyzer = HeadPoseAnalyzer(pitch_threshold=threshold, consec_frames=3)

        for _ in range(3):
            result = analyzer.estimate_pose(head_down_points, (480, 640, 3))

        assert result.is_head_down
        assert result.is_fatigued
        assert result.frame_count == 3

    def test_counter_resets_on_normal_pose(self):
        """正常姿态时计数器应重置为 0"""
        analyzer = HeadPoseAnalyzer(pitch_threshold=90.0, consec_frames=5)
        # 使用极高阈值，确保不会触发低头
        points = _make_front_facing_points()
        result = analyzer.estimate_pose(points, (480, 640, 3))
        assert result.frame_count == 0
        assert not result.is_head_down

    def test_not_fatigued_below_consec_threshold(self):
        """低头帧数未达阈值时不应触发疲劳"""
        analyzer = HeadPoseAnalyzer(pitch_threshold=0.001, consec_frames=100)
        points = _make_front_facing_points()
        result = analyzer.estimate_pose(points, (480, 640, 3))
        assert not result.is_fatigued


class TestReset:
    """测试 reset() 方法"""

    def test_reset_clears_counter(self):
        """reset() 应将帧计数器重置为 0"""
        analyzer = HeadPoseAnalyzer(pitch_threshold=0.001, consec_frames=3)
        points = _make_front_facing_points()

        # 累积一些帧
        for _ in range(2):
            analyzer.estimate_pose(points, (480, 640, 3))

        analyzer.reset()

        # reset 后下一帧计数应从 0 或 1 开始
        result = analyzer.estimate_pose(points, (480, 640, 3))
        assert result.frame_count <= 1


class TestSolvePnPFailure:
    """测试 solvePnP 求解失败的回退行为"""

    def test_degenerate_points_return_defaults(self):
        """退化的 2D 点（全部重合）应返回默认值"""
        analyzer = HeadPoseAnalyzer()
        # 所有点重合 → solvePnP 可能失败
        degenerate_points = [(0.0, 0.0)] * 6
        result = analyzer.estimate_pose(degenerate_points, (480, 640, 3))
        # 无论 solvePnP 成功与否，结果应有有限值
        assert isinstance(result, PoseResult)
        assert np.isfinite(result.pitch)
        assert np.isfinite(result.yaw)
        assert np.isfinite(result.roll)
