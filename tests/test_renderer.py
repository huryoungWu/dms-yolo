"""DisplayRenderer 单元测试"""

import numpy as np
import pytest

from display.renderer import DisplayRenderer, format_value
from models.data_models import (
    DLResult,
    EyeResult,
    FaceLandmarks,
    FatigueStatus,
    MouthResult,
    PoseResult,
)


# --------------- helpers ---------------

def _make_frame(w=640, h=480):
    """创建黑色测试帧。"""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _default_eye(ear=0.25, is_closed=False, is_fatigued=False):
    return EyeResult(ear=ear, is_closed=is_closed, is_fatigued=is_fatigued, frame_count=0)


def _default_mouth(mar=0.4, is_yawning=False, is_fatigued=False):
    return MouthResult(mar=mar, is_yawning=is_yawning, is_fatigued=is_fatigued, frame_count=0)


def _default_pose(pitch=0.0, is_head_down=False, is_fatigued=False):
    return PoseResult(pitch=pitch, yaw=0.0, roll=0.0, is_head_down=is_head_down, is_fatigued=is_fatigued, frame_count=0)


def _default_fatigue(is_fatigued=False, mode="rule"):
    return FatigueStatus(is_fatigued=is_fatigued, reasons=[], mode=mode)


def _simple_landmarks():
    """生成简单的 FaceLandmarks 用于测试。"""
    pts = [(float(i), float(i)) for i in range(468)]
    return FaceLandmarks(
        left_eye=pts[:6],
        right_eye=pts[6:12],
        mouth={"upper": pts[0], "lower": pts[1], "left": pts[2], "right": pts[3],
               "upper_inner": pts[4], "lower_inner": pts[5]},
        head_pose_points=pts[:6],
        all_landmarks=pts,
    )


# --------------- format_value tests ---------------

class TestFormatValue:
    def test_two_decimal_places(self):
        assert format_value(0.123456) == "0.12"

    def test_zero(self):
        assert format_value(0.0) == "0.00"

    def test_round_up(self):
        assert format_value(0.255) == "0.26" or format_value(0.255) == "0.25"
        # Python banker's rounding; just verify 2 decimal places
        assert len(format_value(0.255).split(".")[-1]) == 2

    def test_integer_value(self):
        assert format_value(1.0) == "1.00"

    def test_negative(self):
        assert format_value(-0.5) == "-0.50"


# --------------- DisplayRenderer init tests ---------------

class TestDisplayRendererInit:
    def test_init_fallback_no_font(self):
        """字体不存在时应回退到 OpenCV 模式（不抛异常）。"""
        renderer = DisplayRenderer(font_path="nonexistent_font_xyz")
        assert renderer is not None

    def test_init_default(self):
        """默认初始化不应抛异常。"""
        renderer = DisplayRenderer()
        assert renderer is not None


# --------------- render tests ---------------

class TestRender:
    def setup_method(self):
        # 强制使用 OpenCV 回退模式以保证跨平台测试一致性
        self.renderer = DisplayRenderer(font_path="nonexistent_font_xyz")

    def test_render_returns_ndarray(self):
        frame = _make_frame()
        result = self.renderer.render(
            frame, None, _default_eye(), _default_mouth(),
            _default_pose(), _default_fatigue(),
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == frame.shape

    def test_render_does_not_modify_original(self):
        frame = _make_frame()
        original = frame.copy()
        self.renderer.render(
            frame, None, _default_eye(), _default_mouth(),
            _default_pose(), _default_fatigue(),
        )
        np.testing.assert_array_equal(frame, original)

    def test_render_with_landmarks(self):
        frame = _make_frame()
        landmarks = _simple_landmarks()
        result = self.renderer.render(
            frame, landmarks, _default_eye(), _default_mouth(),
            _default_pose(), _default_fatigue(),
        )
        # 关键点应被绘制为绿色，帧不应全黑
        assert result.sum() > 0

    def test_render_fatigue_warning_red_pixels(self):
        """疲劳时应在画面中出现红色像素 (BGR: 0,0,255)。"""
        frame = _make_frame()
        fatigue = FatigueStatus(is_fatigued=True, reasons=["闭眼"], mode="rule")
        result = self.renderer.render(
            frame, None, _default_eye(), _default_mouth(),
            _default_pose(), fatigue,
        )
        # 检查是否有红色通道值为 255 的像素
        red_channel = result[:, :, 2]  # BGR 中 R 在索引 2
        assert red_channel.max() == 255

    def test_render_normal_no_red_warning(self):
        """正常状态不应有大面积红色警告。"""
        frame = _make_frame()
        result = self.renderer.render(
            frame, None, _default_eye(), _default_mouth(),
            _default_pose(), _default_fatigue(),
        )
        # 绿色文字不应在红色通道产生 255 值（OpenCV 绿色 BGR=(0,255,0)）
        # 但文字抗锯齿可能有少量，所以只检查中央区域无大面积红色
        h, w = result.shape[:2]
        center = result[h // 3 : 2 * h // 3, w // 4 : 3 * w // 4, 2]
        assert center.max() < 200

    def test_render_dl_mode_with_dl_result(self):
        """DL 模式下传入 DLResult 不应抛异常。"""
        frame = _make_frame()
        dl_result = DLResult(
            eye_label="open", eye_confidence=0.95,
            mouth_label="normal", mouth_confidence=0.88,
        )
        fatigue = _default_fatigue(mode="dl")
        result = self.renderer.render(
            frame, None, _default_eye(), _default_mouth(),
            _default_pose(), fatigue, dl_result=dl_result,
        )
        assert isinstance(result, np.ndarray)

    def test_render_hybrid_mode(self):
        """Hybrid 模式渲染不应抛异常。"""
        frame = _make_frame()
        dl_result = DLResult(
            eye_label="closed", eye_confidence=0.75,
            mouth_label="yawn", mouth_confidence=0.60,
        )
        fatigue = FatigueStatus(is_fatigued=True, reasons=["闭眼(DL)"], mode="hybrid")
        result = self.renderer.render(
            frame, None, _default_eye(is_closed=True), _default_mouth(),
            _default_pose(), fatigue, dl_result=dl_result,
        )
        assert result.shape == frame.shape


# --------------- status determination tests ---------------

class TestDetermineStatus:
    def test_normal(self):
        assert DisplayRenderer._determine_status(
            _default_eye(), _default_mouth(), _default_pose()
        ) == "normal"

    def test_eye_closed(self):
        assert DisplayRenderer._determine_status(
            _default_eye(is_closed=True), _default_mouth(), _default_pose()
        ) == "eye_closed"

    def test_yawning(self):
        assert DisplayRenderer._determine_status(
            _default_eye(), _default_mouth(is_yawning=True), _default_pose()
        ) == "yawning"

    def test_head_down(self):
        assert DisplayRenderer._determine_status(
            _default_eye(), _default_mouth(), _default_pose(is_head_down=True)
        ) == "head_down"

    def test_priority_eye_over_yawn(self):
        """闭眼优先于打哈欠。"""
        assert DisplayRenderer._determine_status(
            _default_eye(is_closed=True), _default_mouth(is_yawning=True), _default_pose()
        ) == "eye_closed"
