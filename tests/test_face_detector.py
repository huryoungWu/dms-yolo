"""FaceDetector 单元测试"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from detectors.face_detector import (
    HEAD_POSE_INDICES,
    LEFT_EYE_INDICES,
    MOUTH_INDICES,
    RIGHT_EYE_INDICES,
    FaceDetector,
)
from models.data_models import FaceLandmarks

PATCH_TARGET = "detectors.face_detector.mp.solutions.face_mesh.FaceMesh"


def _make_fake_landmark(x: float, y: float):
    """创建一个模拟的 MediaPipe landmark 对象"""
    lm = MagicMock()
    lm.x = x
    lm.y = y
    return lm


def _build_fake_results(num_landmarks: int = 468, w: int = 640, h: int = 480):
    """构建模拟的 MediaPipe FaceMesh 处理结果（归一化坐标）"""
    landmarks = []
    for i in range(num_landmarks):
        nx = (i % 100) / 100.0
        ny = (i // 100) / 100.0
        landmarks.append(_make_fake_landmark(nx, ny))

    face = MagicMock()
    face.landmark = landmarks

    results = MagicMock()
    results.multi_face_landmarks = [face]
    return results, landmarks


class TestFaceDetectorDetect:
    """测试 detect() 方法"""

    @patch(PATCH_TARGET)
    def test_returns_none_when_no_face(self, mock_mesh_cls):
        """未检测到人脸时返回 None"""
        mock_mesh = MagicMock()
        mock_mesh_cls.return_value = mock_mesh

        no_face_results = MagicMock()
        no_face_results.multi_face_landmarks = None
        mock_mesh.process.return_value = no_face_results

        detector = FaceDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert result is None

    @patch(PATCH_TARGET)
    def test_returns_face_landmarks_when_face_detected(self, mock_mesh_cls):
        """检测到人脸时返回 FaceLandmarks 对象"""
        mock_mesh = MagicMock()
        mock_mesh_cls.return_value = mock_mesh

        w, h = 640, 480
        results, landmarks = _build_fake_results(468, w, h)
        mock_mesh.process.return_value = results

        detector = FaceDetector()
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert result is not None
        assert isinstance(result, FaceLandmarks)

    @patch(PATCH_TARGET)
    def test_left_eye_has_6_points(self, mock_mesh_cls):
        """左眼应包含 6 个关键点"""
        mock_mesh = MagicMock()
        mock_mesh_cls.return_value = mock_mesh

        results, _ = _build_fake_results()
        mock_mesh.process.return_value = results

        detector = FaceDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert len(result.left_eye) == 6

    @patch(PATCH_TARGET)
    def test_right_eye_has_6_points(self, mock_mesh_cls):
        """右眼应包含 6 个关键点"""
        mock_mesh = MagicMock()
        mock_mesh_cls.return_value = mock_mesh

        results, _ = _build_fake_results()
        mock_mesh.process.return_value = results

        detector = FaceDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert len(result.right_eye) == 6

    @patch(PATCH_TARGET)
    def test_mouth_has_all_keys(self, mock_mesh_cls):
        """嘴巴应包含所有 6 个关键点键"""
        mock_mesh = MagicMock()
        mock_mesh_cls.return_value = mock_mesh

        results, _ = _build_fake_results()
        mock_mesh.process.return_value = results

        detector = FaceDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)

        expected_keys = {"upper", "lower", "left", "right", "upper_inner", "lower_inner"}
        assert set(result.mouth.keys()) == expected_keys

    @patch(PATCH_TARGET)
    def test_head_pose_has_6_points(self, mock_mesh_cls):
        """头部姿态应包含 6 个关键点"""
        mock_mesh = MagicMock()
        mock_mesh_cls.return_value = mock_mesh

        results, _ = _build_fake_results()
        mock_mesh.process.return_value = results

        detector = FaceDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert len(result.head_pose_points) == 6

    @patch(PATCH_TARGET)
    def test_all_landmarks_has_468_points(self, mock_mesh_cls):
        """all_landmarks 应包含全部 468 个关键点"""
        mock_mesh = MagicMock()
        mock_mesh_cls.return_value = mock_mesh

        results, _ = _build_fake_results()
        mock_mesh.process.return_value = results

        detector = FaceDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert len(result.all_landmarks) == 468

    @patch(PATCH_TARGET)
    def test_landmark_coordinates_are_pixel_values(self, mock_mesh_cls):
        """关键点坐标应为像素坐标（x*w, y*h）"""
        mock_mesh = MagicMock()
        mock_mesh_cls.return_value = mock_mesh

        w, h = 640, 480
        results, raw_landmarks = _build_fake_results(468, w, h)
        mock_mesh.process.return_value = results

        detector = FaceDetector()
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        result = detector.detect(frame)

        # 验证第一个左眼关键点的像素坐标
        idx = LEFT_EYE_INDICES[0]
        expected_x = raw_landmarks[idx].x * w
        expected_y = raw_landmarks[idx].y * h
        assert result.left_eye[0] == pytest.approx((expected_x, expected_y))


class TestFaceDetectorClose:
    """测试 close() 方法"""

    @patch(PATCH_TARGET)
    def test_close_releases_resources(self, mock_mesh_cls):
        """close() 应调用 FaceMesh.close()"""
        mock_mesh = MagicMock()
        mock_mesh_cls.return_value = mock_mesh

        detector = FaceDetector()
        detector.close()

        mock_mesh.close.assert_called_once()


class TestLandmarkIndices:
    """验证关键点索引常量的正确性"""

    def test_left_eye_indices(self):
        assert LEFT_EYE_INDICES == [33, 160, 158, 133, 153, 144]

    def test_right_eye_indices(self):
        assert RIGHT_EYE_INDICES == [362, 385, 387, 263, 373, 380]

    def test_mouth_indices(self):
        assert MOUTH_INDICES == {
            "upper": 13,
            "lower": 14,
            "left": 78,
            "right": 308,
            "upper_inner": 82,
            "lower_inner": 312,
        }

    def test_head_pose_indices(self):
        assert HEAD_POSE_INDICES == {
            "nose_tip": 1,
            "chin": 152,
            "left_eye_corner": 33,
            "right_eye_corner": 263,
            "left_mouth": 61,
            "right_mouth": 291,
        }
