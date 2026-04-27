"""DLClassifier 单元测试：模型文件不存在异常、输入尺寸自动调整、分类逻辑"""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from classifiers.dl_classifier import DLClassifier


def _make_mock_tf(eye_model=None, mouth_model=None):
    """创建 mock TensorFlow 模块，返回指定的 mock 模型"""
    mock_tf = MagicMock()
    models = [eye_model or MagicMock(), mouth_model or MagicMock()]
    mock_tf.keras.models.load_model.side_effect = models
    return mock_tf


class TestDLClassifierInit:
    """测试 __init__ 方法：模型文件不存在时抛出 FileNotFoundError"""

    def test_eye_model_not_found(self, tmp_path):
        mouth_model = tmp_path / "mouth.h5"
        mouth_model.touch()

        with pytest.raises(FileNotFoundError, match="眼部模型文件不存在"):
            DLClassifier(
                eye_model_path=str(tmp_path / "nonexistent_eye.h5"),
                mouth_model_path=str(mouth_model),
            )

    def test_mouth_model_not_found(self, tmp_path):
        eye_model = tmp_path / "eye.h5"
        eye_model.touch()

        with pytest.raises(FileNotFoundError, match="嘴部模型文件不存在"):
            DLClassifier(
                eye_model_path=str(eye_model),
                mouth_model_path=str(tmp_path / "nonexistent_mouth.h5"),
            )

    def test_both_models_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="眼部模型文件不存在"):
            DLClassifier(
                eye_model_path=str(tmp_path / "no_eye.h5"),
                mouth_model_path=str(tmp_path / "no_mouth.h5"),
            )

    def test_successful_init(self, tmp_path):
        eye_path = tmp_path / "eye.h5"
        mouth_path = tmp_path / "mouth.h5"
        eye_path.touch()
        mouth_path.touch()

        with patch("classifiers.dl_classifier._load_tf", return_value=_make_mock_tf()):
            clf = DLClassifier(str(eye_path), str(mouth_path))
            assert clf.eye_model is not None
            assert clf.mouth_model is not None


@pytest.fixture
def classifier_factory(tmp_path):
    """工厂 fixture，创建带 mock 模型的 DLClassifier"""
    def _make(eye_pred=0.5, mouth_pred=0.5):
        eye_path = tmp_path / "eye.h5"
        mouth_path = tmp_path / "mouth.h5"
        eye_path.touch()
        mouth_path.touch()

        eye_model = MagicMock()
        eye_model.predict.return_value = np.array([[eye_pred]])
        mouth_model = MagicMock()
        mouth_model.predict.return_value = np.array([[mouth_pred]])

        mock_tf = _make_mock_tf(eye_model, mouth_model)
        with patch("classifiers.dl_classifier._load_tf", return_value=mock_tf):
            return DLClassifier(str(eye_path), str(mouth_path))
    return _make


class TestDLClassifierPreprocess:
    """测试 _preprocess 方法：灰度转换、resize、归一化"""

    def test_grayscale_64x64_passthrough(self, classifier_factory):
        clf = classifier_factory()
        img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        result = clf._preprocess(img)
        assert result.shape == (1, 64, 64, 1)
        assert result.dtype == np.float32
        assert 0.0 <= result.min() and result.max() <= 1.0

    def test_resize_from_larger_image(self, classifier_factory):
        clf = classifier_factory()
        img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        result = clf._preprocess(img)
        assert result.shape == (1, 64, 64, 1)

    def test_resize_from_smaller_image(self, classifier_factory):
        clf = classifier_factory()
        img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        result = clf._preprocess(img)
        assert result.shape == (1, 64, 64, 1)

    def test_bgr_to_grayscale_conversion(self, classifier_factory):
        clf = classifier_factory()
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        result = clf._preprocess(img)
        assert result.shape == (1, 64, 64, 1)

    def test_normalization_range(self, classifier_factory):
        clf = classifier_factory()
        img = np.full((64, 64), 255, dtype=np.uint8)
        result = clf._preprocess(img)
        assert np.isclose(result.max(), 1.0)

        img_zero = np.zeros((64, 64), dtype=np.uint8)
        result_zero = clf._preprocess(img_zero)
        assert np.isclose(result_zero.min(), 0.0)

    def test_non_square_resize(self, classifier_factory):
        clf = classifier_factory()
        img = np.random.randint(0, 256, (100, 50), dtype=np.uint8)
        result = clf._preprocess(img)
        assert result.shape == (1, 64, 64, 1)


class TestDLClassifierClassifyEye:
    """测试 classify_eye 方法：sigmoid 阈值判断"""

    def test_high_confidence_closed(self, classifier_factory):
        clf = classifier_factory(eye_pred=0.9)
        img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        result = clf.classify_eye(img)
        assert result["label"] == "closed"
        assert np.isclose(result["confidence"], 0.9)

    def test_low_confidence_open(self, classifier_factory):
        clf = classifier_factory(eye_pred=0.2)
        img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        result = clf.classify_eye(img)
        assert result["label"] == "open"
        assert np.isclose(result["confidence"], 0.8)

    def test_boundary_value_open(self, classifier_factory):
        """sigmoid == 0.5 判定为 open"""
        clf = classifier_factory(eye_pred=0.5)
        img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        result = clf.classify_eye(img)
        assert result["label"] == "open"
        assert np.isclose(result["confidence"], 0.5)

    def test_just_above_threshold_closed(self, classifier_factory):
        """sigmoid 刚超过 0.5 判定为 closed"""
        clf = classifier_factory(eye_pred=0.51)
        img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        result = clf.classify_eye(img)
        assert result["label"] == "closed"


class TestDLClassifierClassifyMouth:
    """测试 classify_mouth 方法：sigmoid 阈值判断"""

    def test_high_confidence_yawn(self, classifier_factory):
        clf = classifier_factory(mouth_pred=0.85)
        img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        result = clf.classify_mouth(img)
        assert result["label"] == "yawn"
        assert np.isclose(result["confidence"], 0.85)

    def test_low_confidence_normal(self, classifier_factory):
        clf = classifier_factory(mouth_pred=0.1)
        img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        result = clf.classify_mouth(img)
        assert result["label"] == "normal"
        assert np.isclose(result["confidence"], 0.9)

    def test_boundary_value_normal(self, classifier_factory):
        """sigmoid == 0.5 判定为 normal"""
        clf = classifier_factory(mouth_pred=0.5)
        img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        result = clf.classify_mouth(img)
        assert result["label"] == "normal"
        assert np.isclose(result["confidence"], 0.5)


class TestDLClassifierClassify:
    """测试 classify 便捷方法：返回 DLResult"""

    def test_classify_returns_dl_result(self, classifier_factory):
        clf = classifier_factory(eye_pred=0.8, mouth_pred=0.3)
        eye_img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        mouth_img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        result = clf.classify(eye_img, mouth_img)

        from models.data_models import DLResult
        assert isinstance(result, DLResult)
        assert result.eye_label == "closed"
        assert np.isclose(result.eye_confidence, 0.8)
        assert result.mouth_label == "normal"
        assert np.isclose(result.mouth_confidence, 0.7)

    def test_classify_with_resized_input(self, classifier_factory):
        """非标准尺寸输入也能正常分类"""
        clf = classifier_factory(eye_pred=0.6, mouth_pred=0.7)
        eye_img = np.random.randint(0, 256, (100, 80), dtype=np.uint8)
        mouth_img = np.random.randint(0, 256, (50, 120, 3), dtype=np.uint8)
        result = clf.classify(eye_img, mouth_img)

        assert result.eye_label == "closed"
        assert result.mouth_label == "yawn"
