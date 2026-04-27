"""阈值校准模块单元测试"""

import json
import math
import os
import tempfile

import numpy as np
import pytest

from calibration.threshold_calibrator import (
    ThresholdCalibrator,
    compute_stats,
)


class TestComputeStats:
    """测试 compute_stats 辅助函数"""

    def test_basic_values(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = compute_stats(values)
        assert result["mean"] == pytest.approx(3.0)
        assert result["min"] == 1.0
        assert result["max"] == 5.0
        expected_std = math.sqrt(sum((x - 3.0) ** 2 for x in values) / 5)
        assert result["std"] == pytest.approx(expected_std)

    def test_single_value(self):
        result = compute_stats([42.0])
        assert result["mean"] == 42.0
        assert result["std"] == 0.0
        assert result["min"] == 42.0
        assert result["max"] == 42.0

    def test_identical_values(self):
        result = compute_stats([7.0, 7.0, 7.0])
        assert result["mean"] == 7.0
        assert result["std"] == 0.0
        assert result["min"] == 7.0
        assert result["max"] == 7.0

    def test_negative_values(self):
        values = [-3.0, -1.0, 0.0, 1.0, 3.0]
        result = compute_stats(values)
        assert result["mean"] == pytest.approx(0.0)
        assert result["min"] == -3.0
        assert result["max"] == 3.0


class TestComputeStatistics:
    """测试 ThresholdCalibrator.compute_statistics()"""

    def test_ear_statistics(self):
        calibrator = ThresholdCalibrator()
        calibrator._ear_data = [
            (0.3, "normal"),
            (0.35, "normal"),
            (0.25, "normal"),
            (0.1, "closed"),
            (0.12, "closed"),
            (0.08, "closed"),
        ]
        calibrator._mar_data = []

        stats = calibrator.compute_statistics()

        assert "ear" in stats
        assert "normal" in stats["ear"]
        assert "closed" in stats["ear"]
        assert stats["ear"]["normal"]["mean"] == pytest.approx(0.3)
        assert stats["ear"]["closed"]["min"] == 0.08
        assert stats["ear"]["closed"]["max"] == 0.12

    def test_empty_data(self):
        calibrator = ThresholdCalibrator()
        stats = calibrator.compute_statistics()
        assert stats == {"ear": {}, "mar": {}}

    def test_mar_statistics(self):
        calibrator = ThresholdCalibrator()
        calibrator._ear_data = []
        calibrator._mar_data = [
            (0.3, "normal"),
            (0.4, "normal"),
            (0.9, "yawn"),
            (1.0, "yawn"),
        ]

        stats = calibrator.compute_statistics()

        assert "mar" in stats
        assert "normal" in stats["mar"]
        assert "yawn" in stats["mar"]
        assert stats["mar"]["normal"]["mean"] == pytest.approx(0.35)
        assert stats["mar"]["yawn"]["mean"] == pytest.approx(0.95)


class TestOptimizeThresholds:
    """测试 ThresholdCalibrator.optimize_thresholds()"""

    def test_with_separable_ear_data(self):
        calibrator = ThresholdCalibrator()
        # Well-separated EAR data: normal ~0.3, closed ~0.1
        np.random.seed(42)
        normal_ears = np.random.normal(0.3, 0.02, 50).tolist()
        closed_ears = np.random.normal(0.1, 0.02, 50).tolist()
        calibrator._ear_data = [(v, "normal") for v in normal_ears] + [
            (v, "closed") for v in closed_ears
        ]
        calibrator._mar_data = []

        result = calibrator.optimize_thresholds()

        # Optimal threshold should be between the two clusters
        assert 0.05 < result.optimal_ear_threshold < 0.35
        assert result.ear_accuracy > 0.8
        assert result.ear_recall > 0.8

    def test_with_separable_mar_data(self):
        calibrator = ThresholdCalibrator()
        calibrator._ear_data = []
        np.random.seed(42)
        normal_mars = np.random.normal(0.3, 0.05, 50).tolist()
        yawn_mars = np.random.normal(0.9, 0.05, 50).tolist()
        calibrator._mar_data = [(v, "normal") for v in normal_mars] + [
            (v, "yawn") for v in yawn_mars
        ]

        result = calibrator.optimize_thresholds()

        assert 0.3 < result.optimal_mar_threshold < 0.9
        assert result.mar_accuracy > 0.8
        assert result.mar_recall > 0.8

    def test_empty_data_returns_defaults(self):
        calibrator = ThresholdCalibrator()
        result = calibrator.optimize_thresholds()
        assert result.optimal_ear_threshold == 0.2
        assert result.optimal_mar_threshold == 0.75

    def test_returns_calibration_result(self):
        from models.data_models import CalibrationResult

        calibrator = ThresholdCalibrator()
        result = calibrator.optimize_thresholds()
        assert isinstance(result, CalibrationResult)


class TestExportConfig:
    """测试 ThresholdCalibrator.export_config()"""

    def test_export_creates_json(self):
        calibrator = ThresholdCalibrator()
        calibrator._ear_data = [(0.3, "normal"), (0.1, "closed")]
        calibrator._mar_data = [(0.3, "normal"), (0.9, "yawn")]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_config.json")
            calibrator.export_config(output_path)

            assert os.path.exists(output_path)
            with open(output_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            assert "ear_threshold" in config
            assert "mar_threshold" in config
            assert "pitch_threshold" in config
            assert config["pitch_threshold"] == 25.0
            assert config["eye_consec_frames"] == 48
            assert config["mouth_consec_frames"] == 25
            assert config["head_consec_frames"] == 50
            assert "calibration_info" in config
            assert "calibrated_at" in config["calibration_info"]

    def test_export_with_precomputed_result(self):
        calibrator = ThresholdCalibrator()
        calibrator.optimize_thresholds()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "config.json")
            calibrator.export_config(output_path)

            with open(output_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            assert config["ear_threshold"] == 0.2
            assert config["mar_threshold"] == 0.75

    def test_export_nested_directory(self):
        calibrator = ThresholdCalibrator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "sub", "dir", "config.json")
            calibrator.export_config(output_path)
            assert os.path.exists(output_path)


class TestLoadDataset:
    """测试 ThresholdCalibrator.load_dataset() 错误处理"""

    def test_invalid_path_raises(self):
        calibrator = ThresholdCalibrator()
        with pytest.raises(ValueError, match="数据集路径无效"):
            calibrator.load_dataset("/nonexistent/path", "kaggle_ddd")

    def test_invalid_dataset_type_raises(self):
        calibrator = ThresholdCalibrator()
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="不支持的数据集类型"):
                calibrator.load_dataset(tmpdir, "unknown_type")

    def test_empty_dataset_dirs(self):
        """空数据集目录不应崩溃"""
        calibrator = ThresholdCalibrator()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create expected subdirectories but leave them empty
            for subdir in ["open", "closed", "yawn", "no_yawn"]:
                os.makedirs(os.path.join(tmpdir, subdir))
            calibrator.load_dataset(tmpdir, "kaggle_ddd")
            assert len(calibrator._ear_data) == 0
            assert len(calibrator._mar_data) == 0
