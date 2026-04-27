"""Tests for main.py DetectionSystem - config loading and initialization logic."""

import json
import sys
from unittest.mock import MagicMock

import pytest

# Mock mediapipe before importing main to avoid hanging
_mp_mock = MagicMock()
sys.modules.setdefault("mediapipe", _mp_mock)
sys.modules.setdefault("mediapipe.solutions", _mp_mock.solutions)
sys.modules.setdefault("mediapipe.solutions.face_mesh", _mp_mock.solutions.face_mesh)

from main import DetectionSystem, _DEFAULTS  # noqa: E402


class TestLoadConfig:
    """Test DetectionSystem._load_config static method."""

    def test_no_config_path_returns_defaults(self):
        config = DetectionSystem._load_config(None)
        assert config == _DEFAULTS

    def test_valid_config_file(self, tmp_path):
        cfg = {
            "ear_threshold": 0.25,
            "mar_threshold": 0.8,
            "pitch_threshold": 30.0,
            "eye_consec_frames": 40,
            "mouth_consec_frames": 20,
            "head_consec_frames": 45,
        }
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps(cfg), encoding="utf-8")

        config = DetectionSystem._load_config(str(cfg_file))
        assert config["ear_threshold"] == 0.25
        assert config["mar_threshold"] == 0.8
        assert config["pitch_threshold"] == 30.0
        assert config["eye_consec_frames"] == 40
        assert config["mouth_consec_frames"] == 20
        assert config["head_consec_frames"] == 45

    def test_missing_config_file_uses_defaults(self, capsys):
        config = DetectionSystem._load_config("/nonexistent/path.json")
        assert config == _DEFAULTS
        captured = capsys.readouterr()
        assert "配置文件不存在" in captured.out

    def test_invalid_json_uses_defaults(self, tmp_path, capsys):
        cfg_file = tmp_path / "bad.json"
        cfg_file.write_text("not valid json {{{", encoding="utf-8")

        config = DetectionSystem._load_config(str(cfg_file))
        assert config == _DEFAULTS
        captured = capsys.readouterr()
        assert "配置文件格式错误" in captured.out

    def test_partial_config_fills_defaults(self, tmp_path):
        cfg = {"ear_threshold": 0.18}
        cfg_file = tmp_path / "partial.json"
        cfg_file.write_text(json.dumps(cfg), encoding="utf-8")

        config = DetectionSystem._load_config(str(cfg_file))
        assert config["ear_threshold"] == 0.18
        assert config["mar_threshold"] == _DEFAULTS["mar_threshold"]
        assert config["pitch_threshold"] == _DEFAULTS["pitch_threshold"]
        assert config["eye_consec_frames"] == _DEFAULTS["eye_consec_frames"]

    def test_null_values_in_config_use_defaults(self, tmp_path):
        cfg = {"ear_threshold": None, "mar_threshold": 0.6}
        cfg_file = tmp_path / "nulls.json"
        cfg_file.write_text(json.dumps(cfg), encoding="utf-8")

        config = DetectionSystem._load_config(str(cfg_file))
        assert config["ear_threshold"] == _DEFAULTS["ear_threshold"]
        assert config["mar_threshold"] == 0.6

    def test_extra_fields_ignored(self, tmp_path):
        cfg = {"ear_threshold": 0.22, "unknown_field": 999}
        cfg_file = tmp_path / "extra.json"
        cfg_file.write_text(json.dumps(cfg), encoding="utf-8")

        config = DetectionSystem._load_config(str(cfg_file))
        assert config["ear_threshold"] == 0.22
        assert "unknown_field" not in config


class TestDetectionSystemInit:
    """Test DetectionSystem initialization."""

    def test_rule_mode_init(self):
        system = DetectionSystem(mode="rule")
        assert system.mode == "rule"
        assert system.dl_classifier is None

    def test_dl_mode_falls_back_to_rule(self, capsys):
        """DL mode should fall back to rule when models are not found."""
        system = DetectionSystem(mode="dl")
        assert system.mode == "rule"
        assert system.dl_classifier is None
        captured = capsys.readouterr()
        assert "DL 模型加载失败" in captured.out or "无法加载 DL 模型" in captured.out

    def test_hybrid_mode_falls_back_to_rule(self):
        """Hybrid mode should fall back to rule when models are not found."""
        system = DetectionSystem(mode="hybrid")
        assert system.mode == "rule"
        assert system.dl_classifier is None

    def test_init_with_config(self, tmp_path):
        cfg = {"ear_threshold": 0.15, "eye_consec_frames": 30}
        cfg_file = tmp_path / "test_config.json"
        cfg_file.write_text(json.dumps(cfg), encoding="utf-8")

        system = DetectionSystem(mode="rule", config_path=str(cfg_file))
        assert system.eye_analyzer.ear_threshold == 0.15
        assert system.eye_analyzer.consec_frames == 30
        assert system.mouth_analyzer.mar_threshold == 0.75


class TestDetectionSystemStop:
    """Test DetectionSystem.stop method."""

    def test_stop_without_camera(self):
        """stop() should not raise even if camera was never opened."""
        system = DetectionSystem(mode="rule")
        system.stop()  # Should not raise
