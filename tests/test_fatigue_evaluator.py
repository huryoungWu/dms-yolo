"""FatigueEvaluator 单元测试"""

import pytest
from models.data_models import EyeResult, MouthResult, PoseResult, DLResult, FatigueStatus
from evaluators.fatigue_evaluator import FatigueEvaluator


@pytest.fixture
def evaluator():
    return FatigueEvaluator()


def _eye(fatigued=False):
    return EyeResult(ear=0.15 if fatigued else 0.25, is_closed=fatigued, is_fatigued=fatigued, frame_count=50 if fatigued else 0)


def _mouth(fatigued=False):
    return MouthResult(mar=0.9 if fatigued else 0.5, is_yawning=fatigued, is_fatigued=fatigued, frame_count=30 if fatigued else 0)


def _pose(fatigued=False):
    return PoseResult(pitch=-30.0 if fatigued else 0.0, yaw=0.0, roll=0.0, is_head_down=fatigued, is_fatigued=fatigued, frame_count=55 if fatigued else 0)


# --- Rule mode tests ---

class TestRuleMode:
    def test_no_fatigue(self, evaluator):
        result = evaluator.evaluate(_eye(), _mouth(), _pose(), mode="rule")
        assert result.is_fatigued is False
        assert result.reasons == []
        assert result.mode == "rule"

    def test_eye_fatigue_only(self, evaluator):
        result = evaluator.evaluate(_eye(True), _mouth(), _pose(), mode="rule")
        assert result.is_fatigued is True
        assert result.reasons == ["闭眼"]

    def test_mouth_fatigue_only(self, evaluator):
        result = evaluator.evaluate(_eye(), _mouth(True), _pose(), mode="rule")
        assert result.is_fatigued is True
        assert result.reasons == ["哈欠"]

    def test_pose_fatigue_only(self, evaluator):
        result = evaluator.evaluate(_eye(), _mouth(), _pose(True), mode="rule")
        assert result.is_fatigued is True
        assert result.reasons == ["低头"]

    def test_all_fatigued(self, evaluator):
        result = evaluator.evaluate(_eye(True), _mouth(True), _pose(True), mode="rule")
        assert result.is_fatigued is True
        assert result.reasons == ["闭眼", "哈欠", "低头"]


# --- DL mode tests ---

class TestDLMode:
    def test_dl_no_fatigue(self, evaluator):
        dl = DLResult(eye_label="open", eye_confidence=0.95, mouth_label="normal", mouth_confidence=0.9)
        result = evaluator.evaluate(_eye(), _mouth(), _pose(), dl_result=dl, mode="dl")
        assert result.is_fatigued is False
        assert result.reasons == []
        assert result.mode == "dl"

    def test_dl_eye_closed(self, evaluator):
        dl = DLResult(eye_label="closed", eye_confidence=0.88, mouth_label="normal", mouth_confidence=0.9)
        result = evaluator.evaluate(_eye(), _mouth(), _pose(), dl_result=dl, mode="dl")
        assert result.is_fatigued is True
        assert result.reasons == ["闭眼(DL)"]

    def test_dl_mouth_yawn(self, evaluator):
        dl = DLResult(eye_label="open", eye_confidence=0.9, mouth_label="yawn", mouth_confidence=0.85)
        result = evaluator.evaluate(_eye(), _mouth(), _pose(), dl_result=dl, mode="dl")
        assert result.is_fatigued is True
        assert result.reasons == ["哈欠(DL)"]

    def test_dl_both(self, evaluator):
        dl = DLResult(eye_label="closed", eye_confidence=0.9, mouth_label="yawn", mouth_confidence=0.85)
        result = evaluator.evaluate(_eye(), _mouth(), _pose(), dl_result=dl, mode="dl")
        assert result.is_fatigued is True
        assert "闭眼(DL)" in result.reasons
        assert "哈欠(DL)" in result.reasons

    def test_dl_none_falls_back_to_rule(self, evaluator):
        """dl_result 为 None 时回退到规则模式"""
        result = evaluator.evaluate(_eye(True), _mouth(), _pose(), dl_result=None, mode="dl")
        assert result.is_fatigued is True
        assert result.reasons == ["闭眼"]
        assert result.mode == "dl"


# --- Hybrid mode tests ---

class TestHybridMode:
    def test_hybrid_no_fatigue(self, evaluator):
        dl = DLResult(eye_label="open", eye_confidence=0.9, mouth_label="normal", mouth_confidence=0.9)
        result = evaluator.evaluate(_eye(), _mouth(), _pose(), dl_result=dl, mode="hybrid")
        assert result.is_fatigued is False
        assert result.reasons == []
        assert result.mode == "hybrid"

    def test_hybrid_rule_only(self, evaluator):
        dl = DLResult(eye_label="open", eye_confidence=0.9, mouth_label="normal", mouth_confidence=0.9)
        result = evaluator.evaluate(_eye(True), _mouth(), _pose(), dl_result=dl, mode="hybrid")
        assert result.is_fatigued is True
        assert "闭眼" in result.reasons

    def test_hybrid_dl_only(self, evaluator):
        dl = DLResult(eye_label="closed", eye_confidence=0.9, mouth_label="normal", mouth_confidence=0.9)
        result = evaluator.evaluate(_eye(), _mouth(), _pose(), dl_result=dl, mode="hybrid")
        assert result.is_fatigued is True
        assert "闭眼(DL)" in result.reasons

    def test_hybrid_both_triggered(self, evaluator):
        dl = DLResult(eye_label="closed", eye_confidence=0.9, mouth_label="yawn", mouth_confidence=0.85)
        result = evaluator.evaluate(_eye(True), _mouth(True), _pose(), dl_result=dl, mode="hybrid")
        assert result.is_fatigued is True
        assert "闭眼" in result.reasons
        assert "哈欠" in result.reasons
        assert "闭眼(DL)" in result.reasons
        assert "哈欠(DL)" in result.reasons

    def test_hybrid_no_dl_result(self, evaluator):
        """hybrid 模式下 dl_result 为 None，仅使用规则"""
        result = evaluator.evaluate(_eye(), _mouth(True), _pose(), dl_result=None, mode="hybrid")
        assert result.is_fatigued is True
        assert result.reasons == ["哈欠"]
