"""综合疲劳判断模块"""

from typing import Optional, List

from models.data_models import EyeResult, MouthResult, PoseResult, DLResult, FatigueStatus


class FatigueEvaluator:
    """汇总各检测模块的疲劳信号，输出最终疲劳状态和触发原因。"""

    def evaluate(
        self,
        eye_result: EyeResult,
        mouth_result: MouthResult,
        pose_result: PoseResult,
        dl_result: Optional[DLResult] = None,
        mode: str = "rule",
    ) -> FatigueStatus:
        """
        综合判断疲劳状态。

        Args:
            eye_result: 眼睛分析结果
            mouth_result: 嘴巴分析结果
            pose_result: 头部姿态分析结果
            dl_result: 深度学习分类结果（可选）
            mode: 检测模式 "rule" | "dl" | "hybrid"

        Returns:
            FatigueStatus 包含是否疲劳、原因列表和当前模式
        """
        if mode == "rule":
            is_fatigued, reasons = self._evaluate_rule(eye_result, mouth_result, pose_result)
        elif mode == "dl":
            if dl_result is None:
                # DL 结果不可用时回退到规则模式
                is_fatigued, reasons = self._evaluate_rule(eye_result, mouth_result, pose_result)
            else:
                is_fatigued, reasons = self._evaluate_dl(dl_result)
        elif mode == "hybrid":
            rule_fatigued, rule_reasons = self._evaluate_rule(eye_result, mouth_result, pose_result)
            if dl_result is not None:
                dl_fatigued, dl_reasons = self._evaluate_dl(dl_result)
            else:
                dl_fatigued, dl_reasons = False, []
            is_fatigued = rule_fatigued or dl_fatigued
            reasons = rule_reasons + dl_reasons
        else:
            # 未知模式回退到规则模式
            is_fatigued, reasons = self._evaluate_rule(eye_result, mouth_result, pose_result)

        return FatigueStatus(is_fatigued=is_fatigued, reasons=reasons, mode=mode)

    def _evaluate_rule(
        self,
        eye_result: EyeResult,
        mouth_result: MouthResult,
        pose_result: PoseResult,
    ) -> tuple:
        """规则模式：任一规则信号触发即为疲劳。"""
        reasons: List[str] = []
        if eye_result.is_fatigued:
            reasons.append("闭眼")
        if mouth_result.is_fatigued:
            reasons.append("哈欠")
        if pose_result.is_fatigued:
            reasons.append("低头")
        return len(reasons) > 0, reasons

    def _evaluate_dl(self, dl_result: DLResult) -> tuple:
        """DL 模式：仅依据 DL 分类结果。"""
        reasons: List[str] = []
        if dl_result.eye_label == "closed":
            reasons.append("闭眼(DL)")
        if dl_result.mouth_label == "yawn":
            reasons.append("哈欠(DL)")
        return len(reasons) > 0, reasons
