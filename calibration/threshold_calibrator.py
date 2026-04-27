"""阈值校准模块，利用数据集统计分析优化 EAR、MAR 等检测阈值"""

import json
import logging
import math
import os
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, roc_curve

from models.data_models import CalibrationResult

logger = logging.getLogger(__name__)

# 关键点索引（与 face_detector.py 保持一致）
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_INDICES = {
    "upper": 13,
    "lower": 14,
    "left": 78,
    "right": 308,
    "upper_inner": 82,
    "lower_inner": 312,
}


def compute_stats(values: list) -> dict:
    """
    计算一组数值的统计信息。

    Args:
        values: 非空浮点数列表

    Returns:
        {"mean": float, "std": float, "min": float, "max": float}
    """
    n = len(values)
    mean = sum(values) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / n)
    return {
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
    }


def _calculate_ear(eye_points: List[Tuple[float, float]]) -> float:
    """计算单只眼睛的 EAR 值"""
    p1, p2, p3, p4, p5, p6 = eye_points
    vertical_1 = math.dist(p2, p6)
    vertical_2 = math.dist(p3, p5)
    horizontal = math.dist(p1, p4)
    if horizontal == 0.0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def _calculate_mar(mouth_points: dict) -> float:
    """计算 MAR 值"""
    horizontal = math.dist(mouth_points["left"], mouth_points["right"])
    if horizontal == 0.0:
        return 0.0
    vertical_inner = math.dist(mouth_points["upper_inner"], mouth_points["lower_inner"])
    vertical_outer = math.dist(mouth_points["upper"], mouth_points["lower"])
    return (vertical_inner + vertical_outer) / (2.0 * horizontal)


class ThresholdCalibrator:
    """加载数据集，统计 EAR/MAR 分布，通过 ROC 分析输出最优阈值"""

    def __init__(self):
        self._ear_data: List[Tuple[float, str]] = []  # (ear_value, label)
        self._mar_data: List[Tuple[float, str]] = []  # (mar_value, label)
        self._dataset_type: str = ""
        self._calibration_result: Optional[CalibrationResult] = None

    def load_dataset(self, dataset_path: str, dataset_type: str) -> None:
        """
        加载数据集并提取 EAR/MAR 值。

        Args:
            dataset_path: 数据集根目录路径
            dataset_type: "kaggle_ddd" | "mrl_eye"
        """
        if not os.path.isdir(dataset_path):
            raise ValueError(f"数据集路径无效: {dataset_path}")

        self._dataset_type = dataset_type
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            refine_landmarks=False,
        )

        try:
            if dataset_type == "kaggle_ddd":
                self._load_kaggle_ddd(dataset_path, face_mesh)
            elif dataset_type == "mrl_eye":
                self._load_mrl_eye(dataset_path, face_mesh)
            else:
                raise ValueError(f"不支持的数据集类型: {dataset_type}")
        finally:
            face_mesh.close()

        logger.info(
            "数据集加载完成: EAR 样本 %d 条, MAR 样本 %d 条",
            len(self._ear_data),
            len(self._mar_data),
        )

    def _load_kaggle_ddd(self, dataset_path: str, face_mesh) -> None:
        """加载 Kaggle DDD 数据集（open/closed/yawn/no_yawn 子目录）"""
        ear_dirs = {
            "open": "normal",
            "closed": "closed",
        }
        mar_dirs = {
            "no_yawn": "normal",
            "yawn": "yawn",
        }

        for subdir, label in ear_dirs.items():
            dir_path = os.path.join(dataset_path, subdir)
            if not os.path.isdir(dir_path):
                logger.warning("子目录不存在: %s", dir_path)
                continue
            self._process_images_for_ear(dir_path, label, face_mesh)

        for subdir, label in mar_dirs.items():
            dir_path = os.path.join(dataset_path, subdir)
            if not os.path.isdir(dir_path):
                logger.warning("子目录不存在: %s", dir_path)
                continue
            self._process_images_for_mar(dir_path, label, face_mesh)

    def _load_mrl_eye(self, dataset_path: str, face_mesh) -> None:
        """加载 MRL Eye Dataset（open/closed 子目录，仅眼部图像）"""
        ear_dirs = {
            "open": "normal",
            "closed": "closed",
        }

        for subdir, label in ear_dirs.items():
            dir_path = os.path.join(dataset_path, subdir)
            if not os.path.isdir(dir_path):
                logger.warning("子目录不存在: %s", dir_path)
                continue
            self._process_images_for_ear(dir_path, label, face_mesh)

    def _process_images_for_ear(self, dir_path: str, label: str, face_mesh) -> None:
        """处理目录中的图像，提取 EAR 值"""
        for filename in os.listdir(dir_path):
            filepath = os.path.join(dir_path, filename)
            ear = self._extract_ear_from_image(filepath, face_mesh)
            if ear is not None:
                self._ear_data.append((ear, label))

    def _process_images_for_mar(self, dir_path: str, label: str, face_mesh) -> None:
        """处理目录中的图像，提取 MAR 值"""
        for filename in os.listdir(dir_path):
            filepath = os.path.join(dir_path, filename)
            mar = self._extract_mar_from_image(filepath, face_mesh)
            if mar is not None:
                self._mar_data.append((mar, label))

    def _extract_ear_from_image(self, filepath: str, face_mesh) -> Optional[float]:
        """从单张图像提取 EAR 值，无人脸时返回 None"""
        image = cv2.imread(filepath)
        if image is None:
            logger.warning("无法读取图像: %s", filepath)
            return None

        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        face = results.multi_face_landmarks[0]
        landmarks = [(lm.x * w, lm.y * h) for lm in face.landmark]

        left_eye = [landmarks[i] for i in LEFT_EYE_INDICES]
        right_eye = [landmarks[i] for i in RIGHT_EYE_INDICES]

        left_ear = _calculate_ear(left_eye)
        right_ear = _calculate_ear(right_eye)
        return (left_ear + right_ear) / 2.0

    def _extract_mar_from_image(self, filepath: str, face_mesh) -> Optional[float]:
        """从单张图像提取 MAR 值，无人脸时返回 None"""
        image = cv2.imread(filepath)
        if image is None:
            logger.warning("无法读取图像: %s", filepath)
            return None

        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        face = results.multi_face_landmarks[0]
        landmarks = [(lm.x * w, lm.y * h) for lm in face.landmark]

        mouth = {key: landmarks[idx] for key, idx in MOUTH_INDICES.items()}
        return _calculate_mar(mouth)


    def compute_statistics(self) -> dict:
        """
        计算各类别的 EAR/MAR 分布统计。

        Returns:
            {
                "ear": {"normal": {mean, std, min, max}, "closed": {...}},
                "mar": {"normal": {mean, std, min, max}, "yawn": {...}},
            }
        """
        result: dict = {"ear": {}, "mar": {}}

        # EAR 统计（按 label 分组）
        ear_groups = {}  # type: dict
        for value, label in self._ear_data:
            ear_groups.setdefault(label, []).append(value)

        for label, values in ear_groups.items():
            result["ear"][label] = compute_stats(values)

        # MAR 统计（按 label 分组）
        mar_groups = {}  # type: dict
        for value, label in self._mar_data:
            mar_groups.setdefault(label, []).append(value)

        for label, values in mar_groups.items():
            result["mar"][label] = compute_stats(values)

        return result

    def optimize_thresholds(self) -> CalibrationResult:
        """
        基于 ROC 曲线分析输出最优 EAR 和 MAR 阈值。

        使用 Youden's J statistic (max(tpr - fpr)) 确定最优阈值。

        Returns:
            CalibrationResult 包含最优阈值和评估指标
        """
        stats = self.compute_statistics()

        # EAR 优化：label "closed" 为正类 (1)，"normal" 为正常 (0)
        ear_optimal, ear_acc, ear_rec = 0.2, 0.0, 0.0
        if self._ear_data:
            ear_values = np.array([v for v, _ in self._ear_data])
            # 对于 EAR，闭眼时值低，所以用 -EAR 作为 score（值越低越可能闭眼）
            ear_scores = -ear_values
            ear_labels = np.array([1 if lab == "closed" else 0 for _, lab in self._ear_data])

            if len(np.unique(ear_labels)) == 2:
                fpr, tpr, thresholds = roc_curve(ear_labels, ear_scores)
                j_scores = tpr - fpr
                best_idx = np.argmax(j_scores)
                # 阈值是对 -EAR 的，需要取反
                ear_optimal = -thresholds[best_idx]

                # 计算该阈值下的 accuracy 和 recall
                ear_preds = (ear_values < ear_optimal).astype(int)
                ear_acc = float(accuracy_score(ear_labels, ear_preds))
                ear_rec = float(recall_score(ear_labels, ear_preds))

        # MAR 优化：label "yawn" 为正类 (1)，"normal" 为正常 (0)
        mar_optimal, mar_acc, mar_rec = 0.75, 0.0, 0.0
        if self._mar_data:
            mar_values = np.array([v for v, _ in self._mar_data])
            mar_scores = mar_values  # MAR 越高越可能哈欠
            mar_labels = np.array([1 if lab == "yawn" else 0 for _, lab in self._mar_data])

            if len(np.unique(mar_labels)) == 2:
                fpr, tpr, thresholds = roc_curve(mar_labels, mar_scores)
                j_scores = tpr - fpr
                best_idx = np.argmax(j_scores)
                mar_optimal = float(thresholds[best_idx])

                # 计算该阈值下的 accuracy 和 recall
                mar_preds = (mar_values > mar_optimal).astype(int)
                mar_acc = float(accuracy_score(mar_labels, mar_preds))
                mar_rec = float(recall_score(mar_labels, mar_preds))

        self._calibration_result = CalibrationResult(
            optimal_ear_threshold=float(ear_optimal),
            optimal_mar_threshold=float(mar_optimal),
            ear_accuracy=ear_acc,
            ear_recall=ear_rec,
            mar_accuracy=mar_acc,
            mar_recall=mar_rec,
            ear_distribution=stats.get("ear", {}),
            mar_distribution=stats.get("mar", {}),
        )

        return self._calibration_result

    def export_config(self, output_path: str) -> None:
        """
        导出 JSON 配置文件。

        Args:
            output_path: 输出 JSON 文件路径
        """
        if self._calibration_result is None:
            self.optimize_thresholds()

        result = self._calibration_result

        config = {
            "ear_threshold": result.optimal_ear_threshold,
            "mar_threshold": result.optimal_mar_threshold,
            "pitch_threshold": 25.0,
            "eye_consec_frames": 48,
            "mouth_consec_frames": 25,
            "head_consec_frames": 50,
            "calibration_info": {
                "ear_accuracy": result.ear_accuracy,
                "ear_recall": result.ear_recall,
                "mar_accuracy": result.mar_accuracy,
                "mar_recall": result.mar_recall,
                "calibrated_at": datetime.now().isoformat(),
                "dataset": self._dataset_type,
            },
        }

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

        logger.info("配置文件已导出: %s", output_path)
