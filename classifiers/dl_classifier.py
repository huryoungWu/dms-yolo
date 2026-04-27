"""深度学习分类模块，基于 CNN 模型对眼部/嘴部图像进行疲劳状态分类"""

import os

import numpy as np

from models.data_models import DLResult


def _load_tf():
    """延迟加载 TensorFlow，处理导入错误"""
    try:
        import tensorflow as tf
        return tf
    except ImportError as e:
        print("警告: 无法导入 TensorFlow，深度学习分类模块不可用")
        raise ImportError(
            "TensorFlow 未安装，请运行 pip install tensorflow 安装"
        ) from e


class DLClassifier:
    """使用 CNN 模型对眼部/嘴部裁剪图像进行二分类"""

    TARGET_SIZE = (64, 64)

    def __init__(self, eye_model_path: str, mouth_model_path: str):
        """
        加载预训练的眼部和嘴部 CNN 模型。

        Args:
            eye_model_path: 眼部模型 .h5 文件路径
            mouth_model_path: 嘴部模型 .h5 文件路径

        Raises:
            FileNotFoundError: 模型文件不存在时抛出
        """
        if not os.path.exists(eye_model_path):
            raise FileNotFoundError(
                f"眼部模型文件不存在: {eye_model_path}"
            )
        if not os.path.exists(mouth_model_path):
            raise FileNotFoundError(
                f"嘴部模型文件不存在: {mouth_model_path}"
            )

        tf = _load_tf()
        self.eye_model = tf.keras.models.load_model(eye_model_path)
        self.mouth_model = tf.keras.models.load_model(mouth_model_path)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        预处理输入图像：灰度转换、resize、归一化、扩展维度。

        Args:
            image: 输入图像（灰度或 BGR）

        Returns:
            形状为 (1, 64, 64, 1) 的归一化数组
        """
        import cv2

        img = image.copy()

        # 转灰度（如果是彩色图像）
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # resize 到目标尺寸
        if img.shape[:2] != self.TARGET_SIZE:
            img = cv2.resize(img, self.TARGET_SIZE)

        # 归一化到 [0, 1]
        img = img.astype(np.float32) / 255.0

        # 扩展维度: (64, 64) -> (1, 64, 64, 1)
        img = np.expand_dims(img, axis=(0, -1))

        return img

    def classify_eye(self, eye_image: np.ndarray) -> dict:
        """
        分类眼部状态。

        Args:
            eye_image: 灰度或 BGR 眼部图像

        Returns:
            {"label": "open"|"closed", "confidence": float}
        """
        preprocessed = self._preprocess(eye_image)
        prediction = self.eye_model.predict(preprocessed, verbose=0)
        confidence = float(prediction[0][0])

        if confidence > 0.5:
            label = "closed"
        else:
            label = "open"
            confidence = 1.0 - confidence

        return {"label": label, "confidence": confidence}

    def classify_mouth(self, mouth_image: np.ndarray) -> dict:
        """
        分类嘴部状态。

        Args:
            mouth_image: 灰度或 BGR 嘴部图像

        Returns:
            {"label": "normal"|"yawn", "confidence": float}
        """
        preprocessed = self._preprocess(mouth_image)
        prediction = self.mouth_model.predict(preprocessed, verbose=0)
        confidence = float(prediction[0][0])

        if confidence > 0.5:
            label = "yawn"
        else:
            label = "normal"
            confidence = 1.0 - confidence

        return {"label": label, "confidence": confidence}

    def classify(self, eye_image: np.ndarray, mouth_image: np.ndarray) -> DLResult:
        """
        同时分类眼部和嘴部状态。

        Args:
            eye_image: 眼部图像
            mouth_image: 嘴部图像

        Returns:
            DLResult 包含眼部和嘴部分类结果
        """
        eye_result = self.classify_eye(eye_image)
        mouth_result = self.classify_mouth(mouth_image)

        return DLResult(
            eye_label=eye_result["label"],
            eye_confidence=eye_result["confidence"],
            mouth_label=mouth_result["label"],
            mouth_confidence=mouth_result["confidence"],
        )
