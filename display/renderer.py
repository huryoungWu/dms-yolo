"""界面渲染模块 - 在视频帧上绘制关键点、数值、状态信息和疲劳警告。"""

from typing import Optional

import cv2
import numpy as np

from models.data_models import (
    DLResult,
    EyeResult,
    FaceLandmarks,
    FatigueStatus,
    MouthResult,
    PoseResult,
)


def format_value(v: float) -> str:
    """格式化浮点数为两位小数字符串。"""
    return f"{v:.2f}"


class DisplayRenderer:
    """在视频帧上绘制检测结果、状态信息和疲劳警告。"""

    # 模式名称映射
    _MODE_NAMES = {
        "rule": "规则",
        "dl": "深度学习",
        "hybrid": "混合",
    }

    # 状态文字映射
    _STATUS_TEXT = {
        "normal": "正常",
        "eye_closed": "闭眼",
        "yawning": "打哈欠",
        "head_down": "低头",
    }

    def __init__(self, font_path: str = "SimHei"):
        """初始化中文字体，字体不存在时回退到 OpenCV 默认英文字体。"""
        self._pil_font = None
        self._pil_font_large = None
        self._use_pil = False

        try:
            from PIL import ImageFont

            # 尝试加载字体
            font = self._try_load_font(font_path)
            if font is not None:
                self._pil_font = font
                self._pil_font_large = ImageFont.truetype(
                    font.path, 48
                )
                self._use_pil = True
        except Exception:
            self._use_pil = False

    @staticmethod
    def _try_load_font(font_path: str):
        """尝试加载字体文件，返回 PIL ImageFont 或 None。"""
        from PIL import ImageFont

        # 直接尝试给定路径
        try:
            return ImageFont.truetype(font_path, 20)
        except (OSError, IOError):
            pass

        # 常见系统路径
        common_paths = [
            "/usr/share/fonts/truetype/simhei/SimHei.ttf",
            "/usr/share/fonts/SimHei.ttf",
            "C:\\Windows\\Fonts\\simhei.ttf",
            "/System/Library/Fonts/STHeiti Medium.ttc",
        ]
        for path in common_paths:
            try:
                return ImageFont.truetype(path, 20)
            except (OSError, IOError):
                continue

        return None

    def render(
        self,
        frame: np.ndarray,
        landmarks: Optional[FaceLandmarks],
        eye_result: EyeResult,
        mouth_result: MouthResult,
        pose_result: PoseResult,
        fatigue_status: FatigueStatus,
        dl_result: Optional[DLResult] = None,
    ) -> np.ndarray:
        """渲染检测结果到视频帧，返回渲染后的帧图像。"""
        output = frame.copy()

        # 绘制人脸关键点
        if landmarks is not None:
            self._draw_landmarks(output, landmarks)

        # 绘制 EAR/MAR 数值和状态
        status_key = self._determine_status(eye_result, mouth_result, pose_result)
        self._draw_info(output, eye_result, mouth_result, status_key)

        # 绘制检测模式（右上角）
        self._draw_mode(output, fatigue_status.mode)

        # DL/hybrid 模式下显示置信度
        if fatigue_status.mode in ("dl", "hybrid") and dl_result is not None:
            self._draw_dl_confidence(output, dl_result)

        # 疲劳警告
        if fatigue_status.is_fatigued:
            self._draw_fatigue_warning(output)

        return output

    @staticmethod
    def _draw_landmarks(frame: np.ndarray, landmarks: FaceLandmarks) -> None:
        """绘制人脸关键点（绿色小圆点）。"""
        for x, y in landmarks.all_landmarks:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

    @staticmethod
    def _determine_status(
        eye_result: EyeResult,
        mouth_result: MouthResult,
        pose_result: PoseResult,
    ) -> str:
        """根据检测结果确定当前状态键。"""
        if eye_result.is_closed:
            return "eye_closed"
        if mouth_result.is_yawning:
            return "yawning"
        if pose_result.is_head_down:
            return "head_down"
        return "normal"

    def _draw_info(
        self,
        frame: np.ndarray,
        eye_result: EyeResult,
        mouth_result: MouthResult,
        status_key: str,
    ) -> None:
        """在左上角绘制 EAR、MAR 数值和状态文字。"""
        ear_text = f"EAR: {format_value(eye_result.ear)}"
        mar_text = f"MAR: {format_value(mouth_result.mar)}"

        if self._use_pil:
            status_label = self._STATUS_TEXT.get(status_key, "正常")
            status_text = f"状态: {status_label}"
            lines = [ear_text, mar_text, status_text]
            self._draw_pil_lines(frame, lines, x=10, y_start=30, color=(0, 255, 0))
        else:
            # 英文回退
            status_map = {
                "normal": "Normal",
                "eye_closed": "Eye Closed",
                "yawning": "Yawning",
                "head_down": "Head Down",
            }
            status_text = f"Status: {status_map.get(status_key, 'Normal')}"
            y = 30
            for text in [ear_text, mar_text, status_text]:
                cv2.putText(
                    frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )
                y += 30

    def _draw_mode(self, frame: np.ndarray, mode: str) -> None:
        """在右上角绘制检测模式。"""
        h, w = frame.shape[:2]

        if self._use_pil:
            mode_name = self._MODE_NAMES.get(mode, mode)
            mode_text = f"模式: {mode_name}"
            self._draw_pil_lines(frame, [mode_text], x=w - 180, y_start=30, color=(255, 255, 0))
        else:
            mode_map = {"rule": "Rule", "dl": "DL", "hybrid": "Hybrid"}
            mode_text = f"Mode: {mode_map.get(mode, mode)}"
            cv2.putText(
                frame, mode_text, (w - 180, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
            )

    def _draw_dl_confidence(self, frame: np.ndarray, dl_result: DLResult) -> None:
        """显示 DL 分类置信度。"""
        eye_text = f"Eye DL: {format_value(dl_result.eye_confidence)}"
        mouth_text = f"Mouth DL: {format_value(dl_result.mouth_confidence)}"
        h, w = frame.shape[:2]

        if self._use_pil:
            self._draw_pil_lines(
                frame, [eye_text, mouth_text],
                x=w - 220, y_start=60, color=(255, 255, 0),
            )
        else:
            cv2.putText(
                frame, eye_text, (w - 220, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
            )
            cv2.putText(
                frame, mouth_text, (w - 220, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
            )

    def _draw_fatigue_warning(self, frame: np.ndarray) -> None:
        """在画面中央显示红色大字体疲劳警告。"""
        h, w = frame.shape[:2]
        warning = "疲劳驾驶！请休息！"

        if self._use_pil:
            from PIL import Image, ImageDraw

            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            bbox = draw.textbbox((0, 0), warning, font=self._pil_font_large)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            x = (w - text_w) // 2
            y = (h - text_h) // 2
            draw.text((x, y), warning, font=self._pil_font_large, fill=(255, 0, 0))
            result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            frame[:] = result
        else:
            warning_en = "FATIGUE! PLEASE REST!"
            font_scale = 1.5
            thickness = 3
            (text_w, text_h), _ = cv2.getTextSize(
                warning_en, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            x = (w - text_w) // 2
            y = (h + text_h) // 2
            cv2.putText(
                frame, warning_en, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness,
            )

    def _draw_pil_lines(
        self,
        frame: np.ndarray,
        lines: list,
        x: int,
        y_start: int,
        color: tuple,
    ) -> None:
        """使用 PIL 在帧上绘制多行文字（BGR color -> RGB fill）。"""
        from PIL import Image, ImageDraw

        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        # BGR -> RGB for PIL fill
        fill = (color[2], color[1], color[0])
        y = y_start
        for line in lines:
            draw.text((x, y), line, font=self._pil_font, fill=fill)
            y += 28
        result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        frame[:] = result
