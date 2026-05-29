"""人脸关键点检测模块，基于 MediaPipe FaceMesh"""

from typing import Optional
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.distance import euclidean
from models.data_models import FaceLandmarks

# 关键点索引常量
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

HEAD_POSE_INDICES = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_corner": 33,
    "right_eye_corner": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}


class FaceDetector:
    """使用 MediaPipe FaceMesh 检测人脸关键点"""

    def __init__(
        self,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        stability_threshold: float = 50.0,  # 关键点位置变化阈值（像素）
        min_face_size_ratio: float = 0.1,   # 最小人脸占画面比例
        max_face_shift_ratio: float = 0.3,  # 最大人脸移动占画面比例
    ):
        """初始化 MediaPipe FaceMesh。兼容不同 mediapipe 发行包结构。"""
        try:
            face_mesh_cls = mp.solutions.face_mesh.FaceMesh
        except AttributeError:
            # 某些发行包不在顶层暴露 mp.solutions，但仍可通过 python.solutions 使用
            from mediapipe.python.solutions.face_mesh import FaceMesh as _FaceMesh

            face_mesh_cls = _FaceMesh

        self._face_mesh = face_mesh_cls(
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
            refine_landmarks=False,
        )
        
        # 人脸稳定性检测参数
        self.stability_threshold = stability_threshold
        self.min_face_size_ratio = min_face_size_ratio
        self.max_face_shift_ratio = max_face_shift_ratio
        
        # 存储上一帧的人脸信息
        self._previous_landmarks = None
        self._previous_face_center = None
        self._previous_face_size = None
        self._stable_face_counter = 0
        self._max_stable_face_counter = 5  # 连续稳定帧数阈值

    def _calculate_face_center_and_size(self, landmarks):
        """计算人脸中心点和大小"""
        if not landmarks:
            return None, 0
        
        # 计算所有关键点的边界框
        xs = [pt[0] for pt in landmarks]
        ys = [pt[1] for pt in landmarks]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # 中心点
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        # 人脸大小（对角线长度）
        size = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)
        
        return (center_x, center_y), size

    def _calculate_landmarks_distance(self, landmarks1, landmarks2):
        """计算两组关键点之间的平均距离"""
        if len(landmarks1) != len(landmarks2):
            return float('inf')
        
        distances = []
        for pt1, pt2 in zip(landmarks1, landmarks2):
            dist = euclidean(pt1, pt2)
            distances.append(dist)
        
        return np.mean(distances)

    def _is_face_stable(self, current_landmarks, frame_shape):
        """检查当前人脸是否稳定，防止突然的人脸切换"""
        h, w = frame_shape[:2]
        
        # 计算当前人脸的中心和大小
        current_center, current_size = self._calculate_face_center_and_size(current_landmarks)
        
        if self._previous_landmarks is None:
            # 第一帧，存储信息并认为是稳定的
            self._previous_landmarks = current_landmarks[:]
            self._previous_face_center = current_center
            self._previous_face_size = current_size
            self._stable_face_counter = 1
            return True
        
        # 检查人脸大小变化
        if self._previous_face_size > 0:
            size_change_ratio = abs(current_size - self._previous_face_size) / self._previous_face_size
            max_size_change = self.max_face_shift_ratio
            
            if size_change_ratio > max_size_change:
                return False
        
        # 检查人脸位置变化
        if self._previous_face_center is not None and current_center is not None:
            position_change = euclidean(self._previous_face_center, current_center)
            max_position_change = min(h, w) * self.max_face_shift_ratio  # 基于画面尺寸的比例
            
            if position_change > max_position_change:
                return False
        
        # 检查关键点整体变化
        avg_landmark_distance = self._calculate_landmarks_distance(
            self._previous_landmarks, current_landmarks
        )
        
        if avg_landmark_distance > self.stability_threshold:
            return False
        
        # 更新存储的信息
        self._previous_landmarks = current_landmarks[:]
        self._previous_face_center = current_center
        self._previous_face_size = current_size
        self._stable_face_counter += 1
        
        return True

    def detect(self, frame: np.ndarray) -> Optional[FaceLandmarks]:
        """
        检测单帧图像中的人脸关键点。

        Args:
            frame: BGR 格式的 OpenCV 图像帧

        Returns:
            FaceLandmarks 对象，包含各区域关键点；未检测到人脸时返回 None
        """
        h, w = frame.shape[:2]
        screen_center = (w / 2, h / 2)

        # BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        results = self._face_mesh.process(rgb_frame)
       # 修改代码以打印所有脸的信息
        if not results.multi_face_landmarks:
            print("未检测到人脸")
            # 如果没检测到人脸，清空之前的存储信息
            self._previous_landmarks = None
            self._previous_face_center = None
            self._previous_face_size = None
            self._stable_face_counter = 0
            return None

        print(f'检测到人脸数量: {len(results.multi_face_landmarks)}')

        # 打印每个人脸的中心点坐标和距离屏幕中心的距离
        screen_center = (w / 2, h / 2)
        for i, face_landmarks in enumerate(results.multi_face_landmarks):
            # 将归一化坐标转换为像素坐标
            all_landmarks = [
                (lm.x * w, lm.y * h) for lm in face_landmarks.landmark
            ]
            
            # 计算人脸中心点
            xs = [pt[0] for pt in all_landmarks]
            ys = [pt[1] for pt in all_landmarks]
            face_center_x = (min(xs) + max(xs)) / 2
            face_center_y = (min(ys) + max(ys)) / 2
            face_center = (face_center_x, face_center_y)
            
            # 计算到屏幕中心的距离
            distance_to_center = euclidean(face_center, screen_center)
            
            # print(f'人脸 {i+1}: 中心点坐标 ({face_center_x:.2f}, {face_center_y:.2f}), '
            #     f'距离屏幕中心({screen_center[0]:.2f}, {screen_center[1]:.2f}) {distance_to_center:.2f} 像素')

        # 从多个检测到的人脸中选择最可能的
        best_face = None
        best_score = -1
        best_all_landmarks = None
        
        for face_landmarks in results.multi_face_landmarks:
            # 将归一化坐标转换为像素坐标
            all_landmarks = [
                (lm.x * w, lm.y * h) for lm in face_landmarks.landmark
            ]
            
            # 计算人脸中心点
            xs = [pt[0] for pt in all_landmarks]
            ys = [pt[1] for pt in all_landmarks]
            face_center_x = (min(xs) + max(xs)) / 2
            face_center_y = (min(ys) + max(ys)) / 2
            face_center = (face_center_x, face_center_y)
            
            # 计算人脸大小（边界框对角线）
            face_width = max(xs) - min(xs)
            face_height = max(ys) - min(ys)
            face_size = np.sqrt(face_width**2 + face_height**2)
            
            # 计算到屏幕中心的距离
            distance_to_center = euclidean(face_center, screen_center)
            
            # 基于多个因素计算综合得分
            # 1. 人脸大小（越大越好）
            size_ratio = face_size / max(w, h)
            
            # 2. 距离屏幕中心的远近（越近越好）
            normalized_distance = distance_to_center / np.sqrt(w**2 + h**2)  # 归一化到0-1
            center_bonus = (1 - normalized_distance) * 0.5  # 给予中心位置奖励
            
            # 3. 人脸大小奖励（优先考虑较大的人脸）
            size_bonus = size_ratio * 0.5
            
            # 综合得分：大小越大越好，中心位置越近越好
            score = size_bonus + center_bonus
            
            if score > best_score:
                best_score = score
                best_face = face_landmarks
                best_all_landmarks = all_landmarks

        # 检查最佳人脸的稳定性
        if best_face is not None:
            # 检查人脸是否足够大
            xs = [pt[0] for pt in best_all_landmarks]
            ys = [pt[1] for pt in best_all_landmarks]
            face_width = max(xs) - min(xs)
            face_height = max(ys) - min(ys)
            face_size = np.sqrt(face_width**2 + face_height**2)
            
            min_required_size = min(h, w) * self.min_face_size_ratio
            if face_size < min_required_size:
                print(f"人脸太小: {face_size:.2f} < {min_required_size:.2f}")
                return None
            
            # 检查人脸稳定性
            if not self._is_face_stable(best_all_landmarks, frame.shape):
                # 返回之前稳定的人脸信息，如果没有则返回None
                if self._stable_face_counter >= self._max_stable_face_counter:
                    # 如果之前已经稳定了一段时间，继续使用之前的信息
                    pass
                return None

        face = best_face

        if face is None:
            return None

        # 将归一化坐标转换为像素坐标（使用已计算的best_all_landmarks）
        all_landmarks = best_all_landmarks

        # 提取眼睛关键点
        left_eye = [all_landmarks[i] for i in LEFT_EYE_INDICES]
        right_eye = [all_landmarks[i] for i in RIGHT_EYE_INDICES]

        # 提取嘴巴关键点
        mouth = {
            key: all_landmarks[idx] for key, idx in MOUTH_INDICES.items()
        }

        # 提取头部姿态关键点（按固定顺序）
        head_pose_points = [
            all_landmarks[HEAD_POSE_INDICES["nose_tip"]],
            all_landmarks[HEAD_POSE_INDICES["chin"]],
            all_landmarks[HEAD_POSE_INDICES["left_eye_corner"]],
            all_landmarks[HEAD_POSE_INDICES["right_eye_corner"]],
            all_landmarks[HEAD_POSE_INDICES["left_mouth"]],
            all_landmarks[HEAD_POSE_INDICES["right_mouth"]],
        ]

        return FaceLandmarks(
            left_eye=left_eye,
            right_eye=right_eye,
            mouth=mouth,
            head_pose_points=head_pose_points,
            all_landmarks=all_landmarks,
        )

    def reset_stability_tracker(self):
        """重置人脸稳定性跟踪器，用于重新开始跟踪"""
        self._previous_landmarks = None
        self._previous_face_center = None
        self._previous_face_size = None
        self._stable_face_counter = 0

    def close(self):
        """释放 MediaPipe 资源"""
        self._face_mesh.close()