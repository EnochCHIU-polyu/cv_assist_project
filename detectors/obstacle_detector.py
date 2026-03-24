"""
障碍物检测模块
==============
基于手部轨迹预测和 MiDaS 深度图，检测前方障碍物并返回警告信息。
"""

import logging
import numpy as np
from typing import Tuple, Optional, List

from .hand_history import HandHistoryBuffer

logger = logging.getLogger(__name__)


class ObstacleDetectionResult:
    """障碍物检测结果"""

    def __init__(self, detected: bool, warning_text: Optional[str] = None,
                 depth_diff: float = 0.0, predicted_point: Optional[Tuple[int, int]] = None,
                 sample_points: Optional[List[Tuple[int, int]]] = None):
        self.detected = detected
        self.warning_text = warning_text
        self.depth_diff = depth_diff
        self.predicted_point = predicted_point
        self.sample_points = sample_points or []

    def __bool__(self):
        return self.detected

    def __repr__(self):
        return f"ObstacleDetectionResult(detected={self.detected}, depth_diff={self.depth_diff:.3f})"


class ObstacleDetector:
    """
    障碍物检测器

    结合手部轨迹历史和 MiDaS 深度图，沿手部运动方向采样深度值，
    检测是否有障碍物（深度值突变点）出现在手部运动轨迹上。

    depth_map 中 depth 值: 0.0=远处, 1.0=近处
    如果预测点的 depth > 手部 depth + threshold，表示前方有物体比手更近 = 障碍物
    """

    def __init__(self,
                 trajectory_frames: int = 10,
                 prediction_distance: float = 50.0,
                 depth_threshold: float = 0.15,
                 warning_cooldown: float = 3.0,
                 sample_count: int = 5,
                 enabled: bool = True):
        self.enabled = enabled
        self.depth_threshold = depth_threshold
        self.prediction_distance = prediction_distance
        self.warning_cooldown = warning_cooldown
        self.sample_count = sample_count

        self.history = HandHistoryBuffer(maxlen=trajectory_frames)
        self._last_warning_time = -1e9  # 初始值为负数，确保首次检测不受冷却限制

        logger.info(f"ObstacleDetector initialized: frames={trajectory_frames}, "
                    f"distance={prediction_distance}px, threshold={depth_threshold}, "
                    f"cooldown={warning_cooldown}s, enabled={enabled}")

    def reset(self) -> None:
        self.history.clear()
        self._last_warning_time = -1e9

    def update(self, hand_center: Tuple[int, int]) -> None:
        self.history.push(hand_center)

    def detect(self,
               depth_map: np.ndarray,
               hand_center: Tuple[int, int],
               current_time: float) -> ObstacleDetectionResult:
        if not self.enabled:
            return ObstacleDetectionResult(False)

        # 检查冷却时间
        if current_time - self._last_warning_time < self.warning_cooldown:
            return ObstacleDetectionResult(False)

        # 更新手部位置历史
        self.update(hand_center)

        # 检查是否有足够的历史帧
        if not self.history.is_ready(min_frames=2):
            return ObstacleDetectionResult(False)

        if depth_map is None or depth_map.size == 0:
            return ObstacleDetectionResult(False)
        img_h, img_w = depth_map.shape[:2]

        # 获取当前手部深度
        hx, hy = hand_center
        hx = max(0, min(img_w - 1, hx))
        hy = max(0, min(img_h - 1, hy))
        hand_depth = float(depth_map[hy, hx])

        # 生成沿运动方向的采样点
        samples = self.history.predict_samples(
            distance=self.prediction_distance,
            num_samples=self.sample_count,
            img_width=img_w,
            img_height=img_h
        )

        if not samples:
            return ObstacleDetectionResult(False)

        # 检查每个采样点的深度
        max_depth_diff = 0.0
        obstacle_detected = False
        obstacle_point = None

        for sx, sy in samples:
            sx = max(0, min(img_w - 1, sx))
            sy = max(0, min(img_h - 1, sy))
            sample_depth = float(depth_map[sy, sx])
            depth_diff = sample_depth - hand_depth

            if depth_diff > self.depth_threshold:
                obstacle_detected = True
                if depth_diff > max_depth_diff:
                    max_depth_diff = depth_diff
                    obstacle_point = (sx, sy)

        # 获取预测的终点位置
        predicted = self.history.predict(self.prediction_distance)
        predicted = (max(0, min(img_w - 1, predicted[0])),
                     max(0, min(img_h - 1, predicted[1])))

        if obstacle_detected:
            self._last_warning_time = current_time
            warning = "前方有障碍物，请小心"
            logger.warning(f"障碍物检测: depth_diff={max_depth_diff:.3f} at {obstacle_point}")
            return ObstacleDetectionResult(
                detected=True,
                warning_text=warning,
                depth_diff=max_depth_diff,
                predicted_point=predicted,
                sample_points=samples
            )

        return ObstacleDetectionResult(
            detected=False,
            predicted_point=predicted,
            sample_points=samples
        )

    def draw(self, frame: np.ndarray, result: ObstacleDetectionResult) -> np.ndarray:
        import cv2
        output = frame.copy()

        # 绘制采样点（绿色小圆点）
        for px, py in result.sample_points:
            cv2.circle(output, (px, py), 3, (0, 255, 0), -1)

        # 绘制预测终点（蓝色圆圈）
        if result.predicted_point:
            cv2.circle(output, result.predicted_point, 8, (255, 0, 0), 2)

        # 如果检测到障碍物，绘制警告
        if result.detected and result.predicted_point:
            cv2.circle(output, result.predicted_point, 12, (0, 0, 255), 3)
            for px, py in result.sample_points:
                cv2.circle(output, (px, py), 5, (0, 0, 255), 1)
            text_pos = (result.predicted_point[0] + 15, result.predicted_point[1])
            cv2.putText(output, "OBSTACLE", text_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return output
