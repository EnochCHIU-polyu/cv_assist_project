"""
手部轨迹历史缓冲
================
追踪手部中心点的历史位置，用于计算运动轨迹和预测前方障碍物。
"""

import logging
from collections import deque
from typing import List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class HandHistoryBuffer:
    """
    手部中心点的历史位置缓冲区

    使用固定大小的环形缓冲区 (deque) 存储最近 N 帧的手部中心点坐标。
    支持计算速度向量、预测未来位置、以及在预测位置采样深度值。
    """

    def __init__(self, maxlen: int = 10):
        self.maxlen = max(2, int(maxlen))
        self._positions: deque = deque(maxlen=self.maxlen)

    def push(self, center: Tuple[int, int]) -> None:
        self._positions.append((int(center[0]), int(center[1])))

    def clear(self) -> None:
        self._positions.clear()

    @property
    def size(self) -> int:
        return len(self._positions)

    def is_ready(self, min_frames: int = 2) -> bool:
        return len(self._positions) >= min_frames

    def get_positions(self) -> List[Tuple[int, int]]:
        return list(self._positions)

    def velocity(self) -> np.ndarray:
        if len(self._positions) < 2:
            return np.array([0.0, 0.0])
        prev = self._positions[-2]
        curr = self._positions[-1]
        dx = float(curr[0] - prev[0])
        dy = float(curr[1] - prev[1])
        return np.array([dx, dy])

    def predict(self, distance: float) -> Tuple[int, int]:
        if not self._positions:
            return (0, 0)
        vel = self.velocity()
        norm = np.linalg.norm(vel)
        if norm < 1e-6:
            return self._positions[-1]
        direction = vel / norm
        offset = direction * distance
        curr = self._positions[-1]
        predicted_x = int(curr[0] + offset[0])
        predicted_y = int(curr[1] + offset[1])
        return (predicted_x, predicted_y)

    def predict_samples(self, distance: float, num_samples: int = 5,
                        img_width: Optional[int] = None,
                        img_height: Optional[int] = None) -> List[Tuple[int, int]]:
        if not self._positions:
            return []
        vel = self.velocity()
        norm = np.linalg.norm(vel)
        if norm < 1e-6:
            return [self._positions[-1]]
        direction = vel / norm
        curr = self._positions[-1]
        samples = []
        for i in range(1, num_samples + 1):
            frac = i / (num_samples + 1)
            offset = direction * distance * frac
            sx = int(curr[0] + offset[0])
            sy = int(curr[1] + offset[1])
            if img_width is not None:
                sx = max(0, min(img_width - 1, sx))
            if img_height is not None:
                sy = max(0, min(img_height - 1, sy))
            samples.append((sx, sy))
        return samples

    def __len__(self) -> int:
        return len(self._positions)

    def __repr__(self) -> str:
        return f"HandHistoryBuffer(maxlen={self.maxlen}, size={self.size})"
