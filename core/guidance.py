"""
引导控制器
==========
根据手部和目标位置生成导航指令。
"""

import cv2
import numpy as np
from typing import Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GuidanceResult:
    """引导结果"""
    instruction: str
    direction_h: str
    direction_v: str
    direction_d: str
    dx: int
    dy: int
    depth_diff: float
    ready_to_grab: bool


class GuidanceController:
    """引导控制器"""
    
    def __init__(self,
                 horizontal_threshold: int = 30,
                 vertical_threshold: int = 30,
                 depth_threshold: float = 0.15):
        self.h_thresh = horizontal_threshold
        self.v_thresh = vertical_threshold
        self.d_thresh = depth_threshold
        
        logger.info(f"引导控制器初始化: h={horizontal_threshold}, v={vertical_threshold}, d={depth_threshold}")
    
    def calculate(self,
                  hand_center: Tuple[int, int],
                  target_center: Tuple[int, int],
                  hand_depth: float,
                  target_depth: float,
                  gesture: str = 'unknown') -> GuidanceResult:
        dx = target_center[0] - hand_center[0]
        dy = target_center[1] - hand_center[1]
        dd = target_depth - hand_depth
        
        if abs(dx) <= self.h_thresh:
            dir_h = 'center'
        else:
            dir_h = 'right' if dx > 0 else 'left'
        
        if abs(dy) <= self.v_thresh:
            dir_v = 'center'
        else:
            dir_v = 'down' if dy > 0 else 'up'
        
        if abs(dd) <= self.d_thresh:
            dir_d = 'hold'
        else:
            dir_d = 'forward' if dd > 0 else 'backward'
        
        ready = (dir_h == 'center' and dir_v == 'center' and dir_d == 'hold')
        
        if ready:
            if gesture == 'open':
                instruction = "抓取! 闭合手掌"
            elif gesture == 'closed':
                instruction = "已抓住!"
            else:
                instruction = "准备抓取!"
        else:
            parts = []
            if dir_h != 'center':
                parts.append(f"向{self._translate(dir_h)}移动")
            if dir_v != 'center':
                parts.append(f"向{self._translate(dir_v)}移动")
            if dir_d != 'hold':
                parts.append(f"向{self._translate(dir_d)}移动")
            instruction = " | ".join(parts) if parts else "保持位置"
        
        return GuidanceResult(
            instruction=instruction,
            direction_h=dir_h,
            direction_v=dir_v,
            direction_d=dir_d,
            dx=dx,
            dy=dy,
            depth_diff=dd,
            ready_to_grab=ready
        )
    
    def _translate(self, direction: str) -> str:
        mapping = {
            'left': '左', 'right': '右',
            'up': '上', 'down': '下',
            'forward': '前', 'backward': '后'
        }
        return mapping.get(direction, direction)
    
    def draw(self, image: np.ndarray, hand_center: Tuple[int, int],
             target_center: Tuple[int, int], result: GuidanceResult) -> np.ndarray:
        output = image.copy()
        
        color = (0, 255, 0) if result.ready_to_grab else (0, 255, 255)
        cv2.line(output, hand_center, target_center, color, 2)
        
        (tw, th), _ = cv2.getTextSize(result.instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(output, (10, 10), (20 + tw, 20 + th + 5), (0, 0, 0), -1)
        
        cv2.putText(output, result.instruction, (15, 15 + th),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        info = f"dx:{result.dx} dy:{result.dy} d:{result.depth_diff:.2f}"
        cv2.putText(output, info, (15, output.shape[0] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return output
