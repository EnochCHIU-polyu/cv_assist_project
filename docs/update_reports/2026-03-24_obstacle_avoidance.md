# 技术更新报告：物理避障系统 (Obstacle Avoidance)

**日期**: 2026-03-24  
**作者**: 核心开发团队  
**主题**: 基于深度图轨迹预测的物理避障与高优先级语音警告

---

## 1. 概述

本次更新为 CV Assist 视觉辅助系统新增了**物理避障功能**。系统利用已有的 MiDaS 深度图，沿用户手部运动轨迹前方采样深度值，检测是否有突出的高点（障碍物），并通过高优先级 TTS 语音立即警告用户，提升使用过程中的物理安全性。

---

## 2. 核心模块

### 2.1 手部轨迹历史缓冲 (`detectors/hand_history.py`)

新增 `HandHistoryBuffer` 类，使用固定大小的环形缓冲区（`collections.deque`）存储最近 N 帧的手部中心点坐标。

- **速度向量计算**: 基于最近两帧的位移差计算运动方向 `(dx, dy)`
- **轨迹预测**: 沿速度方向预测前方指定距离处的像素坐标
- **采样点生成**: 在当前位置到预测位置之间等距分布采样点，用于深度检测
- **边界裁剪**: 所有采样点自动钳位到图像边界内，防止 `IndexError`

### 2.2 障碍物检测器 (`detectors/obstacle_detector.py`)

新增 `ObstacleDetector` 类，整合手部轨迹和 MiDaS 深度图进行障碍物检测。

**检测流程**:
1. 每帧将手部中心点推入 `HandHistoryBuffer`
2. 计算最近两帧的速度向量
3. 沿运动方向生成 N 个等距采样点（默认 5 个）
4. 对每个采样点读取 MiDaS 深度值
5. 若某个采样点的深度值 > 手部深度 + 阈值（默认 0.15），判定为障碍物（更近的物体 = 更高的 depth 值）

**安全机制**:
- **冷却计时器**: 两次警告之间最少间隔 3 秒（可配置），防止 TTS 刷屏
- **启用/禁用开关**: 通过 `enable_obstacle_detection` 配置项控制
- **可视化调试**: `draw()` 方法在画面上绘制采样点（绿色）、预测终点（蓝色）、障碍物警告（红色）

### 2.3 高优先级 TTS 警告

**中断机制**: 检测到障碍物时，系统执行以下操作：
1. `tts_engine.clear_queue()` — 清空待播放队列并停止当前正在播放的音频
2. `tts_engine.speak("前方有障碍物，请小心")` — 立即排队播放避障警告

**TTS 增强**: 对 `pyttsx3` 和 `MiMo` 两个后端的 `clear_queue()` 方法进行了增强，在清空队列之前先调用 `stop()` 停止当前播放，确保警告能被用户立即听到。

---

## 3. 配置参数

新增 `obstacle` 配置段（`config.py` + `config.yaml`）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_obstacle_detection` | `true` | 是否启用避障检测 |
| `obstacle_trajectory_frames` | `10` | 手部位置历史保留帧数 |
| `obstacle_prediction_distance` | `50.0` | 沿运动方向的检测距离（像素） |
| `obstacle_depth_threshold` | `0.15` | 深度差阈值（高于此值判定障碍物） |
| `obstacle_warning_cooldown` | `3.0` | 两次警告之间最短间隔（秒） |
| `obstacle_sample_count` | `5` | 沿轨迹的采样点数量 |

---

## 4. 系统集成

避障检测集成到 `core/system.py` 的主处理流程中：

- **帧处理 (`process_frame`)**: 在手部检测和深度估计之后执行避障检测，结果封装在 `FrameResult.obstacle` 字段中
- **主循环 (`run`)**: 检测到障碍物时执行高优先级 TTS 打断播报
- **可视化 (`draw_results`)**: 在画面上绘制采样轨迹和障碍物标记

---

## 5. 测试覆盖

新增 28 个单元测试，全部通过（40/40）：

| 测试文件 | 测试数 | 覆盖内容 |
|----------|--------|----------|
| `test_hand_history.py` | 15 | 缓冲区溢出、速度计算、预测、边界裁剪 |
| `test_obstacle_detector.py` | 10 | 障碍物检测、冷却机制、阈值判断、边界钳位 |
| `test_tts_clear_queue.py` | 3 | clear_queue 调用 stop()、队列清空行为 |

---

## 6. 修改文件清单

| 文件 | 变更类型 |
|------|----------|
| `detectors/hand_history.py` | 新增 |
| `detectors/obstacle_detector.py` | 新增 |
| `audio/tts/base.py` | 修改（clear_queue 增加 stop 调用） |
| `audio/tts/pyttsx3_backend.py` | 修改（clear_queue 增加 stop 调用） |
| `audio/tts/mimo_backend.py` | 修改（clear_queue 增加 stop 调用） |
| `config.py` | 修改（新增 ObstacleConfig） |
| `config.yaml` | 修改（新增 obstacle 配置段） |
| `core/system.py` | 修改（集成避障检测和 TTS 警告） |
| `tests/test_hand_history.py` | 新增 |
| `tests/test_obstacle_detector.py` | 新增 |
| `tests/test_tts_clear_queue.py` | 新增 |

---

## 7. 使用方法

无需额外操作，避障功能在启用 TTS 后自动生效：

```bash
# 使用 TTS 配置启动（避障警告自动启用）
python main.py --config tts

# 使用 MiMo 云端 TTS
python main.py --config mimo-tts
```

当用户手部向障碍物移动时，系统会自动播报 "前方有障碍物，请小心"。

如需调整参数，编辑 `config.yaml` 中的 `obstacle` 段即可。
