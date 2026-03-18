"""
配置模块
================
集中管理系统所有组件的配置参数。
包含模型选择、优化设置、引导参数等。
"""

import torch
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ModelConfig:
    """模型配置"""
    # OWL-ViT 设置
    owlvit_model: str = "google/owlvit-base-patch32"
    owlvit_input_size: Tuple[int, int] = (384, 384)
    owlvit_confidence_threshold: float = 0.1
    
    # MiDaS 设置
    midas_model: str = "MiDaS_small"
    midas_scale: float = 0.5
    
    # MediaPipe 设置
    hand_max_num: int = 1
    hand_min_confidence: float = 0.5


@dataclass
class OptimizationConfig:
    """优化配置"""
    use_fp16: bool = True
    # 跳帧设置，减少检测的频率以提升性能
    skip_frames_detection: int = 2
    # 跳帧设置，减少深度估计的频率以提升性能
    skip_frames_depth: int = 2
    device: str = "auto"
    
    # 自动选择设备，优先使用 CUDA，如果不可用则回退到 CPU
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            self.use_fp16 = False


@dataclass
class GuidanceConfig:
    """引导配置"""
    horizontal_threshold: int = 30
    vertical_threshold: int = 30
    depth_threshold: float = 0.15
    # 滞回阈值：进入对齐区域更严格，退出对齐区域更宽松
    horizontal_threshold_enter: int = 24
    horizontal_threshold_exit: int = 36
    vertical_threshold_enter: int = 24
    vertical_threshold_exit: int = 36
    depth_threshold_enter: float = 0.12
    depth_threshold_exit: float = 0.18
    # 抓握稳定判定
    grasp_stable_frames: int = 8
    grasp_release_frames: int = 3


@dataclass
class AudioConfig:
    """音频配置"""
    # ASR (语音识别) 设置
    enable_asr: bool = False  # 是否启用语音识别
    whisper_model: str = "base"  # Whisper 模型 (tiny/base/small/medium/large)
    asr_language: str = "zh"  # 识别语言 (zh=中文, en=英文)
    
    # TTS (文本转语音) 设置
    enable_tts: bool = False  # 是否启用语音输出
    tts_rate: int = 150  # 语速 (words per minute)
    tts_volume: float = 1.0  # 音量 (0.0-1.0)
    tts_async: bool = True  # 异步播放 (不阻塞主线程)
    tts_instruction_interval_sec: float = 3.0  # 普通引导最小播报间隔
    tts_grab_repeat_sec: float = 3.0  # 抓握口令重复间隔
    tts_max_queue_size: int = 1  # 异步队列大小，1 表示仅保留最新
    tts_drop_stale: bool = True  # 队列满时是否丢弃旧消息保留新消息
    tts_state_change_bypass: bool = True  # 状态变化时可绕过间隔立即播报
    
    # 录音设置
    record_sample_rate: int = 16000  # 采样率 (Hz)
    record_duration: float = 5.0  # 录音时长 (秒)
    auto_detect_silence: bool = True  # 自动检测静音停止录音
    silence_threshold: float = 0.01  # 静音阈值
    silence_duration: float = 1.5  # 静音持续时长 (秒)


@dataclass
class LoggingConfig:
    """日志配置"""
    log_dir: str = "logs"
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True
    # FPS 统计配置
    enable_fps_stats: bool = True
    fps_window_size: int = 30  # FPS 平滑窗口


@dataclass
class SystemConfig:
    """系统配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    
    # 目标查询列表，用户可以根据需要修改为其他物体
    target_queries: List[str] = field(default_factory=lambda: ["a cup", "a bottle"])
    camera_width: int = 640
    camera_height: int = 480


def get_fast_config() -> SystemConfig:
    """快速配置"""
    config = SystemConfig()
    config.model.owlvit_input_size = (320, 320)
    config.model.midas_scale = 0.5
    config.optimization.skip_frames_detection = 3
    config.optimization.skip_frames_depth = 3
    return config


def get_balanced_config() -> SystemConfig:
    """平衡配置"""
    return SystemConfig()


def get_voice_enabled_config() -> SystemConfig:
    """启用语音功能的配置"""
    config = SystemConfig()
    config.audio.enable_asr = True
    config.audio.enable_tts = True
    config.audio.whisper_model = "base"  # 使用 base 模型平衡速度和准确性
    config.audio.tts_rate = 180  # 稍快的语速
    # 语音交互场景下，为避免阻塞摄像头主循环，启用异步 TTS
    config.audio.tts_async = True
    return config


def get_tts_enabled_config() -> SystemConfig:
    """仅启用 TTS 的配置"""
    config = SystemConfig()
    config.audio.enable_asr = False
    config.audio.enable_tts = True
    config.audio.tts_rate = 180
    # 仅 TTS 模式同样使用异步播放，避免画面卡顿
    config.audio.tts_async = True
    return config


def get_config_by_profile(profile: str = "balanced") -> SystemConfig:
    """根据配置名称获取系统配置。"""
    profile_map = {
        "fast": get_fast_config,
        "balanced": get_balanced_config,
        "voice": get_voice_enabled_config,
        "tts": get_tts_enabled_config,
    }
    return profile_map.get(profile, get_balanced_config)()
