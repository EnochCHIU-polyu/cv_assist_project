"""
配置模块
===============
集中管理系统所有组件的配置参数。
包含模型选择、优化设置、引导参数等。

配置加载优先级（从高到低）:
1. 环境变量（.env 文件）
2. config.yaml 配置值
3. dataclass 默认值
"""

import os
import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
from pathlib import Path

# 可选依赖
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


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
    tts_provider: str = "mimo"  # TTS 提供商: "pyttsx3" (离线) 或 "mimo" (小米 MiMo 云端)
    tts_rate: int = 150  # 语速 (words per minute, pyttsx3 专用)
    tts_volume: float = 1.0  # 音量 (0.0-1.0)
    tts_async: bool = True  # 异步播放 (不阻塞主线程)
    tts_instruction_interval_sec: float = 3.0  # 普通引导最小播报间隔
    tts_grab_repeat_sec: float = 3.0  # 抓握口令重复间隔
    tts_max_queue_size: int = 1  # 异步队列大小，1 表示仅保留最新
    tts_drop_stale: bool = True  # 队列满时是否丢弃旧消息保留新消息
    tts_state_change_bypass: bool = True  # 状态变化时可绕过间隔立即播报

    # MiMo TTS 设置
    mimo_api_key: str = ""  # MiMo API Key（从环境变量 MIMO_API_KEY 读取）
    mimo_voice: str = "mimo_default"  # MiMo 音色: mimo_default / default_zh / default_en
    
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


def _load_env_file(env_path: str = ".env") -> None:
    """加载 .env 文件到环境变量"""
    if not DOTENV_AVAILABLE:
        return
    
    env_file = Path(env_path)
    if env_file.exists():
        load_dotenv(env_file, override=False)  # 不覆盖已有环境变量


def _load_yaml_file(yaml_path: str = "config.yaml") -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    if not YAML_AVAILABLE:
        return {}
    
    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        return {}
    
    with open(yaml_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _apply_yaml_to_config(config: SystemConfig, yaml_data: Dict[str, Any]) -> None:
    """将 YAML 配置应用到 SystemConfig 对象"""
    # 摄像头配置
    if 'camera' in yaml_data:
        cam = yaml_data['camera']
        config.camera_width = cam.get('width', config.camera_width)
        config.camera_height = cam.get('height', config.camera_height)
    
    # 目标查询
    if 'target_queries' in yaml_data:
        config.target_queries = yaml_data['target_queries']
    
    # 模型配置
    if 'model' in yaml_data:
        model = yaml_data['model']
        if 'owlvit_model' in model:
            config.model.owlvit_model = model['owlvit_model']
        if 'owlvit_input_size' in model:
            config.model.owlvit_input_size = tuple(model['owlvit_input_size'])
        if 'owlvit_confidence_threshold' in model:
            config.model.owlvit_confidence_threshold = model['owlvit_confidence_threshold']
        if 'midas_model' in model:
            config.model.midas_model = model['midas_model']
        if 'midas_scale' in model:
            config.model.midas_scale = model['midas_scale']
        if 'hand_max_num' in model:
            config.model.hand_max_num = model['hand_max_num']
        if 'hand_min_confidence' in model:
            config.model.hand_min_confidence = model['hand_min_confidence']
    
    # 优化配置
    if 'optimization' in yaml_data:
        opt = yaml_data['optimization']
        if 'use_fp16' in opt:
            config.optimization.use_fp16 = opt['use_fp16']
        if 'skip_frames_detection' in opt:
            config.optimization.skip_frames_detection = opt['skip_frames_detection']
        if 'skip_frames_depth' in opt:
            config.optimization.skip_frames_depth = opt['skip_frames_depth']
        if 'device' in opt:
            config.optimization.device = opt['device']
    
    # 引导配置
    if 'guidance' in yaml_data:
        guide = yaml_data['guidance']
        for key in ['horizontal_threshold', 'vertical_threshold', 'depth_threshold',
                    'horizontal_threshold_enter', 'horizontal_threshold_exit',
                    'vertical_threshold_enter', 'vertical_threshold_exit',
                    'depth_threshold_enter', 'depth_threshold_exit',
                    'grasp_stable_frames', 'grasp_release_frames']:
            if key in guide:
                setattr(config.guidance, key, guide[key])
    
    # 音频配置
    if 'audio' in yaml_data:
        audio = yaml_data['audio']
        for key in ['enable_asr', 'whisper_model', 'asr_language',
                    'enable_tts', 'tts_provider', 'tts_rate', 'tts_volume',
                    'tts_async', 'tts_instruction_interval_sec', 'tts_grab_repeat_sec',
                    'tts_max_queue_size', 'tts_drop_stale', 'tts_state_change_bypass',
                    'mimo_voice', 'record_sample_rate', 'record_duration',
                    'auto_detect_silence', 'silence_threshold', 'silence_duration']:
            if key in audio:
                setattr(config.audio, key, audio[key])
    
    # 日志配置
    if 'logging' in yaml_data:
        log = yaml_data['logging']
        for key in ['log_dir', 'log_level', 'log_to_file', 'log_to_console',
                    'enable_fps_stats', 'fps_window_size']:
            if key in log:
                setattr(config.logging, key, log[key])


def _apply_env_to_config(config: SystemConfig) -> None:
    """从环境变量读取敏感配置（如 API Key）"""
    # MiMo API Key
    mimo_key = os.environ.get("MIMO_API_KEY") or os.environ.get("XIAOMI_MIMO_API_KEY")
    if mimo_key:
        config.audio.mimo_api_key = mimo_key


def load_config(yaml_path: str = "config.yaml", 
                dotenv_path: str = ".env") -> SystemConfig:
    """
    从 YAML 和 .env 文件加载配置
    
    加载优先级: 环境变量 > config.yaml > 默认值
    
    参数:
        yaml_path: YAML 配置文件路径
        dotenv_path: .env 文件路径
    
    返回:
        SystemConfig 对象
    """
    # 1. 加载 .env 到环境变量
    _load_env_file(dotenv_path)
    
    # 2. 创建默认配置
    config = SystemConfig()
    
    # 3. 从 YAML 加载配置
    yaml_data = _load_yaml_file(yaml_path)
    if yaml_data:
        _apply_yaml_to_config(config, yaml_data)
    
    # 4. 从环境变量读取敏感信息（优先级最高）
    _apply_env_to_config(config)
    
    return config


def load_config_by_profile(yaml_path: str = "config.yaml",
                           dotenv_path: str = ".env") -> SystemConfig:
    """
    根据 YAML 中的 profile 字段加载配置
    
    如果 YAML 中指定了 profile，先应用该 profile 的预设，
    然后再应用 YAML 中的其他配置覆盖。
    """
    # 1. 加载 .env
    _load_env_file(dotenv_path)
    
    # 2. 加载 YAML
    yaml_data = _load_yaml_file(yaml_path)
    
    # 3. 获取 profile 名称
    profile_name = yaml_data.get('profile', 'balanced')
    
    # 4. 根据 profile 创建基础配置
    config = get_config_by_profile(profile_name)
    
    # 5. 应用 YAML 中的其他配置（覆盖 profile 默认值）
    if yaml_data:
        _apply_yaml_to_config(config, yaml_data)
    
    # 6. 从环境变量读取敏感信息
    _apply_env_to_config(config)
    
    return config


# ============ 预设配置函数（保持向后兼容） ============

def get_fast_config() -> SystemConfig:
    """快速配置"""
    config = SystemConfig()
    config.model.owlvit_input_size = (320, 320)
    config.model.midas_scale = 0.5
    config.optimization.skip_frames_detection = 3
    config.optimization.skip_frames_depth = 3
    _apply_env_to_config(config)
    return config


def get_balanced_config() -> SystemConfig:
    """平衡配置"""
    config = SystemConfig()
    _apply_env_to_config(config)
    return config


def get_voice_enabled_config() -> SystemConfig:
    """启用语音功能的配置"""
    config = SystemConfig()
    config.audio.enable_asr = True
    config.audio.enable_tts = True
    config.audio.whisper_model = "base"
    config.audio.tts_rate = 180
    config.audio.tts_async = True
    _apply_env_to_config(config)
    return config


def get_tts_enabled_config() -> SystemConfig:
    """仅启用 TTS 的配置"""
    config = SystemConfig()
    config.audio.enable_asr = False
    config.audio.enable_tts = True
    config.audio.tts_rate = 180
    config.audio.tts_async = True
    _apply_env_to_config(config)
    return config


def get_mimo_tts_config() -> SystemConfig:
    """仅启用 MiMo 云端 TTS 的配置"""
    config = SystemConfig()
    config.audio.enable_asr = False
    config.audio.enable_tts = True
    config.audio.tts_provider = "mimo"
    config.audio.tts_async = True
    _apply_env_to_config(config)
    return config


def get_config_by_profile(profile: str = "balanced") -> SystemConfig:
    """根据配置名称获取系统配置。"""
    profile_map = {
        "fast": get_fast_config,
        "balanced": get_balanced_config,
        "voice": get_voice_enabled_config,
        "tts": get_tts_enabled_config,
        "mimo-tts": get_mimo_tts_config,
    }
    return profile_map.get(profile, get_balanced_config)()
