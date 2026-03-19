"""
配置模块
=======
集中管理系统所有组件的配置参数。

配置加载优先级（从高到低）:
  1. 环境变量（.env 文件）  — 如 MIMO_API_KEY
  2. config.yaml profile overrides
  3. config.yaml 基础配置
  4. dataclass 默认值

用法:
    from config import load_config
    config = load_config(profile="tts")
"""

import os
import sys
import logging
from dataclasses import dataclass, field, fields, is_dataclass
from typing import List, Tuple, Dict, Any, Type, get_type_hints
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    owlvit_model: str = "google/owlvit-base-patch32"
    owlvit_input_size: Tuple[int, int] = (384, 384)
    owlvit_confidence_threshold: float = 0.1
    midas_model: str = "MiDaS_small"
    midas_scale: float = 0.5
    hand_max_num: int = 1
    hand_min_confidence: float = 0.5


@dataclass
class OptimizationConfig:
    use_fp16: bool = True
    skip_frames_detection: int = 2
    skip_frames_depth: int = 2
    device: str = "auto"

    def __post_init__(self):
        if self.device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        if self.device == "cpu":
            self.use_fp16 = False


@dataclass
class GuidanceConfig:
    horizontal_threshold: int = 30
    vertical_threshold: int = 30
    depth_threshold: float = 0.15
    horizontal_threshold_enter: int = 24
    horizontal_threshold_exit: int = 36
    vertical_threshold_enter: int = 24
    vertical_threshold_exit: int = 36
    depth_threshold_enter: float = 0.12
    depth_threshold_exit: float = 0.18
    grasp_stable_frames: int = 8
    grasp_release_frames: int = 3


@dataclass
class AudioConfig:
    enable_asr: bool = False
    whisper_model: str = "base"
    asr_language: str = "zh"
    enable_tts: bool = False
    tts_provider: str = "pyttsx3"
    tts_rate: int = 150
    tts_volume: float = 1.0
    tts_async: bool = True
    tts_instruction_interval_sec: float = 3.0
    tts_grab_repeat_sec: float = 3.0
    tts_max_queue_size: int = 1
    tts_drop_stale: bool = True
    tts_state_change_bypass: bool = True
    mimo_api_key: str = ""
    mimo_voice: str = "mimo_default"
    record_sample_rate: int = 16000
    record_duration: float = 5.0
    auto_detect_silence: bool = True
    silence_threshold: float = 0.01
    silence_duration: float = 1.5


@dataclass
class LoggingConfig:
    log_dir: str = "logs"
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True
    enable_fps_stats: bool = True
    fps_window_size: int = 30


@dataclass
class SystemConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    target_queries: List[str] = field(default_factory=lambda: ["a cup", "a bottle"])
    camera_width: int = 640
    camera_height: int = 480


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _deep_merge(base: Dict, override: Dict) -> Dict:
    """递归合并两个字典，override 中的值覆盖 base。"""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _dict_to_config(data: Dict[str, Any], cls: Type) -> Any:
    """将字典递归转换为 dataclass 实例。

    处理:
    - 嵌套 dataclass（如 SystemConfig.model -> ModelConfig）
    - Tuple 字段（YAML list -> tuple）
    - 未知 key 直接跳过
    """
    if not is_dataclass(cls):
        return data

    field_map = {f.name: f for f in fields(cls)}
    hints = get_type_hints(cls)
    kwargs: Dict[str, Any] = {}

    for name, fld in field_map.items():
        if name not in data:
            continue

        value = data[name]
        field_type = hints.get(name, fld.type)

        # 嵌套 dataclass
        if is_dataclass(field_type) and isinstance(value, dict):
            kwargs[name] = _dict_to_config(value, field_type)

        # Tuple 字段：YAML list -> Python tuple
        elif _is_tuple_type(field_type) and isinstance(value, list):
            kwargs[name] = tuple(value)

        else:
            kwargs[name] = value

    return cls(**kwargs)


def _is_tuple_type(tp) -> bool:
    origin = getattr(tp, '__origin__', None)
    return origin is tuple


def _flatten_camera(data: Dict[str, Any]) -> Dict[str, Any]:
    """将 camera.width / camera.height 映射到 camera_width / camera_height。"""
    if 'camera' in data and isinstance(data['camera'], dict):
        cam = data.pop('camera')
        if 'width' in cam:
            data['camera_width'] = cam['width']
        if 'height' in cam:
            data['camera_height'] = cam['height']
    return data


def _load_env(dotenv_path: str = ".env") -> None:
    if load_dotenv is None:
        return
    p = Path(dotenv_path)
    if p.exists():
        load_dotenv(p, override=False)


def _load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _apply_env(config: SystemConfig) -> None:
    key = os.environ.get("MIMO_API_KEY") or os.environ.get("XIAOMI_MIMO_API_KEY")
    if key:
        config.audio.mimo_api_key = key


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

def load_config(profile: str = "balanced",
                yaml_path: str = "config.yaml",
                dotenv_path: str = ".env") -> SystemConfig:
    """加载系统配置。

    参数:
        profile:    预设名称 (fast / balanced / voice / tts / mimo-tts)
        yaml_path:  YAML 配置文件路径
        dotenv_path:.env 文件路径

    返回:
        SystemConfig 实例
    """
    # 1. 环境变量
    _load_env(dotenv_path)

    # 2. 读 YAML + 应用 profile overrides
    yaml_data = _load_yaml(yaml_path)
    profiles = yaml_data.pop('profiles', {})

    if profile and profile != 'balanced' and profile in profiles:
        yaml_data = _deep_merge(yaml_data, profiles[profile])
        logger.debug("Applied profile overrides: %s", profile)

    # 3. camera: {width, height} -> camera_width, camera_height
    _flatten_camera(yaml_data)

    # 4. 构建 SystemConfig
    config = _dict_to_config(yaml_data, SystemConfig)

    # 5. 环境变量覆盖（最高优先级）
    _apply_env(config)

    return config


# ---------------------------------------------------------------------------
# CLI 快速测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    def _to_dict(obj):
        if is_dataclass(obj) and not isinstance(obj, type):
            return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
        if isinstance(obj, (list, tuple)):
            return [_to_dict(i) for i in obj]
        return obj

    name = sys.argv[1] if len(sys.argv) > 1 else "balanced"
    cfg = load_config(profile=name)
    print(json.dumps(_to_dict(cfg), indent=2, ensure_ascii=False))
