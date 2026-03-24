"""
TTS 抽象基类
定义所有文本转语音服务的统一接口
"""

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseTTS(ABC):
    """
    文本转语音服务抽象基类

    所有 TTS 后端（pyttsx3、MiMo、自定义服务等）均需实现此接口。
    通过统一的 speak / speak_instruction / stop / clear_queue 等方法，
    上层调用方无需关心底层实现细节。
    """

    @abstractmethod
    def speak(self, text: str, block: bool = False):
        """
        播放文本

        参数:
            text: 要播放的文本
            block: 是否阻塞等待播放完成
        """

    @abstractmethod
    def close(self):
        """关闭 TTS 引擎，释放资源"""

    def speak_instruction(self, instruction: str):
        """
        播放引导指令（简化接口）

        参数:
            instruction: 引导指令文本
        """
        self.speak(instruction)

    def stop(self):
        """停止当前播放（默认无操作）"""

    def clear_queue(self):
        """清空播放队列，并停止当前正在播放的音频（子类可覆盖实现）。
        用于高优先级语音（如避障警告）需要立即打断当前播报的场景。
        基类默认调用 stop() 后无操作。
        """
        self.stop()

    def set_rate(self, rate: int):
        """设置语速（默认无操作）"""

    def set_volume(self, volume: float):
        """设置音量（默认无操作）"""

    def list_voices(self):
        """列出可用语音（默认返回空列表）"""
        return []

    def get_debug_info(self) -> dict:
        """返回调试信息"""
        return {'provider': 'unknown'}
