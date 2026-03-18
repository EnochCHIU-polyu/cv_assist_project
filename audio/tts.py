"""
文本转语音模块 (TTS - Text-to-Speech)
将文本转换为语音输出，用于给视觉障碍用户提供语音反馈
"""

import logging
import threading
import queue
from typing import Optional
import platform

logger = logging.getLogger(__name__)

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    logger.warning("pyttsx3 未安装，TTS 功能将不可用")

try:
    import win32com.client  # type: ignore
    SAPI_AVAILABLE = True
except ImportError:
    SAPI_AVAILABLE = False


class TTSEngine:
    """
    文本转语音引擎
    
    使用 pyttsx3 库实现跨平台的文本转语音功能。
    支持异步播放，不会阻塞主程序。
    
    特点:
    - 完全离线，无需网络连接
    - 跨平台支持 (Windows, macOS, Linux)
    - 支持多种语言和语音
    - 可控制语速、音量、声调
    """
    
    def __init__(self,
                 rate: int = 150,
                 volume: float = 1.0,
                 voice_index: Optional[int] = None,
                 async_mode: bool = True,
                 max_queue_size: int = 1,
                 drop_stale: bool = True):
        """
        初始化 TTS 引擎
        
        参数:
            rate: 语速 (words per minute，默认 150，范围通常 100-300)
            volume: 音量 (0.0-1.0，默认 1.0)
            voice_index: 使用的语音索引 (None=默认语音)
            async_mode: 是否异步播放 (True=不阻塞主线程)
        """
        self.rate = rate
        self.volume = volume
        self.voice_id = None
        self.backend = None
        self.max_queue_size = max(1, int(max_queue_size))
        self.drop_stale = drop_stale
        self._voice_index = voice_index
        self._worker_ready = threading.Event()
        self._worker_init_error = None
        self._stop_token = object()

        # 后端选择策略：
        # - 如果需要异步播放，则统一使用 pyttsx3（跨平台且已实现队列+工作线程）
        # - 仅在 Windows 且明确为同步模式时才优先使用 SAPI
        if platform.system().lower() == 'windows' and SAPI_AVAILABLE and not async_mode:
            self.backend = 'sapi'
            self.async_mode = False
        else:
            self.backend = 'pyttsx3'
            self.async_mode = async_mode

        if self.backend == 'pyttsx3' and not PYTTSX3_AVAILABLE:
            raise RuntimeError("pyttsx3 未安装。请运行: pip install pyttsx3")
        
        logger.info("正在初始化 TTS 引擎...")
        
        try:
            if self.backend == 'sapi':
                self._init_sapi(voice_index)
            else:
                self._init_pyttsx3(voice_index)
            
            logger.info(f"TTS 引擎初始化成功 (backend={self.backend})")
            
        except Exception as e:
            logger.error(f"TTS 引擎初始化失败: {e}")
            raise

    def _init_sapi(self, voice_index: Optional[int]):
        """初始化 Windows SAPI 后端"""
        self.sapi_voice = win32com.client.Dispatch("SAPI.SpVoice")
        self.sapi_voice.Rate = self._wpm_to_sapi_rate(self.rate)
        self.sapi_voice.Volume = int(max(0, min(100, self.volume * 100)))

        voices = self.sapi_voice.GetVoices()
        count = voices.Count
        logger.info(f"系统可用语音: {count} 个")

        selected = None
        if voice_index is not None and 0 <= voice_index < count:
            selected = voices.Item(voice_index)
        else:
            selected = self._find_chinese_sapi_voice(voices)
            if selected is None and count > 0:
                selected = voices.Item(0)

        if selected is not None:
            self.sapi_voice.Voice = selected
            self.voice_id = selected.Id
            logger.info(f"使用语音: {selected.GetDescription()}")

    def _init_pyttsx3(self, voice_index: Optional[int]):
        """初始化 pyttsx3 后端"""
        # 关键修复：异步模式下在工作线程中初始化并使用 pyttsx3，避免跨线程调用导致无声/不稳定。
        if self.async_mode:
            self.engine = None
            self.speech_queue = queue.Queue(maxsize=self.max_queue_size)
            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()

            if not self._worker_ready.wait(timeout=3.0):
                logger.warning("TTS 异步工作线程初始化超时，后续将继续尝试播放")
            if self._worker_init_error:
                raise RuntimeError(f"TTS 异步工作线程初始化失败: {self._worker_init_error}")

            logger.info(
                f"TTS 异步模式已启用 (queue_size={self.max_queue_size}, drop_stale={self.drop_stale})"
            )
            return

        self.engine = pyttsx3.init()
        self._configure_pyttsx3_engine(self.engine, voice_index)

    def _configure_pyttsx3_engine(self, engine, voice_index: Optional[int]):
        """配置 pyttsx3 引擎参数与语音。"""
        engine.setProperty('rate', self.rate)
        engine.setProperty('volume', self.volume)

        voices = engine.getProperty('voices')
        logger.info(f"系统可用语音: {len(voices)} 个")

        if voice_index is not None and 0 <= voice_index < len(voices):
            engine.setProperty('voice', voices[voice_index].id)
            self.voice_id = voices[voice_index].id
            logger.info(f"使用语音: {voices[voice_index].name}")
            return

        chinese_voice = self._find_chinese_voice(voices)
        if chinese_voice:
            engine.setProperty('voice', chinese_voice.id)
            self.voice_id = chinese_voice.id
            logger.info(f"自动选择中文语音: {chinese_voice.name}")
            return

        if voices:
            self.voice_id = voices[0].id
        logger.info(f"使用默认语音: {voices[0].name if voices else 'unknown'}")

    def _wpm_to_sapi_rate(self, wpm: int) -> int:
        """将每分钟词数大致映射到 SAPI Rate (-10..10)"""
        mapped = int((wpm - 150) / 10)
        return max(-10, min(10, mapped))

    def _find_chinese_sapi_voice(self, voices):
        """在 SAPI 语音中查找中文语音"""
        chinese_keywords = ['zh', 'chinese', '中文', 'mandarin', 'huihui', 'kangkang']
        for i in range(voices.Count):
            voice = voices.Item(i)
            desc = voice.GetDescription().lower()
            vid = voice.Id.lower()
            if any(k in desc or k in vid for k in chinese_keywords):
                return voice
        return None
    
    def _find_chinese_voice(self, voices) -> Optional[any]:
        """
        查找中文语音
        
        参数:
            voices: 语音列表
            
        返回:
            中文语音对象，如果没找到则返回 None
        """
        # Windows 中文语音标识
        chinese_keywords = ['zh', 'chinese', '中文', 'mandarin', 'huihui', 'kangkang']
        
        for voice in voices:
            voice_name_lower = voice.name.lower()
            voice_id_lower = voice.id.lower()
            
            for keyword in chinese_keywords:
                if keyword in voice_name_lower or keyword in voice_id_lower:
                    return voice
        
        return None
    
    def _worker(self):
        """
        异步播放工作线程
        """
        try:
            self.engine = pyttsx3.init()
            self._configure_pyttsx3_engine(self.engine, self._voice_index)
            self._worker_ready.set()
            logger.debug("TTS 异步工作线程初始化完成")
        except Exception as e:
            self._worker_init_error = e
            self._worker_ready.set()
            logger.error(f"TTS 异步工作线程初始化失败: {e}", exc_info=True)
            return

        while True:
            try:
                payload = self.speech_queue.get()
                if payload is self._stop_token:  # 退出信号
                    self.speech_queue.task_done()
                    break

                text, done_event = payload
                try:
                    logger.debug(f"TTS 播放: '{text}'")
                    self.engine.say(text)
                    self.engine.runAndWait()
                finally:
                    if done_event is not None:
                        done_event.set()
                    self.speech_queue.task_done()
                
            except Exception as e:
                logger.error(f"TTS 播放错误: {e}", exc_info=True)
    
    def speak(self, text: str, block: bool = False):
        """
        播放文本
        
        参数:
            text: 要播放的文本
            block: 是否阻塞等待播放完成 (仅在 async_mode=True 时有效)
        """
        if not text or not text.strip():
            logger.warning("空文本，跳过 TTS 播放")
            return
        
        text = text.strip()
        logger.info(f"TTS 请求: '{text}'")
        
        try:
            if self.backend == 'sapi':
                self.sapi_voice.Speak(text)
                return

            if self.async_mode and not block:
                # 异步播放
                self._enqueue_async(text)
            elif self.async_mode and block:
                # 在异步模式下也通过工作线程执行，避免跨线程访问 pyttsx3
                self._enqueue_async(text, wait=True)
            else:
                # 同步播放
                self.engine.say(text)
                self.engine.runAndWait()
                
        except Exception as e:
            logger.error(f"TTS 播放失败: {e}", exc_info=True)

    def _enqueue_async(self, text: str, wait: bool = False):
        """将文本加入异步队列，支持队列满时去旧保新。"""
        done_event = threading.Event() if wait else None
        payload = (text, done_event)
        queued = False

        try:
            self.speech_queue.put_nowait(payload)
            queued = True
        except queue.Full:
            if not self.drop_stale:
                logger.debug("TTS 队列已满，保留旧消息，丢弃新消息")
                return

        if not queued:
            # 队列满且启用去旧：清空旧指令，仅保留最新文本
            self.clear_queue()
            try:
                self.speech_queue.put_nowait(payload)
                queued = True
            except queue.Full:
                logger.debug("TTS 队列仍然繁忙，丢弃当前消息")
                return

        if wait and done_event is not None:
            if not done_event.wait(timeout=5.0):
                logger.warning("TTS 同步等待超时，文本可能未及时播报")
    
    def speak_instruction(self, instruction: str):
        """
        播放引导指令 (简化版，便于快速反馈)
        
        参数:
            instruction: 引导指令文本
        """
        self.speak(instruction)
    
    def stop(self):
        """
        停止当前播放
        """
        try:
            if self.backend == 'sapi':
                return
            if hasattr(self.engine, 'stop'):
                self.engine.stop()
            logger.debug("TTS 停止播放")
        except Exception as e:
            logger.error(f"TTS 停止失败: {e}")
    
    def clear_queue(self):
        """
        清空播放队列 (异步模式)
        """
        if getattr(self, 'async_mode', False) and hasattr(self, 'speech_queue'):
            while not self.speech_queue.empty():
                try:
                    payload = self.speech_queue.get_nowait()
                    # 如果是阻塞等待任务，通知调用方避免一直等待。
                    if isinstance(payload, tuple) and len(payload) == 2:
                        _, done_event = payload
                        if done_event is not None:
                            done_event.set()
                    self.speech_queue.task_done()
                except queue.Empty:
                    break
            logger.debug("TTS 队列已清空")
    
    def set_rate(self, rate: int):
        """
        设置语速
        
        参数:
            rate: 语速 (words per minute)
        """
        self.rate = rate
        if self.backend == 'sapi':
            self.sapi_voice.Rate = self._wpm_to_sapi_rate(rate)
        else:
            self.engine.setProperty('rate', rate)
        logger.info(f"TTS 语速设置为: {rate}")
    
    def set_volume(self, volume: float):
        """
        设置音量
        
        参数:
            volume: 音量 (0.0-1.0)
        """
        volume = max(0.0, min(1.0, volume))  # 限制范围
        self.volume = volume
        if self.backend == 'sapi':
            self.sapi_voice.Volume = int(volume * 100)
        else:
            self.engine.setProperty('volume', volume)
        logger.info(f"TTS 音量设置为: {volume}")
    
    def list_voices(self):
        """
        列出所有可用语音
        
        返回:
            语音列表 (包含 id 和 name)
        """
        voice_list = []
        if self.backend == 'sapi':
            voices = self.sapi_voice.GetVoices()
            for idx in range(voices.Count):
                voice = voices.Item(idx)
                voice_info = {
                    'index': idx,
                    'id': voice.Id,
                    'name': voice.GetDescription(),
                    'languages': []
                }
                voice_list.append(voice_info)
                logger.info(f"  [{idx}] {voice.GetDescription()} - {voice.Id}")
            return voice_list

        if self.async_mode and not self._worker_ready.is_set():
            logger.warning("TTS 工作线程尚未就绪，暂时无法列出语音")
            return voice_list

        voices = self.engine.getProperty('voices')
        for idx, voice in enumerate(voices):
            voice_info = {
                'index': idx,
                'id': voice.id,
                'name': voice.name,
                'languages': getattr(voice, 'languages', [])
            }
            voice_list.append(voice_info)
            logger.info(f"  [{idx}] {voice.name} - {voice.id}")
        return voice_list
    
    def close(self):
        """
        关闭 TTS 引擎
        """
        try:
            if self.backend == 'sapi':
                logger.info("TTS 引擎已关闭")
                return

            if getattr(self, 'async_mode', False) and hasattr(self, 'speech_queue'):
                # 发送退出信号
                self.clear_queue()
                self.speech_queue.put(self._stop_token)
                if hasattr(self, 'worker_thread'):
                    self.worker_thread.join(timeout=2.0)
            
            # 停止引擎
            if hasattr(self, 'engine'):
                self.stop()
            logger.info("TTS 引擎已关闭")
            
        except Exception as e:
            logger.error(f"关闭 TTS 引擎失败: {e}")

    def get_debug_info(self) -> dict:
        """返回当前 TTS 关键状态，便于定位无声问题。"""
        info = {
            'backend': self.backend,
            'async_mode': self.async_mode,
            'rate': self.rate,
            'volume': self.volume,
            'voice_id': self.voice_id,
            'platform': platform.system(),
            'pyttsx3_available': PYTTSX3_AVAILABLE,
            'sapi_available': SAPI_AVAILABLE,
        }

        if hasattr(self, 'speech_queue'):
            info['queue_size'] = self.speech_queue.qsize()
            info['queue_max_size'] = self.max_queue_size
        if hasattr(self, 'worker_thread'):
            info['worker_alive'] = self.worker_thread.is_alive()
        if self._worker_init_error is not None:
            info['worker_init_error'] = str(self._worker_init_error)

        logger.info(f"TTS 调试信息: {info}")
        return info
    
    def __del__(self):
        """
        析构函数，确保资源释放
        """
        try:
            self.close()
        except:
            pass


# 便捷函数
def quick_speak(text: str, rate: int = 150):
    """
    快速播放文本 (创建临时 TTS 引擎)
    
    参数:
        text: 要播放的文本
        rate: 语速
    """
    if not PYTTSX3_AVAILABLE:
        logger.warning("pyttsx3 不可用，无法播放语音")
        return
    
    try:
        engine = TTSEngine(rate=rate, async_mode=False)
        engine.speak(text, block=True)
        engine.close()
    except Exception as e:
        logger.error(f"快速播放失败: {e}")
