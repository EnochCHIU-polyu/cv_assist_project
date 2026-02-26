#!/usr/bin/env python3
"""
音频功能测试脚本
测试 ASR 和 TTS 模块
"""

import logging
import os
import sys

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.logger import setup_logging

setup_logging(log_to_file=False, log_level="INFO")
logger = logging.getLogger(__name__)


def test_tts():
    logger.info("=== 测试 TTS ===")

    try:
        from audio.tts import TTSEngine

        logger.info("初始化 TTS 引擎...")
        tts = TTSEngine(rate=150, async_mode=False)

        logger.info("\n可用语音:")
        tts.list_voices()

        logger.info("\n测试播放中文:")
        tts.speak("你好，这是文本转语音测试。")

        logger.info("\n测试播放英文:")
        tts.speak("Hello, this is a text to speech test.")

        logger.info("\n测试播放引导指令:")
        tts.speak("向右移动")

        tts.close()
        logger.info("✓ TTS 测试通过")
        return True

    except ImportError:
        logger.error("✗ TTS 模块未安装，请运行: pip install pyttsx3")
        return False
    except Exception as exc:
        logger.error(f"✗ TTS 测试失败: {exc}", exc_info=True)
        return False


def test_asr():
    logger.info("\n=== 测试 ASR ===")

    try:
        from audio.asr import ASREngine
        from audio.audio_utils import AudioRecorder

        logger.info("初始化 ASR 引擎 (这可能需要一些时间下载模型)...")
        asr = ASREngine(model_name="base", device="cpu", language="zh")

        logger.info("✓ ASR 引擎初始化成功")

        logger.info("\n测试指令解析:")
        test_commands = [
            "找到杯子",
            "帮我找一下手机",
            "where is the cup",
            "find the bottle",
        ]

        for cmd in test_commands:
            target = asr.parse_command(cmd)
            logger.info(f"  '{cmd}' -> '{target}'")

        logger.info("\n是否测试语音录制和识别? (y/n)")
        choice = input().strip().lower()

        if choice == "y":
            logger.info("初始化录音器...")
            recorder = AudioRecorder(sample_rate=16000)

            logger.info("\n准备录音 5 秒，请说话...")
            input("按 Enter 键开始录音...")

            audio = recorder.record(duration=5.0)
            logger.info(f"录音完成: {len(audio)} samples")

            logger.info("开始识别...")
            result = asr.transcribe_audio(audio, sample_rate=16000)
            logger.info(f"识别结果: '{result['text']}'")

            target = asr.parse_command(result['text'])
            if target:
                logger.info(f"提取的目标: '{target}'")

        logger.info("✓ ASR 测试通过")
        return True

    except ImportError:
        logger.error("✗ ASR 模块未安装，请运行: pip install openai-whisper sounddevice")
        return False
    except Exception as exc:
        logger.error(f"✗ ASR 测试失败: {exc}", exc_info=True)
        return False


def test_audio_recorder():
    logger.info("\n=== 测试录音器 ===")

    try:
        from audio.audio_utils import AudioRecorder

        logger.info("\n可用音频设备:")
        AudioRecorder.list_devices()

        logger.info("\n初始化录音器...")
        recorder = AudioRecorder(sample_rate=16000)

        logger.info("\n测试录音 3 秒...")
        input("按 Enter 键开始...")

        audio = recorder.record(duration=3.0)
        logger.info(f"录音完成: {len(audio)} samples, {len(audio)/16000:.2f} 秒")

        logger.info("\n保存音频到 test_audio.wav...")
        recorder.save_audio(audio, "test_audio.wav")

        logger.info("加载音频...")
        loaded = recorder.load_audio("test_audio.wav")
        logger.info(f"加载成功: {len(loaded)} samples")

        if os.path.exists("test_audio.wav"):
            os.remove("test_audio.wav")
            logger.info("测试文件已删除")

        logger.info("✓ 录音器测试通过")
        return True

    except ImportError:
        logger.error("✗ 录音模块未安装，请运行: pip install sounddevice")
        return False
    except Exception as exc:
        logger.error(f"✗ 录音器测试失败: {exc}", exc_info=True)
        return False


def main():
    logger.info("=" * 60)
    logger.info(" 音频功能测试")
    logger.info("=" * 60)

    results = {}
    logger.info("\n[1/3] 测试 TTS (文本转语音)")
    results["TTS"] = test_tts()

    logger.info("\n[2/3] 测试录音器")
    results["Recorder"] = test_audio_recorder()

    logger.info("\n[3/3] 测试 ASR (语音识别)")
    results["ASR"] = test_asr()

    logger.info("\n" + "=" * 60)
    logger.info(" 测试总结")
    logger.info("=" * 60)

    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        logger.info(f"{name:12s}: {status}")

    all_passed = all(results.values())
    logger.info("=" * 60)

    if all_passed:
        logger.info("✓ 所有测试通过！")
    else:
        logger.warning("✗ 部分测试失败，请检查错误信息")

    logger.info("\n提示:")
    logger.info("  - 如果 TTS 测试失败: pip install pyttsx3")
    logger.info("  - 如果录音器测试失败: pip install sounddevice")
    logger.info("  - 如果 ASR 测试失败: pip install openai-whisper")
    logger.info("\n完整安装命令:")
    logger.info("  pip install openai-whisper pyttsx3 sounddevice scipy")


if __name__ == "__main__":
    main()
