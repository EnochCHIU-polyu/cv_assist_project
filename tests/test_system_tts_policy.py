import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.system import CVAssistSystem


class _DummyTTS:
    def __init__(self):
        self.messages = []

    def speak_instruction(self, text):
        self.messages.append(text)

    def clear_queue(self):
        pass


class SystemTTSPolicyTests(unittest.TestCase):
    def _build_system_stub(self):
        system = CVAssistSystem.__new__(CVAssistSystem)
        system.config = SimpleNamespace(
            audio=SimpleNamespace(
                tts_instruction_interval_sec=3.0,
                tts_grab_repeat_sec=3.0,
                tts_state_change_bypass=True,
            )
        )
        system.tts_engine = _DummyTTS()
        system._last_instruction = None
        system._last_instruction_state = None
        system._last_instruction_ts = 0.0
        system._last_grab_ts = 0.0
        return system

    def test_state_change_can_bypass_interval(self):
        system = self._build_system_stub()
        moving = SimpleNamespace(instruction="向右移动", ready_to_grab=False, state="moving")
        ready = SimpleNamespace(instruction="准备抓取!", ready_to_grab=True, state="ready")

        with patch("core.system.time.time", return_value=100.0):
            self.assertTrue(system._should_speak_guidance(moving))
            system._speak_guidance(moving)

        # 仅过 1 秒，不满足常规 3 秒间隔；但状态变化 moving->ready，允许立即播报
        with patch("core.system.time.time", return_value=101.0):
            self.assertTrue(system._should_speak_guidance(ready))

    def test_same_moving_instruction_respects_interval(self):
        system = self._build_system_stub()
        moving = SimpleNamespace(instruction="向右移动", ready_to_grab=False, state="moving")

        with patch("core.system.time.time", return_value=200.0):
            system._speak_guidance(moving)

        with patch("core.system.time.time", return_value=201.0):
            self.assertFalse(system._should_speak_guidance(moving))

        with patch("core.system.time.time", return_value=203.2):
            self.assertTrue(system._should_speak_guidance(moving))


if __name__ == "__main__":
    unittest.main()
