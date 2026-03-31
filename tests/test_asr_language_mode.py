import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from audio.asr import ASREngine


class _FakeMel:
    def to(self, _device):
        return self


class _FakeModel:
    def __init__(self, probs):
        self.device = "cpu"
        self._probs = probs

    def detect_language(self, _mel):
        return None, self._probs


class ASRLanguageModeTests(unittest.TestCase):
    def _build_engine(self, probs, language="zh,en"):
        engine = ASREngine.__new__(ASREngine)
        engine.language = language
        engine.device = "cpu"
        engine.model = _FakeModel(probs)
        return engine

    def test_bilingual_mode_prefers_chinese_when_chinese_detected(self):
        engine = self._build_engine({"zh": 0.8, "en": 0.2, "ja": 0.9})

        with patch("audio.asr.whisper.pad_or_trim", return_value=np.zeros(16000, dtype=np.float32)), \
             patch("audio.asr.whisper.log_mel_spectrogram", return_value=_FakeMel()):
            self.assertEqual(engine._resolve_language_for_audio(np.zeros(16000, dtype=np.float32)), "zh")

    def test_bilingual_mode_falls_back_to_english_for_non_chinese(self):
        engine = self._build_engine({"en": 0.4, "fr": 0.9, "de": 0.8})

        with patch("audio.asr.whisper.pad_or_trim", return_value=np.zeros(16000, dtype=np.float32)), \
             patch("audio.asr.whisper.log_mel_spectrogram", return_value=_FakeMel()):
            self.assertEqual(engine._resolve_language_for_audio(np.zeros(16000, dtype=np.float32)), "en")


if __name__ == "__main__":
    unittest.main()
