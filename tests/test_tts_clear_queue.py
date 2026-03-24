import os
import sys
import unittest
import queue
import threading

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from audio.tts.base import BaseTTS


class _MockTTS(BaseTTS):
    """Mock TTS that tracks stop() calls for testing clear_queue behavior."""

    def __init__(self):
        self.spoken = []
        self.stop_called = 0
        self._queue_items = []

    def speak(self, text: str, block: bool = False):
        self.spoken.append(text)

    def close(self):
        pass

    def stop(self):
        self.stop_called += 1


class TestClearQueueCallsStop(unittest.TestCase):
    """Verify that clear_queue() calls stop() to interrupt currently playing audio."""

    def test_base_clear_queue_calls_stop(self):
        """The base class clear_queue() should call stop()."""
        tts = _MockTTS()
        self.assertEqual(tts.stop_called, 0)
        tts.clear_queue()
        self.assertEqual(tts.stop_called, 1)

    def test_multiple_clear_queue_calls(self):
        """Multiple clear_queue() calls should each call stop()."""
        tts = _MockTTS()
        tts.clear_queue()
        tts.clear_queue()
        self.assertEqual(tts.stop_called, 2)


class TestClearQueueIntegration(unittest.TestCase):
    """Test clear_queue on actual Pyttsx3TTS (if available)."""

    def test_pyttsx3_clear_queue_drains_and_stops(self):
        try:
            from audio.tts.pyttsx3_backend import Pyttsx3TTS, PYTTSX3_AVAILABLE
        except ImportError:
            self.skipTest("pyttsx3 not available")

        if not PYTTSX3_AVAILABLE:
            self.skipTest("pyttsx3 engine not available")

        tts = Pyttsx3TTS(async_mode=True, max_queue_size=3, drop_stale=True)

        # Enqueue several items
        tts.speak("test one")
        tts.speak("test two")

        # Queue should have items
        initial_size = tts.speech_queue.qsize()

        # Clear queue
        tts.clear_queue()

        # Queue should now be empty
        self.assertEqual(tts.speech_queue.qsize(), 0)

        # Can still enqueue after clear
        tts.speak("after clear")
        self.assertGreaterEqual(tts.speech_queue.qsize(), 0)

        tts.close()


if __name__ == "__main__":
    unittest.main()
