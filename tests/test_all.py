import os
import sys
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


class SmokeTests(unittest.TestCase):
    def test_imports(self):
        try:
            import detectors
            import core
            import utils
        except Exception as exc:
            self.fail(f"Import failed: {exc}")


if __name__ == "__main__":
    unittest.main()
