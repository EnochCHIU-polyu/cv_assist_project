# test_all.py - basic smoke tests

import unittest

class SmokeTests(unittest.TestCase):
    def test_imports(self):
        try:
            import detectors
            import core
            import utils
        except Exception as e:
            self.fail(f"Import failed: {e}")


if __name__ == "__main__":
    unittest.main()
