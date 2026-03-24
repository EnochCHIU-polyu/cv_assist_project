import os
import sys
import unittest
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from detectors.obstacle_detector import ObstacleDetector, ObstacleDetectionResult


class TestObstacleDetector(unittest.TestCase):

    def _make_depth_map(self, shape=(480, 640), base_depth=0.3, obstacle_region=None, obstacle_depth=0.8):
        """Create a mock depth map with optional obstacle region (y1, y2, x1, x2)."""
        depth = np.full(shape, base_depth, dtype=np.float32)
        if obstacle_region is not None:
            y1, y2, x1, x2 = obstacle_region
            depth[y1:y2, x1:x2] = obstacle_depth
        return depth

    def test_disabled_detector(self):
        detector = ObstacleDetector(enabled=False)
        detector.update((100, 100))
        result = detector.detect(self._make_depth_map(), (100, 100), current_time=1.0)
        self.assertFalse(result.detected)

    def test_no_obstacle_clear_path(self):
        detector = ObstacleDetector(depth_threshold=0.15)
        depth = self._make_depth_map(base_depth=0.3)
        detector.history.push((100, 100))
        detector.history.push((110, 100))
        result = detector.detect(depth, (110, 100), current_time=0.0)
        self.assertFalse(result.detected)

    def test_obstacle_detected_ahead(self):
        """Obstacle placed along the predicted trajectory should be detected."""
        detector = ObstacleDetector(
            depth_threshold=0.15,
            prediction_distance=50.0,
            sample_count=3
        )
        # Hand at (110, 100) moving right at speed 10.
        # detect() calls update(hand_center) which pushes (110,100) again.
        # After pushing (110,100) twice, velocity = (0,0) -> no samples!
        # So we need to NOT let detect() push again.
        # Strategy: pre-push (100,100) then (110,100), then call detect with hand_center=(110,100)
        # detect() will push (110,100) again, making velocity=(0,0).
        # To fix: we need to ensure velocity is non-zero.
        # Let's push (100,100), (110,100), then detect at (120,100) so velocity is (10,0).

        # Obstacle covers x=125-200, y=90-110 (along the trajectory)
        depth = self._make_depth_map(base_depth=0.2, obstacle_region=(90, 110, 125, 200), obstacle_depth=0.8)

        detector.history.push((100, 100))
        detector.history.push((110, 100))
        # detect() will push (120, 100), velocity = (10, 0)
        # Samples at: 120+50*1/4=132.5, 120+50*2/4=145, 120+50*3/4=157.5
        # These all fall inside obstacle region (x=125-200)
        result = detector.detect(depth, (120, 100), current_time=0.0)
        self.assertTrue(result.detected)
        self.assertIsNotNone(result.warning_text)
        self.assertGreater(result.depth_diff, 0.15)
        self.assertIn("障碍物", result.warning_text)

    def test_obstacle_below_threshold(self):
        """Obstacle depth difference below threshold should not trigger."""
        detector = ObstacleDetector(depth_threshold=0.3, prediction_distance=50.0)
        depth = self._make_depth_map(base_depth=0.3, obstacle_region=(90, 110, 125, 200), obstacle_depth=0.4)

        detector.history.push((100, 100))
        detector.history.push((110, 100))
        result = detector.detect(depth, (120, 100), current_time=0.0)
        self.assertFalse(result.detected)

    def test_cooldown(self):
        """Obstacle warning respects cooldown timer."""
        detector = ObstacleDetector(warning_cooldown=3.0, prediction_distance=200.0, depth_threshold=0.15)
        # Obstacle at x=200-350, hand starts at x=100 and never enters obstacle
        depth = self._make_depth_map(obstacle_region=(40, 60, 200, 350), obstacle_depth=0.9)

        detector.history.push((100, 50))
        detector.history.push((110, 50))
        result1 = detector.detect(depth, (120, 50), current_time=0.0)
        self.assertTrue(result1.detected)

        # Second detection at t=1.0 should be in cooldown
        result2 = detector.detect(depth, (130, 50), current_time=1.0)
        self.assertFalse(result2.detected)

        # Third detection at t=5.0 should be allowed (cooldown expired)
        result3 = detector.detect(depth, (140, 50), current_time=5.0)
        self.assertTrue(result3.detected)

    def test_reset(self):
        detector = ObstacleDetector()
        detector.history.push((1, 1))
        detector.history.push((2, 2))
        detector.reset()
        self.assertEqual(detector.history.size, 0)

    def test_result_bool(self):
        r1 = ObstacleDetectionResult(detected=True)
        self.assertTrue(bool(r1))
        r2 = ObstacleDetectionResult(detected=False)
        self.assertFalse(bool(r2))

    def test_result_repr(self):
        r = ObstacleDetectionResult(detected=True, depth_diff=0.25)
        s = repr(r)
        self.assertIn("True", s)
        self.assertIn("0.250", s)

    def test_sample_points_returned(self):
        """Even when no obstacle, sample points should be returned."""
        detector = ObstacleDetector(sample_count=5)
        depth = self._make_depth_map()
        detector.history.push((100, 100))
        detector.history.push((110, 100))
        result = detector.detect(depth, (120, 100), current_time=0.0)
        self.assertFalse(result.detected)
        self.assertEqual(len(result.sample_points), 5)

    def test_predicted_point_clamped(self):
        """Predicted point should be clamped to image bounds."""
        detector = ObstacleDetector(prediction_distance=1000.0)
        depth = self._make_depth_map(shape=(480, 640))
        detector.history.push((100, 100))
        detector.history.push((110, 100))
        result = detector.detect(depth, (120, 100), current_time=0.0)
        px, py = result.predicted_point
        self.assertGreaterEqual(px, 0)
        self.assertLess(px, 640)
        self.assertGreaterEqual(py, 0)
        self.assertLess(py, 480)


if __name__ == "__main__":
    unittest.main()
