import os
import sys
import unittest
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from detectors.hand_history import HandHistoryBuffer


class TestHandHistoryBuffer(unittest.TestCase):

    def test_initial_state(self):
        buf = HandHistoryBuffer(maxlen=5)
        self.assertEqual(buf.size, 0)
        self.assertFalse(buf.is_ready(min_frames=1))

    def test_push_and_size(self):
        buf = HandHistoryBuffer(maxlen=5)
        buf.push((100, 200))
        buf.push((110, 210))
        self.assertEqual(buf.size, 2)
        self.assertTrue(buf.is_ready(min_frames=2))

    def test_eviction(self):
        buf = HandHistoryBuffer(maxlen=3)
        buf.push((1, 1))
        buf.push((2, 2))
        buf.push((3, 3))
        buf.push((4, 4))  # Should evict (1,1)
        positions = buf.get_positions()
        self.assertEqual(len(positions), 3)
        self.assertEqual(positions[0], (2, 2))
        self.assertEqual(positions[-1], (4, 4))

    def test_velocity_stationary(self):
        buf = HandHistoryBuffer(maxlen=5)
        buf.push((100, 100))
        buf.push((100, 100))
        vel = buf.velocity()
        np.testing.assert_array_almost_equal(vel, [0.0, 0.0])

    def test_velocity_moving_right(self):
        buf = HandHistoryBuffer(maxlen=5)
        buf.push((100, 100))
        buf.push((110, 100))
        vel = buf.velocity()
        np.testing.assert_array_almost_equal(vel, [10.0, 0.0])

    def test_velocity_moving_diagonal(self):
        buf = HandHistoryBuffer(maxlen=5)
        buf.push((100, 100))
        buf.push((105, 110))
        vel = buf.velocity()
        np.testing.assert_array_almost_equal(vel, [5.0, 10.0])

    def test_velocity_insufficient_frames(self):
        buf = HandHistoryBuffer(maxlen=5)
        buf.push((100, 100))
        vel = buf.velocity()
        np.testing.assert_array_almost_equal(vel, [0.0, 0.0])

    def test_predict_moving_right(self):
        buf = HandHistoryBuffer(maxlen=5)
        buf.push((100, 100))
        buf.push((110, 100))
        predicted = buf.predict(distance=50.0)
        # Velocity is (10, 0), normalized direction (1, 0), predicted = (160, 100)
        self.assertEqual(predicted, (160, 100))

    def test_predict_moving_down(self):
        buf = HandHistoryBuffer(maxlen=5)
        buf.push((100, 100))
        buf.push((100, 120))
        predicted = buf.predict(distance=50.0)
        self.assertEqual(predicted, (100, 170))

    def test_predict_no_motion(self):
        buf = HandHistoryBuffer(maxlen=5)
        buf.push((100, 100))
        buf.push((100, 100))
        predicted = buf.predict(distance=50.0)
        # No motion -> return current position
        self.assertEqual(predicted, (100, 100))

    def test_predict_samples(self):
        buf = HandHistoryBuffer(maxlen=5)
        buf.push((100, 100))
        buf.push((110, 100))
        samples = buf.predict_samples(distance=50.0, num_samples=3)
        self.assertEqual(len(samples), 3)
        # All samples should be along the x-axis between (100,100) and (160,100)
        for sx, sy in samples:
            self.assertEqual(sy, 100)
            self.assertGreater(sx, 100)
            self.assertLessEqual(sx, 160)

    def test_predict_samples_boundary_clipping(self):
        buf = HandHistoryBuffer(maxlen=5)
        buf.push((100, 100))
        buf.push((110, 100))
        samples = buf.predict_samples(distance=500.0, num_samples=3,
                                       img_width=200, img_height=200)
        for sx, sy in samples:
            self.assertGreaterEqual(sx, 0)
            self.assertLess(sx, 200)
            self.assertGreaterEqual(sy, 0)
            self.assertLess(sy, 200)

    def test_clear(self):
        buf = HandHistoryBuffer(maxlen=5)
        buf.push((1, 1))
        buf.push((2, 2))
        buf.clear()
        self.assertEqual(buf.size, 0)

    def test_min_frames_default(self):
        buf = HandHistoryBuffer(maxlen=5)
        buf.push((1, 1))
        self.assertFalse(buf.is_ready())  # Default min_frames=2
        buf.push((2, 2))
        self.assertTrue(buf.is_ready())

    def test_repr(self):
        buf = HandHistoryBuffer(maxlen=10)
        buf.push((1, 1))
        r = repr(buf)
        self.assertIn("10", r)
        self.assertIn("1", r)


if __name__ == "__main__":
    unittest.main()
