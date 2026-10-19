"""Exercise silent-stall recovery with a simulated clock, without cameras."""

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from channels.channel import VideoChannel
from channels.channel_config import VideoChannelConfig
from channels.gstreamer_capture import GstCapture


class Clock:
    def __init__(self):
        self.now = 0.0
        self.stopped = False

    def is_set(self):
        return self.stopped

    def wait(self, delay):
        self.now += delay
        return self.stopped


class SilentCapture(GstCapture):
    def __init__(self, clock, good_grab=None, closes=False):
        self.clock = clock
        self.good_grab = good_grab
        self.closes = closes
        self.calls = 0
        self.opened = True
        self.released_at = None
        self.frame_at = None

    def isOpened(self):
        return self.opened

    def grab(self, timeout_s=None):
        self.calls += 1
        self.clock.now += timeout_s
        if self.closes:
            self.opened = False
        if self.calls == self.good_grab:
            self.frame_at = self.clock.now
            return True
        return False

    def retrieve(self):
        return True, np.zeros((2, 2, 3), dtype=np.uint8)

    def release(self):
        self.released_at = self.clock.now
        self.clock.stopped = True
        self.opened = False


class RecoveryTests(unittest.TestCase):
    def run_capture(self, fps=3, good_grab=None, closes=False, timeout="5"):
        clock = Clock()
        capture = SilentCapture(clock, good_grab, closes)
        channel = VideoChannel(VideoChannelConfig(camera_uuid="test", sample_fps=fps))
        channel._stop_thread_evt = clock
        with patch.object(channel, "_open_capture", return_value=capture), \
                patch("channels.channel.CONNECT_GATE.wait_turn", return_value=True), \
                patch("channels.channel.time.monotonic", side_effect=lambda: clock.now), \
                patch.dict(os.environ, {"CAPTURE_STALL_TIMEOUT_S": timeout}):
            channel._worker()
        self.assertFalse(channel.status()["connected"])
        return capture

    def test_silent_stream_releases_after_elapsed_budget(self):
        cap = self.run_capture()
        self.assertAlmostEqual(cap.released_at, 5, delta=0.11)
        self.assertLess(cap.calls, 15)

    def test_new_frame_resets_stall_budget(self):
        cap = self.run_capture(good_grab=3)
        self.assertIsNotNone(cap.frame_at)
        self.assertAlmostEqual(cap.released_at - cap.frame_at, 5, delta=0.11)

    def test_low_frame_rate_has_three_intervals_of_grace(self):
        cap = self.run_capture(fps=0.1)
        self.assertAlmostEqual(cap.released_at, 30, delta=0.11)

    def test_closed_capture_does_not_wait_for_stall_budget(self):
        self.assertLess(self.run_capture(closes=True).released_at, 5)

    def test_invalid_timeout_falls_back_to_finite_default(self):
        for value in ("invalid", "nan", "inf", "0", "-1"):
            with self.subTest(value=value):
                self.assertAlmostEqual(self.run_capture(timeout=value).released_at, 5, delta=0.11)


if __name__ == "__main__":
    unittest.main()
