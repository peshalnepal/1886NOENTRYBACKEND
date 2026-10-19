"""Real appsink/frame conversion tests; no camera, GPU or TensorRT required."""

import os
import sys
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from channels.gstreamer_capture import GstCapture, native_gstreamer_available
from channels.channel import VideoChannel
from channels.channel_config import VideoChannelConfig


def source(format="BGR", width=7, count="", extra=""):
    return ("videotestsrc is-live=true pattern=red {} ! {} "
            "video/x-raw,format={},width={},height=5,framerate=10/1 ! "
            "appsink drop=true sync=false max-buffers=1").format(count, extra, format, width)


@unittest.skipUnless(native_gstreamer_available(), "System GStreamer GI bindings required")
class NativeCaptureTests(unittest.TestCase):
    def test_capture_keeps_source_dimensions_until_proportional_resize(self):
        channel = VideoChannel(VideoChannelConfig(camera_uuid="test", resize=(640, 480)))
        for width, height, expected in (
            (1920, 1080, (360, 640)),
            (1280, 960, (480, 640)),
            (720, 1280, (480, 270)),
            (1920, 800, (267, 640)),
            (320, 240, (240, 320)),
        ):
            with self.subTest(source=(width, height)):
                pipeline = (
                    "videotestsrc num-buffers=1 pattern=red ! "
                    "video/x-raw,width={},height={},pixel-aspect-ratio=1/1,framerate=5/1 ! "
                ).format(width, height) + channel._gst_appsink_tail(hw=False)
                cap = GstCapture(pipeline)
                try:
                    ok, frame = cap.read()
                    self.assertTrue(ok)
                    self.assertEqual(frame.shape[:2], (height, width))
                    resized = channel._maybe_resize(frame)
                    self.assertEqual(resized.shape[:2], expected)
                    np.testing.assert_array_equal(resized[0, 0], [0, 0, 255])
                    if width <= 640 and height <= 480:
                        self.assertIs(resized, frame)
                finally:
                    cap.release()

    def test_missing_plugin_is_reported_without_stream_credentials(self):
        with self.assertRaises(RuntimeError) as error:
            GstCapture('uridecodebin uri="rtsp://user:secret@example.test/102" ! '
                       'noentry_missing_decoder ! appsink')
        self.assertIn("noentry_missing_decoder", str(error.exception))
        self.assertNotIn("secret", str(error.exception))
        self.assertNotIn("example.test", str(error.exception))

    def test_failed_candidate_reason_is_visible_after_successful_fallback(self):
        channel = VideoChannel(VideoChannelConfig(camera_uuid="test", source_url="rtsp://unused/102"))
        with patch.object(channel, "_gst_candidates", return_value=[
                ("missing-hw", "noentry_missing_decoder ! appsink"),
                ("working-cpu", source())]):
            cap = channel._open_capture()
        try:
            status = channel.status()
            self.assertEqual(status["backend"], "working-cpu")
            self.assertIn("noentry_missing_decoder", status["gstreamer_failures"]["missing-hw"])
            self.assertTrue(cap.read()[0])
        finally:
            cap.release()

    def test_padded_bgr_rows_and_owned_pixels_survive_release(self):
        for width in (7, 8):
            with self.subTest(width=width):
                cap = GstCapture(source(width=width))
                try:
                    ok, frame = cap.read()
                    self.assertTrue(ok)
                    self.assertEqual(frame.shape, (5, width, 3))
                    self.assertEqual(frame.dtype, np.uint8)
                    self.assertTrue(frame.flags.c_contiguous)
                finally:
                    cap.release()
                np.testing.assert_array_equal(frame, np.tile([0, 0, 255], (5, width, 1)))
                self.assertFalse(cap.isOpened())
                self.assertEqual(cap.read(), (False, None))
                cap.release()  # idempotent

    def test_eos_delivers_final_sample_then_stops(self):
        cap = GstCapture(source(count="num-buffers=1"))
        try:
            self.assertTrue(cap.read()[0])
            started = time.monotonic()
            self.assertEqual(cap.read(), (False, None))
            self.assertLess(time.monotonic() - started, 0.5)
        finally:
            cap.release()

    def test_cancel_interrupts_a_capture_with_no_frames(self):
        stopped = threading.Event()
        cap = GstCapture(source(extra="valve drop=true !"), stop_event=stopped)
        timer = threading.Timer(0.1, stopped.set)
        timer.start()
        try:
            started = time.monotonic()
            self.assertFalse(cap.grab(timeout_s=15))
            self.assertLess(time.monotonic() - started, 1)
        finally:
            cap.release()
            timer.join()

    def test_no_frames_respects_probe_deadline(self):
        cap = GstCapture(source(extra="valve drop=true !"))
        try:
            started = time.monotonic()
            self.assertFalse(cap.grab(timeout_s=0.15))
            self.assertLess(time.monotonic() - started, 0.5)
        finally:
            cap.release()

    def test_non_bgr_sample_is_not_misinterpreted(self):
        cap = GstCapture(source(format="RGB"))
        try:
            self.assertEqual(cap.read(), (False, None))
            self.assertIn("BGR", cap.last_error)
        finally:
            cap.release()

    def test_channel_uses_native_capture_even_without_opencv_gstreamer(self):
        channel = VideoChannel(VideoChannelConfig(camera_uuid="test", source_url="rtsp://unused/102"))
        with patch.object(channel, "_gst_candidates", return_value=[("test-native", source())]), \
                patch("channels.channel.cv2.getBuildInformation", return_value="GStreamer: NO"), \
                patch.object(channel, "_capture_with_timeout", side_effect=AssertionError("OpenCV used")):
            cap = channel._open_capture()
            try:
                self.assertIsInstance(cap, GstCapture)
                self.assertTrue(cap.read()[0])
                self.assertEqual(channel.status()["backend"], "test-native")
                self.assertIsNotNone(channel.status()["last_frame_age_ms"])
            finally:
                cap.release()

    def test_bad_pipeline_does_not_expose_credentials(self):
        with self.assertRaises(RuntimeError) as error:
            GstCapture('missing_plugin location="rtsp://user:secret@example.test/102" ! appsink')
        self.assertNotIn("secret", str(error.exception))
        self.assertNotIn("example.test", str(error.exception))


class PipelineSelectionTests(unittest.TestCase):
    def test_hardware_capture_does_not_force_a_distorted_size(self):
        channel = VideoChannel(VideoChannelConfig(camera_uuid="test", resize=(640, 480)))
        tail = channel._gst_appsink_tail(hw=True)
        self.assertIn("nvvidconv", tail)
        self.assertNotIn("width=", tail)
        self.assertNotIn("height=", tail)

    def test_disabled_resize_preserves_the_original_frame(self):
        channel = VideoChannel(VideoChannelConfig(camera_uuid="test", resize=None))
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.assertIs(channel._maybe_resize(frame), frame)

    def test_fallback_probe_waits_longer_than_wan_jitter_buffer(self):
        from unittest.mock import MagicMock
        channel = VideoChannel(VideoChannelConfig(
            camera_uuid="test", source_url="rtsp://unused/102", gst_latency_ms=4000))
        cap = MagicMock()
        cap.isOpened.return_value = True
        with patch("channels.channel.native_gstreamer_available", return_value=True), \
                patch("channels.channel.GstCapture", return_value=cap), \
                patch.object(channel, "_gst_candidates", return_value=[("first", "a"), ("fallback", "b")]), \
                patch.object(channel, "_probe_first_frame", side_effect=[False, True]) as probe, \
                patch.dict(os.environ, {"CAPTURE_PROBE_S": "12"}):
            self.assertIs(channel._open_capture(), cap)
        self.assertGreater(probe.call_args.args[1], 4.0)
        cap.release.assert_called_once()

    def test_cpu_autodecode_forces_software_decoders(self):
        channel = VideoChannel(VideoChannelConfig(camera_uuid="test", source_url="rtsp://unused/102"))
        candidates = dict(channel._gst_candidates())
        self.assertIn("force-sw-decoders=true", candidates["rtsp-uridecodebin-cpu"])
        self.assertNotIn("force-sw-decoders=true", candidates["rtsp-uridecodebin-hw"])

    def test_failed_opens_back_off_exponentially(self):
        self._check_backoff(opened=False)

    def test_open_handle_without_frames_does_not_reset_backoff(self):
        self._check_backoff(opened=True)

    def _check_backoff(self, opened):
        from unittest.mock import MagicMock
        channel = VideoChannel(VideoChannelConfig(camera_uuid="test", source_url="rtsp://unused/102"))
        cap = MagicMock()
        cap.isOpened.return_value = opened
        clock = [0.0]

        def grab():
            clock[0] += 5.0
            return False

        cap.grab.side_effect = grab
        delays = []

        def wait(delay):
            delays.append(delay)
            return len(delays) == 3

        channel._stop_thread_evt = MagicMock()
        channel._stop_thread_evt.is_set.return_value = False
        channel._stop_thread_evt.wait.side_effect = wait
        with patch.object(channel, "_open_capture", return_value=cap), \
                patch("channels.channel.CONNECT_GATE.wait_turn", return_value=True), \
                patch("channels.channel.random.random", return_value=0.5), \
                patch("channels.channel.time.monotonic", side_effect=lambda: clock[0]), \
                patch.dict(os.environ, {"CAPTURE_STALL_TIMEOUT_S": "5"}):
            channel._worker()
        self.assertEqual(delays, [1.0, 2.0, 4.0])
        self.assertEqual(cap.release.call_count, 3)


if __name__ == "__main__":
    unittest.main()
