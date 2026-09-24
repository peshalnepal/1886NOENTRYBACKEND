"""Native appsink capture, independent of OpenCV's GStreamer videoio bridge.

The capture thread owns the pipeline and releases it. Pulls poll the stop event
so shutdown does not need to release a pipeline concurrently with a read.
"""

import threading
import time
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def gst_modules():
    import gi
    gi.require_version("Gst", "1.0")
    gi.require_version("GstVideo", "1.0")
    from gi.repository import Gst, GstVideo
    Gst.init(None)
    return Gst, GstVideo


def native_gstreamer_available():
    try:
        gst_modules()
        return True
    except (ImportError, ValueError):
        return False


class GstCapture:
    """Small VideoCapture-compatible interface returning owned BGR arrays."""

    def __init__(self, pipeline, stop_event=None, read_timeout_s=5.0):
        self._gst, self._video = gst_modules()
        self._stop = stop_event if stop_event is not None else threading.Event()
        self._read_timeout_s = read_timeout_s
        self._pipeline = None
        self._sink = None
        self._sample = None
        self._failed = False
        self._eos_received = False
        self.last_error = None
        context = self._gst.ParseContext.new()
        try:
            # Generated pipelines end in an unnamed appsink. Give it a stable
            # name; do not depend on OpenCV's appsink discovery or caps changes.
            self._pipeline = self._gst.parse_launch_full(
                pipeline + " name=frame_sink", context, self._gst.ParseFlags.NONE)
            self._sink = self._pipeline.get_by_name("frame_sink")
            if self._sink is None:
                raise RuntimeError("GStreamer pipeline has no frame sink")
            self._bus = self._pipeline.get_bus()
            if self._pipeline.set_state(self._gst.State.PLAYING) == self._gst.StateChangeReturn.FAILURE:
                raise RuntimeError("GStreamer could not start the capture pipeline")
        except Exception:
            self.release()
            missing = context.get_missing_elements()
            if missing:
                raise RuntimeError("GStreamer missing plugins: " + ", ".join(missing)) from None
            # A parse exception can contain the entire credential-bearing URL.
            raise RuntimeError("GStreamer pipeline creation failed; check required plugins") from None

    def _check_error(self):
        if self._pipeline is None:
            return
        message = self._bus.pop_filtered(self._gst.MessageType.ERROR | self._gst.MessageType.EOS)
        if message is not None and message.type == self._gst.MessageType.EOS:
            # Preserve EOS while allowing any last queued sample to be pulled.
            self._eos_received = True
        elif message is not None:
            error, _debug = message.parse_error()
            # Domain/code identify the failure without leaking URL credentials
            # through a plugin's debug string or authentication error message.
            self.last_error = "GStreamer error {}/{}".format(error.domain, error.code)
            self._failed = True

    def isOpened(self):
        self._check_error()
        return self._pipeline is not None and not self._failed

    def grab(self, timeout_s=None):
        self._sample = None
        budget = self._read_timeout_s if timeout_s is None else max(0.0, timeout_s)
        deadline = time.monotonic() + budget
        while not self._stop.is_set() and self.isOpened():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            sample = self._sink.emit("try-pull-sample", int(min(0.1, remaining) * self._gst.SECOND))
            if sample is not None:
                self._sample = sample
                return True
            # appsink's eos property is also true before the sink has started.
            # A slow RTSP handshake must not be mistaken for end-of-stream.
            if self._eos_received:
                self._failed = True
                self.last_error = "GStreamer end of stream"
                return False
        return False

    def retrieve(self):
        sample, self._sample = self._sample, None
        if sample is None:
            return False, None
        caps = sample.get_caps()
        if caps is None or caps.get_structure(0).get_string("format") != "BGR":
            self.last_error = "GStreamer appsink did not negotiate BGR"
            return False, None
        info = self._video.VideoInfo.new_from_caps(caps)
        if info is None:
            return False, None
        buffer = sample.get_buffer()
        if buffer is None:
            return False, None
        # GstVideoMeta can override caps-derived alignment. BGR rows are often
        # padded to four bytes, so reshaping width*height*3 silently corrupts
        # images whose width is not a multiple of four.
        meta = self._video.buffer_get_video_meta(buffer)
        stride = int(meta.stride[0] if meta else info.stride[0])
        offset = int(meta.offset[0] if meta else info.offset[0])
        width, height = int(info.width), int(info.height)
        ok, mapped = buffer.map(self._gst.MapFlags.READ)
        if not ok:
            return False, None
        try:
            if width <= 0 or height <= 0 or stride < width * 3 or offset < 0:
                return False, None
            required = offset + (height - 1) * stride + width * 3
            if len(mapped.data) < required:
                return False, None
            frame = np.ndarray((height, width, 3), dtype=np.uint8,
                               buffer=mapped.data, offset=offset,
                               strides=(stride, 3, 1)).copy()
            return True, frame
        finally:
            # The returned copy remains valid after unmap and pipeline release.
            buffer.unmap(mapped)

    def read(self):
        return self.retrieve() if self.grab() else (False, None)

    def release(self):
        pipeline, self._pipeline = self._pipeline, None
        self._sample = None
        self._sink = None
        if pipeline is not None:
            pipeline.set_state(self._gst.State.NULL)
