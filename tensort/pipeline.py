# simple_model_pipeline.py  (Python 3.6)
import asyncio
import base64
import logging
import threading
import queue
import os
from typing import Any, Dict, Optional, List

try:
    # Script mode (python main.py from Backend/tensort)
    from channels.channel_config import VideoChannelConfig
    from channels.channel import VideoChannel, RTSPEvent
except Exception:
    # Package mode (python -m Backend.tensort.main)
    from .channels.channel_config import VideoChannelConfig
    from .channels.channel import VideoChannel, RTSPEvent

logger = logging.getLogger(__name__)
_DONE = object()


def _encode_thumbnail_data_url(frame_bgr, *, max_edge: int = 200, jpeg_quality: int = 60) -> Optional[str]:
    if frame_bgr is None:
        return None

    try:
        import cv2

        height, width = frame_bgr.shape[:2]
        if height <= 0 or width <= 0:
            return None

        scale = min(float(max_edge) / float(max(height, width)), 1.0)
        thumb = frame_bgr
        if scale < 1.0:
            thumb = cv2.resize(
                frame_bgr,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )

        ok, encoded = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if not ok:
            return None

        return f"data:image/jpeg;base64,{base64.b64encode(encoded.tobytes()).decode('ascii')}"
    except Exception:
        return None


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        v = int(os.getenv(name, str(default)))
    except Exception:
        v = int(default)
    return max(minimum, v)


class Broadcaster:
    def __init__(self):
        self._subscribers = set()
        self._lock = asyncio.Lock()

    async def subscribe(self):
        q = asyncio.Queue(maxsize=1000)
        async with self._lock:
            self._subscribers.add(q)
        return q

    async def unsubscribe(self, q):
        async with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    async def broadcast(self, msg):
        async with self._lock:
            for q in list(self._subscribers):
                try:
                    q.put_nowait(msg)
                except asyncio.QueueFull:
                    pass

class CoalescingBuffer(object):
    def __init__(self, max_pending_keys=1000):
        self._latest = {}  # camera_uuid -> RTSPEvent
        self._pending = asyncio.Queue(maxsize=max_pending_keys)
        self._in_queue = set()
        self._lock = asyncio.Lock()

    async def put(self, ev):
        key = str(ev.camera_uuid)
        async with self._lock:
            self._latest[key] = ev
            if key in self._in_queue:
                return
            try:
                self._pending.put_nowait(key)
                self._in_queue.add(key)
            except asyncio.QueueFull:
                pass

    async def get(self):
        while True:
            key = await self._pending.get()
            async with self._lock:
                if key in self._in_queue:
                    self._in_queue.remove(key)
                ev = self._latest.get(key)
            if ev is not None:
                return ev


class InferenceWorker(object):
    """
    Dedicated inference thread that OWNS:
      - PyCUDA context (via trt_infer import)
      - TRT engines
      - CUDA streams

    You must not call TRT from other threads.
    """
    def __init__(self, loop, max_q=2):
        self._loop = loop
        self._q = queue.Queue(maxsize=max_q)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="trt-infer-worker", daemon=True)

        self._infer = None
        self._thread.start()

    def submit(self, bgr, meta, fut):
        """
        bgr: numpy ndarray (BGR)
        meta: dict
        fut: asyncio.Future (created in event loop thread)
        """
        try:
            self._q.put_nowait((bgr, meta, fut))
            return True
        except queue.Full:
            # drop if overloaded: set a failure result (non-blocking)
            def _set():
                if not fut.done():
                    fut.set_result({
                        "type": "InferenceFailedEvent",
                        "camera_uuid": str(meta.get("camera_uuid", "unknown")),
                        "frame_ts_ms": int(meta.get("frame_ts_ms", 0)),
                        "frame_seq": int(meta.get("frame_seq", 0)),
                        "reason": "Inference queue full (dropped)",
                    })
            self._loop.call_soon_threadsafe(_set)
            return False

    def stop(self):
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except Exception:
            pass

    def join(self, timeout=2.0):
        try:
            self._thread.join(timeout)
        except Exception:
            pass

    def _run(self):
        """
        Runs inside the inference thread.
        Import trt_infer here so PyCUDA context is created in this thread.
        """
        from trt_infer import build_default
        try:
            self._infer = build_default()
            logger.info("TRTInfer initialized inside inference thread.")
        except Exception as e:
            logger.exception("Failed to init TRT infer in worker thread: %s", e)
            self._infer = None

        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.5)
            except Exception:
                continue

            if item is None:
                continue

            bgr, meta, fut = item

            if self._infer is None:
                res = {
                    "type": "InferenceFailedEvent",
                    "camera_uuid": str(meta.get("camera_uuid", "unknown")),
                    "frame_ts_ms": int(meta.get("frame_ts_ms", 0)),
                    "frame_seq": int(meta.get("frame_seq", 0)),
                    "reason": "TRT inference not initialized",
                }
                self._loop.call_soon_threadsafe(self._safe_set_result, fut, res)
                continue

            try:
                res = self._infer.infer_multitask(bgr, meta)
            except Exception as e:
                res = {
                    "type": "InferenceFailedEvent",
                    "camera_uuid": str(meta.get("camera_uuid", "unknown")),
                    "frame_ts_ms": int(meta.get("frame_ts_ms", 0)),
                    "frame_seq": int(meta.get("frame_seq", 0)),
                    "reason": "{}: {}".format(type(e).__name__, e),
                }

            self._loop.call_soon_threadsafe(self._safe_set_result, fut, res)

    def _safe_set_result(self, fut, res):
        try:
            if not fut.done():
                fut.set_result(res)
        except Exception:
            pass


class SimpleInferencePipeline(object):
    def __init__(self, out_queue_max=None, infer_q_max=None):
        if out_queue_max is None:
            out_queue_max = _env_int("PIPELINE_OUT_QUEUE_MAX", 500, minimum=10)
        if infer_q_max is None:
            infer_q_max = _env_int("INFER_QUEUE_MAX", 8, minimum=1)

        self._channels = {}
        self._channel_tasks = {}
        self._closing = False
        self._started = False

        self._buffer = CoalescingBuffer(max_pending_keys=_env_int("PENDING_KEY_MAX", 1000, minimum=100))
        self._out_q = asyncio.Queue(maxsize=out_queue_max)

        self._latest = {}
        self._latest_lock = asyncio.Lock()

        self._lock = asyncio.Lock()
        self._inference_task = None
        self._infer_worker = None  # created on start()
        self._infer_q_max = int(infer_q_max)
        self._log_every_n = _env_int("PIPELINE_LOG_EVERY_N_FRAMES", 0, minimum=0)
        self._stats = {
            "frames_in": 0,
            "infer_ok": 0,
            "infer_fail": 0,
            "infer_dropped": 0,
            "detections_total": 0,
            "alerts_attempted": 0,
        }

        self.broadcaster = Broadcaster()

    async def add_channel(self, cfg):
        camera_key = str(cfg.camera_uuid)
        new_channel = VideoChannel(cfg)

        async with self._lock:
            old_ch = self._channels.pop(camera_key, None)
            old_task = self._channel_tasks.pop(camera_key, None)

            self._channels[camera_key] = new_channel
            should_start = self._started and (not self._closing)

        if old_task is not None and not old_task.done():
            old_task.cancel()
            try:
                await old_task
            except Exception:
                pass

        if old_ch is not None:
            try:
                await old_ch.stop()
            except Exception:
                pass

        if should_start:
            # Record the task inside the lock so that a concurrent remove_channel
            # that runs between the lock release above and here sees the task and
            # can cancel it instead of letting it run orphaned.
            async with self._lock:
                if camera_key in self._channels and not self._closing:
                    self._start_channel_task(camera_key)

    async def remove_channel(self, camera_uuid):
        camera_key = str(camera_uuid)
        async with self._lock:
            ch = self._channels.pop(camera_key, None)
            t = self._channel_tasks.pop(camera_key, None)

        if t is not None and not t.done():
            t.cancel()
            try:
                await t
            except Exception:
                pass

        if ch is not None:
            try:
                await ch.stop()
            except Exception:
                pass

        async with self._latest_lock:
            if camera_key in self._latest:
                del self._latest[camera_key]

    def list_channels(self):
        return list(self._channels.keys())

    async def start(self):
        async with self._lock:
            if self._started:
                return
            self._started = True
            self._closing = False

            loop = asyncio.get_event_loop()
            self._infer_worker = InferenceWorker(loop=loop, max_q=self._infer_q_max)

            self._inference_task = asyncio.ensure_future(self._pump_inference())

            for camera_key in list(self._channels.keys()):
                self._start_channel_task(camera_key)

    async def shutdown(self):
        async with self._lock:
            if not self._started:
                return
            self._closing = True

            for t in list(self._channel_tasks.values()):
                if t is not None and not t.done():
                    t.cancel()

            if self._inference_task is not None and not self._inference_task.done():
                self._inference_task.cancel()

        for t in list(self._channel_tasks.values()):
            try:
                await t
            except Exception:
                pass

        for ch in list(self._channels.values()):
            try:
                await ch.stop()
            except Exception:
                pass

        if self._inference_task is not None:
            try:
                await self._inference_task
            except Exception:
                pass

        if self._infer_worker is not None:
            self._infer_worker.stop()
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(None, self._infer_worker.join, 2.0)
            except Exception:
                pass

        async with self._lock:
            self._channel_tasks = {}
            self._channels = {}
            self._started = False

        await self._put_out(_DONE)

    async def _put_out(self, ev):
        if self._out_q.full():
            try:
                _ = self._out_q.get_nowait()
            except Exception:
                pass
        try:
            self._out_q.put_nowait(ev)
        except Exception:
            pass

    def _start_channel_task(self, camera_key):
        ch = self._channels.get(camera_key)
        if ch is None:
            return
        self._channel_tasks[camera_key] = asyncio.ensure_future(self._pump_channel(camera_key, ch))

    async def _pump_channel(self, camera_key, ch):
        try:
            logger.info("[Jetson] Starting pump for camera %s", camera_key)
            async for ev in ch.stream(event_queue=None):
                if self._closing:
                    break

                if isinstance(ev, RTSPEvent):
                    if getattr(ev, "detection_enabled", True):
                        await self._buffer.put(ev)
                else:
                    await self._put_out(ev)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self._put_out({"type": "ChannelPumpFailed", "camera_uuid": camera_key, "reason": str(e)})
        finally:
            try:
                await ch.stop()
            except Exception:
                pass

    async def _pump_inference(self):
        """
        Sends frames to the TRT inference thread (does NOT call TRT directly here).
        Receives results and sends significant detections to Azure if configured.
        """
        loop = asyncio.get_event_loop()
        notify_url = os.getenv("AZURE_NOTIFY_URL")

        # helper to send alert without blocking pipeline
        def _send_alert(url, payload):
            try:
                import requests
                requests.post(url, json=payload, timeout=2.0)
            except Exception:
                pass # fire and forget

        try:
            while not self._closing:
                rtsp_ev = await self._buffer.get()
                self._stats["frames_in"] += 1
                if self._log_every_n and (int(getattr(rtsp_ev, "seq", 0)) % self._log_every_n == 0):
                    logger.debug(
                        "[Jetson] Received frame camera=%s seq=%s",
                        rtsp_ev.camera_uuid,
                        rtsp_ev.seq,
                    )

                bgr = getattr(rtsp_ev, "frame", None)
                if bgr is None:
                    enc = getattr(rtsp_ev, "encoded", None)
                    if enc is not None:
                        import numpy as np
                        import cv2
                        arr = np.frombuffer(enc, dtype=np.uint8)
                        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                if bgr is None:
                    self._stats["infer_fail"] += 1
                    await self._put_out({
                        "type": "InferenceFailedEvent",
                        "camera_uuid": str(rtsp_ev.camera_uuid),
                        "frame_ts_ms": int(rtsp_ev.ts_ms),
                        "frame_seq": int(rtsp_ev.seq),
                        "reason": "No frame data found (frame=None and encoded=None)",
                    })
                    continue

                meta = {
                    "camera_uuid": str(rtsp_ev.camera_uuid),
                    "channel_id": getattr(rtsp_ev, "channel_id", None),
                    "frame_ts_ms": int(rtsp_ev.ts_ms),
                    "frame_seq": int(rtsp_ev.seq),
                }

                # create an asyncio Future to receive result
                fut = asyncio.Future()

                if self._infer_worker is None:
                    await self._put_out({
                        "type": "InferenceFailedEvent",
                        "camera_uuid": meta["camera_uuid"],
                        "frame_ts_ms": meta["frame_ts_ms"],
                        "frame_seq": meta["frame_seq"],
                        "reason": "Inference worker not running",
                    })
                    continue

                ok = self._infer_worker.submit(bgr, meta, fut)
                if not ok:
                    result = await fut
                else:
                    try:
                        result = await asyncio.wait_for(fut, timeout=2.0)  # tune
                    except asyncio.TimeoutError:
                        result = {
                            "type": "InferenceFailedEvent",
                            "camera_uuid": meta["camera_uuid"],
                            "frame_ts_ms": meta["frame_ts_ms"],
                            "frame_seq": meta["frame_seq"],
                            "reason": "Inference timed out",
                        }
                if isinstance(result, dict) and result.get("type") == "InferenceFailedEvent":
                    self._stats["infer_fail"] += 1
                    reason = str(result.get("reason", "") or "")
                    if "queue full" in reason.lower():
                        self._stats["infer_dropped"] += 1
                    logger.warning(
                        "[Jetson] Inference failed camera=%s seq=%s reason=%s",
                        rtsp_ev.camera_uuid,
                        rtsp_ev.seq,
                        reason or "unknown error",
                    )
                else:
                    self._stats["infer_ok"] += 1
                    if isinstance(result, dict):
                        self._stats["detections_total"] += len(result.get("detections", []) or [])

                if notify_url and result.get("detections"):
                    dets = []
                    for d in result["detections"]:
                        if isinstance(d, dict):
                             dets.append(d)
                        elif hasattr(d, "model_dump"):
                             dets.append(d.model_dump())
                        else:
                             dets.append({
                                 "cls_name": getattr(d, "cls_name", "unknown"),
                                 "conf": getattr(d, "conf", 0.0),
                                 "box": getattr(d, "box", [])
                             })
                    
                    if dets:
                        self._stats["alerts_attempted"] += 1
                        frame_h = result.get("frame_h")
                        frame_w = result.get("frame_w")
                        if (frame_w is None or frame_h is None) and bgr is not None:
                            frame_h, frame_w = bgr.shape[:2]
                        alert_payload = {
                            "camera_uuid": str(meta["camera_uuid"]),
                            "frame_ts_ms": int(meta["frame_ts_ms"]),
                            "frame_seq": int(meta["frame_seq"]),
                            "detections": dets
                        }
                        if frame_w is not None:
                            alert_payload["frame_w"] = int(frame_w)
                        if frame_h is not None:
                            alert_payload["frame_h"] = int(frame_h)
                        image_url = _encode_thumbnail_data_url(bgr)
                        if image_url:
                            alert_payload["image_url"] = image_url
                        loop.run_in_executor(None, _send_alert, notify_url, alert_payload)

                async with self._latest_lock:
                    self._latest[str(rtsp_ev.camera_uuid)] = result

                await self.broadcaster.broadcast(result)
                await self._put_out(result)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self._put_out({"type": "InferencePumpFailed", "reason": str(e)})

    async def events(self):
        if not self._started:
            await self.start()

        while True:
            item = await self._out_q.get()
            if item is _DONE:
                break
            yield item

    async def get_latest(self, camera_uuid):
        async with self._latest_lock:
            return self._latest.get(str(camera_uuid))

    async def get_stats(self) -> Dict[str, Any]:
        async with self._lock:
            channel_count = len(self._channels)
        async with self._latest_lock:
            latest_count = len(self._latest)
        stats = dict(self._stats)
        stats.update({
            "channel_count": int(channel_count),
            "latest_cache_size": int(latest_count),
            "infer_queue_max": int(self._infer_q_max),
            "out_queue_max": int(self._out_q.maxsize),
        })
        return stats
