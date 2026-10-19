"""Camera configuration and the background asyncio pipeline runtime."""

import asyncio
import logging
import math
import os
import threading
from typing import Any, Dict

if __package__:
    from .channels.channel_config import VideoChannelConfig
    from .repositories import CameraRepository
    from .limits import CameraCapacityError, camera_limit
else:
    from channels.channel_config import VideoChannelConfig
    from repositories import CameraRepository
    from limits import CameraCapacityError, camera_limit

logger = logging.getLogger("jetson-app")

DEFAULT_SAMPLE_FPS = float(os.getenv("DEFAULT_SAMPLE_FPS", "5.0"))
DEFAULT_RESIZE_W = int(os.getenv("DEFAULT_RESIZE_W", "640"))
DEFAULT_RESIZE_H = int(os.getenv("DEFAULT_RESIZE_H", "480"))
MAX_SAMPLE_FPS = float(os.getenv("MAX_SAMPLE_FPS", "12.0"))

# rtspsrc jitter buffer, in milliseconds. Raise it for cameras reached over the
# internet (a remote NVR), lower it on a dedicated LAN where latency matters
# more than tolerance to jitter.
DEFAULT_GST_LATENCY_MS = int(os.getenv("GST_LATENCY_MS", "200"))

# Reconnect backoff bounds, in milliseconds. The ceiling matters for cameras
# behind a remote NVR: when the NVR sheds sessions every camera fails at once,
# and a short ceiling means they all re-hammer it forever instead of letting it
# recover.
RECONNECT_BASE_MS = int(os.getenv("RECONNECT_BASE_MS", "1000"))
RECONNECT_MAX_MS = int(os.getenv("RECONNECT_MAX_MS", "30000"))

# -----------------------------
# Pipeline runtime (async loop in background thread)
# -----------------------------
class PipelineRuntime(object):
    def __init__(self, repository=None):
        self.loop = None
        self.pipeline = None
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._camera_update_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closing = False
        self._closed = False
        self._max_cameras = camera_limit()
        self._startup_error = None
        self._cameras = {}
        self._repo = repository or CameraRepository()

        # Initialize database tables
        if not self._repo.initialize_tables():
            logger.error("Failed to initialize database tables")

        self._loop_thread = threading.Thread(target=self._run_loop, name="pipeline-loop", daemon=True)
        self._loop_thread.start()

        if not self._ready.wait(65.0):
            raise RuntimeError("Pipeline thread did not start within timeout.")
        if self._startup_error:
            raise RuntimeError("Pipeline startup failed: {}".format(self._startup_error))
        
        # Restore cameras from database after pipeline is ready
        self._restore_cameras_from_db()

    def _normalize_camera_cfg(self, cfg_data: Dict[str, Any]) -> Dict[str, Any]:
        cfg_data = dict(cfg_data or {})
        for key in ("enabled", "detection_enabled", "notification_enabled"):
            value = cfg_data.get(key, True)
            if isinstance(value, str):
                value = value.strip().lower() in ("1", "true", "yes", "on")
            cfg_data[key] = bool(value)

        try:
            sample_fps = float(cfg_data.get("sample_fps", DEFAULT_SAMPLE_FPS))
        except Exception:
            sample_fps = DEFAULT_SAMPLE_FPS
        if not math.isfinite(sample_fps) or sample_fps <= 0:
            raise ValueError("sample_fps must be finite and greater than zero")

        cfg_data["sample_fps"] = max(0.1, min(sample_fps, MAX_SAMPLE_FPS))
        if "resize" not in cfg_data:
            cfg_data["resize"] = ((DEFAULT_RESIZE_W, DEFAULT_RESIZE_H)
                                  if DEFAULT_RESIZE_W > 0 and DEFAULT_RESIZE_H > 0 else None)
        if not cfg_data.get("gst_latency_ms"):
            cfg_data["gst_latency_ms"] = DEFAULT_GST_LATENCY_MS

        return cfg_data

    def _run_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            if __package__:
                from .pipeline import SimpleInferencePipeline
            else:
                from pipeline import SimpleInferencePipeline
            import cv2
            cv2.setNumThreads(max(1, int(os.getenv("OPENCV_NUM_THREADS", "1"))))
            pipeline = SimpleInferencePipeline()
            loop.run_until_complete(pipeline.start())
            self.loop = loop
            self.pipeline = pipeline
            self._ready.set()
            loop.run_forever()

        except Exception as e:
            logger.exception("Pipeline thread crashed: %s", e)
            self._startup_error = str(e)
            self._ready.set()
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                logger.exception("Failed to close pipeline async generators")
            finally:
                loop.close()

    def close(self):
        """Drain native resources before the server process finalizes Python.

        Called on the main thread. Database camera records are preserved. Keep
        the event loop alive until native workers finish posting their final
        callbacks and releasing decoder/CUDA resources on their owner threads.
        """
        with self._close_lock:
            if self._closed:
                return
            self._closing = True
            logger.info("Stopping camera capture and inference workers")
            # Let an already-admitted HTTP update finish. Further updates fail
            # _require_pipeline before they can start another capture worker.
            with self._camera_update_lock:
                pass
            pipeline, loop = self.pipeline, self.loop
            if pipeline is not None and loop is not None and loop.is_running():
                async def drain():
                    # Inspect loop-owned channels on their event-loop thread.
                    channels = list(pipeline._channels.values())
                    for channel in channels:
                        channel._stop_thread_evt.set()
                    await pipeline.shutdown()
                    return channels

                future = asyncio.run_coroutine_threadsafe(drain(), loop)
                channels = future.result()
                # The ordinary pipeline shutdown uses bounded joins so API
                # operations stay responsive. Process exit must finish those
                # joins instead of abandoning a still-running native worker.
                for channel in channels:
                    thread = channel._thread
                    if thread is not None:
                        thread.join()
                if pipeline._infer_pool is not None:
                    pipeline._infer_pool.join(timeout=None)
                pipeline._snapshot_executor.shutdown(wait=True)
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(loop.stop)
            self._loop_thread.join()
            self._closed = True
            logger.info("Runtime stopped; native workers released")

    def _call(self, coro, timeout_s=10.0):
        if self.loop is None or self.pipeline is None:
            raise RuntimeError("Pipeline not ready")
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return fut.result(timeout=timeout_s)

    def _require_pipeline(self):
        if getattr(self, "_closing", False):
            raise RuntimeError("Pipeline is shutting down")
        pipeline = self.pipeline
        if self.loop is None or pipeline is None:
            raise RuntimeError("Pipeline not ready")
        return pipeline

    def _save_camera_to_db(self, camera_uuid: str, cfg_data: Dict[str, Any]):
        self._call(self._repo.save(camera_uuid, cfg_data), timeout_s=5.0)

    def _delete_camera_from_db(self, camera_uuid: str):
        self._call(self._repo.delete(camera_uuid), timeout_s=5.0)

    def _restore_cameras_from_db(self):
        """
        Restore all cameras from the database on startup.
        """
        try:
            pipeline = self._require_pipeline()
        except RuntimeError:
            logger.error("Skipping camera restore because pipeline is not ready")
            return

        try:
            cameras = self._repo.list_all()
            restored_count = 0

            for record in cameras:
                camera_uuid = record["camera_uuid"]
                try:
                    cfg_data = record["config_json"]
                    cfg_data = self._normalize_camera_cfg(cfg_data)
                    cfg_data["camera_uuid"] = str(camera_uuid)
                    cfg_data.setdefault("channel_id", str(camera_uuid))
                    cfg = None
                    if cfg_data.get("enabled", True):
                        cfg = VideoChannelConfig(**cfg_data)

                    if cfg is not None:
                        self._call(pipeline.add_channel(cfg), timeout_s=15.0)
                        restored_count += 1
                    with self._lock:
                        self._cameras[camera_uuid] = cfg_data
                except Exception as e:
                    logger.exception(f"Failed to restore camera {camera_uuid}: {e}")

            if restored_count > 0:
                logger.info(f"Successfully restored {restored_count} camera(s) from database")
            else:
                logger.info("No cameras to restore from database")
        except Exception as e:
            logger.exception(f"Failed to restore cameras from database: {e}")

    # ---------- camera operations ----------
    def add_camera(self, source_url: str, cfg_patch: Dict[str, Any]) -> Dict[str, Any]:
        cam_id = cfg_patch.get("camera_uuid")

        if not cam_id:
            raise ValueError("camera_uuid must be provided by NVR or backend (do not let Jetson generate IDs)")
        cam_id = str(cam_id)

        cfg_data = {
            "camera_uuid": cam_id,
            "channel_id": cfg_patch.get("channel_id") or cam_id,
            "source_url": source_url,
            "enabled": True,
            "detection_enabled": True,
            "notification_enabled": True,
            "sample_fps": DEFAULT_SAMPLE_FPS,
            "decode_backend": "gstreamer",
            "resize": (DEFAULT_RESIZE_W, DEFAULT_RESIZE_H) if DEFAULT_RESIZE_W > 0 and DEFAULT_RESIZE_H > 0 else None,
            "reconnect_base_ms": RECONNECT_BASE_MS,
            "reconnect_max_ms": RECONNECT_MAX_MS,
        }

        for k, v in (cfg_patch or {}).items():
            if v is not None:
                cfg_data[k] = v

        cfg_data = self._normalize_camera_cfg(cfg_data)

        cfg = VideoChannelConfig(**cfg_data)

        with self._camera_update_lock:
            self._apply_camera(cam_id, cfg_data, cfg)

        return {
            "camera_uuid": cam_id,
            "source_url": source_url,
            "config": dict(cfg_data),
        }

    def _apply_camera(self, camera_uuid, cfg_data, cfg):
        """Validate admission before touching the pipeline, database or roster.

        Caller holds _camera_update_lock, serializing concurrent HTTP
        mutations without holding the roster read lock during I/O.
        """
        with self._lock:
            other_enabled = sum(bool(c.get("enabled", True)) for key, c in self._cameras.items()
                                if key != camera_uuid)
        if cfg.enabled and other_enabled >= self._max_cameras:
            raise CameraCapacityError("Device supports at most {} enabled cameras".format(self._max_cameras))
        pipeline = self._require_pipeline()
        self._call(pipeline.add_channel(cfg), timeout_s=15.0)
        self._save_camera_to_db(camera_uuid, cfg_data)
        with self._lock:
            self._cameras[camera_uuid] = dict(cfg_data)

    def remove_camera(self, camera_uuid: str) -> bool:
        with self._camera_update_lock:
            with self._lock:
                existed = camera_uuid in self._cameras
            pipeline = self._require_pipeline()
            self._call(pipeline.remove_channel(camera_uuid), timeout_s=10.0)
            self._delete_camera_from_db(camera_uuid)
            with self._lock:
                self._cameras.pop(camera_uuid, None)
        return existed

    def list_cameras(self):
        with self._lock:
            cams = []
            for cam_id, cfg in self._cameras.items():
                cams.append({"camera_uuid": cam_id, "source_url": cfg.get("source_url"), "config": dict(cfg)})
            return cams

    def patch_camera(self, camera_uuid: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        with self._camera_update_lock:
            with self._lock:
                cfg_data = self._cameras.get(camera_uuid)
            if cfg_data is None:
                raise KeyError("camera not found")
            cfg_data = dict(cfg_data)
            if str(patch.get("camera_uuid", camera_uuid)) != camera_uuid:
                raise ValueError("camera_uuid cannot be changed")
            for k, v in patch.items():
                if v is not None:
                    cfg_data[k] = v

            cfg_data = self._normalize_camera_cfg(cfg_data)

            new_cfg = VideoChannelConfig(**cfg_data)
            self._apply_camera(camera_uuid, cfg_data, new_cfg)

        return {"camera_uuid": camera_uuid, "config": cfg_data}

    # The peek_* reads are plain dict lookups, so they run directly on the Flask
    # thread instead of being marshalled onto the pipeline loop.
    def get_latest(self, camera_uuid: str) -> Dict[str, Any]:
        return self._require_pipeline().peek_latest(camera_uuid)

    def get_snapshot(self, camera_uuid: str):
        return self._require_pipeline().peek_latest_snapshot(camera_uuid)

    def get_stats(self) -> Dict[str, Any]:
        if self.pipeline is None:
            return {}
        try:
            return self.pipeline.peek_stats()
        except Exception:
            logger.exception("Failed to fetch pipeline stats")
            return {}
