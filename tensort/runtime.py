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
    from .service import DiscoveryService
    from .env_utils import env_bool
    from .limits import CameraCapacityError, camera_limit
else:
    from channels.channel_config import VideoChannelConfig
    from repositories import CameraRepository
    from service import DiscoveryService
    from env_utils import env_bool
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
        self.discovery_managed = env_bool("DISCOVERY_ENABLED", True) and env_bool("DISCOVERY_AUTO_ADD", True)
        self._selected_sources = set()
        self._startup_error = None
        self._cameras = {}
        self._repo = repository or CameraRepository()
        self.discovery = None

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
        if not self.discovery_managed:
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
            logger.info("Stopping discovery, camera capture and inference workers")
            # Let an already-admitted HTTP update finish. Further updates fail
            # _require_pipeline before they can start another capture worker.
            with self._camera_update_lock:
                pass
            if self.discovery is not None:
                self.discovery.stop()
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
            if self.discovery is not None:
                for thread in (self.discovery._thread, self.discovery._requested_scan_thread):
                    if thread is not None:
                        thread.join()
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
                    if getattr(self, "discovery_managed", False):
                        if cfg_data.get("source_url") not in self._selected_sources:
                            continue
                        if camera_uuid in self._cameras:
                            continue
                        # One active UUID per selected endpoint.
                        if any(c.get("source_url") == cfg_data.get("source_url") for c in self._cameras.values()):
                            continue

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

    def select_discovery_sources(self, source_urls):
        """Reserve the selected endpoints; retire other captures without deleting history."""
        if not getattr(self, "discovery_managed", False):
            return
        selected = set(source_urls)
        if len(selected) > self._max_cameras:
            raise ValueError("Discovery selection exceeds MAX_CAMERAS")
        with self._camera_update_lock:
            self._selected_sources = selected
            for camera in self.list_cameras():
                if camera["source_url"] not in selected:
                    self._call(self._require_pipeline().remove_channel(camera["camera_uuid"]))
                    with self._lock:
                        self._cameras.pop(camera["camera_uuid"], None)
            self._restore_cameras_from_db()

    # ---------- camera operations ----------
    def add_camera(self, source_url: str, cfg_patch: Dict[str, Any]) -> Dict[str, Any]:
        cam_id = cfg_patch.get("camera_uuid")

        if not cam_id:
            raise ValueError("camera_uuid must be provided by backend (do not let Jetson generate IDs)")
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

        Caller holds _camera_update_lock, serializing concurrent HTTP and
        discovery mutations without holding the roster read lock during I/O.
        """
        if getattr(self, "discovery_managed", False):
            if cfg_data.get("source_url") not in self._selected_sources:
                raise CameraCapacityError("Camera is not selected by discovery (static NVR, dynamic NVR, local)")
            if any(key != camera_uuid and value.get("source_url") == cfg_data.get("source_url")
                   for key, value in self._cameras.items()):
                raise CameraCapacityError("Selected source already has a camera UUID")
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
            stats = self.pipeline.peek_stats()
            stats["camera_selection"] = "discovery" if getattr(self, "discovery_managed", False) else "manual"
            return stats
        except Exception:
            logger.exception("Failed to fetch pipeline stats")
            return {}

    def verify_source(self, source_url, stop_event):
        """Verify frames on the discovery thread, never the asyncio loop.

        Reuse a running capture's freshness instead of opening another session
        on the same NVR. New candidates get one temporary capture, released
        before admission opens the persistent channel.
        """
        if getattr(self, "_closing", False) or stop_event.is_set():
            return False
        with self._lock:
            existing = [(key, dict(cfg)) for key, cfg in self._cameras.items()
                        if cfg.get("source_url") == source_url and cfg.get("enabled", True)]
        if existing:
            capture = self.get_stats().get("capture", {})
            for key, cfg in existing:
                status = capture.get(key, {})
                age = status.get("last_frame_age_ms")
                max_age = max(15000, int(3000 / float(cfg.get("sample_fps", DEFAULT_SAMPLE_FPS))))
                if age is not None and age <= max_age:
                    return True
            return False

        try:
            from channels.channel import VideoChannel, CONNECT_GATE
        except ImportError:
            from .channels.channel import VideoChannel, CONNECT_GATE
        if not CONNECT_GATE.wait_turn(stop_event):
            return False
        cfg = self._normalize_camera_cfg({"camera_uuid": "discovery-probe", "source_url": source_url})
        channel = VideoChannel(VideoChannelConfig(**cfg))
        channel._stop_thread_evt = stop_event
        cap = None
        try:
            cap = channel._open_capture()
            if cap is None or not cap.isOpened():
                return False
            return (channel._last_frame_monotonic is not None or channel._probe_first_frame(cap))
        except Exception:
            logger.warning("Discovery candidate failed frame verification")
            return False
        finally:
            if cap is not None:
                cap.release()

    # ---------- discovery ----------
    def start_discovery(self) -> None:
        """Start the scheduled Hikvision discovery sweep.

        Called after __init__ so the DB restore has already repopulated
        self._cameras — otherwise the first sweep would see restored cameras
        as unknown and could double-provision them.
        """
        try:
            service = DiscoveryService(self)
            self.discovery = service
            service.start()
        except Exception:
            logger.exception("Failed to start camera discovery service")
            self.discovery = None
