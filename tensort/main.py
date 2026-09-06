# main.py  (Python 3.6)
import asyncio
import logging
import threading
import os
from typing import Any, Dict

from flask import Flask

from channels.channel_config import VideoChannelConfig
from repositories import CameraRepository
from routes import register_routes
from service import DiscoveryService

# IMPORTANT:
# Do NOT import trt_infer / simple_model_pipeline at top-level if they import PyCUDA/TensorRT.
# We import them inside the pipeline thread so CUDA context is created in that same thread.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jetson-app")
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

app = Flask(__name__)
DEFAULT_SAMPLE_FPS = float(os.getenv("DEFAULT_SAMPLE_FPS", "10.0"))
DEFAULT_RESIZE_W = int(os.getenv("DEFAULT_RESIZE_W", "640"))
DEFAULT_RESIZE_H = int(os.getenv("DEFAULT_RESIZE_H", "480"))
MAX_SAMPLE_FPS = float(os.getenv("MAX_SAMPLE_FPS", "12.0"))
# -----------------------------
# Pipeline runtime (async loop in background thread)
# -----------------------------
class PipelineRuntime(object):
    def __init__(self, repository=None):
        self.loop = None
        self.pipeline = None
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._cameras = {}
        self._repo = repository or CameraRepository()
        self.discovery = None

        # Initialize database tables
        if not self._repo.initialize_tables():
            logger.error("Failed to initialize database tables")

        t = threading.Thread(target=self._run_loop, name="pipeline-loop", daemon=True)
        t.start()

        if not self._ready.wait(30.0):
            raise RuntimeError("Pipeline thread did not start within timeout.")
        
        # Restore cameras from database after pipeline is ready
        self._restore_cameras_from_db()

    def _normalize_camera_cfg(self, cfg_data: Dict[str, Any]) -> Dict[str, Any]:
        cfg_data = dict(cfg_data or {})

        try:
            sample_fps = float(cfg_data.get("sample_fps", DEFAULT_SAMPLE_FPS))
        except Exception:
            sample_fps = DEFAULT_SAMPLE_FPS

        cfg_data["sample_fps"] = max(0.1, min(sample_fps, MAX_SAMPLE_FPS))
        if DEFAULT_RESIZE_W > 0 and DEFAULT_RESIZE_H > 0:
            cfg_data["resize"] = (DEFAULT_RESIZE_W, DEFAULT_RESIZE_H)
        else:
            cfg_data["resize"] = None

        return cfg_data

    def _run_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            from pipeline import SimpleInferencePipeline

            pipeline = SimpleInferencePipeline()   # NO build_default here
            loop.run_until_complete(pipeline.start())
            self.loop = loop
            self.pipeline = pipeline
            self._ready.set()
            loop.run_forever()

        except Exception as e:
            logger.exception("Pipeline thread crashed: %s", e)
            self._ready.set()
        finally:
            try:
                loop.stop()
            except Exception:
                pass

    def _call(self, coro, timeout_s=10.0):
        if self.loop is None or self.pipeline is None:
            raise RuntimeError("Pipeline not ready")
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return fut.result(timeout=timeout_s)

    def _require_pipeline(self):
        pipeline = self.pipeline
        if self.loop is None or pipeline is None:
            raise RuntimeError("Pipeline not ready")
        return pipeline

    # ---------- database operations ----------
    # Persistence lives in CameraRepository; these thin wrappers marshal the
    # async repository calls onto the pipeline loop from the Flask thread.
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
                    # Build config first — if this raises (bad stored JSON) we
                    # must NOT add the camera to self._cameras, otherwise it
                    # appears in list_cameras() as "on device" but has no active
                    # pipeline channel (ghost camera that blocks future syncs).
                    cfg = None
                    if cfg_data.get("enabled", True):
                        cfg = VideoChannelConfig(**cfg_data)

                    # Only register in memory after config is validated
                    with self._lock:
                        self._cameras[camera_uuid] = cfg_data

                    if cfg is not None:
                        self._call(pipeline.add_channel(cfg), timeout_s=15.0)
                        restored_count += 1
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
        # Use camera_uuid from patch if provided (from Azure), otherwise generate new
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
            "reconnect_base_ms": 1000,
            "reconnect_max_ms": 8000,
        }

        for k, v in (cfg_patch or {}).items():
            if v is not None:
                cfg_data[k] = v

        cfg_data = self._normalize_camera_cfg(cfg_data)

        cfg = VideoChannelConfig(**cfg_data)  # your simple config class should accept these

        with self._lock:
            self._cameras[cam_id] = dict(cfg_data)

        # persist to database
        self._save_camera_to_db(cam_id, cfg_data)

        # add to pipeline (async)
        pipeline = self._require_pipeline()
        self._call(pipeline.add_channel(cfg), timeout_s=15.0)

        return {
            "camera_uuid": cam_id,
            "source_url": source_url,
            "config": self._cameras[cam_id],
        }

    def remove_camera(self, camera_uuid: str) -> bool:
        with self._lock:
            existed = camera_uuid in self._cameras
            if existed:
                del self._cameras[camera_uuid]

        if existed:
            pipeline = self._require_pipeline()
            self._call(pipeline.remove_channel(camera_uuid), timeout_s=10.0)

        # then delete from database
        self._delete_camera_from_db(camera_uuid)
        return existed

    def list_cameras(self):
        with self._lock:
            cams = []
            for cam_id, cfg in self._cameras.items():
                cams.append({"camera_uuid": cam_id, "source_url": cfg.get("source_url"), "config": cfg})
            return cams

    def patch_camera(self, camera_uuid: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            cfg_data = self._cameras.get(camera_uuid)
            if cfg_data is None:
                raise KeyError("camera not found")
            for k, v in patch.items():
                if v is not None:
                    cfg_data[k] = v

            cfg_data = self._normalize_camera_cfg(cfg_data)

            self._cameras[camera_uuid] = cfg_data
            new_cfg = VideoChannelConfig(**cfg_data)

        # persist updated config to database
        self._save_camera_to_db(camera_uuid, cfg_data)

        pipeline = self._require_pipeline()
        self._call(pipeline.add_channel(new_cfg), timeout_s=15.0)

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


runtime = PipelineRuntime()
runtime.start_discovery()

# -----------------------------
# Routes (blueprints in routes/)
# -----------------------------
register_routes(app, runtime)


if __name__ == "__main__":
    # threaded=True lets Flask handle multiple requests
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), threaded=True)
