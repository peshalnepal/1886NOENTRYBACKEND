# main.py  (Python 3.6)
import asyncio
import logging
import threading
import uuid
import os
from typing import Any, Dict

from flask import Flask, request, jsonify, Response, stream_with_context
import json

from channels.channel_config import VideoChannelConfig
from database import db_manager, AsyncSessionLocal
from database_orm import CameraConfig
from sqlalchemy import select

# IMPORTANT:
# Do NOT import trt_infer / simple_model_pipeline at top-level if they import PyCUDA/TensorRT.
# We import them inside the pipeline thread so CUDA context is created in that same thread.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jetson-app")

app = Flask(__name__)
DEFAULT_SAMPLE_FPS = float(os.getenv("DEFAULT_SAMPLE_FPS", "15.0"))
# 0 disables pre-resize — TRT letterbox handles any input size so this is just wasted work
DEFAULT_RESIZE_W = int(os.getenv("DEFAULT_RESIZE_W", "0"))
DEFAULT_RESIZE_H = int(os.getenv("DEFAULT_RESIZE_H", "0"))
DEFAULT_JPEG_QUALITY = int(os.getenv("DEFAULT_JPEG_QUALITY", "70"))
MAX_SAMPLE_FPS = float(os.getenv("MAX_SAMPLE_FPS", "25.0"))
# -----------------------------
# Pipeline runtime (async loop in background thread)
# -----------------------------
class PipelineRuntime(object):
    def __init__(self):
        self.loop = None
        self.pipeline = None
        self._ready = threading.Event()
        self._lock = threading.Lock()

        # store configs (for PATCH, list, etc.)
        self._cameras = {}  # camera_uuid -> dict(config)
        
        # Initialize database tables
        #logger.info("Initializing SQLite database...")
        if not db_manager.initialize_tables():
            logger.error("Failed to initialize database tables")

        t = threading.Thread(target=self._run_loop, name="pipeline-loop", daemon=True)
        t.start()

        # wait for pipeline to be ready
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

        # Pre-resize is disabled (DEFAULT_RESIZE_W/H == 0). TRT letterbox handles any
        # input resolution, so pre-resizing is pure overhead. Force None so old DB
        # configs with resize=[640,480] don't re-enable it on restore.
        cfg_data["resize"] = None

        try:
            jpeg_quality = int(cfg_data.get("jpeg_quality", DEFAULT_JPEG_QUALITY))
        except Exception:
            jpeg_quality = DEFAULT_JPEG_QUALITY
        cfg_data["jpeg_quality"] = max(30, min(jpeg_quality, 95))

        # force raw path for TRT inference
        cfg_data["emit_format"] = "raw"

        return cfg_data
    
    def _run_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Import heavy TRT modules here so CUDA/PyCUDA init happens in THIS thread
            from pipeline import SimpleInferencePipeline

            pipeline = SimpleInferencePipeline()   # NO build_default here
            loop.run_until_complete(pipeline.start())
            self.loop = loop
            self.pipeline = pipeline
            self._ready.set()

            #logger.info("Pipeline started in background thread.")
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
    async def _save_camera_to_db_async(self, camera_uuid: str, cfg_data: Dict[str, Any]):
        """
        Save camera configuration to SQLite database (async).
        """
        try:
            async with AsyncSessionLocal() as session:
                # Check if camera already exists
                result = await session.execute(
                    select(CameraConfig).filter_by(camera_uuid=camera_uuid)
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    # Update existing camera
                    existing.rtsp_url = cfg_data.get("rtsp_url", existing.rtsp_url)
                    existing.config_json = cfg_data
                    #logger.info(f"Updated camera {camera_uuid} in database")
                else:
                    # Create new camera
                    cam_config = CameraConfig(
                        channel_id=cfg_data.get("channel_id", camera_uuid),
                        camera_uuid=camera_uuid,
                        user_id=cfg_data.get("user_id", 1),  # Default user_id
                        rtsp_url=cfg_data["rtsp_url"],
                        config_json=cfg_data
                    )
                    session.add(cam_config)
                    #logger.info(f"Saved new camera {camera_uuid} to database")
                
                await session.commit()
        except Exception as e:
            logger.exception(f"Failed to save camera {camera_uuid} to database: {e}")
    
    def _save_camera_to_db(self, camera_uuid: str, cfg_data: Dict[str, Any]):
        """
        Synchronous wrapper for async database save (runs in event loop).
        """
        self._call(self._save_camera_to_db_async(camera_uuid, cfg_data), timeout_s=5.0)
    
    async def _delete_camera_from_db_async(self, camera_uuid: str):
        """
        Delete camera configuration from SQLite database (async).
        """
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(CameraConfig).filter_by(camera_uuid=camera_uuid)
                )
                camera = result.scalar_one_or_none()
                if camera:
                    await session.delete(camera)
                    await session.commit()
                    #logger.info(f"Deleted camera {camera_uuid} from database")
        except Exception as e:
            logger.exception(f"Failed to delete camera {camera_uuid} from database: {e}")
    
    def _delete_camera_from_db(self, camera_uuid: str):
        """
        Synchronous wrapper for async database delete.
        """
        self._call(self._delete_camera_from_db_async(camera_uuid), timeout_s=5.0)
    
    def _restore_cameras_from_db(self):
        """
        Restore all cameras from SQLite database on startup (sync version for init).
        """
        try:
            pipeline = self._require_pipeline()
        except RuntimeError:
            logger.error("Skipping camera restore because pipeline is not ready")
            return

        try:
            # Use sync session for startup (before async loop is running heavily)
            session = db_manager.get_session()
            try:
                cameras = session.query(CameraConfig).all()
                restored_count = 0
                
                for cam_config in cameras:
                    try:
                        cfg_data = dict(cam_config.config_json)
                        camera_uuid = cam_config.camera_uuid
                        
                        # Ensure required fields are present
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
                            #logger.info(f"Restored camera {camera_uuid} from database")
                    except Exception as e:
                        logger.exception(f"Failed to restore camera {cam_config.camera_uuid}: {e}")
                
                if restored_count > 0:
                    logger.info(f"Successfully restored {restored_count} camera(s) from database")
                else:
                    logger.info("No cameras to restore from database")
            finally:
                session.close()
        except Exception as e:
            logger.exception(f"Failed to restore cameras from database: {e}")

    # ---------- camera operations ----------
    def add_camera(self, rtsp_url: str, cfg_patch: Dict[str, Any]) -> Dict[str, Any]:
        # Use camera_uuid from patch if provided (from Azure), otherwise generate new
        cam_id = cfg_patch.get("camera_uuid")
        
        if not cam_id:
            raise ValueError("camera_uuid must be provided by backend (do not let Jetson generate IDs)")
        cam_id = str(cam_id)
  
        # defaults for Jetson inference
        cfg_data = {
            "camera_uuid": cam_id,
            "channel_id": cfg_patch.get("channel_id") or cam_id,
            "rtsp_url": rtsp_url,
            "enabled": True,
            "detection_enabled": True,
            "notification_enabled": True,
            "sample_fps": DEFAULT_SAMPLE_FPS,
            "decode_backend": "gstreamer",
            "resize": (DEFAULT_RESIZE_W, DEFAULT_RESIZE_H) if DEFAULT_RESIZE_W > 0 and DEFAULT_RESIZE_H > 0 else None,
            "reconnect_base_ms": 1000,
            "reconnect_max_ms": 8000,
            "emit_format": "raw",
            "jpeg_quality": DEFAULT_JPEG_QUALITY,
        }
        
        # apply patch from request
        for k, v in (cfg_patch or {}).items():
            if v is not None:
                cfg_data[k] = v
                
        cfg_data = self._normalize_camera_cfg(cfg_data)

        cfg = VideoChannelConfig(**cfg_data)  # your simple config class should accept these

        # store config in memory
        with self._lock:
            self._cameras[cam_id] = dict(cfg_data)
        
        # persist to database
        self._save_camera_to_db(cam_id, cfg_data)

        # add to pipeline (async)
        pipeline = self._require_pipeline()
        self._call(pipeline.add_channel(cfg), timeout_s=15.0)

        return {
            "camera_uuid": cam_id,
            "rtsp_url": rtsp_url,
            "config": self._cameras[cam_id],
        }
        
    def remove_camera(self, camera_uuid: str) -> bool:
        with self._lock:
            existed = camera_uuid in self._cameras
            if existed:
                del self._cameras[camera_uuid]

        # stop runtime FIRST (prevents “still inferencing after delete” window)
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
                cams.append({"camera_uuid": cam_id, "rtsp_url": cfg.get("rtsp_url"), "config": cfg})
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

    def get_latest(self, camera_uuid: str) -> Dict[str, Any]:
        # get_latest is async
        pipeline = self._require_pipeline()
        result = self._call(pipeline.get_latest(camera_uuid), timeout_s=5.0)
        return result

    def get_snapshot(self, camera_uuid: str):
        # get_latest_snapshot is async
        pipeline = self._require_pipeline()
        return self._call(pipeline.get_latest_snapshot(camera_uuid), timeout_s=5.0)

    def get_stats(self) -> Dict[str, Any]:
        if self.loop is None or self.pipeline is None:
            return {}
        try:
            return self._call(self.pipeline.get_stats(), timeout_s=2.0)
        except Exception:
            logger.exception("Failed to fetch pipeline stats")
            return {}


runtime = PipelineRuntime()


# -----------------------------
# Helpers
# -----------------------------
def _json():
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def _require_rtsp(url: str):
    if not url or not isinstance(url, str):
        return False
    # basic check
    return url.startswith("rtsp://") or url.startswith("rtsps://")


# -----------------------------
# Routes
# -----------------------------
@app.route("/health", methods=["GET"])
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "pipeline_ready": bool(runtime.pipeline is not None and runtime.loop is not None),
        "stats": runtime.get_stats(),
    })


@app.route("/cameras", methods=["GET"])
@app.route("/api/cameras", methods=["GET"])
def list_cameras():
    return jsonify({"cameras": runtime.list_cameras()})

@app.route("/cameras", methods=["POST"])
@app.route("/api/cameras", methods=["POST"])
def add_camera():
    body = _json()
    rtsp_url = body.get("rtsp_url")
    cfg = body.get("config")
    if not isinstance(cfg, dict):
        cfg = {}
    else:
        cfg = dict(cfg)

    for k, v in body.items():
        if k in {"config", "rtsp_url"}:
            continue
        if k not in cfg and v is not None:
            cfg[k] = v

    if not rtsp_url:
        rtsp_url = cfg.get("rtsp_url")

    if not _require_rtsp(rtsp_url):
        return jsonify({"error": "rtsp_url is required and must start with rtsp://"}), 400

    if "camera_uuid" not in cfg and body.get("camera_uuid"):
        cfg["camera_uuid"] = body.get("camera_uuid")

    if "channel_id" not in cfg and body.get("channel_id"):
        cfg["channel_id"] = body.get("channel_id")

    if cfg.get("emit_format") not in ("jpeg", "raw"):
        cfg["emit_format"] = "raw"

    try:
        out = runtime.add_camera(rtsp_url, cfg)
        return jsonify(out), 201
    except Exception as e:
        logger.exception("add_camera failed: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/cameras/<camera_uuid>", methods=["DELETE"])
@app.route("/api/cameras/<camera_uuid>", methods=["DELETE"])
def delete_camera(camera_uuid):
    try:
        existed = runtime.remove_camera(camera_uuid)
        return jsonify({"deleted": True, "camera_uuid": camera_uuid, "existed": existed})
    except Exception as e:
        logger.exception("delete_camera failed: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/cameras/<camera_uuid>", methods=["PATCH"])
@app.route("/api/cameras/<camera_uuid>", methods=["PATCH"])
def patch_camera(camera_uuid):
    body = _json()
    patch = body.get("config") or body  # allow either {"config": {...}} or direct patch
    if not isinstance(patch, dict) or not patch:
        return jsonify({"error": "No fields to update"}), 400

    try:
        out = runtime.patch_camera(camera_uuid, patch)
        return jsonify(out)
    except KeyError:
        return jsonify({"error": "Camera not found"}), 404
    except Exception as e:
        logger.exception("patch_camera failed: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/cameras/<camera_uuid>/latest", methods=["GET"])
@app.route("/api/cameras/<camera_uuid>/latest", methods=["GET"])
def latest(camera_uuid):
    try:
        result = runtime.get_latest(camera_uuid)
        if result is None:
            return jsonify({"error": "No detections yet"}), 404
        return jsonify(result)
    except Exception as e:
        logger.exception("latest failed: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/cameras/<camera_uuid>/snapshot.jpg", methods=["GET"])
@app.route("/api/cameras/<camera_uuid>/snapshot.jpg", methods=["GET"])
def snapshot(camera_uuid):
    try:
        result = runtime.get_snapshot(camera_uuid)
        if not result:
            return jsonify({"error": "No snapshot available yet"}), 404

        resp = Response(result, mimetype="image/jpeg")
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp
    except Exception as e:
        logger.exception("snapshot failed: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/detection/<camera_uuid>", methods=["GET"])
@app.route("/detections/<camera_uuid>", methods=["GET"])
@app.route("/api/detection/<camera_uuid>", methods=["GET"])
@app.route("/api/detections/<camera_uuid>", methods=["GET"])
def detection(camera_uuid):
    """
    Alternative endpoint for Azure backend compatibility.
    Returns same data as /cameras/<camera_uuid>/latest
    """
    try:
        result = runtime.get_latest(camera_uuid)
        if result is None:
            return jsonify({"error": "No detections yet"}), 404
        return jsonify(result)
    except Exception as e:
        logger.exception("detection failed: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/cameras/detections/stream", methods=["GET"])
@app.route("/api/cameras/detections/stream", methods=["GET"])
def stream_all_detections():
    """
    SSE endpoint for all detections.
    """
    return Response(stream_with_context(_sse_generator()), mimetype="text/event-stream")


@app.route("/cameras/<camera_uuid>/detections/stream", methods=["GET"])
@app.route("/api/cameras/<camera_uuid>/detections/stream", methods=["GET"])
def stream_camera_detections(camera_uuid):
    """
    SSE endpoint for specific camera detections.
    """
    return Response(stream_with_context(_sse_generator(camera_uuid)), mimetype="text/event-stream")


def _sse_generator(target_camera_uuid=None):
    if runtime.pipeline is None:
        return

    # Subscribe
    q_future = asyncio.run_coroutine_threadsafe(
        runtime.pipeline.broadcaster.subscribe(), 
        runtime.loop
    )
    try:
        q = q_future.result(timeout=5.0)
    except Exception:
        return

    try:
        while True:
            # We need to get from queue in a thread-safe way from the async loop
            # But the queue is in the async loop, and we are in a Flask thread.
            # We can use run_coroutine_threadsafe to get an item? 
            # No, that would be very slow for every item.
            # Better: The queue should be thread-safe?
            # asyncio.Queue is NOT thread-safe for cross-thread access.
            #
            # We need a bridge. 
            # Or we just use run_coroutine_threadsafe(q.get(), loop)
            # This is acceptable for SSE scale on Jetson (few clients).
            
            fut = asyncio.run_coroutine_threadsafe(q.get(), runtime.loop)
            try:
                msg = fut.result(timeout=1.0) # Check every second to allow disconnect check
            except Exception:
                # Timeout, send comment/heartbeat to keep alive
                yield ": keepalive\n\n"
                continue

            if not isinstance(msg, dict):
                continue

            # filter if needed
            if target_camera_uuid:
                # check msg camera_uuid
                c_uuid = msg.get("camera_uuid")
                if str(c_uuid) != str(target_camera_uuid):
                    continue

            # Yield SSE
            data_str = json.dumps(msg)
            yield f"data: {data_str}\n\n"

    except GeneratorExit:
        # Client disconnected
        asyncio.run_coroutine_threadsafe(
            runtime.pipeline.broadcaster.unsubscribe(q), 
            runtime.loop
        )
    except Exception as e:
        logger.error(f"SSE stream error: {e}")
        asyncio.run_coroutine_threadsafe(
            runtime.pipeline.broadcaster.unsubscribe(q), 
            runtime.loop
        )


if __name__ == "__main__":
    # threaded=True lets Flask handle multiple requests
    app.run(host="0.0.0.0", port=8080, threaded=True)
