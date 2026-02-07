# main.py  (Python 3.6)
import asyncio
import logging
import threading
import uuid
from typing import Any, Dict

from flask import Flask, request, jsonify

from channels.channel_config import VideoChannelConfig

# IMPORTANT:
# Do NOT import trt_infer / simple_model_pipeline at top-level if they import PyCUDA/TensorRT.
# We import them inside the pipeline thread so CUDA context is created in that same thread.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jetson-app")

app = Flask(__name__)


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

        t = threading.Thread(target=self._run_loop, name="pipeline-loop", daemon=True)
        t.start()

        # wait for pipeline to be ready
        if not self._ready.wait(30.0):
            raise RuntimeError("Pipeline thread did not start within timeout.")

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

            logger.info("Pipeline started in background thread.")
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

    # ---------- camera operations ----------
    def add_camera(self, rtsp_url: str, cfg_patch: Dict[str, Any]) -> Dict[str, Any]:
        cam_id = str(uuid.uuid4())

        # defaults for Jetson inference
        cfg_data = {
            "camera_uuid": cam_id,
            "channel_id": cam_id,
            "rtsp_url": rtsp_url,
            "enabled": True,
            "detection_enabled": True,
            "notification_enabled": True,
            "sample_fps": 5.0,
            "decode_backend": "gstreamer",   # best on Jetson if OpenCV built with GStreamer
            "resize": None,                 # e.g. (640, 360)
            "reconnect_base_ms": 1000,
            "reconnect_max_ms": 8000,
            "emit_format": "raw",           # IMPORTANT: raw for TRT inference
            "jpeg_quality": 80,
        }

        # apply patch from request
        for k, v in (cfg_patch or {}).items():
            if v is not None:
                cfg_data[k] = v

        # normalize resize if list -> tuple
        if cfg_data.get("resize") is not None:
            r = cfg_data["resize"]
            if isinstance(r, (list, tuple)) and len(r) == 2:
                cfg_data["resize"] = (int(r[0]), int(r[1]))
            else:
                cfg_data["resize"] = None

        cfg = VideoChannelConfig(**cfg_data)  # your simple config class should accept these

        # store config
        with self._lock:
            self._cameras[cam_id] = dict(cfg_data)

        # add to pipeline (async)
        self._call(self.pipeline.add_channel(cfg), timeout_s=15.0)

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

        # remove from pipeline
        self._call(self.pipeline.remove_channel(camera_uuid), timeout_s=10.0)
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

            # normalize resize
            if cfg_data.get("resize") is not None:
                r = cfg_data["resize"]
                if isinstance(r, (list, tuple)) and len(r) == 2:
                    cfg_data["resize"] = (int(r[0]), int(r[1]))
                else:
                    cfg_data["resize"] = None

            self._cameras[camera_uuid] = cfg_data
            new_cfg = VideoChannelConfig(**cfg_data)

        self._call(self.pipeline.add_channel(new_cfg), timeout_s=15.0)

        return {"camera_uuid": camera_uuid, "config": cfg_data}

    def get_latest(self, camera_uuid: str) -> Dict[str, Any]:
        # get_latest is async
        result = self._call(self.pipeline.get_latest(camera_uuid), timeout_s=5.0)
        return result


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
def health():
    return jsonify({"ok": True})


@app.route("/cameras", methods=["GET"])
def list_cameras():
    return jsonify({"cameras": runtime.list_cameras()})


@app.route("/cameras", methods=["POST"])
def add_camera():
    body = _json()
    rtsp_url = body.get("rtsp_url")
    if not _require_rtsp(rtsp_url):
        return jsonify({"error": "rtsp_url is required and must start with rtsp://"}), 400

    # optional config overrides
    patch = body.get("config") or {}
    if not isinstance(patch, dict):
        patch = {}

    # strongly recommend raw frames for TRT inference
    if patch.get("emit_format") in ("jpeg", "raw"):
        pass
    else:
        # default raw
        patch["emit_format"] = "raw"

    try:
        out = runtime.add_camera(rtsp_url, patch)
        return jsonify(out), 201
    except Exception as e:
        logger.exception("add_camera failed: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/cameras/<camera_uuid>", methods=["DELETE"])
def delete_camera(camera_uuid):
    try:
        existed = runtime.remove_camera(camera_uuid)
        return jsonify({"deleted": True, "camera_uuid": camera_uuid, "existed": existed})
    except Exception as e:
        logger.exception("delete_camera failed: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/cameras/<camera_uuid>", methods=["PATCH"])
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
def latest(camera_uuid):
    try:
        result = runtime.get_latest(camera_uuid)
        if result is None:
            return jsonify({"error": "No detections yet"}), 404
        return jsonify(result)
    except Exception as e:
        logger.exception("latest failed: %s", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # threaded=True lets Flask handle multiple requests
    app.run(host="0.0.0.0", port=8080, threaded=True)
