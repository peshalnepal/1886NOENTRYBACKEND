# main.py  (Python 3.6)
import asyncio
import logging
import threading
import uuid
from typing import Any, Dict, Optional

from flask import Flask, request, jsonify

from channels.channel_config import VideoChannelConfig
from logging.handlers import RotatingFileHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jetson-app")

LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] [%(threadName)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

# Optional: also write logs to a file (Jetson friendly)
file_handler = RotatingFileHandler("jetson-app.log", maxBytes=5_000_000, backupCount=3)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logging.getLogger().addHandler(file_handler)

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

        self._boot_error = None
        self._boot_trace = None

        self._cameras = {}

        t = threading.Thread(target=self._run_loop, name="pipeline-loop", daemon=True)
        t.start()

        # Wait for boot to finish (success OR failure)
        if not self._ready.wait(30.0):
            raise RuntimeError("Pipeline thread did not signal ready within timeout.")

        # If boot failed, fail hard so you don't serve requests with pipeline=None
        if self._boot_error is not None or self.loop is None or self.pipeline is None:
            raise RuntimeError("Pipeline failed to start: %r" % (self._boot_error,))

    def _run_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            logger.info("Pipeline thread starting...")

            # IMPORTANT: make sure this import path matches your filename
            # If your file is pipeline.py -> keep this
            # If your file is simple_model_pipeline.py -> change import accordingly
            from pipeline import SimpleInferencePipeline

            pipeline = SimpleInferencePipeline()
            logger.info("Created SimpleInferencePipeline, starting...")

            loop.run_until_complete(pipeline.start())

            self.loop = loop
            self.pipeline = pipeline

            logger.info("Pipeline started successfully.")
        except Exception as e:
            import traceback
            self._boot_error = e
            self._boot_trace = traceback.format_exc()
            logger.exception("Pipeline thread crashed during startup: %s", e)
        finally:
            self._ready.set()

        # Only run_forever if startup succeeded
        if self.loop is not None and self.pipeline is not None:
            try:
                loop.run_forever()
            except Exception as e:
                logger.exception("Pipeline loop crashed after startup: %s", e)
            finally:
                try:
                    loop.stop()
                except Exception:
                    pass

    def status(self):
        return {
            "ready": bool(self.loop is not None and self.pipeline is not None),
            "boot_error": None if self._boot_error is None else str(self._boot_error),
            "boot_trace": self._boot_trace,  # include only if you're okay exposing it
            "channels": [] if self.pipeline is None else self.pipeline.list_channels(),
            "cameras": list(self._cameras.keys()),
        }



    def _call(self, coro, timeout_s=10.0):
        if self.loop is None or self.pipeline is None:
            raise RuntimeError("Pipeline not ready")
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return fut.result(timeout=timeout_s)

    def upsert_camera(self, rtsp_url: str, cfg_patch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Upsert semantics:
        - If cfg_patch has camera_uuid, we use it (Azure decides the camera_uuid)
        - Otherwise we generate one (still supported for local testing)
        """
        incoming_id = (
            cfg_patch.get("camera_uuid")
            or cfg_patch.get("camera_id")
            or cfg_patch.get("id")
        )
        cam_id = str(incoming_id or uuid.uuid4())

        cfg_data = {
            "camera_uuid": cam_id,
            "channel_id": cfg_patch.get("channel_id") or cam_id,
            "rtsp_url": rtsp_url,
            "enabled": True,
            "detection_enabled": True,
            "notification_enabled": True,
            "sample_fps": 5.0,
            "decode_backend": "gstreamer",
            "resize": None,
            "reconnect_base_ms": 1000,
            "reconnect_max_ms": 8000,
            "emit_format": "raw",     # IMPORTANT for TRT inference
            "jpeg_quality": 80,
        }

        # apply patch overrides
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

        # strongly recommend raw frames for TRT inference
        if cfg_data.get("emit_format") not in ("raw", "jpeg"):
            cfg_data["emit_format"] = "raw"
        logger.info("######################## UPSERT CAMERA ##########################################")
        logger.info(cfg_data)
        logger.info("####################################################################")
        cfg = VideoChannelConfig(**cfg_data)

        with self._lock:
            existed = cam_id in self._cameras
            self._cameras[cam_id] = dict(cfg_data)
            
        pipe = self.pipeline
        if pipe is None:
            raise RuntimeError("Pipeline not ready (pipeline=None). Check startup logs.")
        self._call(pipe.add_channel(cfg), timeout_s=15.0)

        return {
            "camera_uuid": cam_id,
            "rtsp_url": rtsp_url,
            "config": self._cameras[cam_id],
            "existed": existed,
        }

    def remove_camera(self, camera_uuid: str) -> bool:
        key = str(camera_uuid)
        with self._lock:
            existed = key in self._cameras
            if existed:
                del self._cameras[key]

        # remove from pipeline (idempotent-ish)
        try:
            self._call(self.pipeline.remove_channel(key), timeout_s=10.0)
        except Exception:
            # don't fail DELETE if pipeline already removed it
            logger.warning("remove_channel failed (ignored) camera_uuid=%s", key)

        return existed

    def list_cameras(self):
        with self._lock:
            cams = []
            for cam_id, cfg in self._cameras.items():
                cams.append({"camera_uuid": cam_id, "rtsp_url": cfg.get("rtsp_url"), "config": cfg})
            return cams

    def patch_camera(self, camera_uuid: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        key = str(camera_uuid)
        with self._lock:
            cfg_data = self._cameras.get(key)
            if cfg_data is None:
                raise KeyError("camera not found")

            for k, v in (patch or {}).items():
                if v is not None:
                    cfg_data[k] = v

            # normalize resize
            if cfg_data.get("resize") is not None:
                r = cfg_data["resize"]
                if isinstance(r, (list, tuple)) and len(r) == 2:
                    cfg_data["resize"] = (int(r[0]), int(r[1]))
                else:
                    cfg_data["resize"] = None

            # enforce identity
            cfg_data["camera_uuid"] = key
            if not cfg_data.get("channel_id"):
                cfg_data["channel_id"] = key

            self._cameras[key] = cfg_data
            new_cfg = VideoChannelConfig(**cfg_data)

        self._call(self.pipeline.add_channel(new_cfg), timeout_s=15.0)
        return {"camera_uuid": key, "config": cfg_data}

    def get_latest(self, camera_uuid: str) -> Optional[Dict[str, Any]]:
        key = str(camera_uuid)
        result = self._call(self.pipeline.get_latest(key), timeout_s=5.0)
        return result


runtime = PipelineRuntime()


# -----------------------------
# Helpers
# -----------------------------
def _json():
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}

def _require_rtsp(url: str):
    return bool(url) and isinstance(url, str) and (url.startswith("rtsp://") or url.startswith("rtsps://"))

def _parse_upsert_body() -> (str, Dict[str, Any]):
    """
    Accepts either:
      A) {"rtsp_url": "...", "config": {...}}
      B) {"camera_uuid": "...", "rtsp_url": "...", ...}  # flat
    Returns (rtsp_url, cfg_patch)
    """
    body = _json()
    rtsp_url = body.get("rtsp_url")

    cfg_patch = body.get("config")
    if isinstance(cfg_patch, dict):
        # allow camera_uuid to be passed at top-level too
        if body.get("camera_uuid") and "camera_uuid" not in cfg_patch:
            cfg_patch["camera_uuid"] = body["camera_uuid"]
        if body.get("channel_id") and "channel_id" not in cfg_patch:
            cfg_patch["channel_id"] = body["channel_id"]
        return rtsp_url, cfg_patch

    # flat payload: treat everything except rtsp_url as patch
    cfg_patch = dict(body)
    cfg_patch.pop("rtsp_url", None)
    return rtsp_url, cfg_patch

def _normalize_detection(det: Any, camera_uuid: str) -> Optional[Dict[str, Any]]:
    """
    Ensure we return a dict with top-level keys that Azure expects:
      frame_seq, frame_ts_ms, detections, pose, inference_ms, camera_uuid
    """
    if det is None:
        return None

    if isinstance(det, dict):
        # if older code returns {"camera_uuid":..., "detection": {...}}, unwrap it
        if "detection" in det and isinstance(det.get("detection"), dict):
            det = det["detection"]
        out = dict(det)
    else:
        # last-resort: try best-effort conversion
        try:
            out = dict(det)
        except Exception:
            return None

    out.setdefault("camera_uuid", str(camera_uuid))
    return out


# -----------------------------
# Routes (with /api aliases)
# -----------------------------
@app.route("/health", methods=["GET"])
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


@app.route("/cameras", methods=["GET"])
@app.route("/api/cameras", methods=["GET"])
def list_cameras():
    return jsonify({"cameras": runtime.list_cameras()})


@app.route("/cameras", methods=["POST"])
@app.route("/api/cameras", methods=["POST"])
def upsert_camera():
    rtsp_url, patch = _parse_upsert_body()
    if not _require_rtsp(rtsp_url):
        return jsonify({"error": "rtsp_url is required and must start with rtsp:// or rtsps://"}), 400

    if not isinstance(patch, dict):
        patch = {}

    try:
        out = runtime.upsert_camera(rtsp_url, patch)
        status_code = 200 if out.get("existed") else 201
        return jsonify(out), status_code
    except Exception as e:
        logger.exception("upsert_camera failed: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/cameras/<camera_uuid>", methods=["PATCH"])
@app.route("/api/cameras/<camera_uuid>", methods=["PATCH"])
def patch_camera(camera_uuid):
    body = _json()
    patch = body.get("config") if isinstance(body.get("config"), dict) else body
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


@app.route("/cameras/<camera_uuid>", methods=["DELETE"])
@app.route("/api/cameras/<camera_uuid>", methods=["DELETE"])
def delete_camera(camera_uuid):
    try:
        existed = runtime.remove_camera(camera_uuid)
        return jsonify({"deleted": True, "camera_uuid": str(camera_uuid), "existed": existed})
    except Exception as e:
        logger.exception("delete_camera failed: %s", e)
        return jsonify({"error": str(e)}), 500


# Detection endpoints:
# Azure defaults to GET {device_url}/detection/{camera_uuid}:contentReference[oaicite:4]{index=4}
# Keep /detections too for backward compat.
@app.route("/detection/<camera_uuid>", methods=["GET"])
@app.route("/detections/<camera_uuid>", methods=["GET"])
@app.route("/api/detections/<camera_uuid>", methods=["GET"])
def get_detection(camera_uuid):
    print(f"[Jetson API] Request for {camera_uuid}")
    try:
        det = runtime.get_latest(camera_uuid)
        if det is None:
            print(f"[Jetson API] No detection for {camera_uuid} yet")
            return jsonify({"error": "No detections yet"}), 404
        
        out = _normalize_detection(det, camera_uuid)
        print(f"[Jetson API] Returning detection for {camera_uuid}: {len(out.get('detections', []))} objects")
        return jsonify(out)
    except Exception as e:
        print(f"[Jetson API] ERROR for {camera_uuid}: {e}")
        logger.exception("get_detection failed: %s", e)
        return jsonify({"error": str(e)}), 500


# Older alias some code might call
@app.route("/cameras/<camera_uuid>/latest", methods=["GET"])
@app.route("/api/cameras/<camera_uuid>/latest", methods=["GET"])
def latest(camera_uuid):
    return get_detection(camera_uuid)


if __name__ == "__main__":
    # threaded=True lets Flask handle multiple requests
    # If you ever run with debug=True, also set use_reloader=False to avoid starting the pipeline twice.
    app.run(host="0.0.0.0", port=8080, threaded=True)
