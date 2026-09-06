# routes/camera_routes.py
"""Camera CRUD + per-camera reads (latest detection, snapshot)."""

import logging

from flask import Blueprint, jsonify, Response

try:
    from routes.helpers import json_body, require_source
    from routes.runtime_ref import get_runtime
except Exception:
    from .helpers import json_body, require_source
    from .runtime_ref import get_runtime

logger = logging.getLogger("jetson-app")

bp = Blueprint("cameras", __name__)


@bp.route("/cameras", methods=["GET"])
@bp.route("/api/cameras", methods=["GET"])
def list_cameras():
    return jsonify({"cameras": get_runtime().list_cameras()})


@bp.route("/cameras", methods=["POST"])
@bp.route("/api/cameras", methods=["POST"])
def add_camera():
    body = json_body()
    source_url = body.get("source_url")
    cfg = body.get("config")
    if not isinstance(cfg, dict):
        cfg = {}
    else:
        cfg = dict(cfg)

    for k, v in body.items():
        if k in {"config", "source_url"}:
            continue
        if k not in cfg and v is not None:
            cfg[k] = v

    if not source_url:
        source_url = cfg.get("source_url")

    if not require_source(source_url):
        return jsonify({
            "error": "source_url is required using a supported scheme: "
                     "rtsp/rtsps/webrtc/whep/http/https/rtmp/rtmps/srt."
        }), 400

    if "camera_uuid" not in cfg and body.get("camera_uuid"):
        cfg["camera_uuid"] = body.get("camera_uuid")

    if "channel_id" not in cfg and body.get("channel_id"):
        cfg["channel_id"] = body.get("channel_id")

    try:
        out = get_runtime().add_camera(source_url, cfg)
        return jsonify(out), 201
    except Exception as e:
        logger.exception("add_camera failed: %s", e)
        return jsonify({"error": str(e)}), 500


@bp.route("/cameras/<camera_uuid>", methods=["DELETE"])
@bp.route("/api/cameras/<camera_uuid>", methods=["DELETE"])
def delete_camera(camera_uuid):
    try:
        existed = get_runtime().remove_camera(camera_uuid)
        return jsonify({"deleted": True, "camera_uuid": camera_uuid, "existed": existed})
    except Exception as e:
        logger.exception("delete_camera failed: %s", e)
        return jsonify({"error": str(e)}), 500


@bp.route("/cameras/<camera_uuid>", methods=["PATCH"])
@bp.route("/api/cameras/<camera_uuid>", methods=["PATCH"])
def patch_camera(camera_uuid):
    body = json_body()
    patch = body.get("config") or body
    if not isinstance(patch, dict) or not patch:
        return jsonify({"error": "No fields to update"}), 400

    try:
        out = get_runtime().patch_camera(camera_uuid, patch)
        return jsonify(out)
    except KeyError:
        return jsonify({"error": "Camera not found"}), 404
    except Exception as e:
        logger.exception("patch_camera failed: %s", e)
        return jsonify({"error": str(e)}), 500


@bp.route("/cameras/<camera_uuid>/latest", methods=["GET"])
@bp.route("/api/cameras/<camera_uuid>/latest", methods=["GET"])
def latest(camera_uuid):
    try:
        result = get_runtime().get_latest(camera_uuid)
        if result is None:
            return jsonify({"error": "No detections yet"}), 404
        return jsonify(result)
    except Exception as e:
        logger.exception("latest failed: %s", e)
        return jsonify({"error": str(e)}), 500


@bp.route("/cameras/<camera_uuid>/snapshot.jpg", methods=["GET"])
@bp.route("/api/cameras/<camera_uuid>/snapshot.jpg", methods=["GET"])
def snapshot(camera_uuid):
    try:
        result = get_runtime().get_snapshot(camera_uuid)
        if not result:
            return jsonify({"error": "No snapshot available yet"}), 404

        resp = Response(result, mimetype="image/jpeg")
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp
    except Exception as e:
        logger.exception("snapshot failed: %s", e)
        return jsonify({"error": str(e)}), 500
