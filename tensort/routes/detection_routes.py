# routes/detection_routes.py
"""Detection reads and the SSE detection streams."""

import asyncio
import concurrent.futures
import json
import logging

from flask import Blueprint, jsonify, Response, stream_with_context

try:
    from routes.runtime_ref import get_runtime
except Exception:
    from .runtime_ref import get_runtime

logger = logging.getLogger("jetson-app")

bp = Blueprint("detections", __name__)


@bp.route("/detection/<camera_uuid>", methods=["GET"])
@bp.route("/detections/<camera_uuid>", methods=["GET"])
@bp.route("/api/detection/<camera_uuid>", methods=["GET"])
@bp.route("/api/detections/<camera_uuid>", methods=["GET"])
def detection(camera_uuid):
    """
    Alternative endpoint for Azure backend compatibility.
    Returns same data as /cameras/<camera_uuid>/latest
    """
    try:
        result = get_runtime().get_latest(camera_uuid)
        if result is None:
            return jsonify({"error": "No detections yet"}), 404
        return jsonify(result)
    except Exception as e:
        logger.exception("detection failed: %s", e)
        return jsonify({"error": str(e)}), 500


@bp.route("/cameras/detections/stream", methods=["GET"])
@bp.route("/api/cameras/detections/stream", methods=["GET"])
def stream_all_detections():
    """
    SSE endpoint for all detections.
    """
    return Response(stream_with_context(sse_generator()), mimetype="text/event-stream")


@bp.route("/cameras/<camera_uuid>/detections/stream", methods=["GET"])
@bp.route("/api/cameras/<camera_uuid>/detections/stream", methods=["GET"])
def stream_camera_detections(camera_uuid):
    """
    SSE endpoint for specific camera detections.
    """
    return Response(
        stream_with_context(sse_generator(camera_uuid)),
        mimetype="text/event-stream",
    )


def sse_generator(target_camera_uuid=None):
    runtime = get_runtime()
    if runtime.pipeline is None:
        return

    # Subscribe
    q = None
    q_future = asyncio.run_coroutine_threadsafe(
        runtime.pipeline.broadcaster.subscribe(target_camera_uuid),
        runtime.loop
    )
    try:
        q = q_future.result(timeout=5.0)
    except Exception:
        return

    try:
        while True:
            fut = asyncio.run_coroutine_threadsafe(q.get(), runtime.loop)
            try:
                msg = fut.result(timeout=1.0)  # Check every second to allow disconnect check
            except concurrent.futures.TimeoutError:
                # Cancel lost the race: a detection landed between the timeout
                # and the cancel, so it is already in this future. Deliver it
                # rather than dropping it — the queue slot is gone either way.
                if (not fut.cancel()) and fut.done():
                    try:
                        msg = fut.result()
                    except Exception:
                        msg = None
                    if isinstance(msg, dict):
                        yield f"data: {json.dumps(msg)}\n\n"
                        continue

                yield ": keepalive\n\n"
                continue
            except Exception as e:
                logger.error(f"SSE stream error while waiting for detection: {e}")
                break

            if not isinstance(msg, dict):
                continue

            # Yield SSE
            data_str = json.dumps(msg)
            yield f"data: {data_str}\n\n"

    except GeneratorExit:
        # Client disconnected
        return
    except Exception as e:
        logger.error(f"SSE stream error: {e}")
    finally:
        if q is not None and runtime.pipeline is not None and runtime.loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    runtime.pipeline.broadcaster.unsubscribe(q),
                    runtime.loop
                ).result(timeout=1.0)
            except Exception:
                pass
