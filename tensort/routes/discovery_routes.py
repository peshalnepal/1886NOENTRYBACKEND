# routes/discovery_routes.py
"""
Discovery + sync endpoints.

    GET    /discovery            -> the roster: every camera ever seen, present or not
    GET    /discovery/status     -> scheduler state (running, interval, last sweep)
    GET    /discovery/report     -> the last sweep report, without rescanning
    POST   /discovery/scan       -> force a sweep now, return the fresh report
    DELETE /discovery/<identity> -> forget a decommissioned camera
    POST   /sync                 -> triggers a sweep and returns cameras + report

Note on "list the cameras attached to this Jetson": that already exists as
`GET /cameras` in camera_routes.py, and the cloud's EdgeInferenceClient is
already pointed at it (EDGE_LIST_PATH defaults to /cameras). Adding a second
listing endpoint would give the cloud two disagreeing sources of truth, so the
routes here are strictly additive.

`POST /sync` is the one the cloud calls. The Jetson is the source of truth for
which cameras physically exist, so a sync must reflect the network as it is
*now*, not as it was up to a minute ago — hence the sweep runs inline before
the camera list is rendered.
"""

import logging
from functools import wraps

from flask import Blueprint, jsonify

try:
    from routes.runtime_ref import get_runtime
except ModuleNotFoundError:
    from .runtime_ref import get_runtime

logger = logging.getLogger("jetson-app")

bp = Blueprint("discovery", __name__)


def route(rule, **options):
    """Register a handler at both `<rule>` and `/api<rule>`.

    The cloud calls the /api-prefixed form; local tooling and the Jetson's own
    health checks use the bare one.
    """
    def decorator(fn):
        bp.add_url_rule(rule, fn.__name__, fn, **options)
        bp.add_url_rule("/api" + rule, fn.__name__ + "_api", fn, **options)
        return fn
    return decorator


def with_service(fn):
    """Pass the DiscoveryService as the first argument, or 503 if it never started."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        svc = getattr(get_runtime(), "discovery", None)
        if svc is None:
            return jsonify({"error": "Discovery service is not running"}), 503
        return fn(svc, *args, **kwargs)
    return wrapper


@route("/discovery", methods=["GET"])
@with_service
def list_discovered(svc):
    roster = svc.roster()
    return jsonify({
        "discovered": roster,
        "present": [r for r in roster if r.get("is_present")],
        "missing": [r for r in roster if not r.get("is_present")],
        "status": svc.status(),
    })


@route("/discovery/status", methods=["GET"])
@with_service
def discovery_status(svc):
    return jsonify(svc.status())


@route("/discovery/report", methods=["GET"])
@with_service
def discovery_report(svc):
    """The most recent sweep report. Does NOT rescan — use POST /discovery/scan."""
    report = svc.last_report()
    if report is None:
        return jsonify({"error": "No discovery sweep has completed yet"}), 404
    return jsonify(report)


@route("/discovery/scan", methods=["POST"])
@with_service
def discovery_scan(svc):
    """Force a sweep now and return its report.

    A sweep takes seconds (UDP probe window plus up to a /24 of HTTP probes),
    so callers should use a generous timeout.
    """
    try:
        return jsonify(svc.run_once(force=True))
    except Exception as e:
        logger.exception("discovery_scan failed")
        return jsonify({"error": str(e)}), 500


@route("/discovery/<path:identity>", methods=["DELETE"])
@with_service
def forget_discovered(svc, identity):
    """Drop a roster row so a decommissioned camera stops raising alerts.

    `<path:identity>` because identities look like "serial:DS-2CD1234" and a
    default string converter would still work, but path keeps any future
    identity scheme containing slashes from 404ing.
    """
    existed = svc.forget(identity)
    return jsonify({"forgotten": bool(existed), "identity": identity, "existed": existed})


@route("/sync", methods=["POST"])
def sync():
    """Cloud sync entry point: rescan the network, then report.

    Returns both halves of the truth the cloud needs:
      - `cameras`: what is provisioned on this Jetson right now (same shape as
        GET /cameras, so the cloud's existing parser works unchanged)
      - `discovery`: what the sweep just found, including cameras that have
        gone missing so the frontend can alert on them

    The sweep runs before the camera list is read, so a camera adopted during
    this very sweep is already in `cameras`. Unlike the routes above, a missing
    discovery service is not a 503 here — the cloud still needs the camera list.
    """
    runtime = get_runtime()
    svc = getattr(runtime, "discovery", None)

    report = None
    warnings = []
    if svc is not None:
        try:
            report = svc.run_once(force=True)
        except Exception as e:
            # A failed sweep must not fail the sync — the cloud still needs the
            # camera list, and stale discovery data beats no response at all.
            logger.exception("Discovery sweep failed during sync")
            report = svc.last_report()
            warnings.append("Discovery sweep failed during sync: {}".format(e))

    payload = {
        "cameras": runtime.list_cameras(),
        "discovery": report,
        "status": svc.status() if svc is not None else None,
    }
    if warnings:
        payload["warnings"] = warnings
    return jsonify(payload)
