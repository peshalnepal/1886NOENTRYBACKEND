# routes/health_routes.py
"""Service root + health endpoints."""

from flask import Blueprint, jsonify

try:
    from routes.helpers import runtime_status
except Exception:
    from .helpers import runtime_status

bp = Blueprint("health", __name__)


@bp.route("/", methods=["GET"])
@bp.route("/api", methods=["GET"])
def root():
    return jsonify(runtime_status(include_stats=False))


@bp.route("/health", methods=["GET"])
@bp.route("/api/health", methods=["GET"])
def health():
    return jsonify(runtime_status(include_stats=True))
