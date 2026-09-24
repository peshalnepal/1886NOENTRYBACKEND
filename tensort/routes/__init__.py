"""Register HTTP endpoints and attach the application's runtime."""

from .runtime_ref import get_runtime
from . import health_routes, camera_routes, detection_routes, discovery_routes

BLUEPRINTS = (
    health_routes.bp,
    camera_routes.bp,
    detection_routes.bp,
    discovery_routes.bp,
)


def register_routes(app, runtime):
    """Bind the runtime and register every blueprint on the Flask app."""
    app.extensions["pipeline_runtime"] = runtime
    for bp in BLUEPRINTS:
        app.register_blueprint(bp)
    return app


__all__ = ["register_routes", "get_runtime", "BLUEPRINTS"]
