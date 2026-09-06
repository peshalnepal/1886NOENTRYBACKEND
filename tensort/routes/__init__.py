# routes/ - HTTP boundary for the Jetson edge service.
#
# Each module owns a Flask Blueprint holding the routes it used to declare
# inline in main.py. Blueprints reach the PipelineRuntime through
# `runtime_ref.get_runtime()` (bound once by main.py) so this package never
# imports main.

try:
    # Script mode (python main.py from Backend/tensort)
    from routes.runtime_ref import set_runtime, get_runtime
    from routes import health_routes, camera_routes, detection_routes, discovery_routes
except Exception:
    # Package mode (python -m Backend.tensort.main)
    from .runtime_ref import set_runtime, get_runtime
    from . import health_routes, camera_routes, detection_routes, discovery_routes

BLUEPRINTS = (
    health_routes.bp,
    camera_routes.bp,
    detection_routes.bp,
    discovery_routes.bp,
)


def register_routes(app, runtime):
    """Bind the runtime and register every blueprint on the Flask app."""
    set_runtime(runtime)
    for bp in BLUEPRINTS:
        app.register_blueprint(bp)
    return app


__all__ = ["register_routes", "set_runtime", "get_runtime", "BLUEPRINTS"]
