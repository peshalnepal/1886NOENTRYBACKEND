"""Read the runtime belonging to the Flask application serving this request."""

from flask import current_app


def get_runtime():
    runtime = current_app.extensions.get("pipeline_runtime")
    if runtime is None:
        raise RuntimeError("Pipeline runtime has not been initialized")
    return runtime
