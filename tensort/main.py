"""HTTP entry point. Run one process: it owns one GPU inference runtime."""

import logging
import os

from flask import Flask

# Load configuration before importing the runtime and its database settings.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

if __package__:
    from .runtime import PipelineRuntime
    from .routes import register_routes
else:
    from runtime import PipelineRuntime
    from routes import register_routes

logging.basicConfig(level=logging.INFO)


def create_app(runtime=None):
    app = Flask(__name__)
    if runtime is None:
        runtime = PipelineRuntime()
        runtime.start_discovery()
    register_routes(app, runtime)
    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), threaded=True)
