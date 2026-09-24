"""HTTP entry point. Run one process: it owns one GPU inference runtime."""

import logging
import os
import sys

from flask import Flask

_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
try:
    from dotenv import load_dotenv
    load_dotenv(_ENV_PATH)
except ImportError:
    if os.path.exists(_ENV_PATH) and not os.getenv("HIK_PASSWORD"):
        sys.stderr.write(
            "WARNING: {path} exists but python-dotenv is not installed, so it "
            "was NOT loaded.\n"
            "         The service is running on built-in defaults: no camera "
            "credentials and no NVRs,\n"
            "         which means camera discovery will find nothing.\n"
            "         Fix with:  pip install python-dotenv\n"
            "         (or run under systemd, which loads .env as an "
            "EnvironmentFile)\n".format(path=_ENV_PATH)
        )

if __package__:
    from .runtime import PipelineRuntime
    from .routes import register_routes
    from .lifecycle import serve
else:
    from runtime import PipelineRuntime
    from routes import register_routes
    from lifecycle import serve

logging.basicConfig(level=logging.INFO)


def create_app(runtime=None):
    app = Flask(__name__)
    if runtime is None:
        runtime = PipelineRuntime()
        runtime.start_discovery()
    register_routes(app, runtime)
    return app


if __name__ == "__main__":
    app = create_app()
    serve(app, app.extensions["pipeline_runtime"], host="0.0.0.0",
          port=int(os.getenv("PORT", "8080")), threaded=True, use_reloader=False)
