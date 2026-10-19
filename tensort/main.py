"""HTTP entry point. Run one process: it owns one GPU inference runtime."""

import logging
import re
import os
import sys

from flask import Flask

_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
try:
    from dotenv import load_dotenv
    load_dotenv(_ENV_PATH)
except ImportError:
    if os.path.exists(_ENV_PATH):
        sys.stderr.write(
            "WARNING: {path} exists but python-dotenv is not installed, so it "
            "was NOT loaded.\n"
            "         Export the required inference settings before starting.\n"
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

class _RedactedFormatter(logging.Formatter):
    """Redact the final log text, including exception chains, without mutation."""
    def __init__(self, formatter):
        super().__init__()
        self.formatter = formatter

    def format(self, record):
        text = self.formatter.format(record)
        text = re.sub(r"(\b[a-z][a-z0-9+.-]*://)[^\s/@]+@", r"\1***@", text,
                      flags=re.IGNORECASE)
        return re.sub(r"([?&](?:password|passwd|pwd|token|access_token)=)[^&\s\"']+",
                      r"\1***", text, flags=re.IGNORECASE)


logging.basicConfig(level=logging.INFO)
for handler in logging.getLogger().handlers:
    if not isinstance(handler.formatter, _RedactedFormatter):
        handler.setFormatter(_RedactedFormatter(handler.formatter or logging.Formatter()))


def create_app(runtime=None):
    app = Flask(__name__)
    if runtime is None:
        runtime = PipelineRuntime()
    register_routes(app, runtime)
    return app


if __name__ == "__main__":
    app = create_app()
    serve(app, app.extensions["pipeline_runtime"], host="0.0.0.0",
          port=int(os.getenv("PORT", "8080")), threaded=True, use_reloader=False)
