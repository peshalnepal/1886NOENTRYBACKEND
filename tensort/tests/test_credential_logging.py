"""Credentials stay out of diagnostic output and configured service logs."""

import io
import logging
from pathlib import Path
import runpy
import sys
import types
import unittest
from unittest.mock import Mock, patch

EDGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDGE))


SECRET = "TEST_ONLY_NOT_A_REAL_PASSWORD"
URLS = (
    "rtsp://viewer:" + SECRET + "%24%40@camera.example.test:554/Streaming/Channels/102",
    "RTSPS://viewer:" + SECRET + "@camera.example.test/Streaming/Channels/102",
    "https://camera.example.test/Streaming/Channels/102?password=" + SECRET + "&channel=1",
)


class CredentialLoggingTests(unittest.TestCase):
    def setUp(self):
        self.root = logging.getLogger()
        self.previous_handlers = self.root.handlers[:]
        self.previous_level = self.root.level
        self.output = io.StringIO()
        self.handler = logging.StreamHandler(self.output)
        self.handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        self.root.handlers = [self.handler]
        self.root.setLevel(logging.INFO)
        self.addCleanup(self.restore_logging)
        self.load_main_logging()

    def restore_logging(self):
        self.root.handlers = self.previous_handlers
        self.root.setLevel(self.previous_level)
        self.handler.close()

    def load_main_logging(self):
        # Execute the real entry-point logging setup without creating an app,
        # opening the database, loading .env, or importing native dependencies.
        modules = {}
        for name, attributes in {
            "flask": {"Flask": Mock()},
            "runtime": {"PipelineRuntime": Mock()},
            "routes": {"register_routes": Mock()},
            "lifecycle": {"serve": Mock()},
            "dotenv": {"load_dotenv": Mock()},
        }.items():
            module = types.ModuleType(name)
            module.__dict__.update(attributes)
            modules[name] = module
        with patch.dict(sys.modules, modules):
            runpy.run_path(str(EDGE / "main.py"), run_name="edge_logging_test")


    def test_service_messages_hide_url_credentials(self):
        logger = logging.getLogger("jetson-app")
        for url in URLS:
            logger.error("Cannot open %s", url)
            logger.error("Cannot open %(source)s", {"source": url})
            logger.error("Cannot open " + url)
        text = self.output.getvalue()
        self.assertNotIn(SECRET, text, "Configured service logs exposed a URL password")
        self.assertIn("camera.example.test", text)
        self.assertIn("/Streaming/Channels/102", text)

    def test_service_tracebacks_hide_url_credentials(self):
        try:
            try:
                raise ValueError("Capture rejected " + URLS[0])
            except ValueError as cause:
                raise RuntimeError("SQL parameters: " + repr({"source_url": URLS[0]})) from cause
        except RuntimeError:
            logging.getLogger("repositories.camera_repository").exception("Camera write failed")
        text = self.output.getvalue()
        self.assertNotIn(SECRET, text, "Formatted exception chain exposed a URL password")
        self.assertIn("ValueError", text)
        self.assertIn("RuntimeError", text)
        self.assertIn("Camera write failed", text)

    def test_logging_keeps_original_record_and_url(self):
        record = logging.LogRecord("jetson-app", logging.ERROR, __file__, 1,
                                   "Cannot open %s", (URLS[0],), None)
        self.root.handle(record)
        self.assertEqual(record.args, (URLS[0],))
        self.assertEqual(record.getMessage(), "Cannot open " + URLS[0])

    def test_existing_log_format_and_ordinary_messages_are_preserved(self):
        logging.getLogger("jetson-app").warning("camera 7 disconnected")
        self.assertEqual(self.output.getvalue(), "WARNING:jetson-app:camera 7 disconnected\n")

    def test_repeated_setup_does_not_duplicate_handlers_or_messages(self):
        self.load_main_logging()
        logging.getLogger("jetson-app").warning("one message")
        self.assertEqual(self.root.handlers, [self.handler])
        self.assertEqual(self.output.getvalue(), "WARNING:jetson-app:one message\n")


if __name__ == "__main__":
    unittest.main()
