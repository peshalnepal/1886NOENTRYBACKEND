"""NVR registrations survive restarts without a Jetson discovery service.

Run directly with Python; capture/GPU work is replaced with an in-memory pipeline.
"""
import asyncio
import copy
import importlib
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from flask import Flask
from sqlalchemy import inspect, text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tensort.database import DatabaseManager
from tensort.routes import register_routes
from tensort.runtime import PipelineRuntime

UUID = "11111111-1111-4111-8111-111111111111"
SOURCE = "rtsp://minipc:8554/nvr-example"


class Repository:
    def __init__(self):
        self.saved = {}

    def initialize_tables(self):
        return True

    def list_all(self):
        return [{"camera_uuid": key, "config_json": copy.deepcopy(cfg)}
                for key, cfg in self.saved.items()]

    async def save(self, key, cfg):
        self.saved[key] = copy.deepcopy(cfg)

    async def delete(self, key):
        self.saved.pop(key, None)


class Pipeline:
    def __init__(self):
        self.channels = {}

    async def add_channel(self, cfg):
        if cfg.enabled:
            self.channels[cfg.camera_uuid] = cfg
        else:
            self.channels.pop(cfg.camera_uuid, None)

    async def remove_channel(self, key):
        self.channels.pop(key, None)

    def peek_stats(self):
        return {"inference_ready": True}


def start_pipeline(runtime):
    runtime.pipeline = Pipeline()
    runtime.loop = object()
    runtime._ready.set()


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.patches = [
            patch.object(PipelineRuntime, "_run_loop", start_pipeline),
            patch.object(PipelineRuntime, "_call", lambda self, coro, **kw: asyncio.run(coro)),
            patch.dict(os.environ, {"DISCOVERY_ENABLED": "true", "DISCOVERY_AUTO_ADD": "true",
                                    "MAX_CAMERAS": "1"}),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.repo = Repository()
        self.runtime = PipelineRuntime(self.repo)
        self.runtime._loop_thread.join()
        self.client = register_routes(Flask(__name__), self.runtime).test_client()

    def test_nvr_registration_patch_and_delete(self):
        response = self.client.post("/api/cameras", json={
            "source_url": SOURCE,
            "config": {"camera_uuid": UUID, "discovery_identity": "serial:example"},
        })
        self.assertEqual(response.status_code, 201)
        self.assertEqual(self.repo.saved[UUID]["source_url"], SOURCE)
        self.assertEqual(self.runtime.pipeline.channels[UUID].source_url, SOURCE)
        replacement = SOURCE + "-replacement"
        response = self.client.patch("/cameras/" + UUID, json={"source_url": replacement})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.repo.saved[UUID]["source_url"], replacement)
        self.assertEqual(self.client.get("/cameras").json["cameras"][0]["camera_uuid"], UUID)
        self.assertEqual(self.client.delete("/api/cameras/" + UUID).status_code, 200)
        self.assertEqual(self.repo.saved, {})
        self.assertEqual(self.runtime.pipeline.channels, {})

    def test_restart_restores_saved_uuid_and_disabled_configuration(self):
        self.runtime.add_camera(SOURCE, {"camera_uuid": UUID})
        self.runtime.add_camera(SOURCE + "-disabled", {"camera_uuid": "disabled", "enabled": False})
        restored = PipelineRuntime(self.repo)
        restored._loop_thread.join()
        self.assertEqual(set(restored._cameras), {UUID, "disabled"})
        self.assertEqual(set(restored.pipeline.channels), {UUID})
        self.assertEqual(restored._cameras[UUID]["source_url"], SOURCE)
        self.assertFalse(hasattr(restored, "discovery"))

    def test_capacity_rejection_and_retry_after_removal(self):
        self.runtime.add_camera(SOURCE, {"camera_uuid": UUID})
        payload = {"camera_uuid": "second", "source_url": SOURCE + "-2"}
        self.assertEqual(self.client.post("/cameras", json=payload).status_code, 409)
        self.assertNotIn("second", self.repo.saved)
        self.runtime.remove_camera(UUID)
        self.assertEqual(self.client.post("/cameras", json=payload).status_code, 201)

    def test_registration_still_requires_uuid(self):
        self.assertEqual(self.client.post("/cameras", json={"source_url": SOURCE}).status_code, 400)
        self.assertEqual(self.repo.saved, {})

    def test_discovery_and_sync_routes_are_removed(self):
        for prefix in ("", "/api"):
            for method, path in [("GET", "/discovery"), ("GET", "/discovery/status"),
                                 ("GET", "/discovery/report"), ("POST", "/discovery/scan"),
                                 ("DELETE", "/discovery/serial:example"), ("POST", "/sync")]:
                with self.subTest(path=prefix + path, method=method):
                    self.assertEqual(self.client.open(prefix + path, method=method).status_code, 404)

    def test_health_only_reports_inference_runtime(self):
        for path in ("/", "/api", "/health", "/api/health"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json["pipeline_ready"])
            self.assertNotIn("discovery", response.json)
        self.assertNotIn("camera_selection", self.runtime.get_stats())

    def test_app_startup_needs_only_the_camera_runtime(self):
        # Keep deployment environment files out of this off-device test.
        with patch.dict(sys.modules, {"dotenv": types.SimpleNamespace(load_dotenv=lambda path: None)}):
            main = importlib.import_module("tensort.main")
        with patch.object(main, "PipelineRuntime", return_value=self.runtime) as constructor:
            app = main.create_app()
        constructor.assert_called_once_with()
        self.assertIs(app.extensions["pipeline_runtime"], self.runtime)
        self.assertEqual(app.test_client().get("/api/cameras").status_code, 200)


class SchemaTests(unittest.TestCase):
    def database(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "cameras.db"
        db = DatabaseManager("sqlite+aiosqlite:///" + str(path), "sqlite:///" + str(path))
        self.addCleanup(db.sync_engine.dispose)
        return db

    def test_fresh_database_only_creates_camera_configs(self):
        db = self.database()
        self.assertTrue(db.initialize_tables())
        self.assertEqual(inspect(db.sync_engine).get_table_names(), ["camera_configs"])

    def test_startup_preserves_legacy_roster_without_migrating_it(self):
        db = self.database()
        with db.sync_engine.begin() as conn:
            conn.execute(text("CREATE TABLE discovered_cameras (identity TEXT PRIMARY KEY)"))
            conn.execute(text("INSERT INTO discovered_cameras VALUES ('serial:legacy')"))
        self.assertTrue(db.initialize_tables())
        self.assertTrue(db.initialize_tables())
        with db.sync_engine.connect() as conn:
            self.assertEqual(conn.execute(text("SELECT * FROM discovered_cameras")).all(),
                             [("serial:legacy",)])
        self.assertEqual([c["name"] for c in inspect(db.sync_engine).get_columns("discovered_cameras")],
                         ["identity"])


if __name__ == "__main__":
    unittest.main()
