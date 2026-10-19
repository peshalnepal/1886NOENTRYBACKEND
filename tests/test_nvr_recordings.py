import asyncio
import json
import shutil
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from fastapi import FastAPI, HTTPException

from NVR.config import Settings
from NVR.core.database import Database
from NVR.core.database_orm import Camera
from NVR.recordings import RecordingService, covers_window, parse_event_time, router
from NVR.relay import MediaRelay, _same_setting
from application.services.clip_storage import EventClipService

CAMERA = "92a11140-ed7a-4471-9f72-fba7bbd596de"
EVENT = datetime(2026, 10, 5, 19, 18, tzinfo=timezone.utc)


class WindowTests(unittest.TestCase):
    def test_compact_toronto_time_and_offset_agree(self):
        self.assertEqual(parse_event_time("2026-10-5-15-18", "America/Toronto"), EVENT)
        self.assertEqual(parse_event_time("2026-10-05T15:18:00-04:00"), EVENT)

    def test_ambiguous_missing_and_invalid_timezones_rejected(self):
        for value, zone in [("2026-10-5-15-18", None), ("2026-11-01T01:30:00", "America/Toronto"),
                            ("2026-03-08T02:30:00", "America/Toronto"), ("2026-10-05", "invalid")]:
            with self.assertRaises(HTTPException):
                parse_event_time(value, zone)

    def test_gap_is_not_accepted_as_complete_recording(self):
        start, end = EVENT - timedelta(seconds=90), EVENT + timedelta(seconds=30)
        self.assertTrue(covers_window([{"start": start.isoformat(), "duration": 120}], start, end))
        self.assertFalse(covers_window([{"start": start.isoformat(), "duration": 60},
            {"start": EVENT.isoformat(), "duration": 30}], start, end))

    def test_profiles_preserve_channel_query_and_credentials(self):
        source = "rtsp://u:p@camera:554/Streaming/Channels/1702?transportmode=unicast"
        self.assertEqual(MediaRelay.stream_url(source, 1), source.replace("1702", "1701"))
        self.assertEqual(MediaRelay.stream_url(source.replace("1702", "1701"), 2), source)

    def test_canonical_durations_do_not_restart_recordings(self):
        self.assertTrue(_same_setting("recordDeleteAfter", "168h0m0s", "168h"))
        self.assertFalse(_same_setting("recordDeleteAfter", "24h0m0s", "168h"))


class TowerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.settings = Settings(_env_file=None, nvr_public_rtsp_base_url="rtsp://tower:8554",
            recording_clip_directory=self.temp.name + "/clips", recording_public_base_url="https://tower/api",
            recording_api_key="test-secret-which-is-only-for-local-tests")
        self.db = Database(self.temp.name + "/roster.sqlite")
        await self.db.init()
        async with self.db.session_factory() as session:
            session.add(Camera(camera_uuid=CAMERA, identity="serial:camera1", dedupe_key="serial:camera1",
                source_kind="direct", ip_address="192.168.1.11", rtsp_port=554,
                source_url="rtsp://192.168.1.11:554/Streaming/Channels/102"))
            await session.commit()
        self.service = RecordingService(self.settings, self.db.session_factory)
        await self.service.start()
        self.service._finalize_video = AsyncMock()
        self.requests = []
        self.spans = [{"start": "2020-01-01T00:00:00Z", "duration": 500000000}]
        def playback(request):
            self.requests.append(request)
            if request.url.path == "/list":
                return httpx.Response(200, json=self.spans)
            return httpx.Response(200, content=b"test-video-bytes", headers={"content-type": "video/mp4"})
        await self.service.http.aclose()
        self.service.http = httpx.AsyncClient(base_url="http://local", transport=httpx.MockTransport(playback))
        app = FastAPI()
        app.state.recordings = self.service
        app.include_router(router, prefix="/api")
        self.client = httpx.AsyncClient(base_url="https://tower", transport=httpx.ASGITransport(app))
        self.auth = {"Authorization": "Bearer " + self.settings.recording_api_key}
        # Always historical so the test does not depend on wall clock date.
        self.event = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=10)

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.service.close()
        await self.db.dispose()
        self.temp.cleanup()

    async def capture(self, **kwargs):
        return await self.client.post(f"/api/recordings/cameras/{CAMERA}/events",
            headers=self.auth, json={"event_time": self.event.isoformat()}, **kwargs)

    async def test_export_cached_download_range_and_signature(self):
        response = await self.capture()
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["duration"], 120)
        self.assertEqual(parse_event_time(data["start_time"]), self.event - timedelta(seconds=90))
        self.assertEqual(parse_event_time(data["end_time"]), self.event + timedelta(seconds=30))
        self.assertEqual(self.requests[1].url.params["duration"], "120")
        self.assertEqual(self.requests[1].url.params["path"], MediaRelay.recording_path("serial:camera1"))
        self.assertEqual((await self.capture()).json()["clip_id"], data["clip_id"])
        self.assertEqual(len(self.requests), 2)
        video = await self.client.get(data["view_url"], headers={"Range": "bytes=0-3"})
        self.assertEqual(video.status_code, 206)
        self.assertEqual(video.content, b"test")
        download = await self.client.get(data["download_url"])
        self.assertIn("attachment", download.headers["content-disposition"])
        self.assertEqual((await self.client.get(data["view_url"] + "x")).status_code, 403)

    async def test_auth_unknown_camera_and_gap(self):
        response = await self.client.post(f"/api/recordings/cameras/{CAMERA}/events",
            json={"event_time": self.event.isoformat()})
        self.assertEqual(response.status_code, 401)
        response = await self.client.post("/api/recordings/cameras/unknown/events", headers=self.auth,
            json={"event_time": self.event.isoformat()})
        self.assertEqual(response.status_code, 404)
        self.spans = []
        self.assertEqual((await self.capture()).status_code, 409)
        self.assertEqual(len(self.requests), 1)

    async def test_export_size_limit_removes_partial_file(self):
        self.settings.recording_max_clip_bytes = 2
        self.assertEqual((await self.capture()).status_code, 413)
        self.assertEqual(list(self.service.directory.glob("*.part")), [])

    async def test_simultaneous_requests_share_export(self):
        one, two = await asyncio.gather(self.capture(), self.capture())
        self.assertEqual(one.status_code, 200)
        self.assertEqual(one.json()["clip_id"], two.json()["clip_id"])
        self.assertEqual(len(self.requests), 2)

    async def test_expired_link_and_manifest_are_rejected(self):
        data = (await self.capture()).json()
        expired = f"https://tower/api/recordings/clips/{data['clip_id']}/video?expires=1&signature=" + self.service._signature(data['clip_id'], 1)
        self.assertEqual((await self.client.get(expired)).status_code, 403)
        manifest = self.service.directory / (data["clip_id"] + ".json")
        saved = json.loads(manifest.read_text())
        saved["expires_at"] = 1
        manifest.write_text(json.dumps(saved))
        response = await self.client.get(f"/api/recordings/clips/{data['clip_id']}", headers=self.auth)
        self.assertEqual(response.status_code, 410)

    async def test_future_event_rejected_without_export(self):
        response = await self.client.post(f"/api/recordings/cameras/{CAMERA}/events", headers=self.auth,
            json={"event_time": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.requests, [])

    async def test_historical_export_restores_path_after_recorder_restart(self):
        relay = SimpleNamespace(ensure_recording=AsyncMock(),
                                adapter=SimpleNamespace(render=lambda kind, url: url))
        self.service.relay = relay
        self.assertEqual((await self.capture()).status_code, 200)
        relay.ensure_recording.assert_awaited_once_with("serial:camera1", "rtsp://192.168.1.11:554/Streaming/Channels/102")

    async def test_cloud_persists_tower_link_without_downloading_video(self):
        cloud = object.__new__(EventClipService)
        cloud.enabled, cloud.provider = True, "nvr"
        cloud.nvr_api_key, cloud._http = self.settings.recording_api_key, self.client
        cloud._save_video_record = AsyncMock()
        cloud._download_clip = AsyncMock(side_effect=AssertionError("cloud must not download"))
        result = await cloud.capture_pre_event_clip(camera_uuid=CAMERA,
            ctx=SimpleNamespace(device_url="https://tower/api", camera_code="camera-1"),
            event_ts_ms=int(self.event.timestamp() * 1000), requires_approval=True)
        self.assertEqual(result["provider"], "nvr")
        self.assertTrue(result["recording_url"].startswith("https://tower/api/recordings/"))
        self.assertEqual(cloud._save_video_record.await_args.kwargs["storage_key"], "")
        self.assertTrue(cloud._save_video_record.await_args.kwargs["requires_approval"])
        cloud._download_clip.assert_not_awaited()

    async def test_recording_and_live_paths_have_different_sources_and_lifecycles(self):
        payloads = {}
        def api(request):
            if request.method == "GET":
                return httpx.Response(404)
            payloads[request.url.path.split("/")[-1]] = json.loads(request.content)
            return httpx.Response(200)
        relay = MediaRelay(self.settings, self.db.session_factory, SimpleNamespace(render=lambda kind, url: url),
                           transport=httpx.MockTransport(api))
        try:
            self.assertEqual(await relay.reconcile(["serial:camera1"]), [])
            live, record = payloads[relay.path("serial:camera1")], payloads[relay.recording_path("serial:camera1")]
            self.assertTrue(live["source"].endswith("102"))
            self.assertTrue(live["sourceOnDemand"])
            self.assertFalse(live["record"])
            self.assertTrue(record["source"].endswith("101"))
            self.assertFalse(record["sourceOnDemand"])
            self.assertTrue(record["record"])
        finally:
            await relay.close()


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe required")
class VideoExportTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_video_is_browser_compatible_and_two_minutes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.part"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=64x64:r=5",
                "-t", "120", "-c:v", "libx264", "-f", "mp4", str(source)], check=True)
            settings = Settings(_env_file=None, nvr_public_rtsp_base_url="rtsp://tower:8554")
            service = RecordingService(settings, None)
            try:
                await service._finalize_video(source)
                data = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
                    "format=duration:stream=codec_name", "-of", "json", str(source)]))
                self.assertAlmostEqual(float(data["format"]["duration"]), 120, delta=0.1)
                self.assertEqual(data["streams"][0]["codec_name"], "h264")
            finally:
                await service.close()
