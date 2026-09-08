import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from application.services.clip_storage import EventClipService, PlaybackUnavailableError


class ClipCaptureTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.end = self.start + timedelta(seconds=120)
        self.overlay = {"camera_uuid": "camera-1", "frames": []}
        self.context = SimpleNamespace(camera_code=" camera-stream ")
        self.service = object.__new__(EventClipService)
        self.service.enabled = True
        self.service.storage_enabled = True
        self.service._camera_locks = {}
        self.service._playback_unavailable_until = 0.0
        self.service._get_capture_window = Mock(return_value=(self.start, self.end))
        self.service._fetch_recording_spans = AsyncMock(return_value=[{}])
        self.service._choose_span = Mock(return_value=(self.start, self.end))
        self.service._download_clip = AsyncMock(return_value=b"video")
        self.service._build_storage_key = Mock(return_value="clips/camera-1/clip.mp4")
        self.service._upload_blob = AsyncMock(return_value="https://storage.test/clip.mp4")
        self.service._build_playback_clip_url = Mock(return_value="http://playback.test/clip")
        self.service._save_video_record = AsyncMock()
        self.service._warn_playback_unavailable = Mock()

    async def capture(self):
        return await self.service.capture_pre_event_clip(
            camera_uuid="camera-1",
            ctx=self.context,
            overlay_payload=self.overlay,
            requires_approval=True,
        )

    def assert_saved_result(self, result, *, storage_key, recording_url):
        self.assertEqual(result["storage_key"], storage_key)
        self.assertEqual(result["recording_url"], recording_url)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["duration"], 120)
        self.assertEqual(result["path"], "camera-stream")
        self.assertEqual(result["start_time"], self.start.isoformat())
        self.assertEqual(result["end_time"], self.end.isoformat())
        self.service._save_video_record.assert_awaited_once_with(
            camera_uuid="camera-1",
            external_id=result["external_id"],
            start_time=self.start,
            end_time=self.end,
            duration_s=120,
            status="completed",
            storage_key=storage_key,
            recording_url=recording_url,
            overlay_payload=self.overlay,
            requires_approval=True,
        )

    async def test_uploaded_clip_is_saved_and_returned(self):
        result = await self.capture()
        self.assert_saved_result(
            result,
            storage_key="clips/camera-1/clip.mp4",
            recording_url="https://storage.test/clip.mp4",
        )
        self.service._upload_blob.assert_awaited_once_with(
            blob_name="clips/camera-1/clip.mp4", payload=b"video"
        )
        self.service._build_playback_clip_url.assert_not_called()

    async def test_disabled_storage_skips_download_and_saves_playback_url(self):
        self.service.storage_enabled = False
        result = await self.capture()
        self.assert_saved_result(result, storage_key="", recording_url="http://playback.test/clip")
        self.service._download_clip.assert_not_awaited()
        self.service._upload_blob.assert_not_awaited()

    async def test_upload_failure_falls_back_to_playback_url(self):
        self.service._upload_blob.side_effect = RuntimeError("upload unavailable")
        with self.assertLogs("application.services.clip_storage", level="WARNING"):
            result = await self.capture()
        self.assert_saved_result(result, storage_key="", recording_url="http://playback.test/clip")

    async def test_empty_download_does_not_save_or_upload(self):
        self.service._download_clip.return_value = b""
        self.assertIsNone(await self.capture())
        self.service._save_video_record.assert_not_awaited()
        self.service._upload_blob.assert_not_awaited()
        self.service._build_playback_clip_url.assert_not_called()

    async def test_persistence_failure_still_returns_the_completed_clip(self):
        for storage_enabled in (True, False):
            with self.subTest(storage_enabled=storage_enabled):
                self.service.storage_enabled = storage_enabled
                self.service._save_video_record.reset_mock()
                self.service._save_video_record.side_effect = RuntimeError("database unavailable")
                with self.assertLogs("application.services.clip_storage", level="ERROR"):
                    result = await self.capture()
                self.assertEqual(result["status"], "completed")
                self.service._save_video_record.assert_awaited_once()

    async def test_unavailable_playback_sets_backoff_without_saving(self):
        self.service._fetch_recording_spans.side_effect = PlaybackUnavailableError("offline")
        self.assertIsNone(await self.capture())
        self.assertGreater(self.service._playback_unavailable_until, 0)
        self.service._warn_playback_unavailable.assert_called_once()
        self.service._save_video_record.assert_not_awaited()

    async def test_missing_or_short_recording_is_skipped(self):
        for window in (None, (self.start, self.start + timedelta(seconds=1))):
            with self.subTest(window=window):
                self.service._choose_span.return_value = window
                self.assertIsNone(await self.capture())
                self.service._download_clip.assert_not_awaited()
                self.service._save_video_record.assert_not_awaited()

    async def test_download_failure_records_a_failed_capture(self):
        self.service._download_clip.side_effect = RuntimeError("download unavailable")
        with self.assertLogs("application.services.clip_storage", level="ERROR"):
            self.assertIsNone(await self.capture())
        self.service._save_video_record.assert_awaited_once()
        saved = self.service._save_video_record.call_args.kwargs
        self.assertEqual(saved["status"], "failed")
        self.assertEqual(saved["error"], "download unavailable")
        self.assertEqual(saved["start_time"], self.start)
        self.assertEqual(saved["end_time"], self.end)
