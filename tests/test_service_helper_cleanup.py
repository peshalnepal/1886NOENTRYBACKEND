import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from application.services.clip_storage import (
    EventClipService,
    extract_notification_clip_external_ids,
)
from application.services.report.pdf_report_service import PdfReportGenerator


class ClipExternalIdTests(unittest.TestCase):
    def test_invalid_or_empty_clip_returns_no_ids(self):
        for payload in (None, [], {}, {"clip": None}, {"clip": []}, {"clip": {}}):
            with self.subTest(payload=payload):
                self.assertEqual(extract_notification_clip_external_ids(payload), [])

    def test_external_id_is_trimmed_and_converted_to_string(self):
        for raw, expected in ((" clip-1 ", ["clip-1"]), (123, ["123"]), (" ", []), (None, []), (0, [])):
            with self.subTest(raw=raw):
                self.assertEqual(
                    extract_notification_clip_external_ids({"clip": {"external_id": raw}}),
                    expected,
                )

    def test_only_root_clip_supplies_external_id(self):
        payload = {
            "clip": {"external_id": "root"},
            "msg": {"clip": {"external_id": "nested"}},
            "extra": {"clip": {"external_id": "nested"}},
        }
        self.assertEqual(extract_notification_clip_external_ids(payload), ["root"])


class ClipCaptureConfigurationTests(unittest.TestCase):
    def test_capture_flag_keeps_existing_defaults_and_false_values(self):
        cases = (
            (None, True),
            ("", True),
            ("   ", True),
            ("1", True),
            ("true", True),
            ("unexpected", True),
            ("0", False),
            ("false", False),
            ("no", False),
            ("off", False),
            (" OFF ", False),
        )
        for raw, expected in cases:
            env = {"MEDIAMTX_PLAYBACK_BASE_URL": "http://playback.test"}
            if raw is not None:
                env["VIDEO_CLIP_CAPTURE_ENABLED"] = raw
            with self.subTest(raw=raw), patch.dict("os.environ", env, clear=True):
                with patch("application.services.clip_storage.httpx.AsyncClient"):
                    service = EventClipService()
                self.assertEqual(service.enabled, expected)
                self.assertEqual(service.CLIP_DURATION_S, service.PRE_EVENT_S + service.POST_EVENT_S)

    def test_capture_requires_playback_url(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("application.services.clip_storage.httpx.AsyncClient"):
                self.assertFalse(EventClipService().enabled)


class ReportImageUrlTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_and_https_images_are_fetched(self):
        generator = PdfReportGenerator(session_factory=None)
        for url in ("http://images.test/snapshot.jpg", "https://images.test/snapshot.jpg"):
            with self.subTest(url=url):
                client = SimpleNamespace(
                    get=AsyncMock(return_value=SimpleNamespace(status_code=200, content=b"image"))
                )
                self.assertEqual(await generator._fetch_image_bytes(client, url), b"image")
                client.get.assert_awaited_once_with(url)

    async def test_unsupported_urls_are_not_fetched(self):
        generator = PdfReportGenerator(session_factory=None)
        for url in ("", "file:///snapshot.jpg", "ftp://images.test/snapshot.jpg"):
            with self.subTest(url=url):
                client = SimpleNamespace(get=AsyncMock())
                self.assertIsNone(await generator._fetch_image_bytes(client, url))
                client.get.assert_not_awaited()
