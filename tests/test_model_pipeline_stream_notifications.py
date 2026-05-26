import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from application.repositories.notification_repository import CameraContext
from application.services.pipeline import ModelPipeline


class _FakeConfig:
    def __init__(self, camera_uuid: str, *, scheduled: bool = True):
        self.camera_uuid = camera_uuid
        self.site_uuid = str(uuid.uuid4())
        self.device_uuid = str(uuid.uuid4())
        self.device_url = "http://jetson.example:8080"
        self.enabled = True
        self.notification_enabled = True
        self.notification_trigger_mode = "inherit"
        self._scheduled = bool(scheduled)

    def is_scheduled_now(self, *, now_utc=None):
        return self._scheduled


class _FakeChannel:
    def __init__(self, config: _FakeConfig):
        self.config = config

    def key(self):
        return str(self.config.camera_uuid)

    async def fetch_snapshot_bytes(self):
        return None


class _FakeNotificationService:
    def __init__(self):
        self.hub = SimpleNamespace(publish=AsyncMock())
        self.is_camera_prerecord_eligible = AsyncMock(return_value=False)
        self.record_detection_overlay_frame = AsyncMock()
        self.enqueue_notification = AsyncMock()


class ModelPipelineStreamNotificationTests(unittest.IsolatedAsyncioTestCase):
    def _make_payload(self, camera_uuid: str) -> dict:
        return {
            "camera_uuid": camera_uuid,
            "frame_ts_ms": 1234567890,
            "frame_seq": 7,
            "frame_w": 640,
            "frame_h": 480,
            "image_url": "https://example.com/frame.jpg",
            "detections": [
                {
                    "cls_name": "person",
                    "conf": 0.91,
                    "box": {"x1": 10, "y1": 20, "x2": 110, "y2": 220},
                }
            ],
        }

    async def _make_pipeline(self, *, scheduled: bool = True, trigger_mode: str = "inherit"):
        camera_uuid = str(uuid.uuid4())
        config = _FakeConfig(camera_uuid, scheduled=scheduled)
        config.notification_trigger_mode = trigger_mode
        channel = _FakeChannel(config)
        service = _FakeNotificationService()
        pipeline = ModelPipeline(
            pipeline_id=uuid.uuid4(),
            notify_on_confirmed=False,
            notify_on_roi_enter=False,
            interesting_classes={"person"},
        )
        pipeline.set_notification_service(service)
        pipeline._trigger_mode_resolver.resolve = AsyncMock(return_value="any_detection")
        pipeline._get_camera_ctx = AsyncMock(
            return_value=CameraContext(
                user_id=7,
                site_uuid=uuid.UUID(config.site_uuid),
                site_name="Dock Yard",
                camera_code="CAM-01",
                camera_name="Gate Camera",
                device_uuid=uuid.UUID(config.device_uuid),
                device_name="Jetson A",
            )
        )
        pipeline._persist_and_maybe_email = AsyncMock()
        return pipeline, channel, service

    async def test_stream_payload_restores_summary_and_overlay_side_effects(self):
        pipeline, channel, service = await self._make_pipeline(scheduled=True, trigger_mode="any_detection")
        camera_uuid = channel.key()

        service.is_camera_prerecord_eligible.return_value = True
        pipeline._tracker.update_from_event = Mock(return_value={"tracks": [], "events": []})
        pipeline._build_alert_extra_payload = AsyncMock(return_value={"image_url": "https://example.com/frame.jpg"})
        pipeline._emit_detection_summary_notification = AsyncMock(return_value=True)

        processed = await pipeline._process_detection_payload(camera_uuid, channel, self._make_payload(camera_uuid))

        self.assertTrue(processed)
        service.record_detection_overlay_frame.assert_awaited_once()
        pipeline._emit_detection_summary_notification.assert_awaited_once()
        pipeline._build_alert_extra_payload.assert_awaited_once()
        latest = await pipeline.get_latest_detection(camera_uuid)
        self.assertIsNotNone(latest)
        self.assertEqual(int(latest.frame_seq), 7)

    async def test_stream_payload_honors_notification_schedule_gate(self):
        pipeline, channel, service = await self._make_pipeline(scheduled=False)
        camera_uuid = channel.key()

        pipeline._tracker.update_from_event = Mock(return_value={"tracks": [], "events": []})
        pipeline._build_alert_extra_payload = AsyncMock(return_value={"image_url": "https://example.com/frame.jpg"})
        pipeline._emit_detection_summary_notification = AsyncMock(return_value=True)

        processed = await pipeline._process_detection_payload(camera_uuid, channel, self._make_payload(camera_uuid))

        self.assertTrue(processed)
        pipeline._emit_detection_summary_notification.assert_not_awaited()
        pipeline._build_alert_extra_payload.assert_not_awaited()
        latest = await pipeline.get_latest_detection(camera_uuid)
        self.assertIsNotNone(latest)
        self.assertEqual(int(latest.frame_seq), 7)
        service.record_detection_overlay_frame.assert_not_awaited()

    async def test_stream_payload_prefers_confirmed_track_alerts_over_summary(self):
        pipeline, channel, service = await self._make_pipeline(scheduled=True, trigger_mode="any_detection")
        camera_uuid = channel.key()

        pipeline.notify_on_confirmed = True
        pipeline._tracker.update_from_event = Mock(
            return_value={
                "tracks": [{"track_id": 11, "cls_name": "person", "conf": 0.88, "box": {"x1": 10, "y1": 20, "x2": 110, "y2": 220}}],
                "events": [("track_confirmed", 11)],
            }
        )
        pipeline._build_alert_extra_payload = AsyncMock(return_value={"image_url": "https://example.com/frame.jpg"})
        pipeline._emit_item_detected_notifications = AsyncMock(return_value=True)
        pipeline._emit_detection_summary_notification = AsyncMock(return_value=True)

        processed = await pipeline._process_detection_payload(camera_uuid, channel, self._make_payload(camera_uuid))

        self.assertTrue(processed)
        pipeline._emit_item_detected_notifications.assert_awaited_once()
        pipeline._emit_detection_summary_notification.assert_not_awaited()
        pipeline._build_alert_extra_payload.assert_awaited_once()

    async def test_stream_payload_drops_mismatched_camera_uuid(self):
        pipeline, channel, service = await self._make_pipeline(scheduled=True)
        camera_uuid = channel.key()
        other_camera_uuid = str(uuid.uuid4())

        pipeline._tracker.update_from_event = Mock()

        processed = await pipeline._process_detection_payload(
            camera_uuid,
            channel,
            self._make_payload(other_camera_uuid),
        )

        self.assertFalse(processed)
        pipeline._tracker.update_from_event.assert_not_called()
        latest = await pipeline.get_latest_detection(camera_uuid)
        self.assertIsNone(latest)
        service.record_detection_overlay_frame.assert_not_awaited()

    async def test_remove_channel_clears_cached_detection_and_context(self):
        pipeline, channel, _service = await self._make_pipeline(scheduled=True)
        camera_uuid = channel.key()

        await pipeline.add_channel(channel)
        payload = self._make_payload(camera_uuid)
        processed = await pipeline._process_detection_payload(camera_uuid, channel, payload)

        self.assertTrue(processed)
        self.assertIsNotNone(await pipeline.get_latest_detection(camera_uuid))

        pipeline._ctx_resolver._cache[camera_uuid] = (
            9999999999.0,
            CameraContext(
                user_id=7,
                site_uuid=uuid.UUID(channel.config.site_uuid),
                site_name="Dock Yard",
                camera_code="CAM-01",
                camera_name="Gate Camera",
                device_uuid=uuid.UUID(channel.config.device_uuid),
                device_name="Jetson A",
            ),
        )

        removed = await pipeline.remove_channel(camera_uuid)

        self.assertTrue(removed)
        self.assertIsNone(await pipeline.get_latest_detection(camera_uuid))
        self.assertNotIn(camera_uuid, pipeline._ctx_resolver._cache)


if __name__ == "__main__":
    unittest.main()
