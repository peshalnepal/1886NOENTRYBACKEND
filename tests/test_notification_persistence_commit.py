import unittest
import uuid
from unittest.mock import AsyncMock, patch

from application.repositories.notification_repository import CameraContext
from application.services.notification import NotificationMessage, NotificationService, WebNotificationHub


class _FakeSession:
    def __init__(self):
        self.commit = AsyncMock()
        self.rollback = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class NotificationPersistenceCommitTests(unittest.IsolatedAsyncioTestCase):
    def _make_message(self):
        camera_uuid = str(uuid.uuid4())
        site_uuid = str(uuid.uuid4())
        return NotificationMessage(
            user_id=7,
            id="cam-1-123",
            ts_ms=1234567890,
            camera_uuid=camera_uuid,
            site_uuid=site_uuid,
            site_name="Dock Yard",
            title="Detection: person",
            body="Person detected",
            alert_type="detection_summary",
            cls_names=["person"],
        )

    def _make_context(self, msg):
        return CameraContext(
            user_id=msg.user_id,
            site_uuid=uuid.UUID(msg.site_uuid),
            site_name=msg.site_name,
            camera_code="CAM-01",
            camera_name="Gate Camera",
            device_uuid=None,
            device_name=None,
        )

    async def test_persist_notification_now_commits_after_successful_flush(self):
        service = NotificationService(hub=WebNotificationHub())
        session = _FakeSession()
        service.set_session_factory(lambda: session)

        msg = self._make_message()
        ctx = self._make_context(msg)
        prepared_item = object()

        with patch.object(
            service.flusher,
            "_prepare_notification_item",
            AsyncMock(return_value=prepared_item),
        ) as prepare_mock, patch.object(
            service.flusher,
            "_flush_user_batch",
            AsyncMock(return_value=True),
        ) as flush_mock:
            await service.enqueue_notification(msg, ctx)

        prepare_mock.assert_awaited_once_with(
            msg=msg,
            ctx=ctx,
            extra_payload=None,
        )
        flush_mock.assert_awaited_once_with(int(ctx.user_id), [prepared_item], db=session)
        session.commit.assert_awaited_once()
        session.rollback.assert_not_awaited()

    async def test_persist_notification_now_rolls_back_when_flush_fails(self):
        service = NotificationService(hub=WebNotificationHub())
        session = _FakeSession()
        service.set_session_factory(lambda: session)

        msg = self._make_message()
        ctx = self._make_context(msg)
        prepared_item = object()

        with patch.object(
            service.flusher,
            "_prepare_notification_item",
            AsyncMock(return_value=prepared_item),
        ), patch.object(
            service.flusher,
            "_flush_user_batch",
            AsyncMock(return_value=False),
        ), self.assertLogs("application.services.notification.flusher", level="WARNING") as captured:
            await service.enqueue_notification(msg, ctx)

        session.commit.assert_not_awaited()
        session.rollback.assert_awaited_once()
        self.assertTrue(
            any("Failed to persist notification immediately" in line for line in captured.output)
        )


if __name__ == "__main__":
    unittest.main()
