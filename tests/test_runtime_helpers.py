import asyncio
import unittest
import uuid

from application.channels.channel import VideoChannel
from application.channels.channel_config import VideoChannelConfig
from application.services.common.background import BackgroundTasks
from application.services.notification.hub import WebNotificationHub


class RuntimeHelperTests(unittest.IsolatedAsyncioTestCase):
    async def test_background_tasks_track_shared_runner(self):
        finished = asyncio.Event()
        tasks = BackgroundTasks()

        async def work():
            finished.set()

        task = tasks.spawn(work(), name="test-work")
        await task

        self.assertTrue(finished.is_set())
        self.assertFalse(tasks._tasks)

    async def test_notification_hub_keeps_latest_message_when_queue_is_full(self):
        hub = WebNotificationHub(max_q=1)
        queue = await hub.subscribe(user_id=7)

        await hub.publish_to_users([7], "first")
        await hub.publish_to_users([7], "latest")

        self.assertEqual(await queue.get(), "latest")

    async def test_channel_timeout_uses_stream_floor(self):
        config = VideoChannelConfig(
            camera_uuid=uuid.uuid4(),
            device_url="http://edge.example",
            source_url="rtsp://camera.example/stream",
            request_timeout_s=3.0,
        )
        channel = VideoChannel(config)

        timeout = channel._request_timeout(minimum=15.0)

        self.assertEqual(timeout.read, 15.0)
        self.assertEqual(timeout.connect, 3.0)


if __name__ == "__main__":
    unittest.main()
