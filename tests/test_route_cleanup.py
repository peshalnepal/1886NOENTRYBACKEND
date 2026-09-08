import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import Mock

from routes.device_routes import _cleanup_device_runtime
from routes._runtime_cleanup import cleanup_camera_resources


class RouteCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_camera_resources_are_deleted_once_per_device_and_pipeline(self):
        camera_uuid = uuid.uuid4()
        manager = SimpleNamespace(
            edge=SimpleNamespace(delete_camera=AsyncMock()),
            webrtc=SimpleNamespace(delete_stream=AsyncMock()),
        )
        pipeline = SimpleNamespace(remove_channel=AsyncMock())

        await cleanup_camera_resources(
            manager,
            camera_uuid=camera_uuid,
            camera_code="cam-123",
            device_urls=[" http://edge-a ", "http://edge-a", "http://edge-b"],
            pipeline=pipeline,
            log_prefix="[test]",
        )

        self.assertEqual(manager.edge.delete_camera.await_count, 2)
        manager.edge.delete_camera.assert_any_await(
            device_url="http://edge-a", camera_uuid=str(camera_uuid)
        )
        manager.edge.delete_camera.assert_any_await(
            device_url="http://edge-b", camera_uuid=str(camera_uuid)
        )
        manager.webrtc.delete_stream.assert_awaited_once_with(stream_key="cam-123")
        pipeline.remove_channel.assert_awaited_once_with(camera_uuid)

    async def test_resource_failure_does_not_stop_remaining_cleanup(self):
        camera_uuid = uuid.uuid4()
        edge_delete = AsyncMock(side_effect=[RuntimeError("offline"), None])
        manager = SimpleNamespace(
            edge=SimpleNamespace(delete_camera=edge_delete),
            webrtc=SimpleNamespace(delete_stream=AsyncMock()),
        )

        await cleanup_camera_resources(
            manager,
            camera_uuid=camera_uuid,
            camera_code="cam-456",
            device_urls=["http://edge-a", "http://edge-b"],
            log_prefix="[test]",
        )

        self.assertEqual(edge_delete.await_count, 2)
        manager.webrtc.delete_stream.assert_awaited_once_with(stream_key="cam-456")

    async def test_device_cleanup_keeps_database_path_alive_on_manager_failure(self):
        manager = SimpleNamespace(
            get_activepipeline=AsyncMock(side_effect=RuntimeError("offline")),
        )
        db = Mock()

        await _cleanup_device_runtime(
            manager,
            db,
            device_uuid=uuid.uuid4(),
            owner_id=7,
        )

        manager.get_activepipeline.assert_awaited_once_with(user_id=7)


if __name__ == "__main__":
    unittest.main()
