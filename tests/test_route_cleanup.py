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
            get_loaded_pipeline=Mock(side_effect=RuntimeError("offline")),
            cleanup_device_resources=AsyncMock(),
        )
        db = Mock()
        device_uuid = uuid.uuid4()

        await _cleanup_device_runtime(manager, db, device_uuid=device_uuid, owner_id=7)

        # An unreadable pipeline still leaves the DB-backed teardown to run.
        manager.cleanup_device_resources.assert_awaited_once_with(
            db, device_uuid=device_uuid, active=None
        )

    async def test_device_cleanup_never_starts_a_pipeline_to_tear_one_down(self):
        pipeline = SimpleNamespace(remove_channel=AsyncMock())
        manager = SimpleNamespace(
            get_loaded_pipeline=Mock(return_value=pipeline),
            get_activepipeline=AsyncMock(),
            cleanup_device_resources=AsyncMock(),
        )
        db = Mock()
        device_uuid = uuid.uuid4()

        await _cleanup_device_runtime(manager, db, device_uuid=device_uuid, owner_id=7)

        manager.get_loaded_pipeline.assert_called_once_with(user_id=7)
        manager.get_activepipeline.assert_not_awaited()
        manager.cleanup_device_resources.assert_awaited_once_with(
            db, device_uuid=device_uuid, active=pipeline
        )

    async def test_device_cleanup_survives_failing_cleanup_call(self):
        manager = SimpleNamespace(
            get_loaded_pipeline=Mock(return_value=None),
            cleanup_device_resources=AsyncMock(side_effect=RuntimeError("edge down")),
        )

        await _cleanup_device_runtime(
            manager, Mock(), device_uuid=uuid.uuid4(), owner_id=7
        )

        manager.cleanup_device_resources.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
