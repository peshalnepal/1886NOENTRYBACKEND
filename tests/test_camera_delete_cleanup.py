import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from routes.camera_routes import _cleanup_camera_runtime


class _FakePipeline:
    def __init__(self, cfg_by_camera):
        self._cfg_by_camera = dict(cfg_by_camera)
        self.remove_channel = AsyncMock()

    async def get_channel_config(self, camera_uuid):
        return self._cfg_by_camera.get(str(camera_uuid))


class CameraDeleteCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_cleanup_uses_loaded_pipeline_runtime_device_url(self):
        camera_uuid = uuid.uuid4()
        pipeline = _FakePipeline(
            {
                str(camera_uuid): SimpleNamespace(
                    camera_uuid=camera_uuid,
                    device_url="http://edge.example:19030",
                )
            }
        )
        manager = SimpleNamespace(
            get_loaded_pipeline=Mock(return_value=pipeline),
            _edge=SimpleNamespace(delete_camera=AsyncMock()),
            _webrtc=SimpleNamespace(delete_stream=AsyncMock(return_value=True)),
        )

        await _cleanup_camera_runtime(
            manager,
            user_id=5,
            camera_uuid=camera_uuid,
            camera_code="cam-abcdef12",
            device_urls=[],
        )

        manager.get_loaded_pipeline.assert_called_once_with(user_id=5)
        manager._edge.delete_camera.assert_awaited_once_with(
            device_url="http://edge.example:19030",
            camera_uuid=str(camera_uuid),
        )
        manager._webrtc.delete_stream.assert_awaited_once_with(stream_key="cam-abcdef12")
        pipeline.remove_channel.assert_awaited_once_with(camera_uuid)

    async def test_cleanup_without_loaded_pipeline_still_deletes_edge_and_stream(self):
        camera_uuid = uuid.uuid4()
        manager = SimpleNamespace(
            get_loaded_pipeline=Mock(return_value=None),
            _edge=SimpleNamespace(delete_camera=AsyncMock()),
            _webrtc=SimpleNamespace(delete_stream=AsyncMock(return_value=False)),
        )

        await _cleanup_camera_runtime(
            manager,
            user_id=7,
            camera_uuid=camera_uuid,
            camera_code="cam-12345678",
            device_urls=["http://edge-a.example:19030", "http://edge-a.example:19030"],
        )

        manager.get_loaded_pipeline.assert_called_once_with(user_id=7)
        manager._edge.delete_camera.assert_awaited_once_with(
            device_url="http://edge-a.example:19030",
            camera_uuid=str(camera_uuid),
        )
        manager._webrtc.delete_stream.assert_awaited_once_with(stream_key="cam-12345678")


if __name__ == "__main__":
    unittest.main()
