import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from routes.site_routes import _cleanup_cameras_background


class _FakePipeline:
    def __init__(self, configs_by_camera):
        self._configs_by_camera = dict(configs_by_camera)
        self.remove_channel = AsyncMock()

    def list_channel_ids(self):
        return list(self._configs_by_camera.keys())

    async def get_channel_config(self, camera_uuid):
        return self._configs_by_camera.get(str(camera_uuid))


class SiteDeleteCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_cleanup_includes_runtime_only_site_channels(self):
        site_uuid = uuid.uuid4()
        snap_camera_uuid = uuid.uuid4()
        runtime_only_camera_uuid = uuid.uuid4()
        other_site_camera_uuid = uuid.uuid4()

        pipeline = _FakePipeline(
            {
                str(snap_camera_uuid): SimpleNamespace(
                    camera_uuid=snap_camera_uuid,
                    site_uuid=site_uuid,
                    device_url="http://edge.example:19030",
                ),
                str(runtime_only_camera_uuid): SimpleNamespace(
                    camera_uuid=runtime_only_camera_uuid,
                    site_uuid=site_uuid,
                    device_url="http://edge.example:19030",
                ),
                str(other_site_camera_uuid): SimpleNamespace(
                    camera_uuid=other_site_camera_uuid,
                    site_uuid=uuid.uuid4(),
                    device_url="http://edge.example:19030",
                ),
            }
        )

        manager = SimpleNamespace(
            get_loaded_pipeline=Mock(return_value=pipeline),
            _edge=SimpleNamespace(delete_camera=AsyncMock()),
            _webrtc=SimpleNamespace(delete_stream=AsyncMock()),
        )

        await _cleanup_cameras_background(
            manager,
            [
                {
                    "camera_uuid": snap_camera_uuid,
                    "camera_code": "cam-12345678",
                    "device_urls": ["http://edge.example:19030"],
                }
            ],
            user_id=5,
            site_uuid=site_uuid,
        )

        manager.get_loaded_pipeline.assert_called_once_with(user_id=5)
        manager._edge.delete_camera.assert_any_call(
            device_url="http://edge.example:19030",
            camera_uuid=str(snap_camera_uuid),
        )
        manager._edge.delete_camera.assert_any_call(
            device_url="http://edge.example:19030",
            camera_uuid=str(runtime_only_camera_uuid),
        )
        self.assertEqual(manager._edge.delete_camera.await_count, 2)

        manager._webrtc.delete_stream.assert_awaited_once_with(stream_key="cam-12345678")
        pipeline.remove_channel.assert_any_await(snap_camera_uuid)
        pipeline.remove_channel.assert_any_await(runtime_only_camera_uuid)
        self.assertEqual(pipeline.remove_channel.await_count, 2)


if __name__ == "__main__":
    unittest.main()
