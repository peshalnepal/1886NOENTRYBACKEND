import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import Mock

from application.channels.channel_config import VideoChannelConfig
from application.services.manager.controllers.pipeline import PipelineController
from application.services.manager.helpers import _camera_config_json, build_video_channel_config


class ManagerCleanupTests(unittest.TestCase):
    def test_camera_config_json_returns_copy_and_empty_for_missing_config(self):
        configuration = {"sample_fps": 8, "source_url": "ignored"}
        camera = SimpleNamespace(
            channel_configuration=SimpleNamespace(configuration=configuration)
        )

        result = _camera_config_json(camera)
        result["sample_fps"] = 12

        self.assertEqual(configuration["sample_fps"], 8)
        self.assertEqual(_camera_config_json(SimpleNamespace()), {})
        self.assertEqual(
            _camera_config_json(SimpleNamespace(channel_configuration=None)), {}
        )

    def test_build_config_filters_identity_fields_and_coerces_modes(self):
        camera_uuid = uuid.uuid4()
        site_uuid = uuid.uuid4()
        camera = SimpleNamespace(
            camera_uuid=camera_uuid,
            source_url="rtsp://camera/stream",
            webrtc_url=None,
            site_uuid=site_uuid,
            is_enabled=1,
            is_detection_enabled=0,
            is_notification_enabled=True,
        )

        config = build_video_channel_config(
            camera,
            device_uuid=uuid.uuid4(),
            device_url="http://edge.example",
            schedule_state={
                "timezone": "UTC",
                "schedule": VideoChannelConfig.default_schedule(),
                "use_site_schedule": True,
            },
            cfg_json={
                "camera_uuid": "wrong",
                "source_url": "wrong",
                "notification_trigger_mode": "bad-value",
                "camera_playback_enabled": True,
                "sample_fps": 9,
            },
        )

        self.assertEqual(config.camera_uuid, camera_uuid)
        self.assertEqual(config.source_url, camera.source_url)
        self.assertEqual(config.sample_fps, 9.0)
        self.assertEqual(config.notification_trigger_mode, "inherit")
        self.assertEqual(config.camera_playback_enabled, "always")

    def test_clear_loaded_pipeline_removes_only_selected_user_indexes(self):
        first = Mock()
        second = Mock()
        state = SimpleNamespace(
            pipelines_by_user={1: first, 2: second},
            pipeline_id_by_user={1: "pipeline-1", 2: "pipeline-2"},
        )
        controller = PipelineController(state, Mock(), Mock())

        controller._clear_loaded_pipeline(1)

        self.assertNotIn(1, state.pipelines_by_user)
        self.assertNotIn(1, state.pipeline_id_by_user)
        self.assertIs(state.pipelines_by_user[2], second)
        self.assertEqual(state.pipeline_id_by_user[2], "pipeline-2")

    def test_event_type_reads_objects_and_dicts_and_defaults_missing(self):
        self.assertEqual(
            PipelineController._event_type(SimpleNamespace(event_type="Edit_Channel")),
            "edit_channel",
        )
        self.assertEqual(
            PipelineController._event_type({"event_type": "CREATE_CHANNEL"}),
            "create_channel",
        )
        self.assertEqual(PipelineController._event_type(SimpleNamespace()), "")
        self.assertEqual(PipelineController._event_type({}), "")

    def test_invalidate_camera_roi_state_notifies_service_and_loaded_pipelines(self):
        camera_uuid = uuid.uuid4()
        notification = Mock()
        first_pipeline = Mock()
        second_pipeline = Mock()
        state = SimpleNamespace(
            pipelines_by_user={1: first_pipeline, 2: None, 3: second_pipeline}
        )
        controller = PipelineController(state, Mock(), Mock())

        controller._invalidate_camera_roi_state(camera_uuid, notification)

        notification.invalidate_camera_roi_state.assert_called_once_with(str(camera_uuid))
        first_pipeline.invalidate_camera_roi_state.assert_called_once_with(str(camera_uuid))
        second_pipeline.invalidate_camera_roi_state.assert_called_once_with(str(camera_uuid))

    def test_invalidate_camera_roi_state_is_best_effort(self):
        camera_uuid = uuid.uuid4()
        notification = Mock()
        notification.invalidate_camera_roi_state.side_effect = RuntimeError("gone")
        broken_pipeline = Mock()
        broken_pipeline.invalidate_camera_roi_state.side_effect = RuntimeError("gone")
        state = SimpleNamespace(pipelines_by_user={1: broken_pipeline})
        controller = PipelineController(state, Mock(), Mock())

        controller._invalidate_camera_roi_state(camera_uuid, notification)

        notification.invalidate_camera_roi_state.assert_called_once()
        broken_pipeline.invalidate_camera_roi_state.assert_called_once()


if __name__ == "__main__":
    unittest.main()
