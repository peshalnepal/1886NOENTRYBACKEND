import unittest
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

from routes._camera_serializers import (
    camera_out,
    camera_with_config_out,
    tri_state,
)


def _orm_camera(**overrides):
    """A Camera row as the read paths see it (``is_`` flag names)."""
    base = dict(
        camera_uuid=uuid.uuid4(),
        camera_code="cam-1",
        name="Front door",
        location="Lobby",
        site_uuid=uuid.uuid4(),
        device_uuid=uuid.uuid4(),
        source_url="rtsp://cam/stream",
        webrtc_url=None,
        is_enabled=True,
        is_detection_enabled=True,
        is_notification_enabled=False,
        notification_trigger_mode="roi_enter",
        camera_playback_enabled="always",
        use_site_schedule=False,
        roi={"points": [[0, 0]]},
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _dto_camera(**overrides):
    """A `CameraOut` as the create/edit paths see it (bare flag names)."""
    base = dict(
        camera_uuid=uuid.uuid4(),
        camera_code="cam-2",
        name="Back door",
        location=None,
        site_uuid=uuid.uuid4(),
        device_uuid=uuid.uuid4(),
        source_url="rtsp://cam2/stream",
        webrtc_url=None,
        enabled=True,
        detection_enabled=False,
        notification_enabled=True,
        notification_trigger_mode="inherit",
        camera_playback_enabled="inherit",
        use_site_schedule=True,
        roi=None,
        configuration={"fps": 5},
        timezone="Asia/Kathmandu",
        created_at=None,
        updated_at=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TriStateTests(unittest.TestCase):
    def test_missing_and_empty_values_fall_back_to_inherit(self):
        cam = SimpleNamespace(notification_trigger_mode=None)
        self.assertEqual(tri_state(cam, "notification_trigger_mode"), "inherit")
        self.assertEqual(tri_state(cam, "camera_playback_enabled"), "inherit")

    def test_set_value_is_preserved(self):
        cam = SimpleNamespace(camera_playback_enabled="never")
        self.assertEqual(tri_state(cam, "camera_playback_enabled"), "never")


class CameraOutTests(unittest.TestCase):
    def test_orm_fields_map_onto_the_list_contract(self):
        cam = _orm_camera()
        device_uuid = uuid.uuid4()

        out = camera_out(cam, device_uuid=device_uuid)

        self.assertEqual(out.camera_uuid, cam.camera_uuid)
        self.assertEqual(out.device_uuid, device_uuid)
        self.assertTrue(out.is_enabled)
        self.assertFalse(out.is_notification_enabled)
        self.assertEqual(out.notification_trigger_mode, "roi_enter")
        self.assertFalse(out.use_site_schedule)

    def test_webrtc_url_is_derived_from_the_camera_code(self):
        out = camera_out(_orm_camera(camera_code="cam-xyz"))
        self.assertIn("cam-xyz", str(out.webrtc_url))


class CameraWithConfigOutTests(unittest.TestCase):
    def test_dto_flag_names_are_read_through_their_unprefixed_spelling(self):
        cam = _dto_camera(enabled=True, detection_enabled=False, notification_enabled=True)

        out = camera_with_config_out(cam, webrtc_url="http://whep/cam-2")

        self.assertTrue(out.is_enabled)
        self.assertFalse(out.is_detection_enabled)
        self.assertTrue(out.is_notification_enabled)

    def test_orm_flag_names_are_read_through_their_prefixed_spelling(self):
        cam = _orm_camera(is_enabled=False, is_detection_enabled=True)

        out = camera_with_config_out(
            cam, webrtc_url=None, configuration={}, timezone_name=None
        )

        self.assertFalse(out.is_enabled)
        self.assertTrue(out.is_detection_enabled)

    def test_explicit_configuration_and_timezone_win_over_the_object(self):
        cam = _dto_camera(configuration={"fps": 5}, timezone="Asia/Kathmandu")

        out = camera_with_config_out(
            cam,
            webrtc_url=None,
            configuration={"fps": 30},
            timezone_name="UTC",
        )

        self.assertEqual(out.configuration, {"fps": 30})
        self.assertEqual(out.timezone, "UTC")

    def test_configuration_and_timezone_default_to_the_object(self):
        cam = _dto_camera(configuration={"fps": 5}, timezone="Asia/Kathmandu")

        out = camera_with_config_out(cam, webrtc_url=None)

        self.assertEqual(out.configuration, {"fps": 5})
        self.assertEqual(out.timezone, "Asia/Kathmandu")

    def test_missing_timestamps_fall_back_to_now(self):
        out = camera_with_config_out(
            _dto_camera(created_at=None, updated_at=None), webrtc_url=None
        )

        self.assertIsNotNone(out.created_at)
        self.assertIsNotNone(out.updated_at)

    def test_both_create_paths_produce_the_same_shape(self):
        """The site and camera create routes must not drift apart."""
        cam = _dto_camera()

        from_camera_route = camera_with_config_out(cam, webrtc_url="http://whep/cam-2")
        from_site_route = camera_with_config_out(cam, webrtc_url="http://whep/cam-2")

        self.assertEqual(
            from_camera_route.model_dump(exclude={"created_at", "updated_at"}),
            from_site_route.model_dump(exclude={"created_at", "updated_at"}),
        )


if __name__ == "__main__":
    unittest.main()
