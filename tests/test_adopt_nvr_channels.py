"""Adoption preserves edge UUIDs even when camera streams share a host."""

import asyncio
import unittest
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from application.services.manager.controllers.adopt import CameraAdopter


class _FakeChannelConfig:
    def __init__(self, configuration: Optional[Dict[str, Any]] = None):
        self.configuration = dict(configuration or {})


class _FakeCamera:
    def __init__(self, *, camera_uuid, source_url, camera_code="cam-abc123", config=None):
        self.camera_uuid = camera_uuid
        self.source_url = source_url
        self.camera_code = camera_code
        self.channel_configuration = _FakeChannelConfig(config)


class _FakeSession:
    async def commit(self):
        pass


class _FakeChannelRepo:
    def __init__(self, cameras: List[_FakeCamera]):
        self._by_uuid = {str(c.camera_uuid): c for c in cameras}
        self.cameras = cameras

    async def list_cameras(self, _db, **_kw):
        return list(self.cameras)

    async def get_camera_full(self, _db, *, camera_uuid):
        cam = self._by_uuid.get(str(camera_uuid))
        if cam is None:
            return None
        return cam, cam.channel_configuration, None


class _FakeWebRTC:
    def __init__(self):
        self.updates: List[Dict[str, str]] = []

    async def update_stream(self, *, stream_key: str, source_url: str) -> None:
        self.updates.append({"stream_key": stream_key, "source_url": source_url})


class _FakeState:
    def __init__(self, channel_repo, webrtc, session):
        self.channel_repo = channel_repo
        self.webrtc = webrtc
        self._session = session

    def session_factory(self):
        @asynccontextmanager
        async def _cm():
            yield self._session

        return _cm()


NVR_HOST = "initialsecurity.dvrlists.com"


def _channel_row(channel: int) -> Dict[str, Any]:
    """One roster entry as the edge reports it for a static NVR channel."""
    return {
        "camera_uuid": str(uuid.UUID(int=channel)),
        "identity": "ip:{}#{}".format(NVR_HOST, channel),
        "ip_address": NVR_HOST,
        "serial_number": None,
        "model": "NVR-Static-Channel",
        "device_name": "{} ch{}".format(NVR_HOST.split(".")[0], channel),
        # Every channel shares the host and differs only in the path.
        "source_url": "rtsp://admin:pw@{}:554/Streaming/Channels/{}01".format(NVR_HOST, channel),
        "is_present": True,
    }


class _Harness:
    def __init__(self, *, cameras):
        self.session = _FakeSession()
        self.repo = _FakeChannelRepo(cameras)
        self.webrtc = _FakeWebRTC()
        self.state = _FakeState(self.repo, self.webrtc, self.session)
        self.created: List[Dict[str, Any]] = []

        async def _update_pipeline(*_a, **kw):
            # Stands in for the real channel-create path; record what adoption
            # asked to be created so the test can count distinct cameras.
            self.created.append(dict(kw))
            return uuid.uuid4()

        self.adopter = CameraAdopter(self.state, _update_pipeline)
        self.adopter._sites_for_device = self._sites  # type: ignore[assignment]
        self.adopter._create_camera = self._create  # type: ignore[assignment]
        self._site_uuid = uuid.uuid4()

        import application.services.manager.controllers.adopt as adopt_mod

        class _NoBlocks:
            async def blocked_identities(self, _db, **_kw):
                return set()

        self._orig_inv = adopt_mod.InventoryRepository
        adopt_mod.InventoryRepository = _NoBlocks
        self._adopt_mod = adopt_mod

    async def _sites(self, _db, *, device_uuid):
        return [self._site_uuid]

    async def _create(self, **kw):
        self.created.append(dict(kw))
        return uuid.UUID(kw["entry"]["camera_uuid"])

    def run(self, report, **kw):
        try:
            return asyncio.run(
                self.adopter.adopt_discovered_cameras(
                    device_uuid=uuid.uuid4(),
                    device_url="http://jetson.local:8000",
                    user_id=1,
                    discovery_report=report,
                    **kw,
                )
            )
        finally:
            self._adopt_mod.InventoryRepository = self._orig_inv


class NvrChannelAdoptionTests(unittest.TestCase):
    def test_unverified_and_over_capacity_rows_are_not_auto_adopted(self):
        rows = [_channel_row(c) for c in range(1, 4)]
        rows[0]["verification_state"] = "unverified"
        report = {"roster": rows, "capacity_rejected": [rows[1]["identity"]]}
        self.assertEqual(CameraAdopter._roster_entries(report), [rows[2]])

    def test_minipc_capacity_rejected_rows_are_not_auto_adopted(self):
        """The MiniPC lists whole roster rows (dicts), not bare identities."""
        rows = [_channel_row(c) for c in range(1, 3)]
        report = {"roster": rows, "capacity_rejected": [dict(rows[0])]}
        self.assertEqual(CameraAdopter._roster_entries(report), [rows[1]])

    def test_eight_channels_adopt_as_eight_cameras(self):
        """The reported bug: a first import collapsed 8 channels into 1 row."""
        report = {"roster": [_channel_row(c) for c in range(1, 9)]}
        h = _Harness(cameras=[])

        out = h.run(report)

        self.assertEqual(
            len(out["adopted"]), 8,
            "expected one camera per NVR channel, got {}".format(len(out["adopted"])),
        )
        self.assertEqual(len(set(out["adopted"])), 8, "adopted uuids must be distinct")
        self.assertEqual(out["linked"], [], "no channel should link to another channel")

    def test_channels_are_not_linked_to_an_existing_sibling_channel(self):
        """Channel 1 already in the DB must not absorb channels 2-8."""
        existing = _FakeCamera(
            camera_uuid=uuid.UUID(int=1),
            source_url="rtsp://admin:pw@{}:554/Streaming/Channels/101".format(NVR_HOST),
            config={"discovery_identity": "ip:{}#1".format(NVR_HOST)},
        )
        report = {"roster": [_channel_row(c) for c in range(1, 9)]}
        h = _Harness(cameras=[existing])

        out = h.run(report)

        # ch1 links to its own row by identity; ch2-ch8 are new.
        self.assertEqual(out["linked"], [str(existing.camera_uuid)])
        self.assertEqual(len(out["adopted"]), 7)

    def test_existing_row_is_not_repointed_to_a_sibling_channel(self):
        """The 'last channel wins' half: a link must not rewrite the URL."""
        existing = _FakeCamera(
            camera_uuid=uuid.UUID(int=1),
            source_url="rtsp://admin:pw@{}:554/Streaming/Channels/101".format(NVR_HOST),
            config={"discovery_identity": "ip:{}#1".format(NVR_HOST)},
        )
        report = {"roster": [_channel_row(c) for c in range(1, 9)]}
        h = _Harness(cameras=[existing])

        h.run(report)

        self.assertTrue(
            existing.source_url.endswith("/101"),
            "channel 1's row was repointed to {}".format(existing.source_url),
        )

    def test_plain_camera_links_by_uuid(self):
        handmade = _FakeCamera(
            camera_uuid=uuid.uuid4(),
            source_url="rtsp://admin:pw@10.0.0.50:554/Streaming/Channels/102",
            config={},  # added by hand: no identity
        )
        report = {
            "roster": [
                {
                    "camera_uuid": str(handmade.camera_uuid),
                    "identity": "serial:AAA",  # standalone: no "#channel"
                    "ip_address": "10.0.0.50",
                    "serial_number": "AAA",
                    "model": "DS-2CD",
                    "device_name": "Front Door",
                    "source_url": "rtsp://admin:pw@10.0.0.50:554/Streaming/Channels/102",
                    "is_present": True,
                }
            ]
        }
        h = _Harness(cameras=[handmade])

        out = h.run(report)

        self.assertEqual(out["linked"], [str(handmade.camera_uuid)])
        self.assertEqual(out["adopted"], [])

    def test_relay_streams_with_same_host_keep_distinct_uuids(self):
        rows = [_channel_row(c) for c in range(1, 4)]
        for n, row in enumerate(rows):
            row.update(identity=f"serial:CAM{n}", source_url=f"rtsp://minipc:8554/nvr-{n}")
        h = _Harness(cameras=[])
        out = h.run({"roster": rows})
        self.assertEqual(out["adopted"], [r["camera_uuid"] for r in rows])
        self.assertEqual(out["linked"], [])

    def test_identity_with_different_uuid_is_a_conflict(self):
        row = _channel_row(1)
        existing = _FakeCamera(camera_uuid=uuid.uuid4(), source_url=row["source_url"],
                               config={"discovery_identity": row["identity"]})
        h = _Harness(cameras=[existing])
        out = h.run({"roster": [row]})
        self.assertEqual(out["adopted"], [])
        self.assertEqual(out["linked"], [])
        self.assertIn("UUID conflict", out["errors"][0])
        self.assertEqual(h.created, [])

    def test_missing_or_invalid_uuid_cannot_create_camera(self):
        for value in (None, "", "bad-uuid"):
            row = dict(_channel_row(1), camera_uuid=value)
            h = _Harness(cameras=[])
            out = h.run({"roster": [row]})
            self.assertEqual(out["adopted"], [])
            self.assertIn("camera_uuid", out["errors"][0])
            self.assertEqual(h.created, [])


if __name__ == "__main__":
    unittest.main()
