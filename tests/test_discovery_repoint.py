"""Discovery repoint: a camera that moves address is followed everywhere.

The edge repoints its own pipeline the moment a sweep finds a known camera at a
new IP, but that patch is local to the Jetson. These tests pin the cloud half:
the Camera row, the MediaMTX stream and the reported summary must all follow,
because a stale `source_url` in the DB is silently pushed back down over the
edge's correct one by the next reconcile.
"""

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
    def __init__(self):
        self.commits = 0

    async def commit(self):
        self.commits += 1

    async def execute(self, *_a, **_kw):  # pragma: no cover - unused here
        raise AssertionError("unexpected query")


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
    def __init__(self, fail: bool = False):
        self.updates: List[Dict[str, str]] = []
        self._fail = fail

    async def update_stream(self, *, stream_key: str, source_url: str) -> None:
        if self._fail:
            raise RuntimeError("gateway down")
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


def _report(**overrides) -> Dict[str, Any]:
    row = {
        "identity": "serial:AAA",
        "ip_address": "10.0.0.50",
        "serial_number": "AAA",
        "model": "DS-2CD",
        "device_name": "Front Door",
        "source_url": "rtsp://admin:pw@10.0.0.50:554/Streaming/Channels/102",
        "camera_uuid": str(uuid.UUID(int=1)),
        "is_present": True,
    }
    row.update(overrides)
    return {"roster": [row]}


class _AdopterHarness:
    """A CameraAdopter wired to fakes, with adoption's DB preamble stubbed out.

    `_sites_for_device` and `blocked_identities` are the only things adoption
    needs a real session for; stubbing them keeps these tests on the repoint
    branch rather than on site resolution, which has its own failure modes.
    """

    def __init__(self, *, cameras, site_uuid=None, webrtc_fail=False):
        self.session = _FakeSession()
        self.repo = _FakeChannelRepo(cameras)
        self.webrtc = _FakeWebRTC(fail=webrtc_fail)
        self.state = _FakeState(self.repo, self.webrtc, self.session)
        self.created: List[Dict[str, Any]] = []

        async def _update_pipeline(*_a, **_kw):
            raise AssertionError("a registered camera must never be re-created")

        self.adopter = CameraAdopter(self.state, _update_pipeline)
        self.adopter._sites_for_device = self._sites  # type: ignore[assignment]
        self._site_uuid = site_uuid or uuid.uuid4()

        import application.services.manager.controllers.adopt as adopt_mod

        class _NoBlocks:
            async def blocked_identities(self, _db, **_kw):
                return set()

        self._orig_inv = adopt_mod.InventoryRepository
        adopt_mod.InventoryRepository = _NoBlocks
        self._adopt_mod = adopt_mod

    async def _sites(self, _db, *, device_uuid):
        return [self._site_uuid]

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


class RepointTests(unittest.TestCase):
    def test_moved_camera_updates_row_stream_and_summary(self):
        cam_uuid = uuid.UUID(int=1)
        cam = _FakeCamera(
            camera_uuid=cam_uuid,
            source_url="rtsp://admin:pw@10.0.0.9:554/Streaming/Channels/102",
            config={"discovery_identity": "serial:AAA"},
        )
        h = _AdopterHarness(cameras=[cam])

        out = h.run(_report())

        new_url = "rtsp://admin:pw@10.0.0.50:554/Streaming/Channels/102"
        self.assertEqual(out["linked"], [str(cam_uuid)])
        self.assertEqual(out["adopted"], [])
        self.assertEqual(cam.source_url, new_url)
        self.assertEqual(
            h.webrtc.updates, [{"stream_key": "cam-abc123", "source_url": new_url}]
        )
        self.assertEqual(len(out["repointed"]), 1)
        self.assertEqual(out["repointed"][0]["camera_uuid"], str(cam_uuid))
        self.assertEqual(out["repointed"][0]["source_url"], new_url)

    def test_same_address_writes_nothing(self):
        """The steady state: every sweep sees the camera where it already is."""
        url = "rtsp://admin:pw@10.0.0.50:554/Streaming/Channels/102"
        cam = _FakeCamera(
            camera_uuid=uuid.UUID(int=1),
            source_url=url,
            config={
                "discovered": True,
                "discovery_identity": "serial:AAA",
                "discovery_ip": "10.0.0.50",
                "discovery_mac": None,
                "discovery_model": "DS-2CD",
                "discovery_serial": "AAA",
                "discovery_name": "Front Door",
            },
        )
        h = _AdopterHarness(cameras=[cam])

        out = h.run(_report())

        self.assertEqual(out["repointed"], [])
        self.assertEqual(h.webrtc.updates, [])
        self.assertEqual(h.session.commits, 0, "an unchanged camera must not write")

    def test_credential_or_stream_path_change_alone_is_not_a_repoint(self):
        """Only the host decides. The edge rebuilds the URL from its own config
        every sweep, so comparing whole strings would repoint forever."""
        cam = _FakeCamera(
            camera_uuid=uuid.UUID(int=1),
            source_url="rtsp://admin:pw@10.0.0.50:554/Streaming/Channels/101",
            config={
                "discovered": True,
                "discovery_identity": "serial:AAA",
                "discovery_ip": "10.0.0.50",
                "discovery_model": "DS-2CD",
                "discovery_serial": "AAA",
                "discovery_name": "Front Door",
            },
        )
        h = _AdopterHarness(cameras=[cam])

        out = h.run(_report())

        self.assertEqual(out["repointed"], [])
        self.assertEqual(h.webrtc.updates, [])
        self.assertTrue(cam.source_url.endswith("/101"), "cloud-side URL is preserved")

    def test_gateway_failure_keeps_the_db_repoint(self):
        """MediaMTX is best-effort: the DB is authoritative and the next
        reconcile pushes the new URL regardless."""
        cam = _FakeCamera(
            camera_uuid=uuid.UUID(int=1),
            source_url="rtsp://admin:pw@10.0.0.9:554/Streaming/Channels/102",
            config={"discovery_identity": "serial:AAA"},
        )
        h = _AdopterHarness(cameras=[cam], webrtc_fail=True)

        out = h.run(_report())

        self.assertEqual(len(out["repointed"]), 1)
        self.assertIn("10.0.0.50", cam.source_url)
        self.assertEqual(out["errors"], [])

    def test_dry_run_changes_nothing(self):
        cam = _FakeCamera(
            camera_uuid=uuid.UUID(int=1),
            source_url="rtsp://admin:pw@10.0.0.9:554/Streaming/Channels/102",
            config={"discovery_identity": "serial:AAA"},
        )
        h = _AdopterHarness(cameras=[cam])

        out = h.run(_report(), dry_run=True)

        self.assertEqual(out["repointed"], [])
        self.assertEqual(h.webrtc.updates, [])
        self.assertIn("10.0.0.9", cam.source_url)

    def test_provenance_is_backfilled_on_a_hand_added_camera(self):
        """Matched by host, not identity: the row predates discovery. It gets
        stamped so the next sweep matches on identity instead."""
        cam = _FakeCamera(
            camera_uuid=uuid.UUID(int=1),
            source_url="rtsp://admin:pw@10.0.0.50:554/Streaming/Channels/102",
            config={},
        )
        h = _AdopterHarness(cameras=[cam])

        out = h.run(_report())

        self.assertEqual(out["repointed"], [], "same host — not a move")
        self.assertEqual(cam.channel_configuration.configuration["discovery_identity"], "serial:AAA")
        self.assertTrue(cam.channel_configuration.configuration["discovered"])
        self.assertEqual(h.session.commits, 1)

    def test_camera_without_code_skips_the_stream_patch(self):
        """No camera_code means no MediaMTX path to patch; the row still moves."""
        cam = _FakeCamera(
            camera_uuid=uuid.UUID(int=1),
            source_url="rtsp://admin:pw@10.0.0.9:554/Streaming/Channels/102",
            camera_code=None,
            config={"discovery_identity": "serial:AAA"},
        )
        h = _AdopterHarness(cameras=[cam])

        out = h.run(_report())

        self.assertEqual(len(out["repointed"]), 1)
        self.assertIn("10.0.0.50", cam.source_url)
        self.assertEqual(h.webrtc.updates, [])

    def test_relay_port_change_alone_is_a_repoint(self):
        """A MiniPC relay keeps its host and path when NVR_PUBLIC_RTSP_BASE_URL
        moves to another port; only the port differs, and the stream must follow."""
        cam = _FakeCamera(
            camera_uuid=uuid.UUID(int=1),
            source_url="rtsp://203.0.113.7:8554/nvr-abc",
            config={"discovery_identity": "serial:AAA"},
        )
        h = _AdopterHarness(cameras=[cam])

        new_url = "rtsp://203.0.113.7:9554/nvr-abc"
        out = h.run(_report(source_url=new_url))

        self.assertEqual(len(out["repointed"]), 1)
        self.assertEqual(cam.source_url, new_url)
        self.assertEqual(h.webrtc.updates, [{"stream_key": "cam-abc123", "source_url": new_url}])

    def test_explicit_default_port_is_not_a_repoint(self):
        """`rtsp://h/x` and `rtsp://h:554/x` are the same endpoint."""
        cam = _FakeCamera(
            camera_uuid=uuid.UUID(int=1),
            source_url="rtsp://admin:pw@10.0.0.50/Streaming/Channels/102",
            config={"discovery_identity": "serial:AAA"},
        )
        h = _AdopterHarness(cameras=[cam])

        out = h.run(_report())

        self.assertEqual(out["repointed"], [])
        self.assertEqual(h.webrtc.updates, [])


if __name__ == "__main__":
    unittest.main()
