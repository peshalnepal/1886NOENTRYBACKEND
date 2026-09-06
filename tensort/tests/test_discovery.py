# tests/test_discovery.py  (Python 3.6+, stdlib only)
"""
Tests for discovery.py / service.py / routes/discovery_routes.py.

Runs entirely against a fake Hikvision camera (a local HTTP server that speaks
RFC 2617 digest auth) plus in-memory stand-ins for the repository and the
pipeline runtime. No Jetson, no camera, no database.

Run:  python3 tests/test_discovery.py     (from Backend/tensort)
"""

import hashlib
import os
import re
import sys
import threading
import time
import types
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# service.py imports DiscoveryRepository from `repositories`, which drags in the
# async DB stack. The tests inject their own repository, so stub the package to
# keep this suite runnable without a database driver installed.
if "repositories" not in sys.modules:
    _stub = types.ModuleType("repositories")
    _stub.DiscoveryRepository = object
    sys.modules["repositories"] = _stub

import discovery  # noqa: E402
import service  # noqa: E402
from service import DiscoveryService  # noqa: E402


# ---------------------------------------------------------------------------
# A fake Hikvision camera
# ---------------------------------------------------------------------------

USER, PASSWD = "admin", "s3cr3t"
REALM, NONCE = "IP Camera(C1234)", "abc123nonce"

DEVICE_INFO = b"""<?xml version="1.0" encoding="UTF-8"?>
<DeviceInfo xmlns="http://www.hikvision.com/ver20/XMLSchema">
<deviceName>Front Gate</deviceName>
<model>DS-2CD2143G0-I</model>
<serialNumber>DS-2CD2143G0-I20210101AAWR123</serialNumber>
<firmwareVersion>V5.6.3</firmwareVersion>
<macAddress>44:47:cc:11:22:33</macAddress>
</DeviceInfo>"""


def _md5(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


class _CameraHandler(BaseHTTPRequestHandler):
    mode = "digest"

    def log_message(self, *args):
        pass  # keep test output clean

    def _challenge(self):
        self.send_response(401)
        self.send_header(
            "WWW-Authenticate",
            'Digest qop="auth", realm="%s", nonce="%s"' % (REALM, NONCE),
        )
        self.end_headers()

    def do_GET(self):
        if self.path != discovery.ISAPI_PATH:
            self.send_response(404)
            self.end_headers()
            return

        if self.mode == "html":  # a router admin page on port 80
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"<html>router</html>")
            return
        if self.mode == "no_serial":  # DeviceInfo without a serialNumber
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"<DeviceInfo><model>X</model></DeviceInfo>")
            return

        auth = self.headers.get("Authorization") or ""
        if not auth.lower().startswith("digest "):
            self._challenge()
            return

        p = {k: (a or b) for k, a, b in
             re.findall(r'(\w+)=(?:"([^"]*)"|([^,\s]+))', auth)}
        ha1 = _md5("%s:%s:%s" % (USER, REALM, PASSWD))
        ha2 = _md5("GET:%s" % p.get("uri", ""))
        expected = _md5(":".join([ha1, p.get("nonce", ""), p.get("nc", ""),
                                  p.get("cnonce", ""), p.get("qop", ""), ha2]))
        if p.get("username") != USER or p.get("response") != expected:
            self._challenge()
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/xml")
        self.send_header("Content-Length", str(len(DEVICE_INFO)))
        self.end_headers()
        self.wfile.write(DEVICE_INFO)


def start_camera(mode="digest"):
    """Serve a fake camera on an ephemeral port. Returns (stop_fn, port)."""
    handler = type("Handler", (_CameraHandler,), {"mode": mode})
    srv = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    def stop():
        srv.shutdown()       # stop the serve loop
        srv.server_close()   # and release the listening socket

    return stop, srv.server_address[1]


# ---------------------------------------------------------------------------
# In-memory stand-ins
# ---------------------------------------------------------------------------

class FakeRepo(object):
    """Same contract as DiscoveryRepository, backed by a dict."""

    def __init__(self):
        self.rows = {}

    def list_all(self):
        return [dict(r) for r in self.rows.values()]

    def get_by_identity(self, identity):
        row = self.rows.get(identity)
        return dict(row) if row else None

    def mark_seen(self, device, source_url=None):
        identity = device.identity()
        row = self.rows.get(identity)
        is_new = row is None
        recovered = ip_changed = False

        if row is None:
            row = {"identity": identity, "camera_uuid": None}
            self.rows[identity] = row
        else:
            recovered = not row.get("is_present")
            ip_changed = bool(row.get("ip_address")) and row["ip_address"] != device.ip

        row.update(ip_address=device.ip, serial_number=device.serial_number,
                   source_url=source_url, is_present=True,
                   consecutive_misses=0, alerted=False)
        return {"identity": identity, "is_new": is_new, "recovered": recovered,
                "ip_changed": ip_changed, "row": dict(row)}

    def mark_missing(self, present_identities, miss_threshold=2):
        present = set(present_identities or ())
        newly_missing = []
        for identity, row in self.rows.items():
            if identity in present:
                continue
            row["consecutive_misses"] = row.get("consecutive_misses", 0) + 1
            if row["consecutive_misses"] < miss_threshold:
                continue
            row["is_present"] = False
            if not row.get("alerted"):
                row["alerted"] = True
                newly_missing.append(dict(row))
        return newly_missing

    def attach_camera_uuid(self, identity, camera_uuid, source_url=None):
        row = self.rows.get(identity)
        if row is None:
            return False
        row["camera_uuid"] = camera_uuid
        if source_url:
            row["source_url"] = source_url
        return True

    def forget(self, identity):
        return self.rows.pop(identity, None) is not None


class FakeRuntime(object):
    """Mimics the PipelineRuntime surface the service touches."""

    def __init__(self):
        self.cameras = {}
        self.pipeline = None
        self.loop = None
        self.discovery = None

    def list_cameras(self):
        return [{"camera_uuid": uid, "source_url": c["source_url"], "config": c["config"]}
                for uid, c in self.cameras.items()]

    def add_camera(self, source_url, cfg_patch):
        uid = cfg_patch["camera_uuid"]
        self.cameras[uid] = {"source_url": source_url, "config": dict(cfg_patch)}
        return {"camera_uuid": uid, "config": dict(cfg_patch)}

    def patch_camera(self, camera_uuid, patch):
        cam = self.cameras[camera_uuid]
        if "source_url" in patch:
            cam["source_url"] = patch["source_url"]
        cam["config"].update(patch)
        return {"camera_uuid": camera_uuid}


def make_config(port, password=PASSWD):
    cfg = discovery.DiscoveryConfig()
    cfg.username, cfg.password = USER, password
    cfg.http_port, cfg.http_timeout_s = port, 3.0
    cfg.wsd_timeout_s = 0.3
    cfg.subnets = ["127.0.0.1/32"]
    return cfg


# ---------------------------------------------------------------------------
# discovery.py
# ---------------------------------------------------------------------------

class TestConfig(unittest.TestCase):
    def test_rtsp_url_encodes_credentials(self):
        cfg = discovery.DiscoveryConfig()
        cfg.username, cfg.password = "admin", "p@ss/word"
        self.assertEqual(
            cfg.rtsp_url_for("192.168.1.50"),
            "rtsp://admin:p%40ss%2Fword@192.168.1.50:554/Streaming/Channels/102",
        )

    def test_rtsp_url_channel_encoding(self):
        cfg = discovery.DiscoveryConfig()
        cfg.username = ""
        cfg.rtsp_channel, cfg.rtsp_stream = 2, 1
        # channel*100 + stream, not string concatenation
        self.assertTrue(cfg.rtsp_url_for("10.0.0.1").endswith("/Streaming/Channels/201"))


class TestIdentity(unittest.TestCase):
    def test_prefers_serial_then_mac_then_ip(self):
        make = lambda serial, mac: discovery.HikDevice(  # noqa: E731
            ip="1.2.3.4", serial_number=serial, model=None,
            firmware=None, device_name=None, mac=mac)
        self.assertEqual(make("S1", "AA:BB").identity(), "serial:S1")
        self.assertEqual(make(None, "AA:BB").identity(), "mac:aa:bb")
        self.assertEqual(make(None, None).identity(), "ip:1.2.3.4")

    def test_to_dict_includes_identity(self):
        dev = discovery.HikDevice("1.2.3.4", "S1", "M", "F", "N", "AA:BB")
        self.assertEqual(dev.to_dict()["identity"], "serial:S1")
        self.assertEqual(dev.to_dict()["model"], "M")


class TestCIDR(unittest.TestCase):
    def test_skips_network_and_broadcast(self):
        self.assertEqual(discovery._hosts_in_cidr("192.168.1.0/30"),
                         ["192.168.1.1", "192.168.1.2"])

    def test_slash_32_is_the_single_host(self):
        self.assertEqual(discovery._hosts_in_cidr("10.0.0.7/32"), ["10.0.0.7"])

    def test_slash_24_has_254_hosts(self):
        self.assertEqual(len(discovery._hosts_in_cidr("192.168.1.0/24")), 254)

    def test_rejects_malformed_and_too_wide(self):
        self.assertEqual(discovery._hosts_in_cidr("nonsense"), [])
        self.assertEqual(discovery._hosts_in_cidr("10.0.0.0/8"), [])
        self.assertEqual(discovery._hosts_in_cidr(""), [])

    def test_default_subnet_from_host(self):
        self.assertEqual(discovery._default_subnet_from_host("192.168.5.77"),
                         ["192.168.5.0/24"])
        self.assertEqual(discovery._default_subnet_from_host(None), [])


class TestProbe(unittest.TestCase):
    """ISAPI identification is the authoritative vendor + credential check."""

    @classmethod
    def setUpClass(cls):
        cls.stop, cls.port = start_camera("digest")

    @classmethod
    def tearDownClass(cls):
        cls.stop()

    def test_digest_auth_identifies_camera(self):
        dev = discovery.probe_hikvision("127.0.0.1", make_config(self.port))
        self.assertIsNotNone(dev)
        self.assertEqual(dev.serial_number, "DS-2CD2143G0-I20210101AAWR123")
        self.assertEqual(dev.model, "DS-2CD2143G0-I")
        self.assertEqual(dev.firmware, "V5.6.3")
        self.assertEqual(dev.device_name, "Front Gate")

    def test_wrong_password_is_not_a_camera(self):
        # A camera we cannot authenticate to yields an RTSP URL that connects
        # and never decodes, so it must not be adopted.
        self.assertIsNone(
            discovery.probe_hikvision("127.0.0.1", make_config(self.port, "wrong")))

    def test_generic_web_server_rejected(self):
        stop, port = start_camera("html")
        self.addCleanup(stop)
        self.assertIsNone(discovery.probe_hikvision("127.0.0.1", make_config(port)))

    def test_device_info_without_serial_rejected(self):
        stop, port = start_camera("no_serial")
        self.addCleanup(stop)
        self.assertIsNone(discovery.probe_hikvision("127.0.0.1", make_config(port)))

    def test_closed_port_returns_none(self):
        self.assertIsNone(discovery.probe_hikvision("127.0.0.1", make_config(1)))


class TestScanNetwork(unittest.TestCase):
    def test_sweep_finds_and_dedupes(self):
        stop, port = start_camera("digest")
        self.addCleanup(stop)
        cfg = make_config(port)
        cfg.subnets = ["127.0.0.0/29"]  # 6 hosts, only .1 answers
        found = discovery.scan_network(cfg)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].ip, "127.0.0.1")


# ---------------------------------------------------------------------------
# service.py
# ---------------------------------------------------------------------------

class TestSweepLifecycle(unittest.TestCase):
    def setUp(self):
        stop, port = start_camera("digest")
        self.addCleanup(stop)
        self.cfg = make_config(port)
        self.runtime = FakeRuntime()
        self.repo = FakeRepo()
        self.svc = DiscoveryService(self.runtime, repository=self.repo, config=self.cfg)
        self._real_scan = service.scan_network
        self.addCleanup(setattr, service, "scan_network", self._real_scan)

    def test_adopts_new_camera_with_provenance(self):
        report = self.svc.run_once()
        self.assertEqual(len(report["new_cameras"]), 1)
        entry = report["new_cameras"][0]
        self.assertTrue(entry["adopted"])
        self.assertTrue(entry["camera_uuid"])

        self.assertEqual(len(self.runtime.cameras), 1)
        cfg = list(self.runtime.cameras.values())[0]["config"]
        self.assertTrue(cfg["discovered"])
        self.assertEqual(cfg["discovery_serial"], "DS-2CD2143G0-I20210101AAWR123")

    def test_second_sweep_does_not_re_adopt(self):
        self.svc.run_once()
        report = self.svc.run_once()
        self.assertEqual(report["new_cameras"], [])
        self.assertEqual(len(self.runtime.cameras), 1)

    def test_wiped_roster_links_instead_of_duplicating(self):
        """One physical camera must never end up with two decoding channels."""
        self.svc.run_once()
        self.repo.rows.clear()
        entry = self.svc.run_once()["new_cameras"][0]
        self.assertFalse(entry["adopted"])
        self.assertTrue(entry["already_present"])
        self.assertEqual(len(self.runtime.cameras), 1)

    def test_missing_respects_grace_window_and_alerts_once(self):
        self.svc.run_once()
        service.scan_network = lambda cfg: []  # camera unplugged

        self.assertEqual(self.svc.run_once()["missing_cameras"], [])   # grace
        self.assertEqual(len(self.svc.run_once()["missing_cameras"]), 1)  # alert
        self.assertEqual(self.svc.run_once()["missing_cameras"], [])   # not again

    def test_recovery_after_missing(self):
        self.svc.run_once()
        service.scan_network = lambda cfg: []
        self.svc.run_once()
        self.svc.run_once()
        service.scan_network = self._real_scan
        self.assertEqual(len(self.svc.run_once()["recovered_cameras"]), 1)

    def test_ip_change_repoints_running_channel(self):
        self.svc.run_once()
        uid = list(self.runtime.cameras)[0]
        moved = discovery.scan_network(self.cfg)[0]._replace(ip="127.0.0.9")
        service.scan_network = lambda cfg: [moved]

        report = self.svc.run_once()
        self.assertEqual(len(report["ip_changes"]), 1)
        self.assertEqual(report["errors"], [])
        self.assertIn("127.0.0.9", self.runtime.cameras[uid]["source_url"])
        self.assertEqual(self.runtime.cameras[uid]["config"]["discovery_ip"], "127.0.0.9")

    def test_auto_add_false_reports_without_provisioning(self):
        self.svc.auto_add = False
        report = self.svc.run_once()
        self.assertEqual(len(report["new_cameras"]), 1)
        self.assertFalse(report["new_cameras"][0]["adopted"])
        self.assertEqual(self.runtime.cameras, {})

    def test_report_shape(self):
        report = self.svc.run_once()
        for key in ("type", "generated_at", "duration_s", "scanned", "present",
                    "new_cameras", "missing_cameras", "recovered_cameras",
                    "ip_changes", "roster", "errors"):
            self.assertIn(key, report)
        self.assertEqual(report["type"], "discovery_report")
        self.assertEqual(report["errors"], [])

    def test_concurrent_sweeps_coalesce(self):
        """Two sweeps at once would both see a camera as new and adopt twice."""
        self.svc.run_once()  # seed a report for the coalesced callers

        calls = []

        def slow_scan(cfg):
            calls.append(1)
            time.sleep(0.4)
            return []

        service.scan_network = slow_scan
        results = []
        threads = [threading.Thread(target=lambda: results.append(self.svc.run_once()))
                   for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(calls), 1)
        self.assertEqual(sum(1 for r in results if r.get("coalesced")), 3)
        self.assertTrue(all(r["type"] == "discovery_report" for r in results))


class TestStatus(unittest.TestCase):
    def test_status_before_any_sweep(self):
        cfg = make_config(1)
        svc = DiscoveryService(FakeRuntime(), repository=FakeRepo(), config=cfg)
        status = svc.status()
        self.assertFalse(status["running"])
        self.assertIsNone(status["last_sweep_at"])
        self.assertEqual(status["present_count"], 0)
        self.assertEqual(status["missing_count"], 0)

    def test_disabled_config_returns_empty_report(self):
        cfg = make_config(1)
        cfg.enabled = False
        svc = DiscoveryService(FakeRuntime(), repository=FakeRepo(), config=cfg)
        self.assertFalse(svc.start())
        self.assertEqual(svc.run_once()["reason"], "discovery_disabled")


# ---------------------------------------------------------------------------
# routes/discovery_routes.py
# ---------------------------------------------------------------------------

try:
    from flask import Flask
    HAS_FLASK = True
except ImportError:  # pragma: no cover
    HAS_FLASK = False


@unittest.skipUnless(HAS_FLASK, "Flask is not installed")
class TestRoutes(unittest.TestCase):
    def setUp(self):
        import routes.discovery_routes as dr

        stop, port = start_camera("digest")
        self.addCleanup(stop)

        self.runtime = FakeRuntime()
        self.runtime.discovery = DiscoveryService(
            self.runtime, repository=FakeRepo(), config=make_config(port))

        self.dr = dr
        self._real_get_runtime = dr.get_runtime
        dr.get_runtime = lambda: self.runtime
        self.addCleanup(setattr, dr, "get_runtime", self._real_get_runtime)

        app = Flask(__name__)
        app.register_blueprint(dr.bp)
        self.app = app
        self.client = app.test_client()

    def test_every_route_has_an_api_alias(self):
        rules = [r.rule for r in self.app.url_map.iter_rules() if "static" not in r.rule]
        bare = sorted(r for r in rules if not r.startswith("/api"))
        aliased = sorted(r[len("/api"):] for r in rules if r.startswith("/api"))
        self.assertEqual(bare, aliased)
        self.assertEqual(len(rules), 12)

    def test_report_404_before_first_sweep(self):
        self.assertEqual(self.client.get("/discovery/report").status_code, 404)

    def test_scan_then_list_and_report(self):
        body = self.client.post("/discovery/scan").get_json()
        self.assertEqual(len(body["new_cameras"]), 1)

        listing = self.client.get("/discovery").get_json()
        self.assertEqual(len(listing["discovered"]), 1)
        self.assertEqual(len(listing["present"]), 1)
        self.assertEqual(listing["missing"], [])

        self.assertEqual(self.client.get("/discovery/report").status_code, 200)
        self.assertTrue(self.client.get("/api/discovery/status").get_json()["enabled"])

    def test_forget_is_idempotent(self):
        self.client.post("/discovery/scan")
        identity = self.client.get("/discovery").get_json()["discovered"][0]["identity"]

        first = self.client.delete("/discovery/" + identity).get_json()
        self.assertTrue(first["forgotten"])
        self.assertFalse(self.client.delete("/discovery/" + identity).get_json()["existed"])

    def test_sync_returns_cameras_and_discovery(self):
        body = self.client.post("/sync").get_json()
        self.assertEqual(len(body["cameras"]), 1)
        self.assertIsNotNone(body["discovery"])
        self.assertNotIn("warnings", body)

    def test_sync_survives_a_failing_sweep(self):
        real = service.scan_network
        service.scan_network = lambda cfg: (_ for _ in ()).throw(RuntimeError("boom"))
        self.addCleanup(setattr, service, "scan_network", real)

        response = self.client.post("/sync")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertIn("cameras", body)
        self.assertTrue(any("boom" in w for w in body["warnings"]))

    def test_scan_route_500s_on_a_failing_sweep(self):
        real = service.scan_network
        service.scan_network = lambda cfg: (_ for _ in ()).throw(RuntimeError("boom"))
        self.addCleanup(setattr, service, "scan_network", real)
        self.assertEqual(self.client.post("/discovery/scan").status_code, 500)

    def test_503_when_service_never_started(self):
        self.runtime.discovery = None
        for path in ("/discovery", "/discovery/status", "/discovery/report"):
            self.assertEqual(self.client.get(path).status_code, 503, path)
        self.assertEqual(self.client.post("/discovery/scan").status_code, 503)

    def test_sync_degrades_without_discovery(self):
        """The cloud still needs the camera list even with discovery down."""
        self.runtime.discovery = None
        response = self.client.post("/sync")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertIsNone(body["discovery"])
        self.assertIsNone(body["status"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
