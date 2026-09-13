"""Inventory contracts: roster parsing, credential redaction, HTTP shapes."""

import unittest
import uuid
from datetime import datetime

from pydantic import ValidationError

from application.services.inventory_service import (
    entries_from_roster,
    parse_timestamp,
    redact_source,
)
from core.database_orm import (
    INVENTORY_ADDED,
    INVENTORY_AVAILABLE,
    INVENTORY_REMOVED,
)
from core.schemas import (
    InventoryAddRequest,
    InventoryCameraOut,
    InventoryRefreshOut,
)


def _roster(**overrides):
    row = {
        "identity": "serial:AAA",
        "ip_address": "10.0.0.9",
        "mac_address": "aa:bb:cc",
        "serial_number": "AAA",
        "model": "DS-2CD",
        "firmware": "v5.7",
        "device_name": "Front Door",
        "source_url": "rtsp://10.0.0.9/stream",
        "camera_uuid": "edge-uuid-1",
        "is_present": True,
        "last_seen_at": "2026-01-02T00:00:00Z",
    }
    row.update(overrides)
    return {"roster": [row]}


class RosterParsingTests(unittest.TestCase):
    """Every field the device sends is captured, and nothing else is."""

    def test_all_reported_fields_are_kept(self):
        entry = entries_from_roster(_roster())[0]
        self.assertEqual(entry.discovery_identity, "serial:AAA")
        self.assertEqual(entry.mac_address, "aa:bb:cc")
        self.assertEqual(entry.firmware, "v5.7")
        self.assertEqual(entry.device_name, "Front Door")
        self.assertEqual(entry.edge_camera_uuid, "edge-uuid-1")

    def test_bare_list_is_accepted(self):
        self.assertEqual(len(entries_from_roster(_roster()["roster"])), 1)

    def test_row_without_identity_is_skipped(self):
        self.assertEqual(entries_from_roster({"roster": [{"ip_address": "1.2.3.4"}]}), [])

    def test_non_dict_rows_are_skipped(self):
        self.assertEqual(entries_from_roster({"roster": ["nope", 3, None]}), [])

    def test_missing_roster_is_empty(self):
        for payload in ({}, None, "garbage", 7):
            with self.subTest(payload=payload):
                self.assertEqual(entries_from_roster(payload), [])

    def test_absent_camera_is_kept_and_flagged(self):
        entry = entries_from_roster(_roster(is_present=False))[0]
        self.assertFalse(entry.is_present)

    def test_cloud_owned_fields_are_never_read(self):
        """A device must not be able to assign a site, org or ROI."""
        entry = entries_from_roster(
            _roster(site_uuid=str(uuid.uuid4()), org_id=99, roi={"p": 1}, camera_code="x")
        )[0]
        for forbidden in ("site_uuid", "org_id", "roi", "camera_code", "state"):
            self.assertFalse(hasattr(entry, forbidden))

    def test_timestamp_is_parsed(self):
        self.assertIsInstance(entries_from_roster(_roster())[0].last_seen_at, datetime)

    def test_bad_timestamp_does_not_drop_the_camera(self):
        entry = entries_from_roster(_roster(last_seen_at="not a date"))[0]
        self.assertEqual(entry.discovery_identity, "serial:AAA")
        self.assertIsNone(entry.last_seen_at)


class TimestampTests(unittest.TestCase):
    def test_zulu_is_parsed(self):
        self.assertIsInstance(parse_timestamp("2026-01-01T00:00:00Z"), datetime)

    def test_datetime_passes_through(self):
        now = datetime.now()
        self.assertEqual(parse_timestamp(now), now)

    def test_unparseable_is_none(self):
        for bad in ("", None, "nope", []):
            with self.subTest(bad=bad):
                self.assertIsNone(parse_timestamp(bad))


class RedactionTests(unittest.TestCase):
    """A camera list must never hand out a password."""

    def test_credentials_are_stripped(self):
        self.assertEqual(
            redact_source("rtsp://admin:hunter2@10.0.0.9/stream"),
            "rtsp://10.0.0.9/stream",
        )

    def test_port_is_preserved(self):
        self.assertEqual(
            redact_source("rtsp://user:pw@10.0.0.9:554/s"), "rtsp://10.0.0.9:554/s"
        )

    def test_url_without_credentials_is_unchanged(self):
        self.assertEqual(redact_source("rtsp://10.0.0.9/s"), "rtsp://10.0.0.9/s")

    def test_empty_is_none(self):
        self.assertIsNone(redact_source(None))
        self.assertIsNone(redact_source(""))


class StateTests(unittest.TestCase):
    def test_states_are_distinct(self):
        self.assertEqual(
            len({INVENTORY_AVAILABLE, INVENTORY_ADDED, INVENTORY_REMOVED}), 3
        )


class SchemaTests(unittest.TestCase):
    def test_add_requires_at_least_one_identity(self):
        with self.assertRaises(ValidationError):
            InventoryAddRequest(device_uuid=uuid.uuid4(), discovery_identities=[])

    def test_add_rejects_unknown_fields(self):
        with self.assertRaises(ValidationError):
            InventoryAddRequest(
                device_uuid=uuid.uuid4(),
                discovery_identities=["serial:A"],
                site_uuid=uuid.uuid4(),
            )

    def test_add_is_bounded(self):
        with self.assertRaises(ValidationError):
            InventoryAddRequest(
                device_uuid=uuid.uuid4(),
                discovery_identities=["serial:%d" % i for i in range(201)],
            )

    def test_out_never_exposes_raw_source_url(self):
        self.assertNotIn("source_url", InventoryCameraOut.model_fields)

    def test_out_defaults_to_unlinked(self):
        out = InventoryCameraOut(
            discovery_identity="serial:A", device_uuid=uuid.uuid4(), state="available"
        )
        self.assertIsNone(out.site_uuid)
        self.assertIsNone(out.camera_uuid)

    def test_unreachable_refresh_is_not_an_error(self):
        out = InventoryRefreshOut(fetched=False, detail="unreachable")
        self.assertEqual((out.created, out.updated, out.marked_offline), (0, 0, 0))


def _load_cloud_main():
    """Import the cloud app by path.

    `import main` is ambiguous here: sibling tests prepend `Backend/tensort` to
    sys.path, so the name can resolve to the Jetson's Flask app.
    """
    import importlib.util
    import os

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py"
    )
    spec = importlib.util.spec_from_file_location("cloud_main_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RouteWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        main = _load_cloud_main()
        cls.routes = {
            (path, method)
            for r in main.app.routes
            for path in [getattr(r, "path", "")]
            for method in getattr(r, "methods", set()) or set()
        }

    def test_inventory_routes_exist(self):
        for path, method in (
            ("/api/devices/{device_uuid}/inventory", "GET"),
            ("/api/devices/{device_uuid}/inventory/refresh", "POST"),
            ("/api/sites/{site_uuid}/inventory", "GET"),
            ("/api/sites/{site_uuid}/inventory/add", "POST"),
        ):
            with self.subTest(path=path, method=method):
                self.assertIn((path, method), self.routes)

    def test_existing_routes_are_preserved(self):
        for path, method in (
            ("/api/sites/{site_uuid}/devices", "POST"),
            ("/api/devices/{device_uuid}/edge/reconcile", "POST"),
            ("/api/cameras/{camera_uuid}", "DELETE"),
            ("/api/cameras", "POST"),
        ):
            with self.subTest(path=path, method=method):
                self.assertIn((path, method), self.routes)


if __name__ == "__main__":
    unittest.main()
