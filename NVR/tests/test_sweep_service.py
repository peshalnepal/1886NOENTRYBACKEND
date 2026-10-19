"""Sweep lifecycle over a real (temp-file) SQLite roster with scripted scans."""

import asyncio
import threading
import unittest
from unittest import mock

from sqlalchemy import func, inspect, select

from NVR.core.database_orm import Camera, NvrDevice
from NVR.tests.fakes import NVR_HOST, TowerHarness, direct, hikvision_env, nvr_channel, static_channel


class HarnessCase(unittest.IsolatedAsyncioTestCase):
    env = None
    devices = None

    async def asyncSetUp(self):
        self.tower = await TowerHarness(env=self.env, devices=self.devices).start()

    async def asyncTearDown(self):
        await self.tower.close()

    async def count(self, model):
        async with self.tower.database.session_factory() as db:
            return (await db.execute(select(func.count()).select_from(model))).scalar_one()


class MinipcModeTests(HarnessCase):
    devices = [direct("10.0.0.64", "CAM001"), direct("10.0.0.65", "CAM002")]

    async def test_direct_cameras_fill_the_roster_idempotently(self):
        first = await self.tower.run()
        self.assertEqual(first["nvr_mode"], "minipc")
        self.assertIsNone(first["nvr"])
        self.assertEqual(len(first["new_cameras"]), 2)
        self.assertTrue(all(row["nvr_id"] is None and row["source_kind"] == "direct"
                            and row["camera_uuid"] is None and row["jetson_state"] == "pending"
                            for row in first["roster"]))

        second = await self.tower.run()
        self.assertEqual(second["new_cameras"], [])
        self.assertEqual(len(second["present"]), 2)
        self.assertEqual(await self.count(Camera), 2)
        self.assertEqual(await self.count(NvrDevice), 0)

    async def test_roster_survives_a_restart_without_new_cameras(self):
        await self.tower.run()
        await self.tower.restart()
        report = await self.tower.run()
        self.assertEqual(report["new_cameras"], [])
        self.assertEqual(await self.count(Camera), 2)

    async def test_unexpected_recorder_is_reported_not_adopted(self):
        self.tower.scan.devices.append(nvr_channel(1, serial="X", host="10.0.0.99"))
        report = await self.tower.run()
        self.assertEqual(await self.count(Camera), 2)
        self.assertIn("Unexpected recorder at 10.0.0.99", report["errors"][0])

    async def test_scan_runs_on_the_discovery_thread(self):
        await self.tower.run()
        self.assertTrue(self.tower.scan.threads[0].startswith("nvr-discovery"))
        self.assertNotEqual(self.tower.scan.threads[0], threading.main_thread().name)

    async def test_forget(self):
        await self.tower.run()
        self.assertTrue(await self.tower.sweep.forget("serial:CAM001"))
        self.assertFalse(await self.tower.sweep.forget("serial:CAM001"))
        self.assertEqual([row["identity"] for row in await self.tower.sweep.roster()], ["serial:CAM002"])
        self.assertEqual(await self.count(Camera), 1)

    async def test_database_contains_only_cameras_and_nvr(self):
        async with self.tower.database.engine.connect() as connection:
            tables = await connection.run_sync(lambda sync: inspect(sync).get_table_names())
        self.assertEqual(set(tables), {"cameras", "nvr_devices"})


class HikvisionNvrOnlyTests(HarnessCase):
    env = hikvision_env()
    devices = [nvr_channel(1, serial="CAMA", name="Gate"), nvr_channel(2, serial="CAMB")]

    async def test_channels_are_cameras_behind_the_one_nvr(self):
        report = await self.tower.run()
        self.assertTrue(report["nvr"]["is_present"])
        nvr_id = report["nvr"]["id"]
        rows = self.tower.by_identity(report)
        self.assertEqual({(r["nvr_id"], r["channel_no"], r["source_kind"]) for r in rows.values()},
                         {(nvr_id, 1, "nvr"), (nvr_id, 2, "nvr")})
        self.assertEqual(rows["serial:CAMA#1"]["device_name"], "Gate")
        self.assertEqual(await self.count(NvrDevice), 1)


class HikvisionNvrPlusDirectTests(HarnessCase):
    env = hikvision_env()
    devices = [
        direct("10.0.0.64", "CAM001"),
        nvr_channel(1, serial="CAM001", name="Door"),  # the same camera, via the NVR
        nvr_channel(2, serial="CAM002"),
        direct("10.0.0.70", "CAM009"),  # direct only
    ]

    async def test_one_row_per_physical_camera_direct_route_wins(self):
        report = await self.tower.run()
        rows = self.tower.by_identity(report)
        self.assertEqual(set(rows), {"serial:CAM001", "serial:CAM002#2", "serial:CAM009"})
        door = rows["serial:CAM001"]
        self.assertEqual((door["source_kind"], door["nvr_id"], door["channel_no"]), ("direct", None, None))
        self.assertNotIn("sources", door)
        # Healthy direct route: the NVR route for that camera was not probed.
        self.assertNotIn("rtsp://recorder:nvr-pass@10.0.0.50:554/Streaming/Channels/102", self.tower.verifier.urls)
        self.assertEqual(await self.count(Camera), 3)

    async def test_failover_to_the_nvr_channel_and_back(self):
        await self.tower.run()
        self.tower.verifier.fail_hosts("10.0.0.64")
        report = await self.tower.run()
        change = next(c for c in report["ip_changes"] if c["identity"] == "serial:CAM001")
        self.assertEqual((change["reason"], change["source_kind"], change["channel_no"]), ("source_switch", "nvr", 1))
        door = self.tower.by_identity(report)["serial:CAM001"]
        self.assertEqual((door["source_kind"], door["nvr_id"], door["channel_no"]),
                         ("nvr", report["nvr"]["id"], 1))
        self.assertEqual(report["missing_cameras"], [])

        self.tower.verifier.ok_all()
        back = await self.tower.run()
        self.assertEqual(self.tower.by_identity(back)["serial:CAM001"]["source_kind"], "direct")
        self.assertEqual(len(back["ip_changes"]), 1)

    async def test_camera_first_seen_via_nvr_keeps_its_identity_when_seen_directly(self):
        self.tower.scan.devices = [nvr_channel(1, serial="CAM001")]
        first = await self.tower.run()
        self.assertEqual(list(self.tower.by_identity(first)), ["serial:CAM001#1"])

        self.tower.scan.devices = [nvr_channel(1, serial="CAM001"), direct("10.0.0.64", "CAM001")]
        second = await self.tower.run()
        rows = self.tower.by_identity(second)
        self.assertEqual(list(rows), ["serial:CAM001#1"])  # sticky public identity
        self.assertEqual(second["new_cameras"], [])
        self.assertEqual(rows["serial:CAM001#1"]["source_kind"], "direct")
        self.assertEqual(second["ip_changes"][0]["reason"], "source_switch")

    async def test_nvr_preference(self):
        await self.tower.close()
        self.tower = await TowerHarness(env=hikvision_env(SOURCE_PREFERENCE="nvr"), devices=self.devices).start()
        report = await self.tower.run()
        # The first winning route names the camera: here, the NVR channel.
        door = self.tower.by_identity(report)["serial:CAM001#1"]
        self.assertEqual((door["source_kind"], door["channel_no"]), ("nvr", 1))
        self.assertNotIn("sources", door)


class LifecycleTests(HarnessCase):
    devices = [direct("10.0.0.64", "CAM001")]

    async def test_grace_alert_once_recover_and_ip_repoint(self):
        await self.tower.run()
        self.tower.verifier.fail_hosts("10.0.0.64")

        grace = await self.tower.run()
        self.assertEqual(grace["missing_cameras"], [])
        self.assertTrue(grace["roster"][0]["is_present"])
        self.assertEqual(grace["unverified_candidates"], ["serial:CAM001"])

        alert = await self.tower.run()
        self.assertEqual([m["identity"] for m in alert["missing_cameras"]], ["serial:CAM001"])
        self.assertEqual(alert["roster"][0]["verification_state"], "offline")

        self.assertEqual((await self.tower.run())["missing_cameras"], [])  # no re-alert

        self.tower.verifier.ok_all()
        recovered = await self.tower.run()
        self.assertEqual([r["identity"] for r in recovered["recovered_cameras"]], ["serial:CAM001"])
        self.assertFalse(recovered["roster"][0]["alerted"])

        self.tower.scan.devices = [direct("10.0.0.80", "CAM001")]  # DHCP moved it
        moved = await self.tower.run()
        self.assertEqual(moved["ip_changes"][0]["reason"], "ip_change")
        self.assertEqual(moved["ip_changes"][0]["source_url"], "rtsp://10.0.0.80:554/Streaming/Channels/102")
        self.assertEqual([(r["identity"], r["ip_address"]) for r in moved["roster"]],
                         [("serial:CAM001", "10.0.0.80")])

    async def test_unverified_candidate_never_enters_the_roster(self):
        self.tower.verifier.fail_hosts("10.0.0.64")
        report = await self.tower.run()
        self.assertEqual(report["roster"], [])
        self.assertEqual(report["unverified_candidates"], ["serial:CAM001"])
        self.assertEqual(await self.count(Camera), 0)

    async def test_credentials_used_for_probing_are_never_stored(self):
        report = await self.tower.run()
        self.assertEqual(self.tower.verifier.urls,
                         ["rtsp://admin:cam-pass@10.0.0.64:554/Streaming/Channels/102"])
        self.assertNotIn("cam-pass", str(report))

    async def test_failed_address_change_preserves_verified_route_until_playable(self):
        first = (await self.tower.run())["roster"][0]
        self.tower.scan.devices = [direct("10.0.0.80", "CAM001")]
        self.tower.verifier.fail_hosts("10.0.0.80")

        failed = await self.tower.run()
        camera = failed["roster"][0]
        self.assertEqual(failed["ip_changes"], [])
        self.assertEqual(camera["source_url"], first["source_url"])
        self.assertEqual(camera["last_seen_at"], first["last_seen_at"])
        self.assertEqual(camera["ip_address"], "10.0.0.64")
        self.assertNotIn("sources", camera)

        await self.tower.restart()
        self.tower.verifier.ok_all()
        verified = await self.tower.run()
        self.assertEqual(verified["new_cameras"], [])
        self.assertEqual(verified["ip_changes"][0]["previous_source_url"], first["source_url"])
        camera = verified["roster"][0]
        self.assertEqual(camera["source_url"], "rtsp://10.0.0.80:554/Streaming/Channels/102")
        self.assertEqual(camera["first_frame_at"], first["first_frame_at"])

    async def test_failed_presence_transaction_keeps_already_provisioned_camera(self):
        with mock.patch.object(self.tower.sweep._cameras, "mark_missing", side_effect=RuntimeError("write failed")), \
                mock.patch.object(self.tower.sweep._relay, "reconcile") as relay:
            with self.assertRaisesRegex(RuntimeError, "write failed"):
                await self.tower.run()
            relay.assert_awaited()
        self.assertEqual(await self.count(Camera), len(self.tower.scan.devices))


class CameraMatchingTests(HarnessCase):
    env = hikvision_env()
    devices = (direct("10.0.0.64", "CAM001", mac="44:19:b6:aa:bb:cc"), nvr_channel(1, serial="CAM002"))

    async def test_mac_matches_after_ip_change_and_serial_becomes_available(self):
        self.tower.scan.devices = [direct("10.0.0.64", "", mac="44-19-B6-AA-BB-CC")]
        first = (await self.tower.run())["roster"][0]
        async with self.tower.database.session_factory() as db, db.begin():
            camera = (await db.execute(select(Camera))).scalar_one()
            camera.camera_uuid = "existing-jetson-camera"
        await self.tower.restart()
        self.tower.scan.devices = [direct("10.0.0.80", " cam001 ", mac="44:19:b6:aa:bb:cc")]
        report = await self.tower.run()
        camera = report["roster"][0]
        self.assertEqual(report["new_cameras"], [])
        self.assertEqual(camera["identity"], first["identity"])
        self.assertEqual(camera["camera_uuid"], "existing-jetson-camera")
        self.assertEqual(camera["serial_number"], "CAM001")
        self.assertEqual(camera["ip_address"], "10.0.0.80")
        self.assertEqual(await self.count(Camera), 1)

        # Serial/identity now differ from the MAC-only dedupe key of this scan.
        self.tower.scan.devices = [direct("10.0.0.81", "", mac="44:19:b6:aa:bb:cc")]
        report = await self.tower.run()
        self.assertEqual(report["new_cameras"], [])
        self.assertEqual(report["roster"][0]["camera_uuid"], "existing-jetson-camera")

    async def test_normalized_serial_matches_after_route_and_ip_change(self):
        self.tower.scan.devices = [nvr_channel(1, serial=" cam001 ")]
        first = (await self.tower.run())["roster"][0]
        self.tower.scan.devices = [direct("10.0.0.80", "CAM001")]
        report = await self.tower.run()
        self.assertEqual(report["new_cameras"], [])
        self.assertEqual(report["roster"][0]["identity"], first["identity"])
        self.assertEqual(report["roster"][0]["source_kind"], "direct")
        self.assertEqual(await self.count(Camera), 1)

    async def test_identifier_on_nonwinning_source_matches_existing_camera(self):
        self.tower.scan.devices = [nvr_channel(1, serial="CAM001")]
        first = (await self.tower.run())["roster"][0]
        # Only the alternate source matches the saved camera. The winner and
        # candidate dedupe key both use CAM000, which has never been stored.
        self.tower.scan.devices = [
            direct("10.0.0.64", "CAM000", mac="44:19:b6:aa:bb:cc"),
            nvr_channel(1, serial="CAM001", mac="44:19:b6:aa:bb:cc"),
        ]
        report = await self.tower.run()
        self.assertEqual(report["new_cameras"], [])
        self.assertEqual(report["roster"][0]["identity"], first["identity"])
        self.assertEqual(report["roster"][0]["serial_number"], "CAM000")
        self.assertEqual(report["roster"][0]["source_kind"], "direct")
        self.assertEqual(await self.count(Camera), 1)

        # The sticky identity still recognizes the NVR route after metadata changes.
        await self.tower.restart()
        self.tower.scan.devices = [nvr_channel(1, serial="CAM001")]
        report = await self.tower.run()
        self.assertEqual(report["new_cameras"], [])
        self.assertEqual(report["roster"][0]["identity"], first["identity"])

    async def test_ip_only_camera_matches_same_address_but_not_new_address(self):
        self.tower.scan.devices = [direct("10.0.0.64", "")]
        await self.tower.run()
        self.assertEqual((await self.tower.run())["new_cameras"], [])
        self.tower.scan.devices = [direct("10.0.0.80", "")]
        report = await self.tower.run()
        self.assertEqual([row["identity"] for row in report["new_cameras"]], ["ip:10.0.0.80"])
        self.assertEqual(await self.count(Camera), 2)

    async def test_serial_and_mac_matching_different_rows_is_a_conflict(self):
        self.tower.scan.devices = [
            direct("10.0.0.64", "CAM001"),
            direct("10.0.0.65", "CAM002", mac="44:19:b6:aa:bb:cc"),
        ]
        await self.tower.run()
        self.tower.scan.devices = [direct("10.0.0.80", "CAM001", mac="44:19:b6:aa:bb:cc")]
        report = await self.tower.run()
        self.assertEqual(report["present"], [])
        self.assertEqual(report["new_cameras"], [])
        self.assertIn("Ambiguous camera match", report["errors"][0])
        self.assertEqual({row["ip_address"] for row in report["roster"]}, {"10.0.0.64", "10.0.0.65"})

    async def test_ambiguous_candidate_preserves_both_cameras_and_reports_conflict(self):
        first = await self.tower.run()
        self.tower.scan.devices[1] = nvr_channel(1, serial="CAM002", mac="44:19:b6:aa:bb:cc")
        self.tower.scan.devices.append(direct("10.0.0.90", "UNRELATED"))
        await self.tower.restart()
        report = await self.tower.run()
        self.assertEqual([row["identity"] for row in report["present"]], ["serial:UNRELATED"])
        self.assertEqual([row["identity"] for row in report["new_cameras"]], ["serial:UNRELATED"])
        self.assertEqual(len(report["errors"]), 1)
        self.assertIn("Ambiguous camera match", report["errors"][0])
        rows = self.tower.by_identity(report)
        for original in first["roster"]:
            for field in ("source_url", "serial_number", "mac_address", "camera_uuid", "last_seen_at"):
                self.assertEqual(rows[original["identity"]][field], original[field])
        self.assertEqual(await self.count(Camera), 3)

    async def test_metadata_uses_winner_then_alternates_and_retains_known_values(self):
        camera = direct("10.0.0.64", "CAM001", mac="44:19:b6:aa:bb:cc", name="Front door")
        channel = nvr_channel(1, serial="CAM001", name="Recorder name")._replace(model=None)
        self.tower.scan.devices = [camera, channel]
        self.tower.verifier.fail_hosts(camera.ip)
        first = (await self.tower.run())["roster"][0]
        self.assertEqual(first["device_name"], "Recorder name")
        self.assertEqual((first["model"], first["firmware"], first["mac_address"]),
                         (camera.model, camera.firmware, camera.mac))

        self.tower.scan.devices = [channel._replace(device_name=None)]
        later = (await self.tower.run())["roster"][0]
        for field in ("serial_number", "mac_address", "model", "firmware", "device_name"):
            self.assertEqual(later[field], first[field])


class StaticNvrPresenceTests(HarnessCase):
    env = hikvision_env(NVR_CHANNELS="1-2")
    devices = [static_channel(1), static_channel(2)]

    async def test_rtsp_only_nvr_ages_on_frames_alerts_once_and_recovers(self):
        self.assertTrue((await self.tower.run())["nvr"]["is_present"])
        self.tower.verifier.fail_hosts(NVR_HOST)

        self.assertFalse((await self.tower.run())["nvr"]["missing_alert"])  # grace
        alerted = await self.tower.run()
        self.assertTrue(alerted["nvr"]["missing_alert"])
        self.assertFalse(alerted["nvr"]["is_present"])
        self.assertFalse((await self.tower.run())["nvr"]["missing_alert"])  # once

        self.tower.verifier.ok_all()
        self.assertTrue((await self.tower.run())["nvr"]["recovered"])


class SchedulingTests(HarnessCase):
    devices = [direct("10.0.0.64", "CAM001")]

    async def test_concurrent_sweeps_coalesce(self):
        self.tower.scan.delay_s = 0.3
        first, second = await asyncio.gather(self.tower.run(), self.tower.run())
        self.assertEqual(self.tower.scan.calls, 1)
        self.assertEqual(sorted(bool(r.get("coalesced")) for r in (first, second)), [False, True])
        self.assertEqual(await self.count(Camera), 1)

    async def test_request_scan_honours_the_quiet_period(self):
        self.assertTrue(self.tower.sweep.request_scan())
        await self.tower.sweep._inflight
        self.assertTrue(self.tower.sweep.request_scan())  # quiet period: no new sweep
        self.assertFalse(self.tower.sweep.scan_in_progress())
        self.assertEqual(self.tower.scan.calls, 1)

    async def test_scheduler_start_and_stop(self):
        self.assertTrue(self.tower.sweep.start())
        for _ in range(100):
            if self.tower.sweep.last_report():
                break
            await asyncio.sleep(0.02)
        self.assertEqual(self.tower.sweep.status()["present_count"], 1)
        await self.tower.sweep.stop()
        self.assertFalse(self.tower.sweep.status()["running"])


class DisabledTests(HarnessCase):
    env = {"DISCOVERY_ENABLED": "false"}
    devices = [direct("10.0.0.64", "CAM001")]

    async def test_disabled_discovery_never_scans(self):
        self.assertFalse(self.tower.sweep.start())
        report = await self.tower.run()
        self.assertEqual(report["reason"], "discovery_disabled")
        self.assertEqual(self.tower.scan.calls, 0)


if __name__ == "__main__":
    unittest.main()
