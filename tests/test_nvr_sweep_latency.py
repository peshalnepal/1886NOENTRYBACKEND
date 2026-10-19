"""Incremental discovery regressions using SQLite and no network/cameras."""

import asyncio
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sqlalchemy import select

from NVR.application.dtos import CandidateDTO, NvrDTO, ScanResultDTO, SourceDTO
from NVR.application.services.adoption_service import AdoptionService
from NVR.application.services.frame_verifier import FrameVerifier, VerifyResult
from NVR.application.services.sweep_service import SweepService
from NVR.config import Settings
from NVR.core.database import Database
from NVR.core.database_orm import Camera, NvrDevice
from NVR.relay import MediaRelay
from NVR.jetson import JetsonError


def candidate(name):
    return CandidateDTO(sources=(SourceDTO(
        source_identity="serial:" + name, serial_number=name, kind="direct",
        host=name, rtsp_port=554,
        source_url=f"rtsp://{name}:554/Streaming/Channels/102",
    ),))


class IncrementalSweepTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.database = Database(self.temp.name + "/cameras.sqlite")
        await self.database.init()
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None, discovery_miss_threshold=2,
                                nvr_public_rtsp_base_url="rtsp://tower:8554")
        self.scan = ScanResultDTO(candidates=[candidate("fast"), candidate("slow")])
        self.adapter = SimpleNamespace(scan=AsyncMock(side_effect=lambda: self.scan),
                                       render=lambda kind, url: url)
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()

        async def verify(url):
            if "//slow:" in url:
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    self.cancelled.set()
                    raise
            return VerifyResult(True)

        self.relay = SimpleNamespace(states={}, export=dict,
            jetson_url=lambda identity: "rtsp://tower:8554/" + identity)

        async def configure(identities):
            self.relay.states.update(dict.fromkeys(identities, "configured"))
            return []

        self.relay.reconcile = AsyncMock(side_effect=configure)
        self.registered = asyncio.Event()
        self.remote = []

        async def register(method, path, json):
            self.remote.append(dict(camera_uuid=json["config"]["camera_uuid"],
                                    source_url=json["source_url"], config=json["config"]))
            self.registered.set()

        self.jetson = SimpleNamespace(cameras=AsyncMock(side_effect=lambda: list(self.remote)),
                                      request=AsyncMock(side_effect=register))
        self.adoption = AdoptionService(self.database.session_factory, self.jetson, self.relay)
        self.sweep = SweepService(settings, self.database.session_factory, self.adapter,
                                  SimpleNamespace(verify=verify), self.relay, self.adoption)

    async def asyncTearDown(self):
        await self.sweep.stop()
        await self.database.dispose()
        self.temp.cleanup()

    async def rows(self):
        async with self.database.session_factory() as db:
            return list((await db.execute(select(Camera))).scalars())

    async def test_fast_camera_is_registered_before_slow_probe_finishes(self):
        caller = asyncio.create_task(self.sweep.run_once())
        try:
            await asyncio.wait_for(self.registered.wait(), 3)
            self.assertFalse(caller.done())
            self.assertEqual([row.identity for row in await self.rows()], ["serial:fast"])
            self.assertEqual(self.relay.states, {"serial:fast": "configured"})
            self.release.set()
            report = await asyncio.wait_for(caller, 3)
            self.assertEqual(len(report["present"]), 2)
            self.assertTrue(all(row["adopted"] for row in report["new_cameras"]))
            self.jetson.cameras.assert_awaited_once()
        finally:
            self.release.set()
            await self.sweep.stop()
            await asyncio.gather(caller, return_exceptions=True)

    async def test_cancel_keeps_success_without_aging_missing_and_retry_keeps_uuid(self):
        # Start with both known cameras; a cancellation cannot age the slow one.
        self.release.set()
        await self.sweep.run_once()
        original = {row.identity: row.camera_uuid for row in await self.rows()}
        self.release.clear()
        self.registered.clear()
        reached = asyncio.Event()
        real_reconcile = self.adoption.reconcile

        async def reconcile(identities, **kwargs):
            result = await real_reconcile(identities, **kwargs)
            reached.set()
            return result

        with patch.object(self.adoption, "reconcile", side_effect=reconcile):
            caller = asyncio.create_task(self.sweep.run_once())
            await asyncio.wait_for(reached.wait(), 3)
            await self.sweep.stop()
            with self.assertRaises(asyncio.CancelledError):
                await caller
        self.assertTrue(self.cancelled.is_set())
        rows = await self.rows()
        self.assertTrue(all(row.consecutive_misses == 0 for row in rows))
        self.release.set()
        await self.sweep.run_once()
        self.assertEqual({row.identity: row.camera_uuid for row in await self.rows()}, original)
        self.assertEqual(self.jetson.request.await_count, 2)

    async def test_unreachable_jetson_is_checked_once_and_all_verified_cameras_are_saved(self):
        self.release.set()
        self.jetson.cameras.side_effect = JetsonError(502)
        report = await self.sweep.run_once()
        self.jetson.cameras.assert_awaited_once()
        self.assertEqual(len(await self.rows()), 2)
        self.assertEqual(len(report["present"]), 2)
        self.assertEqual(report["errors"], [str(JetsonError(502))])

    async def test_presence_ages_once_after_all_failed_candidates(self):
        self.release.set()
        self.scan.nvr = NvrDTO(host="recorder", rtsp_port=554, http_port=80,
                              origin="static", responded=True)
        await self.sweep.run_once()
        self.scan.nvr = self.scan.nvr.model_copy(update={"responded": False})
        self.sweep._verifier.verify = AsyncMock(return_value=VerifyResult(False))
        report = await self.sweep.run_once()
        self.assertEqual(report["missing_cameras"], [])
        self.assertTrue(all(row.consecutive_misses == 1 for row in await self.rows()))
        async with self.database.session_factory() as db:
            recorder = (await db.execute(select(NvrDevice))).scalar_one()
            self.assertEqual(recorder.consecutive_misses, 1)
        report = await self.sweep.run_once()
        self.assertEqual(len(report["missing_cameras"]), 2)
        self.assertTrue(report["nvr"]["missing_alert"])

    async def test_final_presence_failure_does_not_rollback_registered_camera(self):
        self.scan.candidates = [candidate("fast")]
        with patch.object(self.sweep._cameras, "mark_missing", side_effect=RuntimeError("write failed")):
            with self.assertRaisesRegex(RuntimeError, "write failed"):
                await self.sweep.run_once()
        rows = await self.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].jetson_state, "pushed")
        self.assertEqual(rows[0].consecutive_misses, 0)


class VerifierFairnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_waiting_recorder_channels_do_not_reserve_global_slots(self):
        verifier = FrameVerifier(max_concurrency=2, per_host=1)
        blocked = asyncio.Event()
        started = asyncio.Event()
        other = asyncio.Event()

        async def run(url):
            if "//recorder/" in url:
                started.set()
                await blocked.wait()
            else:
                other.set()
            return VerifyResult(True)

        verifier._run = run
        tasks = [asyncio.create_task(verifier.verify("rtsp://recorder/1"))]
        try:
            await asyncio.wait_for(started.wait(), 1)
            tasks.append(asyncio.create_task(verifier.verify("rtsp://recorder/2")))
            await asyncio.sleep(0)
            tasks.append(asyncio.create_task(verifier.verify("rtsp://other/1")))
            await asyncio.wait_for(other.wait(), 1)
            self.assertFalse(tasks[0].done())
        finally:
            blocked.set()
            await asyncio.gather(*tasks)

    async def test_live_uses_substream_and_recording_uses_main_stream(self):
        relay = MediaRelay.__new__(MediaRelay)
        relay.settings = SimpleNamespace(recording_enabled=True,
            recording_directory="/recordings", recording_retention="168h")
        relay._ensure_path = AsyncMock()
        source = "rtsp://camera:554/Streaming/Channels/1702"
        await relay.ensure("camera", source)
        await relay.ensure_recording("camera", source)
        live, recording = [call.args[1] for call in relay._ensure_path.await_args_list]
        self.assertEqual(live["source"], source)
        self.assertEqual(recording["source"], source.replace("1702", "1701"))
        self.assertTrue(live["sourceOnDemand"])
        self.assertFalse(recording["sourceOnDemand"])


if __name__ == "__main__":
    unittest.main()
