"""Live delivery regressions; no database, network, cameras or GPU required."""
import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import HTTPException
from application.services.pipeline import DetectionHub, InMemoryObjDetectStore, ObjDetectResponse
from application.services.detection_stream import stream_camera_detections, stream_multi_camera_detections
from routes import camera_routes


def frame(camera="camera", seq=1):
    return ObjDetectResponse(camera_uuid=camera, frame_ts_ms=seq * 1000, frame_seq=seq)


class DeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_running_fan_in_recovers_from_accepted_clock_reset(self):
        store = InMemoryObjDetectStore()
        await store.put(frame(seq=100))
        pipeline = SimpleNamespace(get_latest_detection=store.get_latest, detect_store=store)
        gen = stream_multi_camera_detections(cameras=[("camera", pipeline)],
                                             is_disconnected=AsyncMock(return_value=False))
        self.assertIn('"frame_seq": 100', await anext(gen))
        await store.put(frame(seq=1))
        self.assertIn('"frame_seq": 1', await asyncio.wait_for(anext(gen), 0.1))
        await gen.aclose()

    async def test_explicit_initial_future_cursor_does_not_replay_older_cache(self):
        store = InMemoryObjDetectStore()
        await store.put(frame(seq=1))
        pipeline = SimpleNamespace(get_latest_detection=store.get_latest, detect_store=store)
        gen = stream_camera_detections(pipeline=pipeline, camera_uuid="camera",
            after_ts_ms=100000, after_seq=100, timeout_ms=5,
            is_disconnected=AsyncMock(return_value=False))
        self.assertIn("event: heartbeat", await anext(gen))
        await gen.aclose()

    async def test_each_reader_gets_latest_after_another_clears_signal(self):
        store = InMemoryObjDetectStore()
        newest = frame(seq=2)
        await store.put(newest)
        (await store._evt("camera")).clear()
        results = await asyncio.gather(*[
            store.wait_new("camera", after_ts_ms=1000, after_seq=1, timeout_ms=20)
            for _ in range(3)
        ])
        self.assertEqual(results, [newest] * 3)

    async def test_waiting_readers_all_wake_and_unchanged_frame_times_out(self):
        store = InMemoryObjDetectStore()
        tasks = [asyncio.create_task(store.wait_new(
            "camera", after_ts_ms=0, after_seq=0, timeout_ms=100)) for _ in range(3)]
        await asyncio.sleep(0)
        await store.put(frame())
        self.assertTrue(all(result.frame_seq == 1 for result in await asyncio.gather(*tasks)))
        self.assertIsNone(await store.wait_new(
            "camera", after_ts_ms=1000, after_seq=1, timeout_ms=5))

    async def test_global_stream_coalesces_without_starving_other_cameras(self):
        hub = DetectionHub(max_q=2)
        mailbox = await hub.subscribe()
        await hub.publish(frame("busy"))
        await hub.publish(frame("quiet"))
        for seq in range(2, 250):
            await hub.publish(frame("busy", seq))
        self.assertEqual((await mailbox.get()).frame_seq, 249)
        self.assertEqual((await mailbox.get()).camera_uuid, "quiet")
        await hub.unsubscribe(mailbox)
        self.assertFalse(hub._subs)

    async def test_fan_in_waits_cancel_when_viewer_disconnects(self):
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def wait_new(*args, **kwargs):
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        pipeline = SimpleNamespace(get_latest_detection=AsyncMock(return_value=None),
                                   detect_store=SimpleNamespace(wait_new=wait_new))
        gen = stream_multi_camera_detections(cameras=[("camera", pipeline)],
                                             is_disconnected=AsyncMock(return_value=False))
        next_event = asyncio.create_task(anext(gen))
        await started.wait()
        next_event.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await next_event
        self.assertTrue(cancelled.is_set())


class AuthorizedFanInTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ids = [UUID(int=1), UUID(int=2)]
        self.cameras = {key: SimpleNamespace(camera_uuid=key, user_id=owner,
                                            org_id=10, site_uuid="site")
                        for key, owner in zip(self.ids, (11, 22))}
        self.user = SimpleNamespace(id=99, is_platform_admin=False)
        self.pipelines = {owner: SimpleNamespace(
            get_latest_detection=AsyncMock(return_value=frame(str(key))),
            detect_store=InMemoryObjDetectStore())
            for key, owner in zip(self.ids, (11, 22))}
        self.manager = SimpleNamespace(get_activepipeline=AsyncMock(
            side_effect=lambda user_id: self.pipelines[user_id]))

        @asynccontextmanager
        async def session():
            yield object()

        self.request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(session_factory=session)),
                                       is_disconnected=AsyncMock(return_value=False))

        async def get_camera(db, camera_uuid):
            cam = self.cameras.get(camera_uuid)
            return (cam, None, None) if cam else None

        patches = [
            patch.object(camera_routes, "_resolve_stream_user", AsyncMock(return_value=self.user)),
            patch.object(camera_routes.ChannelRepository, "get_camera_full", side_effect=get_camera),
            patch.object(camera_routes.AuthzService, "resolve_user_org", AsyncMock(return_value=(10, "member"))),
            patch.object(camera_routes.AuthzService, "accessible_site_uuids", AsyncMock(return_value={"site"})),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    async def response(self, ids):
        return await camera_routes.stream_all_detections_sse(
            self.request, camera_uuid=ids, manager=self.manager)

    async def test_members_read_each_camera_owners_pipeline_once(self):
        response = await self.response(self.ids + [self.ids[0]])
        received = []
        for _ in range(2):
            event = await anext(response.body_iterator)
            received.append(json.loads(event.split("data: ", 1)[1])["camera_uuid"])
        await response.body_iterator.aclose()
        self.assertEqual(set(received), set(map(str, self.ids)))
        self.assertEqual(self.manager.get_activepipeline.await_count, 2)
        self.assertEqual(response.headers["x-accel-buffering"], "no")

    async def test_any_forbidden_camera_rejects_before_streaming(self):
        self.cameras[self.ids[1]].org_id = 20
        with self.assertRaises(HTTPException) as caught:
            await self.response(self.ids)
        self.assertEqual(caught.exception.status_code, 404)
        self.manager.get_activepipeline.assert_not_awaited()

    async def test_deleted_camera_rejects_before_streaming(self):
        del self.cameras[self.ids[1]]
        with self.assertRaises(HTTPException) as caught:
            await self.response(self.ids)
        self.assertEqual(caught.exception.status_code, 404)

    async def test_permissions_are_rechecked_for_open_stream(self):
        with patch.object(camera_routes, "stream_multi_camera_detections") as stream:
            stream.return_value = iter(())
            await self.response(self.ids)
            check = stream.call_args.kwargs["still_authorized"]
            self.assertTrue(await check())
            self.cameras[self.ids[1]].org_id = 20
            self.assertFalse(await check())


if __name__ == "__main__":
    unittest.main()
