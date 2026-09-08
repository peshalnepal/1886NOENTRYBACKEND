import asyncio
import unittest
import uuid
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from application.repositories.notification_repository import CameraContext
from application.services.pipeline import InMemoryObjDetectStore, ModelPipeline, ObjDetectResponse


class DetectionOrderingTests(unittest.IsolatedAsyncioTestCase):
    async def test_store_uses_timestamp_then_sequence(self):
        for timestamp, sequence, expected in (
            (101, 0, True), (100, 3, True), (100, 2, False), (99, 99, False)
        ):
            with self.subTest(timestamp=timestamp, sequence=sequence):
                store = InMemoryObjDetectStore()
                response = ObjDetectResponse("camera", timestamp, sequence)
                await store.put(response)
                result = await store.wait_new(
                    "camera", after_ts_ms=100, after_seq=2, timeout_ms=1
                )
                self.assertEqual(result, response if expected else None)

    async def test_pipeline_accepts_regressions_only_after_restart_gap(self):
        pipeline = ModelPipeline(uuid.uuid4())
        pipeline._last_seen["camera"] = (100, 2)
        for timestamp, sequence, elapsed, expected in (
            (101, 0, 0, True), (100, 3, 0, True),
            (100, 2, 6, False), (99, 99, 5, False),
            (99, 99, 6, True), (100, 1, 5, False), (100, 1, 6, True),
        ):
            with self.subTest(timestamp=timestamp, sequence=sequence, elapsed=elapsed):
                pipeline._last_ok_s["camera"] = 100 - elapsed
                with patch("application.services.pipeline.time.monotonic", return_value=100):
                    self.assertEqual(
                        pipeline._is_new_detection(
                            "camera", ObjDetectResponse("camera", timestamp, sequence)
                        ),
                        expected,
                    )


class PollerCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_shutdown_cancels_all_pollers_before_waiting(self):
        second_cancelled = asyncio.Event()

        async def first():
            try:
                await asyncio.Event().wait()
            finally:
                await second_cancelled.wait()

        async def second():
            try:
                await asyncio.Event().wait()
            finally:
                second_cancelled.set()

        pipeline = ModelPipeline(uuid.uuid4())
        pipeline._started = True
        tasks = [asyncio.create_task(first()), asyncio.create_task(second())]
        pipeline._poll_tasks = dict(zip(("first", "second"), tasks))
        await asyncio.sleep(0)
        await asyncio.wait_for(pipeline.shutdown(), timeout=1)
        self.assertTrue(all(task.cancelled() for task in tasks))
        self.assertEqual(pipeline._poll_tasks, {})
        self.assertFalse(pipeline._started)

    async def test_cleanup_logs_failed_task_and_still_cancels_remaining_tasks(self):
        async def failed():
            raise RuntimeError("poller failed")

        failed_task = asyncio.create_task(failed(), name="failed-poller")
        pending_task = asyncio.create_task(asyncio.sleep(60))
        await asyncio.sleep(0)
        with self.assertLogs("application.services.pipeline", level="ERROR"):
            await ModelPipeline._stop_pollers([failed_task, pending_task])
        self.assertTrue(pending_task.cancelled())


class NotificationCleanupTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tasks = []

        def spawn(coro, name):
            task = asyncio.create_task(coro, name=name)
            self.tasks.append(task)
            return task

        self.service = SimpleNamespace(
            hub=SimpleNamespace(publish=AsyncMock()),
            requires_operator_approval=AsyncMock(return_value=False),
            enqueue_notification=AsyncMock(),
        )
        self.pipeline = ModelPipeline(
            uuid.uuid4(), notification_service=self.service, task_spawner=spawn,
            interesting_classes={"person", "car"},
        )
        self.ctx = CameraContext(
            user_id=7, site_uuid=uuid.uuid4(), site_name="Yard",
            camera_code="CAM-01", camera_name="Gate",
            device_uuid=uuid.uuid4(), device_name="Edge",
        )
        self.pipeline._get_camera_ctx = AsyncMock(return_value=self.ctx)
        self.detection = {
            "cls_name": "person", "conf": 0.9,
            "box": {"x1": 1, "y1": 2, "x2": 10, "y2": 20},
        }
        self.track = {**self.detection, "track_id": 3}
        self.alert = {**self.track, "roi_id": "zone"}
        self.response = ObjDetectResponse(
            camera_uuid=str(uuid.uuid4()), frame_ts_ms=100, frame_seq=2,
            frame_w=640, frame_h=480, detections=(self.detection,),
            tracks=(self.track,), track_events=(("track_confirmed", 3),),
            alerts=(self.alert,),
        )

    async def asyncTearDown(self):
        await asyncio.gather(*self.tasks)

    async def _emit(self, kind):
        extra = {"image_url": " https://images.example/frame.jpg "}
        if kind == "roi_enter":
            await self.pipeline._emit_roi_alert_notifications(
                self.response, [self.alert], extra_payload=extra
            )
        elif kind == "item_detected":
            await self.pipeline._emit_item_detected_notifications(
                self.response, extra_payload=extra
            )
        else:
            await self.pipeline._emit_detection_summary_notification(
                self.response, extra_payload=extra
            )
        await asyncio.gather(*self.tasks)

    async def test_notification_variants_share_metadata_and_keep_event_payloads(self):
        for kind in ("roi_enter", "item_detected", "detection_summary"):
            with self.subTest(kind=kind):
                self.service.hub.publish.reset_mock()
                self.service.enqueue_notification.reset_mock()
                await self._emit(kind)
                msg = self.service.hub.publish.call_args.args[0]
                self.assertEqual(msg.alert_type, kind)
                self.assertEqual(msg.camera_uuid, self.response.camera_uuid)
                self.assertEqual(msg.site_uuid, str(self.ctx.site_uuid))
                self.assertEqual((msg.user_id, msg.ts_ms, msg.frame_seq), (7, 100, 2))
                self.assertEqual((msg.frame_w, msg.frame_h), (640, 480))
                self.assertEqual((msg.camera_name, msg.device_name), ("Gate", "Edge"))
                self.assertEqual(msg.image_url, "https://images.example/frame.jpg")
                self.assertEqual(msg.cls_names, ["person"])
                self.assertEqual(msg.max_conf, 0.9)
                self.assertTrue(msg.detections)
                call = self.service.enqueue_notification.call_args
                self.assertEqual(call.args, (msg, self.ctx))
                payload = call.kwargs["extra_payload"]
                if kind == "roi_enter":
                    self.assertEqual((msg.track_id, msg.roi_id), (3, "zone"))
                    self.assertEqual(payload["alert"], self.alert)
                elif kind == "item_detected":
                    self.assertEqual(msg.track_id, 3)
                    self.assertEqual(payload["track"], self.track)
                    self.assertEqual(payload["event"], "track_confirmed")
                else:
                    self.assertIsNone(msg.track_id)
                    self.assertIsNone(msg.roi_id)
                    self.assertEqual(payload["detections"], [self.detection])
                    self.assertEqual(payload["event"], "detection_summary")

    async def test_operator_approval_suppresses_realtime_but_preserves_persistence(self):
        self.service.requires_operator_approval.return_value = True
        await self._emit("detection_summary")
        self.service.hub.publish.assert_not_awaited()
        self.service.enqueue_notification.assert_awaited_once()

    async def test_summary_filters_classes_consistently_and_limits_persisted_detections(self):
        kept = [{**self.detection, "cls_name": " person "} for _ in range(25)]
        ignored = [None, {}, {"cls_name": "dog", "conf": 1.0}]
        self.response = replace(self.response, detections=tuple(ignored + kept))
        self.assertEqual(self.pipeline._interesting_detection_classes(self.response), ["person"])
        await self._emit("detection_summary")
        msg = self.service.hub.publish.call_args.args[0]
        self.assertEqual(msg.cls_names, ["person"])
        self.assertEqual(msg.max_conf, 0.9)
        payload = self.service.enqueue_notification.call_args.kwargs["extra_payload"]
        self.assertEqual(payload["detections"], kept[:20])
