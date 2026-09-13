"""Off-device capacity, lifecycle, backpressure and concurrent-admission tests."""

import asyncio
import os
import sys
import threading
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.modules.setdefault("cv2", MagicMock())
repo_stub = types.ModuleType("repositories")
repo_stub.CameraRepository = object
repo_stub.DiscoveryRepository = object
sys.modules["repositories"] = repo_stub

from channels.channel import RTSPEvent, VideoChannel
from channels.channel_config import VideoChannelConfig
from limits import CameraCapacityError, camera_limit
from pipeline import Broadcaster, FramePool, InferenceWorkerPool, SimpleInferencePipeline
from runtime import PipelineRuntime


def config(key, **kwargs):
    return VideoChannelConfig(camera_uuid=key, source_url="rtsp://camera/live", **kwargs)


class AsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_pipeline_rejects_ninth_and_allows_replacement_and_disable(self):
        pipeline = SimpleInferencePipeline()
        for i in range(8):
            await pipeline.add_channel(config(str(i)))
        with self.assertRaises(CameraCapacityError):
            await pipeline.add_channel(config("ninth"))
        await pipeline.add_channel(config("0"))
        self.assertEqual(len(pipeline._channels), 8)
        await pipeline.add_channel(config("0", enabled=False))
        await pipeline.add_channel(config("ninth"))
        self.assertEqual(len(pipeline._channels), 8)

    async def test_eight_camera_overload_is_bounded_fair_and_ordered(self):
        pool = FramePool(capacity=16, max_age_s=0)
        sequences = {str(i): [] for i in range(8)}
        for tick in range(100):
            for key in sequences:
                for offset in range(4 if key == "0" else 1):
                    pool.put(RTSPEvent(key, key, tick * 10 + offset, tick * 4 + offset, object()))
                    self.assertLessEqual(pool.depth(), 16)
            if tick % 2 == 1:
                batch = await pool.get_batch(8)
                self.assertEqual({ev.camera_uuid for ev in batch}, set(sequences))
                for ev in batch:
                    sequences[ev.camera_uuid].append(ev.seq)
        self.assertGreater(pool.evicted_total, 0)
        for emitted in sequences.values():
            self.assertEqual(len(emitted), 50)
            self.assertEqual(emitted, sorted(set(emitted)))
        pool.discard_camera("7")
        batch = await pool.get_batch(8)
        self.assertTrue(batch)
        self.assertNotIn("7", {ev.camera_uuid for ev in batch})

    async def test_capture_handoff_holds_only_newest_and_one_callback(self):
        channel = VideoChannel(config("cam"))
        channel._loop = MagicMock()
        for seq in range(1000):
            channel._push_from_thread(RTSPEvent("cam", "cam", 0, seq, object()))
        self.assertEqual(channel._loop.call_soon_threadsafe.call_count, 1)
        self.assertEqual(channel._handoff_dropped, 999)
        channel._drain_handoff()
        self.assertEqual(channel._out_q.get_nowait().seq, 999)
        self.assertIsNone(channel._pending_frame)

    async def test_slow_subscriber_receives_latest_message(self):
        broadcaster = Broadcaster()
        subscriber = await broadcaster.subscribe("cam")
        for seq in range(250):
            broadcaster.broadcast({"camera_uuid": "cam", "frame_seq": seq})
        self.assertEqual(subscriber.qsize(), 200)
        self.assertEqual(subscriber.get_nowait()["frame_seq"], 50)

    async def test_old_generation_result_cannot_repopulate_removed_camera(self):
        pipeline = SimpleInferencePipeline()
        await pipeline.add_channel(config("cam"))
        generation = pipeline._channel_generation["cam"]
        await pipeline.remove_channel("cam")
        meta = {"camera_uuid": "cam", "_generation": generation, "_bgr_ref": object()}
        pipeline._handle_result({"type": "DetectionsProducedEvent", "detections": []}, meta)
        self.assertIsNone(pipeline.peek_latest("cam"))
        self.assertNotIn("_bgr_ref", meta)

    async def test_snapshot_work_stays_bounded_when_encoder_is_slow(self):
        pipeline = SimpleInferencePipeline()
        pipeline._snapshot_on_detection_only = False
        pipeline._snapshot_min_interval_ms = 0
        gate = threading.Event()
        started = threading.Event()

        def slow_encode(*args, **kwargs):
            started.set()
            gate.wait(3)
            return b"jpeg"

        try:
            with patch("pipeline._encode_jpeg_bytes", side_effect=slow_encode):
                for seq in range(100):
                    for key in map(str, range(8)):
                        pipeline._handle_result({"type": "DetectionsProducedEvent", "detections": []},
                                                {"camera_uuid": key, "frame_seq": seq,
                                                 "_ts_ms": int(time.time() * 1000), "_bgr_ref": object()})
                    await asyncio.sleep(0)
                self.assertTrue(started.is_set())
                self.assertEqual(len(pipeline._snapshot_tasks), 8)
                gate.set()
                await asyncio.gather(*list(pipeline._snapshot_tasks.values()))
                await asyncio.sleep(0)
                self.assertFalse(pipeline._snapshot_tasks)
                self.assertEqual(len(pipeline._latest_snapshots), 8)
        finally:
            gate.set()
            pipeline._snapshot_executor.shutdown(wait=True)

    async def test_worker_initialization_failure_is_reported(self):
        pool = InferenceWorkerPool.__new__(InferenceWorkerPool)
        initialized = threading.Event()
        initialized.set()
        pool._workers = [types.SimpleNamespace(initialized=initialized, init_error="bad engine")]
        with self.assertRaisesRegex(RuntimeError, "bad engine"):
            await pool.wait_ready()

    async def test_worker_reports_actual_engine_batch_capacity(self):
        pool = InferenceWorkerPool.__new__(InferenceWorkerPool)
        initialized = threading.Event()
        initialized.set()
        pool._workers = [types.SimpleNamespace(initialized=initialized, init_error=None,
                                             _infer=types.SimpleNamespace(max_batch=1))]
        self.assertEqual(await pool.wait_ready(), 1)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.runtime = PipelineRuntime.__new__(PipelineRuntime)
        self.runtime._lock = threading.Lock()
        self.runtime._camera_update_lock = threading.Lock()
        self.runtime._cameras = {}
        self.runtime._max_cameras = 8
        self.runtime._require_pipeline = lambda: types.SimpleNamespace(add_channel=lambda cfg: cfg,
                                                                      remove_channel=lambda key: key)
        self.runtime._call = lambda *args, **kwargs: time.sleep(0.001)
        self.runtime._save_camera_to_db = MagicMock()
        self.runtime._delete_camera_from_db = MagicMock()

    def add(self, key, **kwargs):
        return self.runtime.add_camera("rtsp://camera/live", dict(camera_uuid=key, **kwargs))

    def test_concurrent_adds_never_admit_more_than_eight(self):
        def add(key):
            try:
                self.add(key)
                return True
            except CameraCapacityError:
                return False
        with ThreadPoolExecutor(max_workers=16) as executor:
            accepted = list(executor.map(add, map(str, range(16))))
        self.assertEqual(sum(accepted), 8)
        self.assertEqual(len(self.runtime.list_cameras()), 8)
        self.assertEqual(self.runtime._save_camera_to_db.call_count, 8)

    def test_failed_patch_leaves_original_configuration_unchanged(self):
        for i in range(8):
            self.add(str(i))
        self.add("disabled", enabled=False)
        with self.assertRaises(CameraCapacityError):
            self.runtime.patch_camera("disabled", {"enabled": True})
        self.assertFalse(self.runtime._cameras["disabled"]["enabled"])
        with self.assertRaises(ValueError):
            self.runtime.patch_camera("0", {"sample_fps": float("nan")})
        self.assertEqual(self.runtime._cameras["0"]["sample_fps"], 5)
        with self.assertRaises(ValueError):
            self.runtime.patch_camera("0", {"camera_uuid": "different"})

    def test_disabled_camera_and_custom_resize_are_preserved(self):
        result = self.add("cam", enabled="false", resize=(640, 480))
        self.assertFalse(result["config"]["enabled"])
        self.assertEqual(result["config"]["resize"], (640, 480))

    def test_site_limit_cannot_exceed_eight(self):
        with patch.dict(os.environ, {"MAX_CAMERAS": "100"}):
            self.assertEqual(camera_limit(), 8)
        with patch.dict(os.environ, {"MAX_CAMERAS": "4"}):
            self.assertEqual(camera_limit(), 4)

    def test_gstreamer_does_not_drop_compressed_packets(self):
        channel = VideoChannel(config("cam", resize=(640, 360)))
        pipeline = channel._build_rtsp_pipeline('rtsp://user:p"ass@camera/live', "nvv4l2decoder")
        self.assertNotIn("leaky", pipeline.split("nvv4l2decoder")[0])
        self.assertIn('location="rtsp://user:p\\"ass@camera/live"', pipeline)
        self.assertLess(pipeline.index("videorate"), pipeline.index("videoconvert"))
        self.assertIn("framerate=5/1", pipeline)

    def test_discovery_retries_admission_after_capacity_is_freed(self):
        from test_discovery import FakeRepo
        from discovery import DiscoveryConfig, HikDevice
        from service import DiscoveryService
        repo = FakeRepo()
        device = HikDevice(ip="192.168.1.10", serial_number="test-camera",
                           model="test", firmware="", device_name="", mac="")
        discovery_service = DiscoveryService(self.runtime, repository=repo, config=DiscoveryConfig())
        for i in range(8):
            self.add(str(i))
        with patch("service.scan_network", return_value=[device]):
            with self.assertLogs("jetson-discovery-service", level="ERROR"):
                first = discovery_service.run_once(force=True)
            self.assertTrue(first["errors"])
            self.assertIsNone(repo.get_by_identity(device.identity())["camera_uuid"])
            self.runtime.remove_camera("0")
            second = discovery_service.run_once(force=True)
            self.assertTrue(second["new_cameras"][0]["adopted"])
            self.assertEqual(len(self.runtime.list_cameras()), 8)
            self.assertFalse(discovery_service.run_once(force=True)["new_cameras"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
