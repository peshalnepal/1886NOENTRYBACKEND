"""Offline reproductions of B3/B4 against the reviewed source.

These assert the CURRENT defects, so a correct future fix should make them
fail. They are evidence examples, not permanent acceptance tests.

No database/camera/GPU is opened. Existing edge test fixtures install native
and repository import stubs, so run this script in its own Python process.
The test-fixture imports require the same available Python packages as those
edge tests (including NumPy). Application source is never modified.
"""

import ast
import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

EDGE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(EDGE / "tests"), str(EDGE)]
import test_orin_capacity  # installs isolated repository/native test stubs
from channels.channel_config import VideoChannelConfig
from pipeline import SimpleInferencePipeline


def inventory_uuid():
    source = (EDGE.parent / "application/services/inventory_service.py").read_text()
    node = next(n for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.FunctionDef) and n.name == "_adoption_entry")
    # Execute only this pure method, avoiding cloud imports/engine setup.
    namespace = {}
    exec(compile("from __future__ import annotations\n" + ast.get_source_segment(source, node),
                 "<isolated-current-adoption-method>", "exec"), namespace)
    row = SimpleNamespace(discovery_identity="serial:S1", source_url="rtsp://192.168.50.21/Streaming/Channels/102",
                          ip_address="192.168.50.21", device_name="Gate", model="Example", serial_number="S1",
                          edge_camera_uuid="11111111-1111-4111-8111-111111111111")
    entry = namespace["_adoption_entry"](row)
    assert "camera_uuid" not in entry
    print("B3 reproduced: manual inventory adoption omits existing edge UUID.")


async def identical_upsert():
    pipeline = SimpleInferencePipeline()
    cfg = VideoChannelConfig(camera_uuid="edge-A", source_url="rtsp://192.168.50.21/Streaming/Channels/102")
    # Pipeline is not started: no threads or GPU load. add_channel still runs
    # its real replacement/cache invalidation logic.
    try:
        await pipeline.add_channel(cfg)
        generation = pipeline._channel_generation["edge-A"]
        pipeline._latest["edge-A"] = {"frame_seq": 42}
        await pipeline.add_channel(cfg)
        assert pipeline._channel_generation["edge-A"] is not generation
        assert pipeline.peek_latest("edge-A") is None
        print("B4 reproduced: identical channel config changes generation and clears latest result.")
    finally:
        pipeline._snapshot_executor.shutdown(wait=True)


if __name__ == "__main__":
    with patch.dict("os.environ", {}, clear=True):
        inventory_uuid()
        asyncio.run(identical_upsert())
