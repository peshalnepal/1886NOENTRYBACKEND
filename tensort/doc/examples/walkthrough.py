"""Offline arithmetic/identity examples. No camera, database, or GPU access.

Run from any directory: python3 /path/to/doc/examples/walkthrough.py
Uses only the standard library and the edge's frame pool.
"""

import asyncio
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

EDGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EDGE))
from frame_pool import FramePool


async def pool_example():
    pool = FramePool(capacity=4, max_age_s=1)
    def event(camera, seq, timestamp):
        return SimpleNamespace(camera_uuid=camera, seq=seq, ts_ms=timestamp)
    with patch("frame_pool.time.time", return_value=1.0):
        for item in [event("A", 1, 100), event("A", 2, 200),
                     event("A", 3, 300), event("B", 1, 150), event("B", 2, 350)]:
            pool.put(item)
        batch = await pool.get_batch(2)
    assert pool.evicted_total == 1
    assert [(item.camera_uuid, item.seq) for item in batch] == [("B", 1), ("A", 2)]
    print("Fair pool batch:", [(item.camera_uuid, item.seq) for item in batch])


def main():
    height, width, size = 360, 640, 640
    scale = min(size / height, size / width)
    pad_x, pad_y = (size - round(width * scale)) // 2, (size - round(height * scale)) // 2
    assert (scale, pad_x, pad_y) == (1.0, 0, 140)
    model_box = (100, 190, 300, 390)
    box = tuple((value - pad) / scale for value, pad in zip(model_box, (pad_x, pad_y, pad_x, pad_y)))
    assert box == (100, 50, 300, 250)
    norm = (box[0]/width, box[1]/height, (box[2]-box[0])/width, (box[3]-box[1])/height)
    assert math.isclose(norm[1], 0.1388888888888889)
    print("Original box:", box, "normalized:", norm)
    print("RGB normalized pixel:", [value / 255 for value in (200, 20, 10)])

    row_mib = 3 * 640 * 640 * 4 / (1024**2)
    assert row_mib == 4.6875
    assert row_mib * 8 == 37.5
    print("FP32 input MiB: one row =", row_mib, "eight rows =", row_mib * 8)
    assert 8 * 3 == 24
    assert 8 / .4 == 20
    print("Example demand: 24 images/s; hypothetical 400 ms batch capacity: 20 images/s")
    asyncio.run(pool_example())
    print("All walkthrough assertions passed.")


if __name__ == "__main__":
    main()
