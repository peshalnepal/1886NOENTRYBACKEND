"""On-device batch correctness and preprocessing/inference/postprocessing timing."""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("batch", type=int, nargs="?", default=8)
    parser.add_argument("--seconds", type=float, default=15)
    args = parser.parse_args()
    if not 1 <= args.batch <= 8 or args.seconds <= 0:
        parser.error("Use batch 1..8 and a positive duration")
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
    except ImportError:
        pass
    import cv2
    import numpy as np
    from trt_infer import build_default

    cv2.setNumThreads(1)
    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit("Cannot read image: " + args.image)
    infer = build_default()
    try:
        if infer.max_batch < args.batch:
            raise SystemExit("Engine max batch is {}; rebuild for {}".format(infer.max_batch, args.batch))
        frames = [frame] * args.batch
        runner = infer.det_runner
        reference = runner.run(frame)
        if not reference:
            raise SystemExit("Use an image with visible allowed objects; empty results cannot verify row correctness")

        def numbers(dets):
            return np.array([[d["conf"]] + [d["box"][k] for k in ("x1", "y1", "x2", "y2")] for d in dets])

        # Exercise partial batches too: a quiet/offline camera must not break inference.
        for size in sorted({1, max(1, args.batch // 2), args.batch}):
            for row in runner.run_batch(frames[:size]):
                if [d["cls_name"] for d in row] != [d["cls_name"] for d in reference]:
                    raise RuntimeError("Batch class/count mismatch at batch {}".format(size))
                delta = np.abs(numbers(row) - numbers(reference))
                if not np.all(delta <= np.array([0.03, 3, 3, 3, 3])):
                    raise RuntimeError("Batch output differs beyond FP16 tolerance at batch {}".format(size))
        for _ in range(10):
            runner.run_batch(frames)
        batches = 0
        durations = []
        started = time.perf_counter()
        while time.perf_counter() - started < args.seconds:
            t0 = time.perf_counter()
            runner.run_batch(frames)
            durations.append((time.perf_counter() - t0) * 1000)
            batches += 1
        elapsed = time.perf_counter() - started
        fps = batches * args.batch / elapsed
        print(json.dumps({"batch_rows_match": True, "batch_size": args.batch,
                          "images_per_second": round(fps, 2),
                          "batch_ms_p95": round(float(np.percentile(durations, 95)), 2),
                          "suggested_fps_per_camera_with_25_percent_headroom": round(fps / 8 / 1.25, 2),
                          "includes": "preprocessing, GPU inference, postprocessing; excludes video decode and SSE"}, indent=2))
    finally:
        infer.close()


if __name__ == "__main__":
    main()
