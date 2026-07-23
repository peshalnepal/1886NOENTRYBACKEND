# diag_batch.py — does the TRT engine actually compute every batch row?
#
# Run from Backend/tensort (so `import trt_infer` and the .env resolve):
#     python tests/diag_batch.py /path/to/frame_with_an_object.jpg
#
# It feeds the engine B identical copies of ONE image. A correctly-batched
# engine returns IDENTICAL non-empty detections for every row. A row-0-only
# engine (NMS baked at batch=1) returns detections for row 0 and empty for the
# rest — which proves the engine, not the pipeline, is dropping the cameras.

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from trt_infer import build_default, preprocess


def main():
    if len(sys.argv) < 2:
        print("usage: python tests/diag_batch.py <image_path> [batch]")
        return
    img_path = sys.argv[1]
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    bgr = cv2.imread(img_path)
    if bgr is None:
        print("could not read image:", img_path)
        return

    infer = build_default()
    runner = infer.det_runner
    eng = runner.trt
    print("engine max_batch  :", eng.max_batch)
    print("engine dynamic    :", eng._dynamic)
    print("input name/shape  :", eng.input_name, eng._decl_shapes[eng.input_index])
    for oi in eng.output_indices:
        print("output name/shape :", eng.tensor_names[oi], eng._decl_shapes[oi])

    B = min(B, eng.max_batch)
    print("\n--- feeding %d IDENTICAL copies of %s ---" % (B, img_path))

    # Build the (B,3,H,W) batch exactly like run_batch does.
    x, r, (padx, pady) = preprocess(bgr, runner.imgsz)
    batch = np.concatenate([x] * B, axis=0)
    print("batch tensor shape:", batch.shape, "contiguous:", batch.flags["C_CONTIGUOUS"])

    outs = eng.infer(batch)
    pred = outs[0]
    print("raw output shape  :", pred.shape, "dtype:", pred.dtype)

    # Per-row signal: how much of each batch row is non-zero. A working batch
    # engine shows ~equal non-zero counts on every row; a broken one is non-zero
    # only on row 0.
    print("\nper-row raw non-zero element count (should be ~equal for all rows):")
    for i in range(pred.shape[0]):
        nz = int(np.count_nonzero(pred[i]))
        print("  row %2d: nonzero=%d  max=%.4f" % (i, nz, float(np.max(np.abs(pred[i])))))

    # Per-row parsed detections (after conf/class filtering + de-letterbox).
    print("\nper-row parsed detections (should be IDENTICAL for all rows):")
    H0, W0 = bgr.shape[:2]
    for i in range(pred.shape[0]):
        dets = runner._parse_pred(pred[i:i + 1], H0, W0, r, padx, pady)
        print("  row %2d: %d detections  %s" % (
            i, len(dets),
            [(d["cls_name"], round(d["conf"], 2)) for d in dets[:5]],
        ))


if __name__ == "__main__":
    main()
