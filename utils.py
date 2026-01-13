import cv2
import numpy as np
from typing import Optional, Iterable, Tuple

from domain.events import PoseResult  # your pydantic model
from domain.model_pipeline import ObjDetectResponse  # your dataclass
from domain.events import DetectionItem

# COCO-17 skeleton edges (Ultralytics YOLO pose uses COCO order by default)
COCO17_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
)

def _frame_from_rtsp_event(ev) -> Optional[np.ndarray]:
    """
    Get BGR frame from RTSPEvent.
    - If ev.frame exists (raw), use it.
    - If ev.encoded is jpeg/png, decode it.
    """
    frame = getattr(ev, "frame", None)
    if frame is not None:
        return frame

    enc = getattr(ev, "encoded", None)
    fmt = getattr(ev, "format", None)
    if enc is None:
        return None

    if fmt in ("jpeg", "png"):
        arr = np.frombuffer(enc, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img

    return None

def _draw_bbox(frame: np.ndarray, det: DetectionItem) -> None:
    b = det.box
    x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    label = f"{det.cls_name} {det.conf:.2f}"
    # background for text
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y0 = max(0, y1 - th - 6)
    cv2.rectangle(frame, (x1, y0), (x1 + tw + 6, y0 + th + 6), (0, 255, 0), -1)
    cv2.putText(frame, label, (x1 + 3, y0 + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def _draw_pose(frame: np.ndarray, pose: PoseResult, *, kp_min_conf: float = 0.2) -> None:
    """
    Draw skeleton keypoints + edges.
    Assumes pose.format is "xy" (pixel) or "xyn" (normalized).
    """
    H, W = frame.shape[:2]
    norm = (pose.format == "xyn")

    for sk in pose.skeletons:
        # optional: draw pose box in a different color
        b = sk.box
        cv2.rectangle(frame, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (255, 0, 0), 2)

        kps = sk.keypoints or []
        if not kps:
            continue

        # convert keypoints to pixel coords
        pts = []
        confs = []
        for kp in kps:
            x = float(kp.x) * W if norm else float(kp.x)
            y = float(kp.y) * H if norm else float(kp.y)
            c = float(kp.conf) if kp.conf is not None else 1.0
            pts.append((int(x), int(y)))
            confs.append(c)

        # draw edges
        for a, bidx in COCO17_EDGES:
            if a >= len(pts) or bidx >= len(pts):
                continue
            if confs[a] < kp_min_conf or confs[bidx] < kp_min_conf:
                continue
            cv2.line(frame, pts[a], pts[bidx], (255, 0, 0), 2)

        # draw keypoints
        for (x, y), c in zip(pts, confs):
            if c < kp_min_conf:
                continue
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

def render_frame_with_overlays(ev, detect: Optional[ObjDetectResponse]) -> Optional[bytes]:
    """
    Returns JPEG bytes (ready for MJPEG chunk).
    """
    frame = _frame_from_rtsp_event(ev)
    if frame is None:
        return None

    out = frame.copy()

    if detect is not None:
        # Draw only non-skeleton boxes (pose will be drawn from detect.pose)
        for d in (detect.detections or ()):
            if d.cls_name == "skeleton":
                continue
            _draw_bbox(out, d)

        if detect.pose is not None and detect.pose.skeletons:
            _draw_pose(out, detect.pose)

    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return None
    return buf.tobytes()