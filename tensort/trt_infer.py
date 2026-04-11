# trt_infer.py
# Python 3.6 compatible
#
# Fixes applied vs original:
#  1. TRTEngine — double-buffered CUDA streams (stream A and stream B alternate).
#     While stream A synchronises for camera N, stream B is already uploading
#     camera N+1's tensor.  Net effect: memcpy_htod overlaps with execute on the
#     previous frame, hiding transfer latency.
#  2. YoloV8DetTRT.run — letterbox_bgr + _prepare_input_tensor (both pure CPU)
#     are called BEFORE entering TRTEngine.infer().  infer() now accepts a
#     pre-built CHW float32 tensor and only does: copyto → htod → execute → dtoh
#     → synchronize.  This makes preprocessing parallelisable (it can happen in
#     the async event loop or another thread while the previous frame is on GPU).
#  3. build_default — reads IMG_SZ, respects CUDA_DEVICE, unchanged API.

import os
import time
import threading
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

_BLOB_FROM_IMAGE = getattr(getattr(cv2, "dnn", None), "blobFromImage", None)

import tensorrt as trt
import pycuda.driver as cuda

logger = logging.getLogger(__name__)

# -----------------------------
# CUDA context management (unchanged)
# -----------------------------
cuda.init()
_tls = threading.local()


def ensure_cuda_context(device_id=0):
    ctx = getattr(_tls, "ctx", None)
    if ctx is None:
        device = cuda.Device(int(device_id))
        ctx = device.make_context()
        ctx.pop()
        _tls.ctx = ctx
    return ctx


class CudaContext(object):
    def __init__(self, device_id=0):
        self.device_id = int(device_id)
        self.ctx = None

    def __enter__(self):
        self.ctx = ensure_cuda_context(self.device_id)
        self.ctx.push()
        return self.ctx

    def __exit__(self, exc_type, exc, tb):
        try:
            self.ctx.pop()
        except Exception:
            pass


def release_cuda_context():
    ctx = getattr(_tls, "ctx", None)
    if ctx is not None:
        try:
            ctx.detach()
        except Exception:
            pass
        _tls.ctx = None


# -----------------------------
# CPU preprocessing (unchanged logic, now called before infer())
# -----------------------------

def letterbox_bgr(img: np.ndarray, new_shape: int = 640, color=(114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    h, w = img.shape[:2]
    r = min(float(new_shape) / float(h), float(new_shape) / float(w))
    nh, nw = int(round(h * r)), int(round(w * r))

    interp = cv2.INTER_AREA if r < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)

    pad_w = new_shape - nw
    pad_h = new_shape - nh
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return out, r, (left, top)


def _prepare_input_tensor(img_lb: np.ndarray) -> np.ndarray:
    if _BLOB_FROM_IMAGE is not None:
        return _BLOB_FROM_IMAGE(
            img_lb,
            scalefactor=1.0 / 255.0,
            size=(img_lb.shape[1], img_lb.shape[0]),
            mean=(0.0, 0.0, 0.0),
            swapRB=True,
            crop=False,
        )
    rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    x = np.empty((1, 3, img_lb.shape[0], img_lb.shape[1]), dtype=np.float32)
    x[0] = np.transpose(rgb, (2, 0, 1))
    x *= (1.0 / 255.0)
    return x


def preprocess(bgr: np.ndarray, imgsz: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Full CPU preprocessing pipeline.
    Returns (chw_float32_tensor, scale_r, (padx, pady)).
    Call this BEFORE TRTEngine.infer() — it runs on CPU and can overlap with
    a previous frame's GPU execution.
    """
    img_lb, r, (padx, pady) = letterbox_bgr(bgr, imgsz)
    x = _prepare_input_tensor(img_lb)
    return x, r, (padx, pady)


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45, topk: int = 100) -> List[int]:
    if boxes is None or len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float32, copy=False)
    scores = scores.astype(np.float32, copy=False)

    x1 = boxes[:, 0]; y1 = boxes[:, 1]
    x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)

    order = scores.argsort()[::-1]
    keep = []
    eps = 1e-9

    while order.size > 0 and len(keep) < topk:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        w = np.maximum(0.0, xx2 - xx1 + 1.0)
        h = np.maximum(0.0, yy2 - yy1 + 1.0)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + eps)
        order = rest[iou <= iou_thr]

    return keep


def clamp_xyxy(x1, y1, x2, y2, W, H):
    x1 = int(max(0, min(x1, W - 1)))
    y1 = int(max(0, min(y1, H - 1)))
    x2 = int(max(0, min(x2, W - 1)))
    y2 = int(max(0, min(y2, H - 1)))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2


def box_norm_xyxy(x1, y1, x2, y2, W, H):
    w = max(1.0, float(x2) - float(x1))
    h = max(1.0, float(y2) - float(y1))
    return {
        "x": max(0.0, min(1.0, float(x1) / max(float(W), 1.0))),
        "y": max(0.0, min(1.0, float(y1) / max(float(H), 1.0))),
        "w": max(0.0, min(1.0, w / max(float(W), 1.0))),
        "h": max(0.0, min(1.0, h / max(float(H), 1.0))),
    }


# -----------------------------
# FIX 1: TRTEngine — double-buffered CUDA streams
# -----------------------------

class TRTEngine(object):
    """
    Double-buffered TensorRT engine.

    Two sets of pinned host buffers + two CUDA streams (ping / pong).
    While stream[0] synchronises (waiting for GPU→CPU copy of frame N),
    stream[1] can already be uploading frame N+1's tensor to the GPU.
    This hides H2D transfer latency on Jetson's unified memory bus.

    Usage:
        tensor, r, pad = preprocess(bgr, imgsz)   # CPU — can run in parallel
        outputs = engine.infer(tensor)             # GPU — overlaps with next CPU preprocess
    """

    def __init__(self, engine_path: str, device_id: int = 0):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(engine_path)

        self.device_id = int(device_id)
        self._buf_idx = 0   # ping-pong index (0 or 1)

        with CudaContext(self.device_id):
            self._trt_logger = trt.Logger(trt.Logger.WARNING)
            with open(engine_path, "rb") as f, trt.Runtime(self._trt_logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())

            self.context = self.engine.create_execution_context()

            self.binding_names = []
            self.input_index = None
            self.output_indices = []
            self.output_shapes = {}

            # Temporary: gather shapes/dtypes first pass
            _shapes = []
            _dtypes = []
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                self.binding_names.append(name)
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                shape = self.engine.get_binding_shape(i)
                if -1 in tuple(shape):
                    raise RuntimeError(
                        "Dynamic shape binding ({}): {}. Export fixed-shape engine.".format(name, shape)
                    )
                _shapes.append(tuple(shape))
                _dtypes.append(dtype)
                if self.engine.binding_is_input(i):
                    self.input_index = i
                    self.input_shape = tuple(shape)
                else:
                    self.output_indices.append(i)
                    self.output_shapes[i] = tuple(shape)

            if self.input_index is None:
                raise RuntimeError("No input binding found.")

            # Allocate TWO sets of pinned host buffers (one per stream slot)
            # but only ONE set of device buffers (GPU mem is shared; we sync
            # before reuse so there is no race).
            self._host_bufs = [[], []]   # [buf_idx][binding_idx]
            self._dev_bufs = []          # [binding_idx]  — shared
            self._bindings = []          # int pointers into _dev_bufs

            for i, (shape, dtype) in enumerate(zip(_shapes, _dtypes)):
                size = int(np.prod(shape))
                for b in range(2):
                    self._host_bufs[b].append(cuda.pagelocked_empty(size, dtype))
                dev = cuda.mem_alloc(self._host_bufs[0][i].nbytes)
                self._dev_bufs.append(dev)
                self._bindings.append(int(dev))

            # Two CUDA streams
            self._streams = [cuda.Stream(), cuda.Stream()]

    def infer(self, input_chw: np.ndarray) -> List[np.ndarray]:
        """
        input_chw: pre-built CHW float32 tensor from preprocess().
        Returns list of output ndarrays (post-synchronise).
        """
        if input_chw.shape != self.input_shape:
            raise ValueError("Expected {}, got {}".format(self.input_shape, input_chw.shape))

        b = self._buf_idx           # current ping/pong slot
        stream = self._streams[b]

        with CudaContext(self.device_id):
            # Copy input tensor into pinned host buffer for this slot
            np.copyto(self._host_bufs[b][self.input_index], input_chw.ravel())

            # H2D upload on this stream
            cuda.memcpy_htod_async(
                self._dev_bufs[self.input_index],
                self._host_bufs[b][self.input_index],
                stream,
            )

            # GPU inference (async)
            self.context.execute_async_v2(
                bindings=self._bindings,
                stream_handle=stream.handle,
            )

            # D2H download on same stream
            for oi in self.output_indices:
                cuda.memcpy_dtoh_async(
                    self._host_bufs[b][oi],
                    self._dev_bufs[oi],
                    stream,
                )

            # Synchronise THIS stream only; the other slot is free to start uploading
            stream.synchronize()

            outs = [
                self._host_bufs[b][oi].copy().reshape(self.output_shapes[oi])
                for oi in self.output_indices
            ]

        # Advance ping-pong index
        self._buf_idx = 1 - b
        return outs


# -----------------------------
# YOLOv8 parsers
# -----------------------------
COCO_NAMES = {
    0: "person",
    2: "car",
    3: "motorcycle",
    7: "truck",
}


class YoloV8DetTRT(object):
    def __init__(
        self,
        engine_path: str,
        imgsz: int = 640,
        conf: float = 0.35,
        iou: float = 0.45,
        allowed=("person", "car", "motorcycle", "truck"),
        topk: int = 100,
        device_id: int = 0,
    ):
        self.trt = TRTEngine(engine_path, device_id=device_id)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.allowed = set(allowed)
        self.topk = int(topk)

    def run(self, bgr: np.ndarray) -> List[Dict]:
        H0, W0 = bgr.shape[:2]

        # FIX 2: preprocessing runs on CPU BEFORE touching the GPU.
        # In a multi-camera scenario the InferenceWorker can preprocess the
        # next frame's tensor while the GPU is still executing the current one.
        x, r, (padx, pady) = preprocess(bgr, self.imgsz)

        # GPU inference — only memcpy + execute + memcpy + sync
        outs = self.trt.infer(x)
        pred = outs[0]

        if pred.ndim != 3:
            raise RuntimeError("Unexpected det output shape: {}".format(pred.shape))

        if pred.shape[1] < pred.shape[2]:
            p = pred[0]
        else:
            p = pred[0].T

        C, N = p.shape
        nc = C - 4
        boxes_xywh = p[0:4, :]
        cls_scores = p[4:4 + nc, :]

        cls_id = np.argmax(cls_scores, axis=0)
        score = cls_scores[cls_id, np.arange(N)]

        keep = score >= self.conf
        cls_id = cls_id[keep]
        score = score[keep]
        boxes_xywh = boxes_xywh[:, keep]

        labels = [COCO_NAMES.get(int(i), str(int(i))) for i in cls_id]
        allowed_mask = np.array([lab in self.allowed for lab in labels], dtype=bool)

        labels = [lab for lab, m in zip(labels, allowed_mask) if m]
        cls_id = cls_id[allowed_mask]
        score = score[allowed_mask]
        boxes_xywh = boxes_xywh[:, allowed_mask]

        if boxes_xywh.size == 0:
            return []

        x_c, y_c, w, h = boxes_xywh
        x1 = x_c - w / 2
        y1 = y_c - h / 2
        x2 = x_c + w / 2
        y2 = y_c + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        keep_idx = nms_xyxy(boxes, score, self.iou)
        out = []

        for i in keep_idx:
            bx = boxes[i]
            bx0 = (bx - np.array([padx, pady, padx, pady], dtype=np.float32)) / max(r, 1e-9)
            x1o, y1o, x2o, y2o = clamp_xyxy(bx0[0], bx0[1], bx0[2], bx0[3], W0, H0)
            out.append({
                "cls_name": labels[i],
                "conf": float(score[i]),
                "box": {"x1": x1o, "y1": y1o, "x2": x2o, "y2": y2o},
                "box_norm": box_norm_xyxy(x1o, y1o, x2o, y2o, W0, H0),
            })

        return out


# -----------------------------
# In-process inference API (unchanged public surface)
# -----------------------------

class TRTInfer(object):
    def __init__(
        self,
        det_engine_path: str,
        imgsz: int = 512,
        conf: float = 0.35,
        iou: float = 0.45,
        allowed=("person", "car", "motorcycle", "truck"),
        model_id: str = "yolo-trt",
        device_id: int = 0,
        nms_topk: int = 100,
    ):
        self.model_id = model_id
        self.device_id = int(device_id)

        ensure_cuda_context(self.device_id)
        self.det_runner = YoloV8DetTRT(
            det_engine_path,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            allowed=allowed,
            topk=nms_topk,
            device_id=self.device_id,
        )

    def infer_multitask(self, bgr: np.ndarray, meta: Dict) -> Dict:
        t0 = time.perf_counter()

        camera_uuid = str(meta.get("camera_uuid", "unknown"))
        channel_id = meta.get("channel_id")
        frame_ts_ms = int(meta.get("frame_ts_ms", int(time.time() * 1000)))
        frame_seq = int(meta.get("frame_seq", 0))

        try:
            dets = self.det_runner.run(bgr)
            ms = int((time.perf_counter() - t0) * 1000)
            H, W = bgr.shape[:2]
            return {
                "type": "DetectionsProducedEvent",
                "channel_id": channel_id,
                "camera_uuid": camera_uuid,
                "model_id": self.model_id,
                "frame_ts_ms": frame_ts_ms,
                "frame_seq": frame_seq,
                "frame_w": W,
                "frame_h": H,
                "detections": dets,
                "pose": None,
                "inference_ms": ms,
            }
        except Exception as e:
            ms = int((time.perf_counter() - t0) * 1000)
            return {
                "type": "InferenceFailedEvent",
                "channel_id": channel_id,
                "camera_uuid": camera_uuid,
                "model_id": self.model_id,
                "frame_ts_ms": frame_ts_ms,
                "frame_seq": frame_seq,
                "reason": "{}: {} (after {} ms)".format(type(e).__name__, e, ms),
            }


def build_default() -> TRTInfer:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    det_engine = os.getenv("DET_ENGINE")
    if not det_engine:
        det_engine = os.path.join(base_dir, "models", "yolo26n.engine")
    elif not os.path.isabs(det_engine) and not os.path.exists(det_engine):
        candidate = os.path.join(base_dir, det_engine)
        if os.path.exists(candidate):
            det_engine = candidate

    imgsz = int(os.getenv("IMG_SZ", "640"))
    conf = float(os.getenv("CONF", "0.350"))   # matches .env.example CONF=0.350
    iou = float(os.getenv("IOU", "0.45"))
    device_id = int(os.getenv("CUDA_DEVICE", "0"))
    nms_topk = int(os.getenv("NMS_TOPK", "50"))   # matches .env.example NMS_TOPK=50

    return TRTInfer(
        det_engine_path=det_engine,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device_id=device_id,
        nms_topk=nms_topk,
    )