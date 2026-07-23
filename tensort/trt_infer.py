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
    Dynamic-batch TensorRT engine.

    The engine is expected to be built with a dynamic batch axis and an
    optimization profile (min=1 .. max=N). We allocate ONE set of pinned host
    + device buffers sized for the profile's MAX batch, then per infer() call
    set the actual batch with set_input_shape and transfer only the rows in use.

    A fixed-shape (batch=1) engine still works: it is treated as max_batch=1,
    so single-frame inference keeps running unchanged. This is the fallback for
    an engine that was NOT re-exported with a dynamic axis.

    Note: the previous ping-pong double-buffering (two streams hiding H2D
    latency for serial single-frame inference) is intentionally removed —
    throughput now comes from batching, and variable batch sizes don't map
    cleanly onto fixed dual buffers. One stream, one buffer set.

    Usage:
        tensor, r, pad = preprocess(bgr, imgsz)   # CPU, (1,3,H,W)
        outputs = engine.infer(batch_tensor)       # GPU, (B,...) outputs
    """

    def __init__(self, engine_path: str, device_id: int = 0):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(engine_path)

        self.device_id = int(device_id)

        with CudaContext(self.device_id):
            self._trt_logger = trt.Logger(trt.Logger.WARNING)
            with open(engine_path, "rb") as f, trt.Runtime(self._trt_logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())

            self.context = self.engine.create_execution_context()

            # TensorRT 10 name-based tensor I/O API.
            self.tensor_names = []
            self.input_index = None
            self.input_name = None
            self.output_indices = []
            self.output_names = []

            self._dtypes = {}        # index -> numpy dtype
            self._decl_shapes = {}   # index -> declared shape (may contain -1)

            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                self.tensor_names.append(name)
                self._dtypes[i] = trt.nptype(self.engine.get_tensor_dtype(name))
                self._decl_shapes[i] = tuple(int(d) for d in self.engine.get_tensor_shape(name))
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.input_index = i
                    self.input_name = name
                else:
                    self.output_indices.append(i)
                    self.output_names.append(name)

            if self.input_index is None:
                raise RuntimeError("No input tensor found.")

            in_decl = self._decl_shapes[self.input_index]
            self._dynamic = any(d < 0 for d in in_decl)

            if self._dynamic:
                # profile 0's MAX shape gives the largest batch the engine accepts
                _min, _opt, _max = self.engine.get_tensor_profile_shape(self.input_name, 0)
                self.max_batch = int(tuple(_max)[0])
            else:
                self.max_batch = int(in_decl[0])

            # C,H,W are the fixed (non-batch) input dims.
            self._fixed_chw = tuple(int(x) for x in in_decl[1:])
            self._in_row_size = int(np.prod(self._fixed_chw)) if self._fixed_chw else 1

            # Resolve a MAX shape per tensor (dynamic batch dim -> max_batch) so
            # the single allocation below is large enough for any B <= max_batch.
            self._max_sizes = {}     # index -> element count at max batch
            for i in range(self.engine.num_io_tensors):
                resolved = tuple(self.max_batch if d < 0 else d for d in self._decl_shapes[i])
                self._max_sizes[i] = int(np.prod(resolved)) if resolved else 1

            # ONE set of pinned host + device buffers, sized for max batch.
            self._host_bufs = []     # [index] -> pagelocked ndarray
            self._dev_bufs = []      # [index] -> device allocation
            for i in range(self.engine.num_io_tensors):
                host = cuda.pagelocked_empty(self._max_sizes[i], self._dtypes[i])
                dev = cuda.mem_alloc(host.nbytes)
                self._host_bufs.append(host)
                self._dev_bufs.append(dev)
                # Device buffers are reused for the engine's life, so the
                # addresses are stable and can be bound once here.
                self.context.set_tensor_address(self.tensor_names[i], int(dev))

            self._stream = cuda.Stream()

    def infer(self, input_chw: np.ndarray) -> List[np.ndarray]:
        """
        input_chw: pre-built (B,C,H,W) float32 tensor, B in [1, max_batch].
        Returns list of output ndarrays with their actual batched shapes
        (post-synchronise).
        """
        if input_chw.ndim != 4:
            raise ValueError("Expected 4D (B,C,H,W), got {}".format(input_chw.shape))
        b = int(input_chw.shape[0])
        if tuple(int(x) for x in input_chw.shape[1:]) != self._fixed_chw:
            raise ValueError("Expected (B,{}), got {}".format(self._fixed_chw, input_chw.shape))
        if b < 1 or b > self.max_batch:
            raise ValueError("Batch {} out of range [1,{}]".format(b, self.max_batch))

        ii = self.input_index
        stream = self._stream

        with CudaContext(self.device_id):
            # Tell TRT the actual batch for this call; resolves dynamic dims so
            # get_tensor_shape() below returns concrete output shapes.
            self.context.set_input_shape(self.input_name, (b,) + self._fixed_chw)

            # H2D: copy only the rows in use into the pinned buffer, then upload.
            n_in = b * self._in_row_size
            host_in = self._host_bufs[ii]
            np.copyto(host_in[:n_in], input_chw.ravel())
            cuda.memcpy_htod_async(self._dev_bufs[ii], host_in[:n_in], stream)

            # GPU inference (async); tensor addresses were bound in __init__.
            self.context.execute_async_v3(stream_handle=stream.handle)

            # D2H: resolve each output's real shape for this batch and copy only
            # that many elements back.
            pending = []
            for oi in self.output_indices:
                out_shape = tuple(int(x) for x in self.context.get_tensor_shape(self.tensor_names[oi]))
                n_out = int(np.prod(out_shape)) if out_shape else 1
                cuda.memcpy_dtoh_async(self._host_bufs[oi][:n_out], self._dev_bufs[oi], stream)
                pending.append((oi, n_out, out_shape))

            stream.synchronize()

            outs = [
                self._host_bufs[oi][:n_out].copy().reshape(out_shape)
                for (oi, n_out, out_shape) in pending
            ]

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
        # Largest batch this engine actually accepts (1 for a fixed batch=1
        # engine). Callers must not feed more than this.
        self.max_batch = int(getattr(self.trt, "max_batch", 1))

    def run(self, bgr: np.ndarray) -> List[Dict]:
        """Single-frame convenience path: preprocess -> infer (B=1) -> parse."""
        H0, W0 = bgr.shape[:2]
        x, r, (padx, pady) = preprocess(bgr, self.imgsz)
        outs = self.trt.infer(x)               # x is (1,3,H,W)
        return self._parse_pred(outs[0], H0, W0, r, padx, pady)

    def run_batch(self, bgr_list: List[np.ndarray]) -> List[List[Dict]]:
        """
        Batched path: preprocess every frame, stack into a single (B,3,H,W)
        tensor, run ONE GPU inference, then parse each image slice back to its
        own original coordinates (each camera keeps its own scale + padding).
        Returns one detection list per input frame, in the same order.
        """
        if not bgr_list:
            return []

        # Preprocess every frame up front (CPU). Each frame keeps its own scale
        # + padding so it can be de-letterboxed back to its own resolution.
        prepped = []   # (tensor(1,3,H,W), (H0,W0,r,padx,pady))
        for bgr in bgr_list:
            H0, W0 = bgr.shape[:2]
            x, r, (padx, pady) = preprocess(bgr, self.imgsz)
            prepped.append((x, (H0, W0, r, padx, pady)))

        # Never feed the engine more than it accepts. A dynamic engine takes the
        # whole batch in one call; a fixed batch=1 engine processes one frame per
        # call (chunk size 1). This keeps things correct regardless of whether
        # the dynamic engine has been re-exported yet.
        eng_max = max(1, int(self.trt.max_batch))

        results = []
        for start in range(0, len(prepped), eng_max):
            chunk = prepped[start:start + eng_max]
            tensors = [c[0] for c in chunk]
            batch = tensors[0] if len(tensors) == 1 else np.concatenate(tensors, axis=0)
            pred = self.trt.infer(batch)[0]    # (b, ...)
            for i, (_, (H0, W0, r, padx, pady)) in enumerate(chunk):
                # pred[i:i+1] keeps the leading dim so _parse_pred sees (1, ...)
                results.append(self._parse_pred(pred[i:i + 1], H0, W0, r, padx, pady))
        return results

    def _parse_pred(self, pred: np.ndarray, H0: int, W0: int, r: float, padx: int, pady: int) -> List[Dict]:
        if pred.ndim != 3:
            raise RuntimeError("Unexpected det output shape: {}".format(pred.shape))
        if pred.shape[2] == 6 and pred.shape[1] != 6:
            dets = pred[0]                 # (num_det, 6)
            inv_r = 1.0 / max(r, 1e-9)
            out = []
            for det in dets:
                score = float(det[4])
                if score < self.conf:
                    continue
                label = COCO_NAMES.get(int(det[5]), str(int(det[5])))
                if label not in self.allowed:
                    continue
                # de-letterbox back to original frame coords
                x1o, y1o, x2o, y2o = clamp_xyxy(
                    (float(det[0]) - padx) * inv_r,
                    (float(det[1]) - pady) * inv_r,
                    (float(det[2]) - padx) * inv_r,
                    (float(det[3]) - pady) * inv_r,
                    W0, H0,
                )
                out.append({
                    "cls_name": label,
                    "conf": score,
                    "box": {"x1": x1o, "y1": y1o, "x2": x2o, "y2": y2o},
                    "box_norm": box_norm_xyxy(x1o, y1o, x2o, y2o, W0, H0),
                })
            return out

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

    def _fail_event(self, meta: Dict, reason: str) -> Dict:
        return {
            "type": "InferenceFailedEvent",
            "channel_id": meta.get("channel_id"),
            "camera_uuid": str(meta.get("camera_uuid", "unknown")),
            "model_id": self.model_id,
            "frame_ts_ms": int(meta.get("frame_ts_ms", int(time.time() * 1000))),
            "frame_seq": int(meta.get("frame_seq", 0)),
            "reason": reason,
        }

    def infer_multitask_batch(self, bgr_list: List[np.ndarray], meta_list: List[Dict]) -> List[Dict]:
        """
        Run a batch of frames through ONE GPU inference and return one event per
        frame, in the same order as the inputs.

        On a whole-batch failure every frame gets an InferenceFailedEvent so the
        caller can still resolve each pending future (no frame is left hanging).
        """
        if not bgr_list:
            return []

        t0 = time.perf_counter()
        try:
            dets_list = self.det_runner.run_batch(bgr_list)
        except Exception as e:
            ms = int((time.perf_counter() - t0) * 1000)
            reason = "{}: {} (after {} ms)".format(type(e).__name__, e, ms)
            return [self._fail_event(m, reason) for m in meta_list]

        # Batch wall-time; shared across the batch (per-frame split isn't
        # meaningful since they run in one GPU call).
        ms = int((time.perf_counter() - t0) * 1000)
        batch_size = len(bgr_list)
        results = []
        for bgr, meta, dets in zip(bgr_list, meta_list, dets_list):
            H, W = bgr.shape[:2]
            results.append({
                "type": "DetectionsProducedEvent",
                "channel_id": meta.get("channel_id"),
                "camera_uuid": str(meta.get("camera_uuid", "unknown")),
                "model_id": self.model_id,
                "frame_ts_ms": int(meta.get("frame_ts_ms", int(time.time() * 1000))),
                "frame_seq": int(meta.get("frame_seq", 0)),
                "frame_w": W,
                "frame_h": H,
                "detections": dets,
                "pose": None,
                "inference_ms": ms,
                "batch_size": batch_size,
            })
        return results


def build_default() -> TRTInfer:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    det_engine = os.getenv("DET_ENGINE")
    if not det_engine:
        det_engine = os.path.join(base_dir, "models", "yolo26n.engine")
    elif not os.path.isabs(det_engine) and not os.path.exists(det_engine):
        candidate = os.path.join(base_dir, det_engine)
        if os.path.exists(candidate):
            det_engine = candidate

    if not os.path.exists(det_engine):
        raise FileNotFoundError("DET_ENGINE not found: {}".format(det_engine))

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
