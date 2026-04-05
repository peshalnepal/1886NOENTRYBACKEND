# trt_infer.py
# Python 3.6 compatible
#
# Key points:
# - NO pycuda.autoinit
# - Explicit CUDA driver init
# - Thread-local CUDA context with push/pop around CUDA operations
#
# IMPORTANT:
# - Create TRTInfer/TRTEngine in the SAME thread where you'll call infer().
#   Do not create in main thread and call infer in another thread.

import os
import time
import threading
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

import tensorrt as trt
import pycuda.driver as cuda

logger = logging.getLogger(__name__)

# -----------------------------
# Explicit CUDA init + per-thread context
# -----------------------------
cuda.init()
_tls = threading.local()


def ensure_cuda_context(device_id=0):
    """
    Ensure CUDA context exists for CURRENT thread.
    Creates a thread-local context (stored in TLS). Does NOT leave it pushed.
    """
    ctx = getattr(_tls, "ctx", None)
    if ctx is None:
        device = cuda.Device(int(device_id))
        ctx = device.make_context()  # pushes immediately
        ctx.pop()                    # pop now; we'll push only when needed
        _tls.ctx = ctx
    return ctx


class CudaContext(object):
    """
    Push/pop thread-local CUDA context around CUDA/TRT calls.
    """
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
    """
    Optional cleanup for worker threads: call when thread exits.
    """
    ctx = getattr(_tls, "ctx", None)
    if ctx is not None:
        try:
            ctx.detach()
        except Exception:
            pass
        _tls.ctx = None


# -----------------------------
# Utils
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
    dnn_mod = getattr(cv2, "dnn", None)
    blob_from_image = getattr(dnn_mod, "blobFromImage", None) if dnn_mod is not None else None
    if callable(blob_from_image):
        return blob_from_image(
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
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
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
# TensorRT engine wrapper
# -----------------------------
class TRTEngine(object):
    def __init__(self, engine_path: str, device_id: int = 0):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(engine_path)

        self.device_id = int(device_id)

        # Ensure context exists and is pushed while we allocate CUDA resources
        with CudaContext(self.device_id):
            self.logger = trt.Logger(trt.Logger.WARNING)
            with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())

            self.context = self.engine.create_execution_context()

            self.bindings = []
            self.host_mem = []
            self.device_mem = []
            self.binding_names = []
            self.output_shapes = {}

            self.input_index = None
            self.output_indices = []

            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                self.binding_names.append(name)

                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                shape = self.engine.get_binding_shape(i)

                if -1 in tuple(shape):
                    raise RuntimeError(
                        "Dynamic shape binding found ({}): {}. Export fixed-shape engine.".format(name, shape)
                    )

                size = int(np.prod(shape))
                host = cuda.pagelocked_empty(size, dtype)
                dev = cuda.mem_alloc(host.nbytes)

                self.host_mem.append(host)
                self.device_mem.append(dev)
                self.bindings.append(int(dev))

                if self.engine.binding_is_input(i):
                    self.input_index = i
                    self.input_shape = tuple(shape)
                else:
                    self.output_indices.append(i)
                    self.output_shapes[i] = tuple(shape)

            if self.input_index is None:
                raise RuntimeError("No input binding found.")

            self.stream = cuda.Stream()

    def infer(self, input_chw: np.ndarray) -> List[np.ndarray]:
        if input_chw.shape != self.input_shape:
            raise ValueError("Expected input {}, got {}".format(self.input_shape, input_chw.shape))

        # Push the same thread-local context for CUDA calls
        with CudaContext(self.device_id):
            np.copyto(self.host_mem[self.input_index], input_chw.ravel())

            cuda.memcpy_htod_async(
                self.device_mem[self.input_index],
                self.host_mem[self.input_index],
                self.stream,
            )

            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

            for oi in self.output_indices:
                cuda.memcpy_dtoh_async(self.host_mem[oi], self.device_mem[oi], self.stream)

            self.stream.synchronize()

            outs = []
            for oi in self.output_indices:
                out = self.host_mem[oi].copy().reshape(self.output_shapes[oi])
                outs.append(out)

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
        conf: float = 0.25,
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
        img_lb, r, (padx, pady) = letterbox_bgr(bgr, self.imgsz)

        x = _prepare_input_tensor(img_lb)

        outs = self.trt.infer(x)
        pred = outs[0]

        if pred.ndim != 3:
            raise RuntimeError("Unexpected det output shape: {}".format(pred.shape))

        # normalize to (C, N)
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

        cls_id = cls_id[allowed_mask]
        score = score[allowed_mask]
        boxes_xywh = boxes_xywh[:, allowed_mask]
        labels = [COCO_NAMES.get(int(i), str(int(i))) for i in cls_id]

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


class YoloV8PoseTRT(object):
    def __init__(self, engine_path: str, imgsz: int = 640, conf: float = 0.25, iou: float = 0.45, kpts: int = 17, device_id: int = 0):
        self.trt = TRTEngine(engine_path, device_id=device_id)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.kpts = int(kpts)

    def run(self, bgr: np.ndarray) -> Tuple[List[Dict], Optional[Dict]]:
        H0, W0 = bgr.shape[:2]
        img_lb, r, (padx, pady) = letterbox_bgr(bgr, self.imgsz)

        x = _prepare_input_tensor(img_lb)

        outs = self.trt.infer(x)
        pred = outs[0]

        if pred.ndim != 3:
            raise RuntimeError("Unexpected pose output shape: {}".format(pred.shape))

        # normalize to (C, N)
        if pred.shape[1] < pred.shape[2]:
            p = pred[0]
        else:
            p = pred[0].T

        C, N = p.shape
        expected_min = 4 + 1 + self.kpts * 3
        if C < expected_min:
            raise RuntimeError("Pose output channels too small: C={}, expected>={}".format(C, expected_min))

        boxes_xywh = p[0:4, :]
        score = p[4, :]
        kps = p[5:5 + self.kpts * 3, :]

        keep = score >= self.conf
        score = score[keep]
        boxes_xywh = boxes_xywh[:, keep]
        kps = kps[:, keep]

        if boxes_xywh.size == 0:
            return [], None

        x_c, y_c, w, h = boxes_xywh
        x1 = x_c - w / 2
        y1 = y_c - h / 2
        x2 = x_c + w / 2
        y2 = y_c + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        keep_idx = nms_xyxy(boxes, score, self.iou)

        det_items = []
        skeletons = []

        for i in keep_idx:
            bx = boxes[i]
            bx0 = (bx - np.array([padx, pady, padx, pady], dtype=np.float32)) / max(r, 1e-9)

            x1o, y1o, x2o, y2o = clamp_xyxy(bx0[0], bx0[1], bx0[2], bx0[3], W0, H0)

            pts = kps[:, i].reshape(self.kpts, 3)
            kp_list = []

            for (kx, ky, kc) in pts:
                kx0 = (kx - padx) / max(r, 1e-9)
                ky0 = (ky - pady) / max(r, 1e-9)
                kp_list.append({"x": float(kx0), "y": float(ky0), "conf": float(kc)})

            det_items.append({
                "cls_name": "skeleton",
                "conf": float(score[i]),
                "box": {"x1": x1o, "y1": y1o, "x2": x2o, "y2": y2o},
                "box_norm": box_norm_xyxy(x1o, y1o, x2o, y2o, W0, H0),
            })

            skeletons.append({
                "conf": float(score[i]),
                "box": {"x1": x1o, "y1": y1o, "x2": x2o, "y2": y2o},
                "keypoints": kp_list,
            })

        pose = {"format": "xy", "skeletons": skeletons} if skeletons else None
        return det_items, pose


# -----------------------------
# In-process inference API
# -----------------------------
class TRTInfer(object):
    """
    In-process inference wrapper.
    Use: infer.infer_multitask(bgr, meta_dict) -> event dict
    """
    def __init__(
        self,
        det_engine_path: str,
        pose_engine_path: Optional[str] = None,
        imgsz: int = 512,
        conf: float = 0.25,
        iou: float = 0.45,
        allowed=("person", "car", "motorcycle", "truck"),
        kpts: int = 17,
        model_id: str = "yolo-trt",
        device_id: int = 0,
        enable_pose: bool = False,
        nms_topk: int = 100,
    ):
        self.model_id = model_id
        self.device_id = int(device_id)
        self.enable_pose = bool(enable_pose)

        # Ensure context exists for this thread before building engines
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
        self.pose_runner = None
        if self.enable_pose and pose_engine_path:
            self.pose_runner = YoloV8PoseTRT(
                pose_engine_path, imgsz=imgsz, conf=conf, iou=iou, kpts=kpts, device_id=self.device_id
            )

    def infer_multitask(self, bgr: np.ndarray, meta: Dict) -> Dict:
        t0 = time.perf_counter()

        camera_uuid = str(meta.get("camera_uuid", "unknown"))
        channel_id = meta.get("channel_id")
        frame_ts_ms = int(meta.get("frame_ts_ms", int(time.time() * 1000)))
        frame_seq = int(meta.get("frame_seq", 0))

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[TRT] Starting inference for %s (seq=%s)", camera_uuid, frame_seq)

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
        det_engine = os.path.join(base_dir, "models", "yolov8n.engine")
    elif not os.path.isabs(det_engine) and not os.path.exists(det_engine):
        candidate = os.path.join(base_dir, det_engine)
        if os.path.exists(candidate):
            det_engine = candidate

    imgsz = int(os.getenv("IMG_SZ", "512"))
    conf = float(os.getenv("CONF", "0.25"))
    iou = float(os.getenv("IOU", "0.45"))
    device_id = int(os.getenv("CUDA_DEVICE", "0"))
    nms_topk = int(os.getenv("NMS_TOPK", "100"))

    return TRTInfer(
        det_engine_path=det_engine,
        pose_engine_path=None,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device_id=device_id,
        enable_pose=False,
        nms_topk=nms_topk,
    )