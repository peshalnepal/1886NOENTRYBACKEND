# agents/application/models/yolo_model.py

import asyncio
import time
from typing import Any, Dict, List, Literal, Optional, Tuple
import httpx
import numpy as np
import cv2
from pydantic import BaseModel, Field
from application.models.yolo_config import YoloModelConfig
from domain.events import (
    RTSPEvent,
    DetectionBox,
    DetectionItem,
    DetectionsProducedEvent,
    InferenceFailedEvent,
    PoseResult,       
    PoseKeypoint,     
    SkeletonItem,     

)


from domain.model import VisionModel  # or from domain.model import VisionTask
from application.models.vision_config import VisionTask

import logging
logger = logging.getLogger(__name__)
class YoloMultiTaskModel(VisionModel):

    def __init__(self, cfg: YoloModelConfig):
        if isinstance(cfg, dict):
            cfg = YoloModelConfig.model_validate(cfg)
        self.cfg = cfg
        # Decide which parts are enabled based on enum task
        self._do_det = cfg.task in (VisionTask.OBJECT_DETECTION, VisionTask.MULTI_TASK)
        self._do_pose = cfg.task in (VisionTask.POSE_ESTIMATION, VisionTask.MULTI_TASK)

        self._det_model = None
        self._pose_model = None
        self._allowed_det_class_ids: Optional[set[int]] = None

        # Load ONLY what we need
        # if self._do_det:
        #     if not self.cfg.det_weights:
        #         raise ValueError(
        #             f"VisionTask={cfg.task} requires det_weights, but det_weights is empty/None."
        #         )
        #     self._det_model = YOLO(self.cfg.det_weights)
        #     self._allowed_det_class_ids = self._resolve_allowed_class_ids(
        #         getattr(self._det_model, "names", None),
        #         allowed_names=set(self.cfg.allowed_det_labels),
        #     )

        # if self._do_pose:
        #     if not self.cfg.pose_weights:
        #         raise ValueError(
        #             f"VisionTask={cfg.task} requires pose_weights, but pose_weights is empty/None."
        #         )
        #     self._pose_model = YOLO(self.cfg.pose_weights)

        self._predict_lock = asyncio.Lock()
        self._pose_interval_ms: int = int(getattr(cfg, "pose_interval_ms", 100))
        self._last_pose_run_ms: Dict[str, int] = {}

    def _should_run_pose_now(self, camera_uuid: str, ts_ms: int) -> bool:
        """
        Per-camera throttle: True only if >= pose_interval_ms since last pose run.
        Must be called on the event-loop thread (we call it inside _predict_lock).
        """
        last = self._last_pose_run_ms.get(camera_uuid)
        if last is None or (ts_ms - last) >= self._pose_interval_ms:
            self._last_pose_run_ms[camera_uuid] = ts_ms
            return True
        return False

    async def _remote_infer(self, rtsp_ev: RTSPEvent,do_pose_now) -> dict:
        """
        Call Jetson TRT service. Uses encoded jpeg if available (best).
        """
        base = (self.cfg.remote_url or "").rstrip("/")
        if self._do_det and self._do_pose and do_pose_now:
            url = f"{base}/v1/multitask"
        elif self._do_pose and do_pose_now:
            url = f"{base}/v1/pose"
        else:
            url = f"{base}/v1/detect"

        # Use existing JPEG bytes if you already emit jpeg from VideoChannel
        img_bytes = getattr(rtsp_ev, "encoded", None)
        if img_bytes is None:
            frame = getattr(rtsp_ev, "frame", None)
            if frame is None:
                raise ValueError("No frame/encoded available to send to remote inference.")
            ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                raise ValueError("Failed to JPEG-encode frame for remote inference.")
            img_bytes = enc.tobytes()

        data = {
            "camera_uuid": str(getattr(rtsp_ev, "camera_uuid", "unknown")),
            "frame_ts_ms": str(int(getattr(rtsp_ev, "ts_ms", 0) or 0)),
            "frame_seq": str(int(getattr(rtsp_ev, "seq", 0) or 0)),
            "channel_id": str(getattr(rtsp_ev, "channel_id", "") or ""),
            "model_id": str(getattr(self.cfg, "model_id", "yolo-remote") or "yolo-remote"),
        }

        files = {
            "image": ("frame.jpg", img_bytes, "image/jpeg")
        }

        timeout = httpx.Timeout(connect=2.0, read=5.0, write=5.0, pool=2.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, data=data, files=files)
            r.raise_for_status()
            return r.json()

    # def _predict_all(self, frame: np.ndarray, *, do_pose_now: bool) -> Tuple[List[DetectionItem], List[DetectionItem], Optional[PoseResult]]:
    #     det_items = self._predict_det(frame) if self._do_det else []
    #     pose_items: List[DetectionItem] = []
    #     pose_result: Optional[PoseResult] = None

    #     # Pose runs only if enabled AND throttle allows it
    #     if self._do_pose and do_pose_now:
    #         # Optional: only run pose when a person is present (saves a lot of compute)
    #         if (not self._do_det) or any(d.cls_name == "person" for d in det_items):
    #             pose_items, pose_result = self._predict_pose(frame)

    #     return det_items, pose_items, pose_result
    
    async def infer(self, rtsp_ev: RTSPEvent):
        t0 = time.perf_counter()
        cam = str(rtsp_ev.camera_uuid)
        ts = int(rtsp_ev.ts_ms)
        seq = int(rtsp_ev.seq)
        
        detection_enabled = getattr(rtsp_ev, "detection_enabled", False)
        if detection_enabled is None:
            detection_enabled = bool(getattr(self.cfg, "detection_enabled", False))

        # If disabled -> do not run YOLO at all
        if not detection_enabled:
            # Return "empty detections" event (better than InferenceFailed for UI)
            return DetectionsProducedEvent(
                channel_id=getattr(rtsp_ev, "channel_id", None),
                camera_uuid=cam,
                model_id=self.cfg.model_id,
                frame_ts_ms=ts,
                frame_seq=seq,
                detections=[],
                inference_ms=0,
                pose=None,
            )
        try:
            frame = self._get_frame_bgr(rtsp_ev)
            if frame is None:
                return InferenceFailedEvent(
                    camera_uuid=cam,
                    model_id=self.cfg.model_id,
                    frame_ts_ms=ts,
                    frame_seq=seq,
                    reason="No frame data found (rtsp_ev.frame is None and decode from encoded failed).",
                )

            det_items: List[DetectionItem] = []
            pose_items: List[DetectionItem] = []
            pose_result: Optional[PoseResult] = None

            async with self._predict_lock:
                do_pose_now = self._should_run_pose_now(cam, ts) if self._do_pose else False
                results= results = await self._remote_infer(rtsp_ev, do_pose_now)
                logger.info(results)
                det_items=results["detections"]
                pose_result=results["pose"]
            detections: List[DetectionItem] = []
            detections.extend(det_items)

            inference_ms = int((time.perf_counter() - t0) * 1000)
            
            return DetectionsProducedEvent(
                channel_id=getattr(rtsp_ev, "channel_id", None),
                camera_uuid=cam,
                model_id=self.cfg.model_id,
                frame_ts_ms=ts,
                frame_seq=seq,
                detections=detections,
                inference_ms=inference_ms,
                pose=pose_result,  # will be None for 9/10 frames (per camera) if pose enabled
            )

        except Exception as e:
            inference_ms = int((time.perf_counter() - t0) * 1000)
            return InferenceFailedEvent(
                camera_uuid=cam,
                model_id=self.cfg.model_id,
                frame_ts_ms=ts,
                frame_seq=seq,
                reason=f"{type(e).__name__}: {e} (after {inference_ms} ms)",
            )

    def _get_frame_bgr(self, rtsp_ev: RTSPEvent) -> Optional[np.ndarray]:
        frame = getattr(rtsp_ev, "frame", None)
        if frame is not None:
            return frame

        enc = getattr(rtsp_ev, "encoded", None)
        fmt = getattr(rtsp_ev, "format", None)
        if enc is None:
            return None

        if fmt in ("jpeg", "png"):
            arr = np.frombuffer(enc, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img

        return None


    def _predict_det(self, frame_bgr: np.ndarray) -> List[DetectionItem]:
        """
        Detection model => ONLY person/car.
        """
        results = self._det_model.predict(
            source=frame_bgr,
            device=self.cfg.device,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            max_det=self.cfg.max_det,
            half=self.cfg.half,
            verbose=False,
        )

        if not results:
            return []

        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        names = getattr(r0, "names", None) or getattr(self._det_model, "names", {}) or {}
        items: List[DetectionItem] = []

        # tensors -> cpu numpy
        xyxy = boxes.xyxy
        cls = boxes.cls
        conf = boxes.conf

        xyxy_np = xyxy.detach().cpu().numpy() if hasattr(xyxy, "detach") else np.asarray(xyxy)
        cls_np = cls.detach().cpu().numpy() if hasattr(cls, "detach") else np.asarray(cls)
        conf_np = conf.detach().cpu().numpy() if hasattr(conf, "detach") else np.asarray(conf)

        H, W = frame_bgr.shape[:2]

        for i in range(xyxy_np.shape[0]):
            cls_id = int(cls_np[i])
            if self._allowed_det_class_ids is not None and cls_id not in self._allowed_det_class_ids:
                continue

            label = names.get(cls_id, str(cls_id))
            if label not in self.cfg.allowed_det_labels:
                continue

            x1, y1, x2, y2 = xyxy_np[i].tolist()
            box = self._clamp_box(int(x1), int(y1), int(x2), int(y2), W, H)

            items.append(
                DetectionItem(
                    cls_name=label,
                    conf=float(conf_np[i]),
                    box=DetectionBox(**box),
                )
            )

        return items

    def _predict_pose(self, frame_bgr: np.ndarray) -> Tuple[List[DetectionItem], Optional[PoseResult]]:
        """
        Pose model => label detections as "skeleton" and (optionally) include keypoints in payload.
        """
        results = self._pose_model.predict(
            source=frame_bgr,
            device=self.cfg.device,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            max_det=self.cfg.max_det,
            half=self.cfg.half,
            verbose=False,
        )

        if not results:
            return [], None

        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        kps = getattr(r0, "keypoints", None)

        if boxes is None or len(boxes) == 0:
            return [], {}

        # tensors -> cpu numpy
        xyxy = boxes.xyxy
        conf = boxes.conf
        xyxy_np = xyxy.detach().cpu().numpy() if hasattr(xyxy, "detach") else np.asarray(xyxy)
        conf_np = conf.detach().cpu().numpy() if hasattr(conf, "detach") else np.asarray(conf)

        H, W = frame_bgr.shape[:2]

        items: List[DetectionItem] = []
        skeletons: List[SkeletonItem] = []

        # keypoints extraction (robust to ultralytics version)
        kp_xy = None
        kp_conf = None
        if kps is not None:
            if self.cfg.keypoints_format == "xyn" and hasattr(kps, "xyn"):
                kp_xy = kps.xyn
            elif hasattr(kps, "xy"):
                kp_xy = kps.xy
            else:
                kp_xy = getattr(kps, "data", None)

            kp_conf = getattr(kps, "conf", None)

            if kp_xy is not None and hasattr(kp_xy, "detach"):
                kp_xy = kp_xy.detach().cpu().numpy()
            elif kp_xy is not None:
                kp_xy = np.asarray(kp_xy)

            if kp_conf is not None and hasattr(kp_conf, "detach"):
                kp_conf = kp_conf.detach().cpu().numpy()
            elif kp_conf is not None:
                kp_conf = np.asarray(kp_conf)
                
        for i in range(xyxy_np.shape[0]):
            x1, y1, x2, y2 = xyxy_np[i].tolist()
            box_dict = self._clamp_box(int(x1), int(y1), int(x2), int(y2), W, H)
            box = DetectionBox(**box_dict)

            # keep "skeleton" as a normal DetectionItem too (for overlays/filtering)
            items.append(
                DetectionItem(
                    cls_name=self.cfg.skeleton_label,
                    conf=float(conf_np[i]),
                    box=box,
                )
            )

            kps_list: List[PoseKeypoint] = []
            if self.cfg.include_keypoints_in_payload and kp_xy is not None and i < len(kp_xy):
                pts = kp_xy[i]
                confs = kp_conf[i] if (kp_conf is not None and i < len(kp_conf)) else None
                for j in range(len(pts)):
                    x, y = float(pts[j][0]), float(pts[j][1])
                    c = float(confs[j]) if confs is not None else None
                    kps_list.append(PoseKeypoint(x=x, y=y, conf=c))

            skeletons.append(
                SkeletonItem(
                    conf=float(conf_np[i]),
                    box=box,
                    keypoints=kps_list,
                )
            )

        pose_result = PoseResult(
            format=self.cfg.keypoints_format,
            skeletons=skeletons,
        ) if skeletons else None

        return items, pose_result


    def _resolve_allowed_class_ids(self, names_obj: Any, allowed_names: set[str]) -> Optional[set[int]]:
        """
        Convert allowed label names -> class ids, if possible.
        Ultralytics uses names like {0:"person", 1:"bicycle", ...}.
        """
        if not names_obj:
            # fallback to COCO ids if model doesn't expose names (person=0, car=2)
            coco = {"person": 0, "car": 2}
            ids = {coco[n] for n in allowed_names if n in coco}
            return ids if ids else None

        # names might be dict or list
        if isinstance(names_obj, dict):
            items = names_obj.items()
        else:
            items = enumerate(list(names_obj))

        out = set()
        for k, v in items:
            if str(v) in allowed_names:
                out.add(int(k))
        return out if out else None

    def _clamp_box(self, x1: int, y1: int, x2: int, y2: int, W: int, H: int) -> Dict[str, int]:
        x1 = max(0, min(x1, W - 1))
        y1 = max(0, min(y1, H - 1))
        x2 = max(0, min(x2, W - 1))
        y2 = max(0, min(y2, H - 1))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
