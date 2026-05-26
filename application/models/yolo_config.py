
import asyncio
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import cv2
from pydantic import BaseModel, Field
from application.models.vision_config import VisionModelConfig, VisionTask

class YoloModelConfig(VisionModelConfig):
    """
    YOLO wrapper config.

    We use:
      - det_weights for object detection (person + car)
      - pose_weights for pose/skeleton (person boxes + keypoints => we label as "skeleton")

    Notes:
      - You can pass .pt, .onnx, or .engine (TensorRT) paths depending on your environment.
      - device examples: "cpu", "cuda:0"
    """
    task: VisionTask = VisionTask.MULTI_TASK
    det_weights: str = Field(default="yolov8n.pt")
    pose_weights: Optional[str] = Field(default="yolov8n-pose.pt")
    half: bool = Field(default=False)

    allowed_det_labels: Tuple[str, ...] = ("person", "car")
    skeleton_label: str = "skeleton"
    remote_url:str="http://0.0.0.0:8081"
    include_keypoints_in_payload: bool = True
    keypoints_format: Literal["xy", "xyn"] = "xy"

