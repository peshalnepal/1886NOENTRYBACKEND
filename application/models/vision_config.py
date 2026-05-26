from __future__ import annotations

from enum import Enum
from typing import Tuple

from pydantic import BaseModel, Field


class VisionTask(str, Enum):
    OBJECT_DETECTION = "object_detection"
    POSE_ESTIMATION = "pose_estimation"
    MULTI_TASK = "multi_task"  # det + pose, etc.


class VisionModelConfig(BaseModel):
    """
    Domain-level model config (NO vendor-specific fields here).
    This stays stable even if you swap YOLO -> TensorRT -> DeepStream -> etc.
    """
    model_id: str = Field(default="vision-model")
    task: VisionTask = Field(default=VisionTask.MULTI_TASK)
    device: str = Field(default="cuda:0")
    imgsz: int = Field(default=640)
    conf: float = Field(default=0.25)
    iou: float = Field(default=0.45)
    max_det: int = Field(default=50)
    allowed_labels: Tuple[str, ...] = Field(default_factory=tuple)

    class Config:
        extra = "forbid"
