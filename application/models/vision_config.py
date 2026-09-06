from __future__ import annotations

from enum import Enum
from typing import Tuple

from pydantic import BaseModel, ConfigDict, Field


class VisionTask(str, Enum):
    OBJECT_DETECTION = "object_detection"
    POSE_ESTIMATION = "pose_estimation"
    MULTI_TASK = "multi_task"  # detection + pose


class VisionModelConfig(BaseModel):
    """Domain-level model config, free of vendor-specific fields so it survives
    swapping YOLO for TensorRT, DeepStream or anything else."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(default="vision-model")
    task: VisionTask = Field(default=VisionTask.MULTI_TASK)
    device: str = Field(default="cuda:0")
    imgsz: int = Field(default=640)
    conf: float = Field(default=0.25)
    iou: float = Field(default=0.45)
    max_det: int = Field(default=50)
    allowed_labels: Tuple[str, ...] = Field(default_factory=tuple)
