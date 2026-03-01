from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple,Literal
import uuid
from pydantic import BaseModel, EmailStr, Field, field_validator,SecretStr
from datetime import datetime, timezone
from pydantic import BaseModel, EmailStr, Field, SecretStr, ConfigDict, field_validator, model_validator
import re

from abc import ABC


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
        
        
        
class ChannelConfig(ABC):
    """
    Abstract base class for channel-specific configurations. This serves as a
    marker interface for all concrete channel config implementations.
    """

    channel_id: str
    camera_uuid: Optional[uuid.UUID]
    rtsp_url: str
    enabled: bool
    sample_fps: float
    decode_backend: Literal["gstreamer", "opencv"]
    resize: Optional[Tuple[int, int]]
    reconnect_base_ms: int
    reconnect_max_ms: int
    emit_format: Literal["raw", "jpeg"]
    jpeg_quality: int



def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

PASSWORD_MIN_LEN = 8

class SignupRequestCode(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    user_name: str = Field(..., min_length=1, max_length=100)
    user_email: EmailStr

    @field_validator("user_email")
    @classmethod
    def normalize_email(cls, v: EmailStr) -> EmailStr:
        return EmailStr(str(v).lower())


class SignupConfirm(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    user_name: str = Field(..., min_length=1, max_length=100)
    user_email: EmailStr
    code: str = Field(..., min_length=6, max_length=10)  # e.g., "123456"
    password: SecretStr = Field(..., min_length=PASSWORD_MIN_LEN)

    @field_validator("user_email")
    @classmethod
    def normalize_email(cls, v: EmailStr) -> EmailStr:
        return EmailStr(str(v).lower())

    @field_validator("password")
    @classmethod
    def password_complexity(cls, v: SecretStr) -> SecretStr:
        pw = v.get_secret_value()
        if len(pw) < PASSWORD_MIN_LEN:
            raise ValueError("Password must be at least 8 characters")
        if not re.search(r"[A-Z]", pw):
            raise ValueError("Password must contain at least 1 uppercase letter")
        if not re.search(r"\d", pw):
            raise ValueError("Password must contain at least 1 digit")
        return v


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    user_email: EmailStr
    password: SecretStr
    # If you want the request time too (optional):
    login_at: datetime = Field(default_factory=utc_now)

    @field_validator("user_email")
    @classmethod
    def normalize_email(cls, v: EmailStr) -> EmailStr:
        return EmailStr(str(v).lower())