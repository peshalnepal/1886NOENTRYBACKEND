"""Pydantic models and dataclasses used across the notification package.

Extracted verbatim from the former monolithic application/services/notification.py.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from application.repositories.notification_repository import (
    CameraContext,
    SitePrerecordSettings,
)


@dataclass(frozen=True)
class CameraMode:
    playback_enabled: bool = True
    detection_enabled: bool = True
    notification_enabled: bool = True


class NotificationMessage(BaseModel):
    user_id: int
    id: str
    ts_ms: int
    camera_uuid: str
    site_uuid: str
    site_name: str

    title: str
    body: str
    alert_type: str = "item_detected"  # item_detected | roi_enter | detection_summary

    cls_names: List[str] = Field(default_factory=list)
    max_conf: Optional[float] = None
    frame_w: Optional[int] = None
    frame_h: Optional[int] = None
    frame_seq: Optional[int] = None
    detections: List[Dict[str, Any]] = Field(default_factory=list)

    device_name: Optional[str] = None
    camera_name: Optional[str] = None
    roi_id: Optional[str] = None
    track_id: Optional[int] = None
    image_url: Optional[str] = None
    clip_url: Optional[str] = None
    clip_status: Optional[str] = None
    db_id: Optional[int] = None  # Set after DB persistence so the frontend can delete by ID


@dataclass(frozen=True)
class BufferedNotification:
    msg: NotificationMessage
    ctx: CameraContext
    extra_payload: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class SitePrerecordPlan:
    settings: SitePrerecordSettings
    ordered_camera_uuids: List[uuid.UUID]
    contexts_by_camera: Dict[uuid.UUID, CameraContext]


class EmailConfig(BaseModel):
    enabled: bool = True

    smtp_host: str
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_pass: Optional[str] = None
    use_tls: bool = True

    from_email: str
    to_emails: List[str] = Field(default_factory=list)

    subject_prefix: str = "[1886NOENTRY Alert]"
    dashboard_base_url: Optional[str] = None
    camera_path_template: str = "/app?camera={camera_uuid}"
