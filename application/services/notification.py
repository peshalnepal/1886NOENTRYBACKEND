# application/notifications/notification_service.py

import asyncio
from collections import defaultdict, deque
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import html
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
import uuid
from sqlalchemy import update
from sqlalchemy.exc import OperationalError
from core.database_orm import Notification

from pydantic import BaseModel, Field
import numpy as np

from domain.events import DetectionsProducedEvent, DetectionItem
from application.services.tracker import MultiCameraByteTrack, ROIAlertEngine, ROI
from sqlalchemy import and_, update, select
from application.services.alert_image_storage import (
    AlertImageStorageService,
    extract_image_storage_key,
)
# NEW: repository
from application.repositories.notification_repository import (
    NotificationRepository,
    CameraContext,
    SitePrerecordSettings,
    dt_from_ts_ms,
)
from application.services.alert_image_storage import AlertImageStorageService
from application.services.clip_storage import EventClipService

SMTP_USERNAME = os.environ.get("SMTP_USERNAME")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None else int(default)
    except (TypeError, ValueError):
        value = int(default)
    return max(minimum, value)


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    try:
        value = float(raw) if raw is not None else float(default)
    except (TypeError, ValueError):
        value = float(default)
    return max(minimum, value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _json_safe(value.model_dump())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_positive_int(value: Any) -> Optional[int]:
    parsed = _coerce_int(value)
    if parsed is None or parsed <= 0:
        return None
    return parsed


def _normalize_overlay_box(raw_box: Any) -> Optional[Dict[str, int]]:
    if isinstance(raw_box, dict):
        keys = ("x1", "y1", "x2", "y2")
        if not all(key in raw_box for key in keys):
            return None
        values = tuple(_coerce_int(raw_box.get(key)) for key in keys)
    elif isinstance(raw_box, (list, tuple)) and len(raw_box) >= 4:
        values = tuple(_coerce_int(raw_box[idx]) for idx in range(4))
    else:
        values = tuple(_coerce_int(getattr(raw_box, key, None)) for key in ("x1", "y1", "x2", "y2"))

    if any(value is None for value in values):
        return None

    x1, y1, x2, y2 = values
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _normalize_box_norm(raw: Any) -> Optional[Dict[str, float]]:
    if isinstance(raw, dict):
        raw_dict = raw
    elif hasattr(raw, "x") and hasattr(raw, "y") and hasattr(raw, "w") and hasattr(raw, "h"):
        raw_dict = {"x": raw.x, "y": raw.y, "w": raw.w, "h": raw.h}
    else:
        return None
    try:
        x = float(raw_dict["x"])
        y = float(raw_dict["y"])
        w = float(raw_dict["w"])
        h = float(raw_dict["h"])
    except (KeyError, TypeError, ValueError):
        return None
    return {"x": x, "y": y, "w": w, "h": h}


def _normalize_overlay_detection(raw_detection: Any) -> Optional[Dict[str, Any]]:
    if isinstance(raw_detection, dict):
        box = _normalize_overlay_box(raw_detection.get("box") or raw_detection.get("bbox"))
        cls_name = str(raw_detection.get("cls_name") or raw_detection.get("class") or "obj")
        conf = float(raw_detection.get("conf", 0.0) or 0.0)
        raw_box_norm = raw_detection.get("box_norm")
    else:
        box = _normalize_overlay_box(getattr(raw_detection, "box", None) or getattr(raw_detection, "bbox", None))
        cls_name = str(
            getattr(raw_detection, "cls_name", None)
            or getattr(raw_detection, "class_name", None)
            or getattr(raw_detection, "class", None)
            or "obj"
        )
        conf = float(getattr(raw_detection, "conf", 0.0) or 0.0)
        raw_box_norm = getattr(raw_detection, "box_norm", None)

    if box is None:
        return None

    result: Dict[str, Any] = {
        "cls_name": cls_name,
        "conf": conf,
        "box": box,
    }
    box_norm = _normalize_box_norm(raw_box_norm)
    if box_norm is not None:
        result["box_norm"] = box_norm
    return result


def _normalize_overlay_frame(
    *,
    camera_uuid: Optional[str],
    frame_ts_ms: Any,
    frame_seq: Any,
    frame_w: Any,
    frame_h: Any,
    detections: Any,
) -> Optional[Dict[str, Any]]:
    ts_ms = _coerce_int(frame_ts_ms)
    seq = _coerce_int(frame_seq)
    if ts_ms is None or seq is None:
        return None

    normalized_detections: List[Dict[str, Any]] = []
    seen = set()
    for raw_detection in list(detections or []):
        normalized = _normalize_overlay_detection(raw_detection)
        if normalized is None:
            continue
        box = normalized["box"]
        key = (
            normalized["cls_name"],
            normalized["conf"],
            box["x1"],
            box["y1"],
            box["x2"],
            box["y2"],
        )
        if key in seen:
            continue
        seen.add(key)
        normalized_detections.append(normalized)

    if not normalized_detections:
        return None

    frame: Dict[str, Any] = {
        "frame_ts_ms": int(ts_ms),
        "frame_seq": int(seq),
        "detections": normalized_detections,
    }
    if camera_uuid:
        frame["camera_uuid"] = str(camera_uuid)
    normalized_w = _coerce_positive_int(frame_w)
    normalized_h = _coerce_positive_int(frame_h)
    if normalized_w is not None:
        frame["frame_w"] = normalized_w
    if normalized_h is not None:
        frame["frame_h"] = normalized_h
    return frame


def _merge_overlay_frames(*sources: Any) -> List[Dict[str, Any]]:
    merged: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for source in sources:
        if not isinstance(source, list):
            continue
        for raw_frame in source:
            if not isinstance(raw_frame, dict):
                continue
            normalized = _normalize_overlay_frame(
                camera_uuid=raw_frame.get("camera_uuid"),
                frame_ts_ms=raw_frame.get("frame_ts_ms"),
                frame_seq=raw_frame.get("frame_seq"),
                frame_w=raw_frame.get("frame_w"),
                frame_h=raw_frame.get("frame_h"),
                detections=raw_frame.get("detections"),
            )
            if normalized is None:
                continue
            merged[(normalized["frame_ts_ms"], normalized["frame_seq"])] = normalized

    return sorted(
        merged.values(),
        key=lambda item: (int(item.get("frame_ts_ms", 0)), int(item.get("frame_seq", 0))),
    )


def _select_overlay_reference_frame(
    frames: List[Dict[str, Any]],
    *,
    preferred_ts_ms: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    if not frames:
        return None
    if preferred_ts_ms is None:
        return frames[-1]
    return min(
        frames,
        key=lambda item: (
            abs(int(item.get("frame_ts_ms", 0)) - int(preferred_ts_ms)),
            abs(int(item.get("frame_seq", 0))),
            int(item.get("frame_ts_ms", 0)),
        ),
    )


def _build_overlay_payload_from_frames(
    *,
    camera_uuid: str,
    frames: List[Dict[str, Any]],
    preferred_ts_ms: Optional[int] = None,
    clip_start_time: Optional[datetime] = None,
    clip_end_time: Optional[datetime] = None,
    timeline_source: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    merged_frames = _merge_overlay_frames(frames)
    reference = _select_overlay_reference_frame(merged_frames, preferred_ts_ms=preferred_ts_ms)
    if reference is None:
        return None

    payload: Dict[str, Any] = {
        "camera_uuid": str(camera_uuid),
        "frame_ts_ms": int(reference["frame_ts_ms"]),
        "frame_seq": int(reference["frame_seq"]),
        "detections": list(reference.get("detections") or []),
        "frames": merged_frames,
    }
    if reference.get("frame_w") is not None:
        payload["frame_w"] = int(reference["frame_w"])
    if reference.get("frame_h") is not None:
        payload["frame_h"] = int(reference["frame_h"])
    if clip_start_time is not None:
        payload["clip_start_time"] = clip_start_time.astimezone(timezone.utc).isoformat()
    if clip_end_time is not None:
        payload["clip_end_time"] = clip_end_time.astimezone(timezone.utc).isoformat()
    if timeline_source:
        payload["timeline_source"] = str(timeline_source)
    return payload


def _parse_utc_datetime(raw: Any) -> Optional[datetime]:
    if isinstance(raw, datetime):
        dt = raw
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _event_overlay_payload(
    det_ev: DetectionsProducedEvent,
    *,
    frame_w: Optional[int] = None,
    frame_h: Optional[int] = None,
) -> Dict[str, Any]:
    frame = _normalize_overlay_frame(
        camera_uuid=str(det_ev.camera_uuid),
        frame_ts_ms=det_ev.frame_ts_ms,
        frame_seq=det_ev.frame_seq,
        frame_w=frame_w,
        frame_h=frame_h,
        detections=list(det_ev.detections or []),
    )
    if frame is None:
        payload: Dict[str, Any] = {
            "camera_uuid": str(det_ev.camera_uuid),
            "frame_ts_ms": int(det_ev.frame_ts_ms),
            "frame_seq": int(det_ev.frame_seq),
            "detections": [],
            "frames": [],
        }
        if frame_w is not None:
            payload["frame_w"] = int(frame_w)
        if frame_h is not None:
            payload["frame_h"] = int(frame_h)
        return payload

    return _build_overlay_payload_from_frames(
        camera_uuid=str(det_ev.camera_uuid),
        frames=[frame],
        preferred_ts_ms=int(det_ev.frame_ts_ms),
        timeline_source="event",
    ) or {
        "camera_uuid": str(det_ev.camera_uuid),
        "frame_ts_ms": int(det_ev.frame_ts_ms),
        "frame_seq": int(det_ev.frame_seq),
        "detections": [],
        "frames": [],
    }

@dataclass(frozen=True)
class BufferedDeletion:
    user_id: int
    notification_ids: Tuple[int, ...]

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


class WebNotificationHub:
    def __init__(self, max_q: int = 200):
        self._subs_by_user: Dict[int, Set[asyncio.Queue[NotificationMessage]]] = {}
        self._lock = asyncio.Lock()
        self._max_q = max_q

    async def subscribe(self, *, user_id: int) -> asyncio.Queue[NotificationMessage]:
        q: asyncio.Queue[NotificationMessage] = asyncio.Queue(maxsize=self._max_q)
        async with self._lock:
            uid = int(user_id)
            bucket = self._subs_by_user.get(uid)
            if bucket is None:
                bucket = set()
                self._subs_by_user[uid] = bucket
            bucket.add(q)
        return q

    async def unsubscribe(self, *, user_id: int, q: asyncio.Queue[NotificationMessage]) -> None:
        async with self._lock:
            uid = int(user_id)
            bucket = self._subs_by_user.get(uid)
            if not bucket:
                return
            bucket.discard(q)
            if not bucket:
                self._subs_by_user.pop(uid, None)

    async def publish(self, msg: NotificationMessage) -> None:
        uid = int(msg.user_id)
        async with self._lock:
            subs = list(self._subs_by_user.get(uid, set()))

        for q in subs:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                try:
                    _ = q.get_nowait()
                except Exception:
                    pass
                try:
                    q.put_nowait(msg)
                except Exception:
                    pass


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


class EmailNotifier:
    def __init__(self, cfg: EmailConfig):
        self.cfg = cfg

    async def send(self, msg: NotificationMessage, to_emails: Optional[List[str]] = None) -> None:
        if not self.cfg.enabled:
            return

        recipients = (to_emails or []) or self.cfg.to_emails
        if not recipients:
            return

        await asyncio.to_thread(self._send_sync, msg, recipients)

    async def send_digest(
        self,
        messages: List[NotificationMessage],
        to_emails: Optional[List[str]] = None,
    ) -> None:
        ordered = sorted(messages or [], key=lambda msg: int(msg.ts_ms), reverse=True)
        if not ordered:
            return
        if len(ordered) == 1:
            await self.send(ordered[0], to_emails=to_emails)
            return

        if not self.cfg.enabled:
            return

        recipients = (to_emails or []) or self.cfg.to_emails
        if not recipients:
            return

        await asyncio.to_thread(self._send_digest_sync, ordered, recipients)

    def _send_sync(self, msg: NotificationMessage, to_emails: List[str]) -> None:
        subject = f"{self.cfg.subject_prefix} {msg.site_name} — {msg.title}"

        text_body = self._render_text(msg)
        html_body = self._render_html(msg)

        m = MIMEMultipart("alternative")
        m["Subject"] = subject
        m["From"] = self.cfg.from_email
        m["To"] = ", ".join(to_emails)

        m.attach(MIMEText(text_body, "plain", "utf-8"))
        m.attach(MIMEText(html_body, "html", "utf-8"))

        with smtplib.SMTP(self.cfg.smtp_host, self.cfg.smtp_port, timeout=10) as server:
            if self.cfg.use_tls:
                server.starttls()
            if self.cfg.smtp_user and self.cfg.smtp_pass:
                server.login(self.cfg.smtp_user, self.cfg.smtp_pass)
            server.sendmail(self.cfg.from_email, to_emails, m.as_string())

    def _send_digest_sync(self, messages: List[NotificationMessage], to_emails: List[str]) -> None:
        first = messages[0]
        total = len(messages)
        subject = f"{self.cfg.subject_prefix} {first.site_name} — {total} alerts"

        text_body = self._render_digest_text(messages)
        html_body = self._render_digest_html(messages)

        m = MIMEMultipart("alternative")
        m["Subject"] = subject
        m["From"] = self.cfg.from_email
        m["To"] = ", ".join(to_emails)

        m.attach(MIMEText(text_body, "plain", "utf-8"))
        m.attach(MIMEText(html_body, "html", "utf-8"))

        with smtplib.SMTP(self.cfg.smtp_host, self.cfg.smtp_port, timeout=10) as server:
            if self.cfg.use_tls:
                server.starttls()
            if self.cfg.smtp_user and self.cfg.smtp_pass:
                server.login(self.cfg.smtp_user, self.cfg.smtp_pass)
            server.sendmail(self.cfg.from_email, to_emails, m.as_string())

    def _render_text(self, msg: NotificationMessage) -> str:
        ts = self._fmt_ts(msg.ts_ms)
        classes = ", ".join(msg.cls_names) if msg.cls_names else "unknown"
        conf = f"{msg.max_conf:.2f}" if msg.max_conf is not None else "n/a"

        lines = [
            "1886NOENTRY — AI Monitor Alert",
            "",
            f"Site: {msg.site_name}",
            f"Camera: {msg.camera_name or msg.camera_uuid}",
            f"Time (UTC): {ts}",
            f"Detected: {classes}",
            f"Max confidence: {conf}",
            "",
            msg.body,
        ]

        if self.cfg.dashboard_base_url:
            url = self._camera_url(msg.camera_uuid)
            lines += ["", f"Open dashboard: {url}"]

        lines += [
            "",
            "If you did not expect this email, check your notification settings in the 1886NOENTRY dashboard.",
        ]
        return "\n".join(lines)

    def _render_html(self, msg: NotificationMessage) -> str:
        site = html.escape(msg.site_name or "Unknown Site")
        title = html.escape(msg.title or "Alert")
        body = html.escape(msg.body or "")
        cam = html.escape(msg.camera_name or msg.camera_uuid)
        dev = html.escape(msg.device_name or "")
        ts = html.escape(self._fmt_ts(msg.ts_ms))
        classes = ", ".join(html.escape(c) for c in (msg.cls_names or ["unknown"]))
        conf = f"{msg.max_conf:.2f}" if msg.max_conf is not None else "n/a"

        button_html = ""
        if self.cfg.dashboard_base_url:
            url = html.escape(self._camera_url(msg.camera_uuid))
            button_html = f"""
              <tr>
                <td style="padding: 14px 24px 24px 24px;">
                  <a href="{url}"
                     style="display:inline-block;text-decoration:none;background:#0ea5e9;color:#ffffff;
                            padding:12px 16px;border-radius:10px;font-weight:700;font-family:Arial,sans-serif;">
                    Open Dashboard
                  </a>
                </td>
              </tr>
            """

        return f"""\
<!doctype html>
<html>
  <body style="margin:0;padding:0;background:#f4f6fb;">
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background:#f4f6fb;padding:24px 0;">
      <tr>
        <td align="center">
          <table role="presentation" width="640" cellspacing="0" cellpadding="0"
                 style="background:#ffffff;border-radius:16px;overflow:hidden;border:1px solid #e6e8ef;">
            <tr>
              <td style="padding:18px 24px;background:#0b1220;color:#ffffff;font-family:Arial,sans-serif;">
                <div style="font-size:14px;opacity:0.85;font-weight:700;letter-spacing:0.3px;">1886NOENTRY</div>
                <div style="font-size:22px;font-weight:800;margin-top:6px;">AI Monitor Alert</div>
              </td>
            </tr>

            <tr>
              <td style="padding:18px 24px;font-family:Arial,sans-serif;">
                <div style="display:inline-block;background:#fee2e2;color:#991b1b;padding:6px 10px;border-radius:999px;
                            font-weight:700;font-size:12px;">
                  ALERT
                </div>

                <div style="margin-top:12px;font-size:18px;font-weight:800;color:#111827;">
                  {title}
                </div>

                <div style="margin-top:6px;color:#475569;font-size:14px;line-height:1.5;">
                  A detection event was triggered at <strong>{site}</strong>.
                </div>

                <table role="presentation" width="100%" cellspacing="0" cellpadding="0"
                       style="margin-top:14px;border-collapse:separate;border-spacing:0;background:#f8fafc;
                              border:1px solid #e2e8f0;border-radius:12px;overflow:hidden;">
                  <tr>
                    <td style="padding:12px 14px;border-bottom:1px solid #e2e8f0;color:#0f172a;font-weight:700;font-size:13px;">Site</td>
                    <td style="padding:12px 14px;border-bottom:1px solid #e2e8f0;color:#334155;font-size:13px;">{site}</td>
                  </tr>
                  <tr>
                    <td style="padding:12px 14px;border-bottom:1px solid #e2e8f0;color:#0f172a;font-weight:700;font-size:13px;">Camera</td>
                    <td style="padding:12px 14px;border-bottom:1px solid #e2e8f0;color:#334155;font-size:13px;">{cam}</td>
                  </tr>
                  {f'''
                  <tr>
                    <td style="padding:12px 14px;border-bottom:1px solid #e2e8f0;color:#0f172a;font-weight:700;font-size:13px;">Device</td>
                    <td style="padding:12px 14px;border-bottom:1px solid #e2e8f0;color:#334155;font-size:13px;">{dev}</td>
                  </tr>
                  ''' if dev else ''}
                  <tr>
                    <td style="padding:12px 14px;border-bottom:1px solid #e2e8f0;color:#0f172a;font-weight:700;font-size:13px;">Detected</td>
                    <td style="padding:12px 14px;border-bottom:1px solid #e2e8f0;color:#334155;font-size:13px;">{classes}</td>
                  </tr>
                  <tr>
                    <td style="padding:12px 14px;border-bottom:1px solid #e2e8f0;color:#0f172a;font-weight:700;font-size:13px;">Max Confidence</td>
                    <td style="padding:12px 14px;border-bottom:1px solid #e2e8f0;color:#334155;font-size:13px;">{conf}</td>
                  </tr>
                  <tr>
                    <td style="padding:12px 14px;color:#0f172a;font-weight:700;font-size:13px;">Time (UTC)</td>
                    <td style="padding:12px 14px;color:#334155;font-size:13px;">{ts}</td>
                  </tr>
                </table>

                <div style="margin-top:14px;color:#0f172a;font-size:14px;line-height:1.6;">
                  {body}
                </div>
              </td>
            </tr>

            {button_html}

            <tr>
              <td style="padding:14px 24px 18px 24px;background:#f8fafc;border-top:1px solid #e2e8f0;
                         font-family:Arial,sans-serif;color:#64748b;font-size:12px;line-height:1.5;">
                You are receiving this alert because notifications are enabled for this site/camera in 1886NOENTRY.
                If this is unexpected, adjust notification settings in your dashboard.
              </td>
            </tr>

          </table>

          <div style="width:640px;text-align:center;color:#94a3b8;font-family:Arial,sans-serif;font-size:12px;margin-top:10px;">
            © {datetime.now().year} 1886NOENTRY
          </div>
        </td>
      </tr>
    </table>
  </body>
</html>
"""

    def _render_digest_text(self, messages: List[NotificationMessage]) -> str:
        first = messages[0]
        shown = messages[:20]
        remaining = max(0, len(messages) - len(shown))

        lines = [
            "1886NOENTRY — Buffered Alert Digest",
            "",
            f"Site: {first.site_name}",
            f"Alerts in batch: {len(messages)}",
            "",
        ]

        for msg in shown:
            lines.append(
                f"- [{self._fmt_ts(msg.ts_ms)} UTC] {msg.title}: {msg.body}"
            )

        if remaining > 0:
            lines += ["", f"... and {remaining} more alerts in this batch."]

        if self.cfg.dashboard_base_url:
            lines += ["", f"Open dashboard: {self._camera_url(first.camera_uuid)}"]

        return "\n".join(lines)

    def _render_digest_html(self, messages: List[NotificationMessage]) -> str:
        first = messages[0]
        shown = messages[:20]
        remaining = max(0, len(messages) - len(shown))

        rows = []
        for msg in shown:
            rows.append(
                f"""
                <tr>
                  <td style="padding:10px 12px;border-bottom:1px solid #e2e8f0;color:#0f172a;font-size:13px;white-space:nowrap;">{html.escape(self._fmt_ts(msg.ts_ms))}</td>
                  <td style="padding:10px 12px;border-bottom:1px solid #e2e8f0;color:#0f172a;font-size:13px;font-weight:700;">{html.escape(msg.title or "Alert")}</td>
                  <td style="padding:10px 12px;border-bottom:1px solid #e2e8f0;color:#334155;font-size:13px;">{html.escape(msg.body or "")}</td>
                </tr>
                """
            )

        more_html = ""
        if remaining > 0:
            more_html = f"""
              <div style="margin-top:12px;color:#64748b;font-size:13px;">
                ... and {remaining} more alerts in this batch.
              </div>
            """

        button_html = ""
        if self.cfg.dashboard_base_url:
            url = html.escape(self._camera_url(first.camera_uuid))
            button_html = f"""
              <div style="margin-top:16px;">
                <a href="{url}"
                   style="display:inline-block;text-decoration:none;background:#0ea5e9;color:#ffffff;
                          padding:12px 16px;border-radius:10px;font-weight:700;font-family:Arial,sans-serif;">
                  Open Dashboard
                </a>
              </div>
            """

        return f"""\
<!doctype html>
<html>
  <body style="margin:0;padding:24px;background:#f4f6fb;font-family:Arial,sans-serif;">
    <div style="max-width:760px;margin:0 auto;background:#ffffff;border:1px solid #e2e8f0;border-radius:16px;overflow:hidden;">
      <div style="padding:18px 24px;background:#0b1220;color:#ffffff;">
        <div style="font-size:14px;opacity:0.85;font-weight:700;">1886NOENTRY</div>
        <div style="font-size:22px;font-weight:800;margin-top:6px;">Buffered Alert Digest</div>
      </div>
      <div style="padding:18px 24px;">
        <div style="color:#0f172a;font-size:16px;font-weight:800;">{html.escape(first.site_name or "Unknown Site")}</div>
        <div style="margin-top:6px;color:#475569;font-size:14px;">{len(messages)} alerts were buffered and delivered together.</div>
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0"
               style="margin-top:16px;border-collapse:separate;border-spacing:0;background:#f8fafc;
                      border:1px solid #e2e8f0;border-radius:12px;overflow:hidden;">
          <thead>
            <tr>
              <th style="padding:10px 12px;border-bottom:1px solid #e2e8f0;text-align:left;color:#0f172a;font-size:12px;">Time (UTC)</th>
              <th style="padding:10px 12px;border-bottom:1px solid #e2e8f0;text-align:left;color:#0f172a;font-size:12px;">Title</th>
              <th style="padding:10px 12px;border-bottom:1px solid #e2e8f0;text-align:left;color:#0f172a;font-size:12px;">Details</th>
            </tr>
          </thead>
          <tbody>
            {"".join(rows)}
          </tbody>
        </table>
        {more_html}
        {button_html}
      </div>
    </div>
  </body>
</html>
"""

    def _fmt_ts(self, ts_ms: int) -> str:
        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def _camera_url(self, camera_uuid: str) -> str:
        base = (self.cfg.dashboard_base_url or "").rstrip("/")
        path = self.cfg.camera_path_template.format(camera_uuid=camera_uuid)
        return f"{base}{path}"


class NotificationService:
    def __init__(
        self,
        *,
        hub: WebNotificationHub,
        email: Optional[EmailNotifier] = None,
        interesting_classes: Iterable[str] = ("person", "car", "motorcycle", "truck"),
        cooldown_s: float = 10.0,
        enable_tracking: bool = True,
        tracker_cfg: Optional[dict] = None,
        roi_provider=None,
        notify_on_confirmed: bool = False,
        notify_on_roi_enter: bool = True,
        clip_service: Optional[EventClipService] = None,
        image_service: Optional[AlertImageStorageService] = None,
    ):
        self.hub = hub
        self.email = email
        self.interesting = set(interesting_classes)
        self.cooldown_s = float(cooldown_s)

        self._last_sent: Dict[Tuple[str, str], float] = {}
        self._lock = asyncio.Lock()

        self.enable_tracking = bool(enable_tracking)
        self.notify_on_confirmed = bool(notify_on_confirmed)
        self.notify_on_roi_enter = bool(notify_on_roi_enter)
        self.roi_provider = roi_provider

        cfg = tracker_cfg or {}
        self._tracker = MultiCameraByteTrack(**cfg)
        self._roi_engine = ROIAlertEngine()
        self._session_factory = None
        self._repo = NotificationRepository()
        self._ctx_cache: Dict[str, Tuple[float, Optional[CameraContext]]] = {}
        self._ctx_ttl_s = 60.0  # reduce DB hits on frequent detections
        self._ctx_miss_ttl_s = _env_float("NOTIFICATION_CAMERA_CONTEXT_MISS_CACHE_TTL_S", 5.0, minimum=0.0)
        self._ctx_cache_lock = asyncio.Lock()
        self._ctx_inflight: Dict[str, asyncio.Future] = {}
        self._ctx_lookup_limit = asyncio.Semaphore(
            _env_int("NOTIFICATION_CAMERA_CONTEXT_MAX_CONCURRENT_DB_LOOKUPS", 4, minimum=1)
        )
        self._roi_cache: Dict[str, Tuple[float, List[ROI]]] = {}
        self._roi_ttl_s = 15.0
        self._recipient_cache: Dict[Tuple[int, str], Tuple[float, List[str]]] = {}
        self._recipient_ttl_s = _env_float("NOTIFICATION_RECIPIENT_CACHE_TTL_S", 60.0, minimum=1.0)

        self._buffer_lock = asyncio.Lock()
        self._flush_event = asyncio.Event()
        # Changed from flat list to hierarchical: user_id -> site_uuid -> camera_uuid -> alerts
        self._pending_by_user: Dict[int, Dict[str, Dict[str, List[BufferedNotification]]]] = {}
        self._pending_delete_ids_by_user: Dict[int, Set[int]] = {}
        self._delete_event = asyncio.Event()
        self._delete_task: Optional[asyncio.Task] = None
        self._pending_since: Dict[int, float] = {}
        self._active_flush_users: Set[int] = set()
        self._flush_task: Optional[asyncio.Task] = None
        self._closing = False
        self._buffer_max_items = _env_int("NOTIFICATION_BUFFER_MAX_ITEMS", 100, minimum=1)
        self._buffer_max_age_s = _env_float("NOTIFICATION_BUFFER_MAX_AGE_S", 60.0, minimum=1.0)
        self._buffer_poll_s = _env_float("NOTIFICATION_BUFFER_POLL_S", 1.0, minimum=0.2)
        self._site_prerecord_timeout_s = _env_float("SITE_PRERECORD_TIMEOUT_S", 30.0, minimum=1.0)
        self._trigger_camera_timeout_s = _env_float("TRIGGER_CAMERA_TIMEOUT_S", 15.0, minimum=1.0)
        self._clip_overlay_history_ttl_s = _env_float("CLIP_OVERLAY_HISTORY_TTL_S", 180.0, minimum=30.0)
        self.overlay_duration = _env_float("VIDEO_CLIP_DURATION_S",120.0, minimum=60.0)+60.0
        self._clip_overlay_history_max_frames = _env_int(
            "CLIP_OVERLAY_HISTORY_MAX_FRAMES_PER_CAMERA",
            self.overlay_duration* 60,
            minimum=1,
        )
        self._overlay_history_by_camera: Dict[str, deque[Dict[str, Any]]] = {}
        self._overlay_history_lock = asyncio.Lock()
        self._clip_service = clip_service or EventClipService()
        self._image_service = image_service or AlertImageStorageService()

    def set_session_factory(self, session_factory):
        self._session_factory = session_factory
        if self._clip_service is not None:
            try:
                self._clip_service.set_session_factory(session_factory)
            except Exception:
                logger.exception("Failed to set session factory on EventClipService")


    async def _delete_alert_blob_keys(self, storage_keys: List[str]) -> None:
        unique_keys = [key for key in dict.fromkeys(str(key or "").strip() for key in storage_keys) if key]
        if not unique_keys:
            return

        image_service = self._image_service or AlertImageStorageService()
        close_when_done = image_service is not self._image_service

        try:
            for storage_key in unique_keys:
                try:
                    await image_service.delete_blob(blob_name=storage_key)
                except Exception:
                    logger.warning(
                        "Failed deleting alert image blob %s after alert hide",
                        storage_key,
                        exc_info=True,
                    )
        finally:
            if close_when_done:
                try:
                    await image_service.close()
                except Exception:
                    logger.exception("Failed closing temporary alert image service")
                    
    async def record_detection_overlay_frame(
        self,
        *,
        camera_uuid: str,
        frame_ts_ms: Any,
        frame_seq: Any,
        frame_w: Any = None,
        frame_h: Any = None,
        detections: Any = None,
    ) -> None:
        frame = _normalize_overlay_frame(
            camera_uuid=str(camera_uuid),
            frame_ts_ms=frame_ts_ms,
            frame_seq=frame_seq,
            frame_w=frame_w,
            frame_h=frame_h,
            detections=detections,
        )
        if frame is None:
            return

        history_cutoff_ms = int(frame["frame_ts_ms"]) - int(self._clip_overlay_history_ttl_s * 1000.0)
        async with self._overlay_history_lock:
            bucket = self._overlay_history_by_camera.get(str(camera_uuid))
            if bucket is None:
                bucket = deque(maxlen=int(self._clip_overlay_history_max_frames))
                self._overlay_history_by_camera[str(camera_uuid)] = bucket

            while bucket and int(bucket[0].get("frame_ts_ms", 0)) < history_cutoff_ms:
                bucket.popleft()

            if bucket:
                last = bucket[-1]
                if (
                    int(last.get("frame_ts_ms", -1)) == int(frame["frame_ts_ms"])
                    and int(last.get("frame_seq", -1)) == int(frame["frame_seq"])
                ):
                    bucket[-1] = frame
                    return

            bucket.append(frame)

    def _build_clip_overlay_payload(
        self,
        *,
        msg: NotificationMessage,
        extra_payload: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        payload = dict(extra_payload or {})

        detections = list(msg.detections or [])
        if not detections and isinstance(payload.get("detections"), list):
            detections = list(payload.get("detections") or [])

        frame_w = msg.frame_w if msg.frame_w is not None else payload.get("frame_w")
        frame_h = msg.frame_h if msg.frame_h is not None else payload.get("frame_h")
        frame_seq = msg.frame_seq if msg.frame_seq is not None else payload.get("frame_seq")

        if not detections and frame_w is None and frame_h is None:
            return None

        frame = _normalize_overlay_frame(
            camera_uuid=str(msg.camera_uuid),
            frame_ts_ms=msg.ts_ms,
            frame_seq=frame_seq or 0,
            frame_w=frame_w,
            frame_h=frame_h,
            detections=detections,
        )
        if frame is None:
            return None

        result = _build_overlay_payload_from_frames(
            camera_uuid=str(msg.camera_uuid),
            frames=[frame],
            preferred_ts_ms=int(msg.ts_ms),
            timeline_source="trigger_frame",
        )
        if result is not None and msg.alert_type:
            result["alert_type"] = str(msg.alert_type)
        return result

    async def _clip_overlay_frames_for_window(
        self,
        *,
        camera_uuid: str,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
    ) -> List[Dict[str, Any]]:
        if start_time is None or end_time is None:
            return []

        start_ts_ms = int(start_time.astimezone(timezone.utc).timestamp() * 1000.0)
        end_ts_ms = int(end_time.astimezone(timezone.utc).timestamp() * 1000.0)
        if end_ts_ms < start_ts_ms:
            start_ts_ms, end_ts_ms = end_ts_ms, start_ts_ms

        async with self._overlay_history_lock:
            bucket = list(self._overlay_history_by_camera.get(str(camera_uuid), ()))

        return [
            dict(frame)
            for frame in bucket
            if start_ts_ms <= int(frame.get("frame_ts_ms", -1)) <= end_ts_ms
        ]

    async def _trim_clip_overlay_history(
        self,
        *,
        camera_uuid: str,
        through_time: Optional[datetime],
    ) -> None:
        if through_time is None:
            return

        through_ts_ms = int(through_time.astimezone(timezone.utc).timestamp() * 1000.0)
        async with self._overlay_history_lock:
            bucket = self._overlay_history_by_camera.get(str(camera_uuid))
            if not bucket:
                return

            while bucket and int(bucket[0].get("frame_ts_ms", 0)) <= through_ts_ms:
                bucket.popleft()

            if not bucket:
                self._overlay_history_by_camera.pop(str(camera_uuid), None)

    async def _finalize_captured_clip(
        self,
        *,
        camera_uuid: str,
        clip: Optional[Dict[str, Any]],
        preferred_ts_ms: Optional[int],
        fallback_overlay_payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(clip, dict):
            return clip

        start_time = _parse_utc_datetime(clip.get("start_time"))
        end_time = _parse_utc_datetime(clip.get("end_time"))
        frames = await self._clip_overlay_frames_for_window(
            camera_uuid=str(camera_uuid),
            start_time=start_time,
            end_time=end_time,
        )

        fallback_frames = list(fallback_overlay_payload.get("frames") or []) if isinstance(fallback_overlay_payload, dict) else []
        if not fallback_frames and isinstance(fallback_overlay_payload, dict):
            fallback_frame = _normalize_overlay_frame(
                camera_uuid=str(camera_uuid),
                frame_ts_ms=fallback_overlay_payload.get("frame_ts_ms"),
                frame_seq=fallback_overlay_payload.get("frame_seq"),
                frame_w=fallback_overlay_payload.get("frame_w"),
                frame_h=fallback_overlay_payload.get("frame_h"),
                detections=fallback_overlay_payload.get("detections"),
            )
            if fallback_frame is not None:
                fallback_frames = [fallback_frame]

        overlay_payload = _build_overlay_payload_from_frames(
            camera_uuid=str(camera_uuid),
            frames=_merge_overlay_frames(frames, fallback_frames),
            preferred_ts_ms=preferred_ts_ms,
            clip_start_time=start_time,
            clip_end_time=end_time,
            timeline_source="clip_history",
        )

        if overlay_payload is None:
            return dict(clip)

        clip_service = self._clip_service
        external_id = str(clip.get("external_id") or "").strip()
        update_overlay = getattr(clip_service, "update_overlay_payload", None) if clip_service is not None else None
        if callable(update_overlay) and external_id:
            try:
                merged_overlay = await update_overlay(
                    camera_uuid=str(camera_uuid),
                    external_id=external_id,
                    overlay_payload=overlay_payload,
                )
                if isinstance(merged_overlay, dict):
                    overlay_payload = merged_overlay
            except Exception:
                logger.exception(
                    "Failed finalizing clip overlay camera=%s external_id=%s",
                    camera_uuid,
                    external_id,
                )

        await self._trim_clip_overlay_history(
            camera_uuid=str(camera_uuid),
            through_time=end_time,
        )

        finalized = dict(clip)
        finalized["overlay_payload"] = overlay_payload
        return finalized
        
    def invalidate_recipient_cache(
        self,
        *,
        user_id: int,
        site_uuid: Optional[uuid.UUID] = None,
    ) -> None:
        uid = int(user_id)
        if site_uuid is not None:
            self._recipient_cache.pop((uid, str(site_uuid)), None)
            return

        for key in list(self._recipient_cache.keys()):
            if key[0] == uid:
                self._recipient_cache.pop(key, None)

    async def purge_deleted_site_runtime_state(
        self,
        *,
        user_id: int,
        site_uuid: Optional[uuid.UUID] = None,
        camera_uuids: Optional[List[uuid.UUID]] = None,
    ) -> None:
        """Drop cached/buffered runtime state for cameras that are being deleted.

        This prevents:
        - buffered alerts for the deleted site from being flushed after delete
        - cached camera context from continuing to resolve deleted cameras
        - stale overlay/ROI cooldown state from hanging around
        """
        uid = int(user_id)
        site_key = str(site_uuid) if site_uuid is not None else None
        camera_keys = {str(value) for value in (camera_uuids or []) if value is not None}

        async with self._buffer_lock:
            hierarchical = self._pending_by_user.get(uid)

            if hierarchical is not None:
                if site_key is not None:
                    hierarchical.pop(site_key, None)

                if camera_keys:
                    for raw_site_key in list(hierarchical.keys()):
                        per_camera = hierarchical.get(raw_site_key) or {}
                        for cam_key in list(per_camera.keys()):
                            if cam_key in camera_keys:
                                per_camera.pop(cam_key, None)
                        if not per_camera:
                            hierarchical.pop(raw_site_key, None)

                if hierarchical:
                    self._pending_by_user[uid] = hierarchical
                else:
                    self._pending_by_user.pop(uid, None)
                    self._pending_since.pop(uid, None)

        self.invalidate_recipient_cache(user_id=uid, site_uuid=site_uuid)

        for cam_key in camera_keys:
            self._ctx_cache.pop(cam_key, None)
            self._ctx_inflight.pop(cam_key, None)
            self._roi_cache.pop(cam_key, None)
            self._overlay_history_by_camera.pop(cam_key, None)
            self.invalidate_camera_roi_state(cam_key)

        if camera_keys:
            for key in list(self._last_sent.keys()):
                if key[0] in camera_keys:
                    self._last_sent.pop(key, None)

    def _fire_and_forget(self, coro):
        async def _runner():
            try:
                await coro
            except Exception:
                logger.exception("Notification background task failed")
        asyncio.create_task(_runner())

    def _ensure_delete_task(self) -> None:
        if self._closing:
            return
        task = self._delete_task
        if task is None or task.done():
            self._delete_task = asyncio.create_task(
                self._delete_loop(),
                name="notification_delete_queue",
            )

    async def _delete_loop(self) -> None:
        while True:
            try:
                await asyncio.wait_for(self._delete_event.wait(), timeout=self._buffer_poll_s)
            except asyncio.TimeoutError:
                pass

            self._delete_event.clear()
            force_all = bool(self._closing)

            await self._flush_delete_queue(force_all=force_all)

            if force_all:
                return

    async def _batch_hide_notifications(
        self,
        *,
        user_id: int,
        notification_ids: List[int],
        batch_size: int = 1000,
    ) -> List[str]:
        """Hide notifications in batches to prevent table lock exhaustion.

        Each batch is a separate transaction, releasing locks between commits.
        Returns: list of blob storage keys to cleanup.
        """
        if not notification_ids:
            return []

        sorted_ids = sorted(set(int(nid) for nid in notification_ids if nid > 0))
        all_storage_keys: List[str] = []
        total_hidden = 0

        logger.info(
            "[Notification Batch Hide] starting hide of %d notifications for user=%s in batches of %d",
            len(sorted_ids),
            user_id,
            batch_size,
        )

        for batch_num, i in enumerate(range(0, len(sorted_ids), batch_size), start=1):
            batch_ids = sorted_ids[i : i + batch_size]

            try:
                async with self._session_factory() as db:
                    # Fetch blob keys for this batch BEFORE hiding
                    rows = (
                        await db.execute(
                            select(Notification.id, Notification.payload)
                            .where(
                                Notification.user_id == int(user_id),
                                Notification.id.in_(batch_ids),
                                Notification.visible.is_(True),
                            )
                        )
                    ).all()

                    matched_ids: List[int] = []
                    for notification_id, notification_payload in rows:
                        try:
                            parsed_id = int(notification_id)
                        except Exception:
                            continue

                        if parsed_id <= 0:
                            continue

                        matched_ids.append(parsed_id)
                        storage_key = extract_image_storage_key(notification_payload)
                        if storage_key:
                            all_storage_keys.append(storage_key)

                    if not matched_ids:
                        logger.info(
                            "[Notification Batch Hide] batch %d: no matching notifications to hide",
                            batch_num,
                        )
                        continue

                    # Hide this batch (set visible=False)
                    result = await db.execute(
                        update(Notification)
                        .where(
                            Notification.user_id == int(user_id),
                            Notification.id.in_(matched_ids),
                            Notification.visible.is_(True),
                        )
                        .values(visible=False)
                        .execution_options(synchronize_session=False)
                    )
                    await db.commit()

                    hidden_count = result.rowcount or len(matched_ids)
                    total_hidden += hidden_count

                    logger.info(
                        "[Notification Batch Hide] batch %d: hid %d notifications, total=%d, remaining=%d",
                        batch_num,
                        hidden_count,
                        total_hidden,
                        len(sorted_ids) - total_hidden,
                    )
            except Exception as exc:
                logger.exception(
                    "[Notification Batch Hide] batch %d failed for user=%s with %d ids: %s",
                    batch_num,
                    user_id,
                    len(batch_ids),
                    str(exc),
                )
                raise

        logger.info(
            "[Notification Batch Hide] complete: %d notifications hidden, %d storage keys to cleanup",
            total_hidden,
            len(all_storage_keys),
        )
        return all_storage_keys

    async def _flush_delete_queue(self, *, force_all: bool = False) -> None:
        if not self._session_factory:
            return

        ready: Dict[int, List[int]] = {}

        async with self._buffer_lock:
            for user_id, ids in list(self._pending_delete_ids_by_user.items()):
                clean_ids = sorted({int(x) for x in ids if int(x) > 0})
                if clean_ids:
                    ready[int(user_id)] = clean_ids
            self._pending_delete_ids_by_user.clear()

        if not ready:
            return

        for user_id, ids in ready.items():
            try:
                # Use batched hiding instead of single large transaction
                storage_keys = await self._batch_hide_notifications(
                    user_id=int(user_id),
                    notification_ids=ids,
                    batch_size=1000,
                )

                if storage_keys:
                    self._fire_and_forget(self._delete_alert_blob_keys(storage_keys))

            except Exception:
                logger.exception(
                    "[Notification Batch Hide] failed to apply queued notification hide user=%s ids=%s",
                    user_id,
                    ids,
                )
                if not force_all:
                    async with self._buffer_lock:
                        bucket = self._pending_delete_ids_by_user.setdefault(int(user_id), set())
                        bucket.update(ids)
                    self._delete_event.set()


    async def handle_deletion_event(
        self,
        *,
        user_id: int,
        notification_ids: Optional[List[int]] = None,
        site_uuid: Optional[str] = None,
        camera_uuid: Optional[str] = None,
    ) -> Dict[str, Any]:
        ids = sorted(
            {
                int(raw_id)
                for raw_id in (notification_ids or [])
                if raw_id is not None and int(raw_id) > 0
            }
        )
        if not ids and (site_uuid or camera_uuid) and self._session_factory:
            try:
                su = uuid.UUID(site_uuid) if site_uuid else None
            except ValueError:
                su = None
            try:
                cu = uuid.UUID(camera_uuid) if camera_uuid else None
            except ValueError:
                cu = None

            if su is not None or cu is not None:
                try:
                    conds = [
                        Notification.user_id == int(user_id),
                        Notification.visible.is_(True),
                    ]
                    if su is not None:
                        conds.append(Notification.site_uuid == su)
                    if cu is not None:
                        conds.append(Notification.camera_uuid == cu)

                    # Fetch IDs in batches to avoid loading millions of rows at once
                    id_batch_size = 10000
                    id_offset = 0
                    all_ids: set[int] = set()
                    while True:
                        async with self._session_factory() as db:
                            rows = (
                                await db.execute(
                                    select(Notification.id)
                                    .where(and_(*conds))
                                    .order_by(Notification.id)
                                    .offset(id_offset)
                                    .limit(id_batch_size)
                                )
                            ).scalars().all()

                        if not rows:
                            break
                        for r in rows:
                            if r is not None and int(r) > 0:
                                all_ids.add(int(r))
                        id_offset += id_batch_size
                        if len(rows) < id_batch_size:
                            break

                    ids = sorted(all_ids)
                except Exception:
                    logger.exception(
                        "Failed to resolve notification IDs for bulk delete user=%s site=%s camera=%s",
                        user_id,
                        site_uuid,
                        camera_uuid,
                    )

        if not ids:
            return {"ok": True, "deleted": 0}

        async with self._buffer_lock:
            bucket = self._pending_delete_ids_by_user.setdefault(int(user_id), set())
            bucket.update(ids)

        self._ensure_delete_task()
        self._delete_event.set()

        return {
            "ok": True,
            "deleted": len(ids),
            "notification_ids": ids[:1000],
        }

    def _ensure_flush_task(self) -> None:
        if self._closing:
            return
        task = self._flush_task
        if task is None or task.done():
            self._flush_task = asyncio.create_task(
                self._flush_loop(),
                name="notification_buffer_flush",
            )

    async def shutdown(self) -> None:
        self._closing = True
        self._flush_event.set()
        task = self._flush_task
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Notification flush task shutdown failed")
        self._flush_task = None
        clip_service = self._clip_service
        if clip_service is not None:
            try:
                await clip_service.close()
            except Exception:
                logger.exception("Event clip service shutdown failed")
        image_service = self._image_service
        if image_service is not None:
            try:
                await image_service.close()
            except Exception:
                logger.exception("Alert image service shutdown failed")

    async def _materialize_alert_image_payload(
        self,
        *,
        msg: NotificationMessage,
        extra_payload: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Optional[str], Optional[str]]:
        payload = dict(extra_payload or {})
        image_url = str(payload.get("image_url") or msg.image_url or "").strip()
        if not image_url:
            return payload, msg.image_url, None

        image_service = self._image_service
        if image_service is None:
            return payload, image_url, str(payload.get("image_storage_key") or "").strip() or None

        if not image_url.startswith("data:"):
            return payload, image_url, str(payload.get("image_storage_key") or "").strip() or None

        try:
            stored = await image_service.store_image_data_url(
                image_data_url=image_url,
                camera_uuid=str(msg.camera_uuid),
                ts_ms=int(msg.ts_ms),
            )
        except Exception:
            logger.warning(
                "Alert image upload failed for camera=%s msg_id=%s; keeping inline image payload",
                msg.camera_uuid,
                msg.id,
                exc_info=True,
            )
            return payload, image_url, None
        if not stored:
            return payload, image_url, None

        stored_url = str(stored.get("image_url") or "").strip() or image_url
        stored_key = str(stored.get("image_storage_key") or "").strip() or None
        payload["image_url"] = stored_url
        if stored_key:
            payload["image_storage_key"] = stored_key
        return payload, stored_url, stored_key

    def _site_prerecord_clip_payload(
        self,
        *,
        camera_uuid: str,
        camera_ctx: CameraContext,
        clip: Dict[str, Any],
        trigger_camera_uuid: str,
    ) -> Dict[str, Any]:
        payload = dict(clip or {})
        payload["camera_uuid"] = str(camera_uuid)
        payload["camera_name"] = camera_ctx.camera_name
        payload["camera_code"] = camera_ctx.camera_code
        payload["site_uuid"] = str(camera_ctx.site_uuid)
        payload["is_trigger_camera"] = str(camera_uuid) == str(trigger_camera_uuid)
        return payload

    def _site_prerecord_trigger_matches(
        self,
        *,
        trigger_mode: str,
        alert_type: str,
    ) -> bool:
        normalized_mode = str(trigger_mode or "").strip().lower()
        normalized_alert = str(alert_type or "").strip().lower()

        if normalized_mode == "any_detection":
            return normalized_alert in {"roi_enter", "item_detected", "detection_summary"}

        return normalized_alert == "roi_enter"

    async def _load_site_prerecord_plan(
        self,
        *,
        msg: NotificationMessage,
        ctx: CameraContext,
    ) -> Optional[SitePrerecordPlan]:
        if self._session_factory is None:
            return None

        try:
            trigger_camera_uuid = uuid.UUID(str(msg.camera_uuid))
        except Exception:
            return None

        async with self._session_factory() as db:
            settings = await self._repo.get_site_prerecord_settings(
                db,
                user_id=int(ctx.user_id),
                site_uuid=ctx.site_uuid,
            )
            if not settings.enabled:
                return None
            if not self._site_prerecord_trigger_matches(
                trigger_mode=settings.trigger_mode,
                alert_type=msg.alert_type,
            ):
                return None

            selected_camera_uuids = list(settings.camera_uuids or [])
            if trigger_camera_uuid not in set(selected_camera_uuids):
                return None

            contexts_by_camera = await self._repo.list_camera_contexts(
                db,
                user_id=int(ctx.user_id),
                camera_uuids=selected_camera_uuids,
            )

        ordered_camera_uuids = [
            camera_uuid
            for camera_uuid in settings.camera_uuids
            if camera_uuid in contexts_by_camera
        ]
        if not ordered_camera_uuids or trigger_camera_uuid not in contexts_by_camera:
            return None

        return SitePrerecordPlan(
            settings=settings,
            ordered_camera_uuids=ordered_camera_uuids,
            contexts_by_camera=contexts_by_camera,
        )

    async def _capture_site_prerecord_clips(
        self,
        *,
        msg: NotificationMessage,
        ctx: CameraContext,
        plan: SitePrerecordPlan,
        trigger_clip: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        clip_service = self._clip_service
        if clip_service is None:
            return []

        out: List[Dict[str, Any]] = []
        pending_camera_uuids: List[uuid.UUID] = []
        pending_tasks: List[asyncio.Future] = []
        trigger_camera_uuid_str = str(msg.camera_uuid)
        trigger_overlay_payload = self._build_clip_overlay_payload(
            msg=msg,
            extra_payload=None,
        )

        for camera_uuid in plan.ordered_camera_uuids:
            camera_uuid_str = str(camera_uuid)
            camera_ctx = plan.contexts_by_camera.get(camera_uuid)
            if camera_ctx is None:
                continue

            if camera_uuid_str == trigger_camera_uuid_str and trigger_clip:
                out.append(
                    self._site_prerecord_clip_payload(
                        camera_uuid=camera_uuid_str,
                        camera_ctx=camera_ctx,
                        clip=trigger_clip,
                        trigger_camera_uuid=msg.camera_uuid,
                    )
                )
                continue

            pending_camera_uuids.append(camera_uuid)
            pending_tasks.append(
                clip_service.capture_pre_event_clip(
                    camera_uuid=camera_uuid_str,
                    ctx=camera_ctx,
                    event_ts_ms=msg.ts_ms,
                    trigger=f"site_prerecord:{msg.camera_uuid}:{plan.settings.trigger_mode}:{msg.alert_type}",
                    overlay_payload=(
                        trigger_overlay_payload
                        if camera_uuid_str == trigger_camera_uuid_str
                        else None
                    ),
                )
            )

        if pending_tasks:
            results = []
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*pending_tasks, return_exceptions=True),
                    timeout=self._site_prerecord_timeout_s,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Site prerecord clip capture timed out after %.1fs trigger_camera=%s pending_count=%s",
                    self._site_prerecord_timeout_s,
                    msg.camera_uuid,
                    len(pending_camera_uuids),
                )
                # Return partial results; remaining cameras will be skipped
                return out
            
            for camera_uuid, result in zip(pending_camera_uuids, results):
                camera_uuid_str = str(camera_uuid)
                if isinstance(result, Exception):
                    logger.exception(
                        "Failed capturing site prerecord clip trigger_camera=%s target_camera=%s",
                        msg.camera_uuid,
                        camera_uuid_str,
                        exc_info=(type(result), result, result.__traceback__),
                    )
                    continue
                if not result:
                    continue
                camera_ctx = plan.contexts_by_camera.get(camera_uuid)
                if camera_ctx is None:
                    continue
                finalized_clip = await self._finalize_captured_clip(
                    camera_uuid=camera_uuid_str,
                    clip=result,
                    preferred_ts_ms=int(msg.ts_ms),
                    fallback_overlay_payload=(
                        trigger_overlay_payload
                        if camera_uuid_str == trigger_camera_uuid_str
                        else None
                    ),
                )
                out.append(
                    self._site_prerecord_clip_payload(
                        camera_uuid=camera_uuid_str,
                        camera_ctx=camera_ctx,
                        clip=finalized_clip or result,
                        trigger_camera_uuid=msg.camera_uuid,
                    )
                )

        return out

    async def _attach_clip_payload(
        self,
        *,
        msg: NotificationMessage,
        ctx: CameraContext,
        extra_payload: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        clip_service = self._clip_service
        if clip_service is None:
            return extra_payload

        # Check the site prerecord setting. If recording is disabled, skip ALL
        # clip capture (both the trigger camera and multi-camera prerecord).
        # If recording is enabled but the trigger mode doesn't match, also skip.
        if self._session_factory is not None:
            try:
                trigger_cam_uuid = uuid.UUID(str(msg.camera_uuid))
            except Exception:
                trigger_cam_uuid = None

            if trigger_cam_uuid is not None:
                async with self._session_factory() as _precheck_db:
                    _precheck_settings = await self._repo.get_site_prerecord_settings(
                        _precheck_db,
                        user_id=int(ctx.user_id),
                        site_uuid=ctx.site_uuid,
                    )
                # Recording disabled → no clips at all
                if not _precheck_settings.enabled:
                    return extra_payload
                # Recording enabled but trigger mode doesn't match for this camera
                if (
                    trigger_cam_uuid in set(_precheck_settings.camera_uuids)
                    and not self._site_prerecord_trigger_matches(
                        trigger_mode=_precheck_settings.trigger_mode,
                        alert_type=msg.alert_type,
                    )
                ):
                    return extra_payload

        overlay_payload = self._build_clip_overlay_payload(
            msg=msg,
            extra_payload=extra_payload,
        )
        plan = await self._load_site_prerecord_plan(msg=msg, ctx=ctx)

        merged = dict(extra_payload or {})
        try:
            trigger_camera_uuid = uuid.UUID(str(msg.camera_uuid))
        except Exception:
            trigger_camera_uuid = None
        trigger_ctx = (
            plan.contexts_by_camera.get(trigger_camera_uuid)
            if plan is not None and trigger_camera_uuid is not None
            else None
        )
        trigger_mode = plan.settings.trigger_mode if plan is not None else "single_camera"
        
        # Capture trigger camera clip with timeout
        clip = None
        try:
            clip = await asyncio.wait_for(
                clip_service.capture_pre_event_clip(
                    camera_uuid=msg.camera_uuid,
                    ctx=trigger_ctx or ctx,
                    event_ts_ms=msg.ts_ms,
                    trigger=f"site_prerecord:{msg.camera_uuid}:{trigger_mode}:{msg.alert_type}",
                    overlay_payload=overlay_payload,
                ),
                timeout=self._trigger_camera_timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Trigger camera clip capture timed out after %.1fs camera=%s",
                self._trigger_camera_timeout_s,
                msg.camera_uuid,
            )
        except Exception:
            logger.exception(
                "Trigger camera clip capture failed camera=%s",
                msg.camera_uuid,
            )
        
        if clip:
            clip = await self._finalize_captured_clip(
                camera_uuid=str(msg.camera_uuid),
                clip=clip,
                preferred_ts_ms=int(msg.ts_ms),
                fallback_overlay_payload=overlay_payload,
            )
            merged["clip"] = clip

        if plan is None:
            return merged or extra_payload

        multi_clips = await self._capture_site_prerecord_clips(
            msg=msg,
            ctx=ctx,
            plan=plan,
            trigger_clip=clip,
        )
        if multi_clips:
            merged["multi_camera_prerecordings"] = multi_clips

        return merged or extra_payload

    async def _flush_loop(self) -> None:
        while True:
            try:
                await asyncio.wait_for(self._flush_event.wait(), timeout=self._buffer_poll_s)
            except asyncio.TimeoutError:
                pass
            self._flush_event.clear()

            force_all = bool(self._closing)
            await self._flush_ready_users(force_all=force_all)
            if force_all:
                return

    async def _flatten_user_alerts_preserving_order(
        self,
        hierarchical: Dict[str, Dict[str, List[BufferedNotification]]],
    ) -> List[BufferedNotification]:
        """Flatten hierarchical buffer (site -> camera -> alerts) while preserving timestamp order."""
        all_items: List[BufferedNotification] = []
        for sites in hierarchical.values():
            for alerts in sites.values():
                all_items.extend(alerts)
        # Sort by timestamp to preserve order across all sites/cameras
        all_items.sort(key=lambda x: int(x.msg.ts_ms))
        return all_items

    async def _flush_ready_users(self, *, force_all: bool = False) -> None:
        now = time.monotonic()
        ready: Dict[int, List[BufferedNotification]] = {}

        async with self._buffer_lock:
            for user_id, hierarchical in list(self._pending_by_user.items()):
                if user_id in self._active_flush_users:
                    continue

                # Count total items across all sites/cameras
                total_items = sum(
                    len(alerts)
                    for sites in hierarchical.values()
                    for alerts in sites.values()
                )
                
                if total_items == 0:
                    self._pending_by_user.pop(user_id, None)
                    self._pending_since.pop(user_id, None)
                    continue

                since = self._pending_since.get(user_id, now)
                if force_all or total_items >= self._buffer_max_items or (now - since) >= self._buffer_max_age_s:
                    # Flatten hierarchical structure while preserving timestamp order
                    items = await self._flatten_user_alerts_preserving_order(hierarchical)
                    ready[user_id] = items
                    self._pending_by_user.pop(user_id, None)
                    self._pending_since.pop(user_id, None)
                    self._active_flush_users.add(user_id)

        if not ready:
            return

        results = await asyncio.gather(
            *[
                self._flush_user_batch(user_id, items)
                for user_id, items in ready.items()
            ],
            return_exceptions=True,
        )

        for (user_id, items), result in zip(ready.items(), results):
            requeue_items = False
            if isinstance(result, Exception):
                logger.error(
                    "Notification batch flush failed user=%s",
                    user_id,
                    exc_info=(type(result), result, result.__traceback__),
                )
                requeue_items = not force_all
            elif result is False:
                requeue_items = not force_all

            async with self._buffer_lock:
                self._active_flush_users.discard(user_id)

                if requeue_items:
                    # Requeue: restructure flattened items back into hierarchical form
                    hierarchical: Dict[str, Dict[str, List[BufferedNotification]]] = {}
                    for item in items:
                        site_uuid_str = str(item.ctx.site_uuid)
                        camera_uuid_str = str(item.msg.camera_uuid)
                        
                        if site_uuid_str not in hierarchical:
                            hierarchical[site_uuid_str] = {}
                        if camera_uuid_str not in hierarchical[site_uuid_str]:
                            hierarchical[site_uuid_str][camera_uuid_str] = []
                        
                        hierarchical[site_uuid_str][camera_uuid_str].append(item)
                    
                    self._pending_by_user[user_id] = hierarchical
                    self._pending_since[user_id] = time.monotonic() - self._buffer_max_age_s
                    self._flush_event.set()

                # Check if buffer exceeds limit after potential requeue
                if user_id in self._pending_by_user:
                    total_items = sum(
                        len(alerts)
                        for sites in self._pending_by_user[user_id].values()
                        for alerts in sites.values()
                    )
                    if total_items >= self._buffer_max_items:
                        self._flush_event.set()

    async def _get_recipients_for_sites_cached(
        self,
        *,
        user_id: int,
        site_uuids: List[uuid.UUID],
    ) -> Dict[uuid.UUID, List[str]]:
        now = time.monotonic()
        out: Dict[uuid.UUID, List[str]] = {}
        missing: List[uuid.UUID] = []

        for site_uuid in site_uuids:
            cache_key = (int(user_id), str(site_uuid))
            hit = self._recipient_cache.get(cache_key)
            if hit and hit[0] > now:
                out[site_uuid] = list(hit[1])
            else:
                missing.append(site_uuid)

        if missing and self._session_factory:
            async with self._session_factory() as db:
                loaded = await self._repo.list_notification_emails_for_sites(
                    db,
                    user_id=int(user_id),
                    site_uuids=missing,
                    only_enabled=True,
                )

            for site_uuid in missing:
                emails = list(loaded.get(site_uuid, []))
                self._recipient_cache[(int(user_id), str(site_uuid))] = (
                    now + self._recipient_ttl_s,
                    emails,
                )
                out[site_uuid] = list(emails)

        return out

    async def _flush_user_batch(self, user_id: int, items: List[BufferedNotification]) -> bool:
        if not items:
            return True
        if not self._session_factory:
            logger.warning("Skipping notification batch flush because session factory is missing user=%s", user_id)
            return False

        site_groups: Dict[uuid.UUID, List[int]] = defaultdict(list)
        create_rows: List[Dict[str, Any]] = []
        for idx, item in enumerate(items):
            stored_extra_payload, stored_image_url, stored_image_key = await self._materialize_alert_image_payload(
                msg=item.msg,
                extra_payload=item.extra_payload,
            )
            if stored_image_url:
                updated_msg = item.msg.model_copy(update={"image_url": stored_image_url})
                items[idx] = BufferedNotification(
                    msg=updated_msg,
                    ctx=item.ctx,
                    extra_payload=stored_extra_payload,
                )
                item = items[idx]
            elif stored_extra_payload != (item.extra_payload or {}):
                items[idx] = BufferedNotification(
                    msg=item.msg,
                    ctx=item.ctx,
                    extra_payload=stored_extra_payload,
                )
                item = items[idx]

            msg_payload = item.msg.model_dump()
            if stored_image_url:
                msg_payload["image_url"] = stored_image_url
            if stored_image_key:
                msg_payload["image_storage_key"] = stored_image_key
            site_groups[item.ctx.site_uuid].append(idx)
            create_rows.append(
                {
                    "user_id": int(item.ctx.user_id),
                    "site_uuid": item.ctx.site_uuid,
                    "camera_uuid": uuid.UUID(item.msg.camera_uuid),
                    "device_uuid": item.ctx.device_uuid,
                    "event_type": item.msg.alert_type,
                    "title": item.msg.title,
                    "message": item.msg.body,
                    "payload": _json_safe(
                        {
                            "msg": msg_payload,
                            "extra": stored_extra_payload,
                        }
                    ),
                    "detected_at": dt_from_ts_ms(item.msg.ts_ms),
                    "status": "created",
                    "sent_at": None,
                }
            )

        recipients_by_site = await self._get_recipients_for_sites_cached(
            user_id=int(user_id),
            site_uuids=list(site_groups.keys()),
        )

        notification_ids_by_site: Dict[uuid.UUID, List[int]] = {}
        rows = None
        for _attempt in range(3):
            try:
                async with self._session_factory() as db:
                    rows = await self._repo.create_notifications(db, rows=create_rows)
                    await db.commit()
                break
            except OperationalError as exc:
                if _attempt < 2 and "1205" in str(exc):
                    wait_s = 0.5 * (2 ** _attempt)
                    logger.warning(
                        "Notification INSERT lock timeout (attempt %s/3) user=%s — retrying in %.1fs",
                        _attempt + 1, user_id, wait_s,
                    )
                    await asyncio.sleep(wait_s)
                    continue
                logger.exception("Failed to persist buffered notifications user=%s count=%s", user_id, len(items))
                return False
            except Exception:
                logger.exception("Failed to persist buffered notifications user=%s count=%s", user_id, len(items))
                return False
        if rows is None:
            return False

        for site_uuid, indices in site_groups.items():
            notification_ids_by_site[site_uuid] = [
                int(rows[idx].id)
                for idx in indices
                if idx < len(rows) and getattr(rows[idx], "id", None) is not None
            ]
        for idx in range(len(items)):
            if idx >= len(rows):
                break
            row_id = getattr(rows[idx], "id", None)
            if row_id is None:
                continue
            item = items[idx]
            await self.hub.publish(item.msg.model_copy(update={"db_id": int(row_id)}))

        if not self.email:
            return True

        sent_ids: List[int] = []
        failed_ids: List[int] = []
        sent_at = datetime.now(timezone.utc)

        for site_uuid, indices in site_groups.items():
            recipients = recipients_by_site.get(site_uuid, [])
            if not recipients:
                continue

            site_messages = [items[idx].msg for idx in indices]
            try:
                await self.email.send_digest(site_messages, to_emails=recipients)
                sent_ids.extend(notification_ids_by_site.get(site_uuid, []))
            except Exception:
                logger.exception(
                    "Buffered email digest send failed user=%s site=%s count=%s",
                    user_id,
                    site_uuid,
                    len(site_messages),
                )
                failed_ids.extend(notification_ids_by_site.get(site_uuid, []))

        if not sent_ids and not failed_ids:
            return True

        try:
            async with self._session_factory() as db:
                if sent_ids:
                    await self._repo.mark_notifications_sent(
                        db,
                        notification_ids=sent_ids,
                        sent_at=sent_at,
                    )
                if failed_ids:
                    await self._repo.mark_notifications_failed(
                        db,
                        notification_ids=failed_ids,
                    )
                await db.commit()
        except Exception:
            logger.exception("Failed to update buffered notification statuses user=%s", user_id)
            return True

        return True

    async def _get_camera_ctx_cached(self, camera_uuid_str: str) -> Optional[CameraContext]:
        if not self._session_factory:
            return None

        now = time.monotonic()
        leader = False
        pending: Optional[asyncio.Future] = None

        async with self._ctx_cache_lock:
            hit = self._ctx_cache.get(camera_uuid_str)
            if hit and hit[0] > now:
                return hit[1]
            pending = self._ctx_inflight.get(camera_uuid_str)
            if pending is None:
                pending = asyncio.get_running_loop().create_future()
                self._ctx_inflight[camera_uuid_str] = pending
                leader = True

        try:
            cam_uuid = uuid.UUID(str(camera_uuid_str))
        except Exception:
            if leader:
                async with self._ctx_cache_lock:
                    future = self._ctx_inflight.pop(camera_uuid_str, None)
                    if future is not None and not future.done():
                        future.set_result(None)
            return None

        if not leader and pending is not None:
            return await pending

        ctx: Optional[CameraContext] = None
        cancelled = False
        try:
            async with self._ctx_lookup_limit:
                async with self._session_factory() as db:
                    ctx = await self._repo.get_camera_context(db, camera_uuid=cam_uuid)
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception:
            logger.exception("Failed to load cached CameraContext camera=%s", camera_uuid_str)
            ctx = None
        finally:
            async with self._ctx_cache_lock:
                if not cancelled:
                    ttl_s = self._ctx_ttl_s if ctx else self._ctx_miss_ttl_s
                    if ttl_s > 0.0:
                        self._ctx_cache[camera_uuid_str] = (now + ttl_s, ctx)
                    else:
                        self._ctx_cache.pop(camera_uuid_str, None)
                future = self._ctx_inflight.pop(camera_uuid_str, None)
                if future is not None and not future.done():
                    if cancelled:
                        future.cancel()
                    else:
                        future.set_result(ctx)

        return ctx
    
    async def _prepare_notification_item(
        self,
        msg: NotificationMessage,
        ctx: CameraContext,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> BufferedNotification:
        extra_payload = await self._attach_clip_payload(
            msg=msg,
            ctx=ctx,
            extra_payload=extra_payload,
        )

        updated_fields: Dict[str, Any] = {}

        next_image_url = str((extra_payload or {}).get("image_url") or msg.image_url or "").strip()
        if next_image_url and not next_image_url.startswith("data:"):
            if next_image_url != str(msg.image_url or "").strip():
                updated_fields["image_url"] = next_image_url

        clip_payload = extra_payload.get("clip") if isinstance(extra_payload, dict) else None
        if isinstance(clip_payload, dict):
            next_clip_url = str(clip_payload.get("recording_url") or "").strip()
            next_clip_status = str(clip_payload.get("status") or "").strip()

            if next_clip_url and next_clip_url != str(msg.clip_url or "").strip():
                updated_fields["clip_url"] = next_clip_url
            if next_clip_status and next_clip_status != str(msg.clip_status or "").strip():
                updated_fields["clip_status"] = next_clip_status

        if updated_fields:
            msg = msg.model_copy(update=updated_fields)

        return BufferedNotification(
            msg=msg,
            ctx=ctx,
            extra_payload=extra_payload,
        )

    async def _persist_notification_now(
        self,
        msg: NotificationMessage,
        ctx: CameraContext,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._session_factory:
            return

        item = await self._prepare_notification_item(
            msg=msg,
            ctx=ctx,
            extra_payload=extra_payload,
        )

        ok = await self._flush_user_batch(int(ctx.user_id), [item])
        if not ok:
            logger.warning(
                "Failed to persist notification immediately user=%s camera=%s msg_id=%s",
                ctx.user_id,
                msg.camera_uuid,
                msg.id,
            )
            
    async def enqueue_notification(
        self,
        msg: NotificationMessage,
        ctx: CameraContext,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        await self._persist_notification_now(
            msg=msg,
            ctx=ctx,
            extra_payload=extra_payload,
        )
        
    async def _persist_and_send(
        self,
        msg: NotificationMessage,
        ctx: CameraContext,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        await self._persist_notification_now(
            msg=msg,
            ctx=ctx,
            extra_payload=extra_payload,
        )
        
    def _parse_roi_points(self, raw_points: Any) -> List[Tuple[float, float]]:
        if not isinstance(raw_points, (list, tuple)):
            return []
        points: List[Tuple[float, float]] = []
        for p in raw_points:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                continue
            try:
                x = float(p[0])
                y = float(p[1])
            except Exception:
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            points.append((x, y))
        return points

    def _coerce_roi_normalized(self, raw_normalized: Any, points: List[Tuple[float, float]]) -> bool:
        if isinstance(raw_normalized, bool):
            return raw_normalized
        if isinstance(raw_normalized, (int, float)):
            return bool(raw_normalized)
        if isinstance(raw_normalized, str):
            s = raw_normalized.strip().lower()
            if s in {"true", "1", "yes", "on"}:
                return True
            if s in {"false", "0", "no", "off"}:
                return False
        if not points:
            return True
        return all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for (x, y) in points)

    def _coerce_positive_int(self, raw_value: Any) -> Optional[int]:
        try:
            value = int(raw_value)
        except Exception:
            return None
        return value if value > 0 else None

    def _parse_roi_frame_size(self, raw_roi: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
        frame_w = self._coerce_positive_int(
            raw_roi.get("frame_w") or raw_roi.get("frameWidth") or raw_roi.get("width")
        )
        frame_h = self._coerce_positive_int(
            raw_roi.get("frame_h") or raw_roi.get("frameHeight") or raw_roi.get("height")
        )
        return frame_w, frame_h

    def _clamp_unit_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        return [(max(0.0, min(1.0, x)), max(0.0, min(1.0, y))) for (x, y) in points]

    def invalidate_camera_roi_state(self, camera_uuid: str) -> None:
        cam = str(camera_uuid)
        self._roi_cache.pop(cam, None)
        self._roi_engine.reset_camera(cam)

    async def _get_rois(self, camera_uuid: str) -> List[ROI]:
        if not self._session_factory:
            return []

        now = time.monotonic()
        hit = self._roi_cache.get(camera_uuid)
        if hit and hit[0] > now:
            return hit[1]

        try:
            from core.database_orm import Camera
            from sqlalchemy import select

            cam_uuid = uuid.UUID(str(camera_uuid))  # FIX: compare UUID to UUID

            async with self._session_factory() as session:
                result = await session.execute(
                    select(Camera.roi).where(Camera.camera_uuid == cam_uuid)
                )
                row = result.scalar_one_or_none()

            if not row or not isinstance(row, dict):
                rois: List[ROI] = []
                self._roi_cache[camera_uuid] = (now + self._roi_ttl_s, rois)
                return rois

            points = self._parse_roi_points(row.get("points", []))
            normalized = self._coerce_roi_normalized(row.get("normalized"), points)
            roi_frame_w, roi_frame_h = self._parse_roi_frame_size(row)

            if not points or len(points) < 3:
                rois = []
                self._roi_cache[camera_uuid] = (now + self._roi_ttl_s, rois)
                return rois

            roi_points = self._clamp_unit_points(points) if normalized else points
            rois = [
                ROI(
                    roi_id=f"{camera_uuid}-roi",
                    points=roi_points,
                    normalized=normalized,
                    frame_w=roi_frame_w,
                    frame_h=roi_frame_h,
                )
            ]
            self._roi_cache[camera_uuid] = (now + self._roi_ttl_s, rois)
            return rois

        except Exception as e:
            stale = self._roi_cache.get(camera_uuid)
            if stale:
                return stale[1]
            logger.warning("Failed to fetch ROI for %s: %s", camera_uuid, e)
            return []

    async def handle_detection_event(
        self,
        det_ev: DetectionsProducedEvent,
        *,
        camera_mode: CameraMode,
        frame_w: Optional[int] = None,
        frame_h: Optional[int] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        cam = str(det_ev.camera_uuid)
        ts_ms = int(det_ev.frame_ts_ms)

        if bool(getattr(camera_mode, "playback_enabled", True)):
            await self.record_detection_overlay_frame(
                camera_uuid=cam,
                frame_ts_ms=ts_ms,
                frame_seq=int(det_ev.frame_seq),
                frame_w=frame_w,
                frame_h=frame_h,
                detections=list(det_ev.detections or []),
            )

        if not camera_mode.notification_enabled or not camera_mode.detection_enabled:
            return

        matches = self._extract_interesting(det_ev.detections)
        if not matches:
            return

        ctx = await self._get_camera_ctx_cached(cam)
        if ctx is None:
            logger.warning("Skipping alert publish because camera context was not found camera=%s", cam)
            return

        site_name = ctx.site_name
        site_uuid_str = str(ctx.site_uuid)
        device_name = ctx.device_name
        camera_name = ctx.camera_name
        raw_image_url = str((extra_payload or {}).get("image_url") or "").strip() or None
        # Don't send base64 data: URLs over SSE — materialization to blob happens during flush.
        # extra_payload still carries raw_image_url so the flush path can store it correctly.
        image_url = None if (raw_image_url or "").startswith("data:") else raw_image_url
        overlay_payload = _event_overlay_payload(
            det_ev,
            frame_w=frame_w,
            frame_h=frame_h,
        )
        msg_frame_w = overlay_payload.get("frame_w")
        msg_frame_h = overlay_payload.get("frame_h")
        msg_frame_seq = overlay_payload.get("frame_seq")
        msg_detections = list(overlay_payload.get("detections") or [])

        # -------------------------
        # Tracking path
        # -------------------------
        if self.enable_tracking:
            dets = []
            for d in det_ev.detections or []:
                cls_name = getattr(d, "cls_name", None)
                if cls_name not in self.interesting:
                    continue
                conf = float(getattr(d, "conf", 0.0) or 0.0)
                b = getattr(d, "box", None)
                if not b:
                    continue
                dets.append({
                    "bbox": np.array([b.x1, b.y1, b.x2, b.y2], dtype=np.float32),
                    "cls_name": str(cls_name),
                    "conf": conf,
                })

            if not dets:
                return

            res = self._tracker.update(cam, dets, ts_ms=ts_ms)
            tracks = res["tracks"]
            events = res["events"]

            _any_notification_fired = False

            # A) notify-on-confirmed-track
            if self.notify_on_confirmed:
                for ev_type, track_id in events:
                    if ev_type != "track_confirmed":
                        continue
                    tr = next((t for t in tracks if int(t["track_id"]) == int(track_id)), None)
                    if not tr:
                        continue

                    msg = NotificationMessage(
                        user_id=int(ctx.user_id),
                        id=f"{cam}-trk{track_id}-confirm-{ts_ms}",
                        ts_ms=ts_ms,
                        camera_uuid=cam,
                        site_uuid=site_uuid_str,
                        site_name=site_name,
                        title=f"Item Detected: {tr['cls_name']}",
                        body=f"{tr['cls_name']} confirmed (track_id={track_id}, conf={float(tr['conf']):.2f})",
                        alert_type="item_detected",
                        cls_names=[tr["cls_name"]],
                        max_conf=float(tr["conf"]),
                        frame_w=msg_frame_w,
                        frame_h=msg_frame_h,
                        frame_seq=msg_frame_seq,
                        detections=msg_detections,
                        track_id=int(track_id),
                        device_name=device_name,
                        camera_name=camera_name,
                        image_url=image_url,
                    )

                    _any_notification_fired = True

                    self._fire_and_forget(
                        self._persist_and_send(
                                msg,
                                ctx,
                                extra_payload={
                                    **overlay_payload,
                                    **(extra_payload or {}),
                                    "track": _json_safe(tr),
                                    "event": "track_confirmed",
                                },
                            )
                        )

            # B) notify-on-ROI-enter
            if self.notify_on_roi_enter:
                rois = await self._get_rois(cam)
                if rois:
                    # Use frame dims from the alert payload; fall back to the dimensions
                    # recorded when the ROI polygon was drawn (stored on the ROI itself).
                    eff_frame_w = frame_w if frame_w is not None else rois[0].frame_w
                    eff_frame_h = frame_h if frame_h is not None else rois[0].frame_h
                    if eff_frame_w is None or eff_frame_h is None:
                        logger.warning(
                            "Skipping ROI check: no frame dimensions available camera=%s "
                            "(send frame_w/frame_h in the alert payload or store them when drawing the ROI)",
                            cam,
                        )
                    else:
                        roi_alerts = self._roi_engine.process(
                            camera_uuid=cam,
                            tracks=tracks,
                            rois=rois,
                            frame_w=int(eff_frame_w),
                            frame_h=int(eff_frame_h),
                            ts_ms=ts_ms,
                        )
                        for a in roi_alerts:
                            msg = NotificationMessage(
                                user_id=int(ctx.user_id),
                                id=f"{cam}-trk{a['track_id']}-roi{a['roi_id']}-{ts_ms}",
                                ts_ms=ts_ms,
                                camera_uuid=cam,
                                site_uuid=site_uuid_str,
                                site_name=site_name,
                                title=f"ROI Enter: {a['cls_name']}",
                                body=f"{a['cls_name']} entered ROI={a['roi_id']} (track_id={a['track_id']})",
                                alert_type="roi_enter",
                                cls_names=[a["cls_name"]],
                                max_conf=float(a["conf"]),
                                frame_w=msg_frame_w,
                                frame_h=msg_frame_h,
                                frame_seq=msg_frame_seq,
                                detections=msg_detections,
                                roi_id=str(a["roi_id"]),
                                track_id=int(a["track_id"]),
                                device_name=device_name,
                                camera_name=camera_name,
                                image_url=image_url,
                            )

                            await self.hub.publish(msg)
                            _any_notification_fired = True

                            self._fire_and_forget(
                                self._persist_and_send(
                                    msg,
                                    ctx,
                                    extra_payload={
                                        **overlay_payload,
                                        **(extra_payload or {}),
                                        "alert": _json_safe(a),
                                        "event": "roi_enter",
                                    },
                                )
                            )

            if not _any_notification_fired:
                now = time.monotonic()
                async with self._lock:
                    send_classes: List[str] = []
                    for cls_name, _conf in matches:
                        key = (cam, cls_name)
                        last = self._last_sent.get(key, 0.0)
                        if (now - last) >= self.cooldown_s:
                            self._last_sent[key] = now
                            send_classes.append(cls_name)

                if send_classes:
                    max_conf = max([c for (_n, c) in matches] or [0.0])
                    msg = NotificationMessage(
                        user_id=int(ctx.user_id),
                        id=f"{cam}-{ts_ms}-summary",
                        ts_ms=ts_ms,
                        camera_uuid=cam,
                        site_uuid=site_uuid_str,
                        site_name=site_name,
                        title=f"Detection: {', '.join(sorted(set(send_classes)))}",
                        body=f"Detected {', '.join(sorted(set(send_classes)))} (max_conf={max_conf:.2f})",
                        alert_type="detection_summary",
                        cls_names=sorted(set(send_classes)),
                        max_conf=float(max_conf),
                        frame_w=msg_frame_w,
                        frame_h=msg_frame_h,
                        frame_seq=msg_frame_seq,
                        detections=msg_detections,
                        device_name=device_name,
                        camera_name=camera_name,
                        image_url=image_url,
                    )

                    await self.hub.publish(msg)

                    self._fire_and_forget(
                        self._persist_and_send(
                            msg,
                            ctx,
                            extra_payload={
                                **overlay_payload,
                                **(extra_payload or {}),
                                "event": "detection_summary",
                            },
                        )
                    )

            return

        now = time.monotonic()
        async with self._lock:
            send_classes: List[str] = []
            for cls_name, _conf in matches:
                key = (cam, cls_name)
                last = self._last_sent.get(key, 0.0)
                if (now - last) >= self.cooldown_s:
                    self._last_sent[key] = now
                    send_classes.append(cls_name)

        if not send_classes:
            return

        max_conf = max([c for (_n, c) in matches] or [0.0])

        msg = NotificationMessage(
            user_id=int(ctx.user_id),
            id=f"{det_ev.camera_uuid}-{det_ev.frame_ts_ms}-{det_ev.frame_seq}",
            ts_ms=ts_ms,
            camera_uuid=cam,
            site_uuid=site_uuid_str,
            site_name=site_name,
            title=f"Detection: {', '.join(sorted(set(send_classes)))}",
            body=f"Detected {', '.join(sorted(set(send_classes)))} (max_conf={max_conf:.2f})",
            alert_type="detection_summary",
            cls_names=sorted(set(send_classes)),
            max_conf=float(max_conf),
            frame_w=msg_frame_w,
            frame_h=msg_frame_h,
            frame_seq=msg_frame_seq,
            detections=msg_detections,
            device_name=device_name,
            camera_name=camera_name,
            image_url=image_url,
        )

        await self.hub.publish(msg)

        self._fire_and_forget(
            self._persist_and_send(
                msg,
                ctx,
                extra_payload={
                    **overlay_payload,
                    **(extra_payload or {}),
                    "event": "detection_summary",
                },
            )
        )

    def _extract_interesting(self, detections: List[DetectionItem]) -> List[Tuple[str, float]]:
        out: List[Tuple[str, float]] = []
        for d in (detections or []):
            cls_name = getattr(d, "cls_name", None)
            conf = float(getattr(d, "conf", 0.0) or 0.0)
            if cls_name in self.interesting:
                out.append((cls_name, conf))
        return out
