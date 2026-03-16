# application/notifications/notification_service.py

import asyncio
from collections import defaultdict
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

from pydantic import BaseModel, Field
import numpy as np

from domain.events import DetectionsProducedEvent, DetectionItem
from application.services.tracker import MultiCameraByteTrack, ROIAlertEngine, ROI

# NEW: repository
from application.repositories.notification_repository import (
    NotificationRepository,
    CameraContext,
    dt_from_ts_ms,
)
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


@dataclass(frozen=True)
class CameraMode:
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

    device_name: Optional[str] = None
    camera_name: Optional[str] = None
    roi_id: Optional[str] = None
    track_id: Optional[int] = None


@dataclass(frozen=True)
class BufferedNotification:
    msg: NotificationMessage
    ctx: CameraContext
    extra_payload: Optional[Dict[str, Any]] = None


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
        self._ctx_cache: Dict[str, Tuple[float, CameraContext]] = {}
        self._ctx_ttl_s = 60.0  # reduce DB hits on frequent detections
        self._roi_cache: Dict[str, Tuple[float, List[ROI]]] = {}
        self._roi_ttl_s = 15.0
        self._recipient_cache: Dict[Tuple[int, str], Tuple[float, List[str]]] = {}
        self._recipient_ttl_s = _env_float("NOTIFICATION_RECIPIENT_CACHE_TTL_S", 60.0, minimum=1.0)

        self._buffer_lock = asyncio.Lock()
        self._flush_event = asyncio.Event()
        self._pending_by_user: Dict[int, List[BufferedNotification]] = {}
        self._pending_since: Dict[int, float] = {}
        self._active_flush_users: Set[int] = set()
        self._flush_task: Optional[asyncio.Task] = None
        self._closing = False
        self._buffer_max_items = _env_int("NOTIFICATION_BUFFER_MAX_ITEMS", 100, minimum=1)
        self._buffer_max_age_s = _env_float("NOTIFICATION_BUFFER_MAX_AGE_S", 60.0, minimum=1.0)
        self._buffer_poll_s = _env_float("NOTIFICATION_BUFFER_POLL_S", 1.0, minimum=0.2)
        self._clip_service = clip_service or EventClipService()

    def set_session_factory(self, session_factory):
        self._session_factory = session_factory
        if self._clip_service is not None:
            try:
                self._clip_service.set_session_factory(session_factory)
            except Exception:
                logger.exception("Failed to set session factory on EventClipService")

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

    def _fire_and_forget(self, coro):
        async def _runner():
            try:
                await coro
            except Exception:
                logger.exception("Notification background task failed")
        asyncio.create_task(_runner())

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
        if msg.alert_type != "roi_enter":
            return extra_payload

        clip = await clip_service.capture_pre_event_clip(
            camera_uuid=msg.camera_uuid,
            ctx=ctx,
            event_ts_ms=msg.ts_ms,
            trigger=msg.alert_type,
        )
        if not clip:
            return extra_payload

        merged = dict(extra_payload or {})
        merged["clip"] = clip
        return merged

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

    async def _flush_ready_users(self, *, force_all: bool = False) -> None:
        now = time.monotonic()
        ready: Dict[int, List[BufferedNotification]] = {}

        async with self._buffer_lock:
            for user_id, items in list(self._pending_by_user.items()):
                if user_id in self._active_flush_users:
                    continue

                since = self._pending_since.get(user_id, now)
                if force_all or len(items) >= self._buffer_max_items or (now - since) >= self._buffer_max_age_s:
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
                    existing = self._pending_by_user.get(user_id)
                    if existing:
                        self._pending_by_user[user_id] = list(items) + existing
                    else:
                        self._pending_by_user[user_id] = list(items)
                        self._pending_since[user_id] = time.monotonic() - self._buffer_max_age_s
                    self._flush_event.set()

                if user_id in self._pending_by_user and len(self._pending_by_user[user_id]) >= self._buffer_max_items:
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
                            "msg": item.msg.model_dump(),
                            "extra": item.extra_payload or {},
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
        try:
            async with self._session_factory() as db:
                rows = await self._repo.create_notifications(db, rows=create_rows)
                await db.commit()
        except Exception:
            logger.exception("Failed to persist buffered notifications user=%s count=%s", user_id, len(items))
            return False

        for site_uuid, indices in site_groups.items():
            notification_ids_by_site[site_uuid] = [
                int(rows[idx].id)
                for idx in indices
                if idx < len(rows) and getattr(rows[idx], "id", None) is not None
            ]

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
        hit = self._ctx_cache.get(camera_uuid_str)
        if hit and hit[0] > now:
            return hit[1]

        try:
            cam_uuid = uuid.UUID(str(camera_uuid_str))
        except Exception:
            return None

        async with self._session_factory() as db:
            ctx = await self._repo.get_camera_context(db, camera_uuid=cam_uuid)

        if ctx:
            self._ctx_cache[camera_uuid_str] = (now + self._ctx_ttl_s, ctx)

        return ctx

    async def enqueue_notification(
        self,
        msg: NotificationMessage,
        ctx: CameraContext,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._session_factory:
            return

        extra_payload = await self._attach_clip_payload(msg=msg, ctx=ctx, extra_payload=extra_payload)
        self._ensure_flush_task()
        user_id = int(ctx.user_id)
        item = BufferedNotification(
            msg=msg,
            ctx=ctx,
            extra_payload=extra_payload,
        )

        async with self._buffer_lock:
            bucket = self._pending_by_user.get(user_id)
            if bucket is None:
                bucket = []
                self._pending_by_user[user_id] = bucket
                self._pending_since[user_id] = time.monotonic()
            bucket.append(item)
            should_flush = len(bucket) >= self._buffer_max_items

        if should_flush:
            self._flush_event.set()

    async def _persist_and_send(
        self,
        msg: NotificationMessage,
        ctx: CameraContext,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        await self.enqueue_notification(msg, ctx, extra_payload=extra_payload)

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

            if not points or len(points) < 3:
                rois = []
                self._roi_cache[camera_uuid] = (now + self._roi_ttl_s, rois)
                return rois

            roi_points = self._clamp_unit_points(points) if normalized else points
            rois = [ROI(roi_id=f"{camera_uuid}-roi", points=roi_points, normalized=normalized)]
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
    ) -> None:
        if not camera_mode.notification_enabled or not camera_mode.detection_enabled:
            return

        matches = self._extract_interesting(det_ev.detections)
        if not matches:
            return

        cam = str(det_ev.camera_uuid)
        ts_ms = int(det_ev.frame_ts_ms)

        ctx = await self._get_camera_ctx_cached(cam)
        if ctx is None:
            logger.warning("Skipping alert publish because camera context was not found camera=%s", cam)
            return

        site_name = ctx.site_name
        site_uuid_str = str(ctx.site_uuid)
        device_name = ctx.device_name
        camera_name = ctx.camera_name

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
                        track_id=int(track_id),
                        device_name=device_name,
                        camera_name=camera_name,
                    )

                    await self.hub.publish(msg)

                    self._fire_and_forget(
                        self._persist_and_send(
                            msg,
                            ctx,
                            extra_payload={"track": _json_safe(tr), "event": "track_confirmed"},
                        )
                    )

            # B) notify-on-ROI-enter
            if self.notify_on_roi_enter:
                if frame_w is None or frame_h is None:
                    return

                rois = await self._get_rois(cam)
                if rois:
                    alerts = self._roi_engine.process(
                        camera_uuid=cam,
                        tracks=tracks,
                        rois=rois,
                        frame_w=int(frame_w),
                        frame_h=int(frame_h),
                        ts_ms=ts_ms,
                    )
                    for a in alerts:
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
                            roi_id=str(a["roi_id"]),
                            track_id=int(a["track_id"]),
                            device_name=device_name,
                            camera_name=camera_name,
                        )

                        await self.hub.publish(msg)

                        self._fire_and_forget(
                            self._persist_and_send(
                                msg,
                                ctx,
                                extra_payload={"alert": _json_safe(a), "event": "roi_enter"},
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
            device_name=device_name,
            camera_name=camera_name,
        )

        await self.hub.publish(msg)

        self._fire_and_forget(
            self._persist_and_send(
                msg,
                ctx,
                extra_payload={
                    "detections": _json_safe(
                        [
                            {
                                "cls_name": cls_name,
                                "conf": conf,
                            }
                            for cls_name, conf in matches
                        ]
                    ),
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
