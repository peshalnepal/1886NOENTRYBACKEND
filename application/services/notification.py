# application/notifications/notification_service.py

import asyncio
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

SMTP_USERNAME = os.environ.get("SMTP_USERNAME")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")

logger = logging.getLogger(__name__)


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
        notify_on_confirmed: bool = True,
        notify_on_roi_enter: bool = True,
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

    def set_session_factory(self, session_factory):
        self._session_factory = session_factory

    def _fire_and_forget(self, coro):
        async def _runner():
            try:
                await coro
            except Exception:
                logger.exception("Notification background task failed")
        asyncio.create_task(_runner())

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

    async def _persist_and_send(self, msg: NotificationMessage, ctx: CameraContext) -> None:
        """
        Store notification row in DB (site/camera/user scoped),
        then send email (site-scoped recipients),
        then mark sent/failed.
        """
        if not self._session_factory:
            return

        notification_id: Optional[int] = None
        recipients: List[str] = []

        try:
            # 1) insert row (status=created) + load recipients
            async with self._session_factory() as db:
                notif = await self._repo.create_notification(
                    db,
                    user_id=ctx.user_id,
                    site_uuid=ctx.site_uuid,
                    camera_uuid=uuid.UUID(msg.camera_uuid),
                    device_uuid=ctx.device_uuid,
                    event_type=msg.alert_type,
                    title=msg.title,
                    message=msg.body,
                    payload={"msg": msg.model_dump()},
                    detected_at=dt_from_ts_ms(msg.ts_ms),
                    status="created",
                    sent_at=None,
                )
                notification_id = int(notif.id)

                recipients = await self._repo.list_notification_emails_for_site(
                    db,
                    user_id=ctx.user_id,
                    site_uuid=ctx.site_uuid,
                    only_enabled=True,
                )

                await db.commit()

            # 2) email send (optional)
            sent_ok = False
            if self.email:
                try:
                    await self.email.send(msg, to_emails=recipients if recipients else None)
                    sent_ok = True
                except Exception:
                    logger.exception("Email send failed camera=%s", msg.camera_uuid)
                    sent_ok = False

            # 3) mark sent/failed
            if notification_id is not None:
                async with self._session_factory() as db2:
                    if sent_ok:
                        await self._repo.mark_notification_sent(db2, notification_id=notification_id)
                    else:
                        # If email is disabled/no recipients, you can keep status=created,
                        # but if you want a failure marker when email attempted and failed:
                        if self.email and recipients:
                            await self._repo.mark_notification_failed(db2, notification_id=notification_id)
                    await db2.commit()

        except Exception:
            logger.exception("Persist+send failed camera=%s", msg.camera_uuid)
            if notification_id is not None and self._session_factory:
                try:
                    async with self._session_factory() as db3:
                        await self._repo.mark_notification_failed(db3, notification_id=notification_id)
                        await db3.commit()
                except Exception:
                    logger.exception("Failed to mark notification failed id=%s", notification_id)

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

    async def _get_rois(self, camera_uuid: str) -> List[ROI]:
        if not self._session_factory:
            return []

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
                return []

            points = self._parse_roi_points(row.get("points", []))
            normalized = self._coerce_roi_normalized(row.get("normalized"), points)

            if not points or len(points) < 3:
                return []

            roi_points = self._clamp_unit_points(points) if normalized else points
            return [ROI(roi_id=f"{camera_uuid}-roi", points=roi_points, normalized=normalized)]

        except Exception as e:
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

                    self._fire_and_forget(self._persist_and_send(msg, ctx))

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

                        self._fire_and_forget(self._persist_and_send(msg, ctx))

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

        self._fire_and_forget(self._persist_and_send(msg, ctx))

    def _extract_interesting(self, detections: List[DetectionItem]) -> List[Tuple[str, float]]:
        out: List[Tuple[str, float]] = []
        for d in (detections or []):
            cls_name = getattr(d, "cls_name", None)
            conf = float(getattr(d, "conf", 0.0) or 0.0)
            if cls_name in self.interesting:
                out.append((cls_name, conf))
        return out
