# application/notifications/notification_service.py

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from email.mime.text import MIMEText
from typing import Dict, Iterable, List, Optional, Set, Tuple
import html
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import smtplib
from pydantic import BaseModel, Field

from domain.events import DetectionsProducedEvent, DetectionItem  # your existing events
import numpy as np
from application.services.tracker import (
    MultiCameraByteTrack,
    ROIAlertEngine,
    ROI,
)
import os

SMTP_USERNAME=os.environ.get("SMTP_USERNAME")
SMTP_PASSWORD=os.environ.get("SMTP_PASSWORD")
logger = logging.getLogger(__name__)
@dataclass(frozen=True)
class CameraMode:
    detection_enabled: bool = True
    notification_enabled: bool = True



class NotificationMessage(BaseModel):
    id: str
    ts_ms: int
    camera_uuid: str
    site_uuid: str
    site_name: str

    title: str
    body: str

    cls_names: List[str] = Field(default_factory=list)
    max_conf: Optional[float] = None

    # Optional: handy for email templates / UI later
    device_name: Optional[str] = None
    camera_name: Optional[str] = None


class WebNotificationHub:
    """
    Simple in-memory pubsub for browser notifications.
    Each connected client gets its own asyncio.Queue.
    """
    def __init__(self, max_q: int = 200):
        self._subs: Set[asyncio.Queue[NotificationMessage]] = set()
        self._lock = asyncio.Lock()
        self._max_q = max_q

    async def subscribe(self) -> asyncio.Queue[NotificationMessage]:
        q: asyncio.Queue[NotificationMessage] = asyncio.Queue(maxsize=self._max_q)
        async with self._lock:
            self._subs.add(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue[NotificationMessage]) -> None:
        async with self._lock:
            self._subs.discard(q)

    async def publish(self, msg: NotificationMessage) -> None:
        async with self._lock:
            subs = list(self._subs)

        # best-effort: never block pipeline
        for q in subs:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                # drop oldest then try once
                try:
                    _ = q.get_nowait()
                except Exception:
                    pass
                try:
                    q.put_nowait(msg)
                except Exception:
                    pass


# -----------------------------
# Email notifier (SMTP)
# -----------------------------
class EmailConfig(BaseModel):
    enabled: bool = True

    smtp_host: str
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_pass: Optional[str] = None
    use_tls: bool = True

    from_email: str

    # FIX: recipients should live here OR be provided at send-time (we do both)
    to_emails: List[str] = Field(default_factory=list)

    subject_prefix: str = "[1886NOENTRY Alert]"

    # Optional: link users back to your frontend (button in email)
    dashboard_base_url: Optional[str] = None  # e.g. "https://your-frontend.com"
    camera_path_template: str = "/app?camera={camera_uuid}"  # customize for your UI


class EmailNotifier:
    def __init__(self, cfg: EmailConfig):
        self.cfg = cfg

    async def send(self, msg: NotificationMessage, to_emails: Optional[List[str]] = None) -> None:
        if not self.cfg.enabled:
            return

        recipients = (to_emails or []) or self.cfg.to_emails
        if not recipients:
            # No recipients configured; quietly skip (or log if you prefer)
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

        # Plain text first, HTML second
        m.attach(MIMEText(text_body, "plain", "utf-8"))
        m.attach(MIMEText(html_body, "html", "utf-8"))

        with smtplib.SMTP(self.cfg.smtp_host, self.cfg.smtp_port, timeout=10) as server:
            if self.cfg.use_tls:
                server.starttls()
            if self.cfg.smtp_user and self.cfg.smtp_pass:
                server.login(self.cfg.smtp_user, self.cfg.smtp_pass)

            # FIX: sendmail needs recipients list
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

        # optional button
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
            <!-- Header -->
            <tr>
              <td style="padding:18px 24px;background:#0b1220;color:#ffffff;font-family:Arial,sans-serif;">
                <div style="font-size:14px;opacity:0.85;font-weight:700;letter-spacing:0.3px;">1886NOENTRY</div>
                <div style="font-size:22px;font-weight:800;margin-top:6px;">AI Monitor Alert</div>
              </td>
            </tr>

            <!-- Summary card -->
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

            <!-- Footer -->
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


# -----------------------------
# Main service (filters + cooldown + fan-out)
# -----------------------------

class NotificationService:
    """
    Consumes DetectionsProducedEvent and sends:
      - Email notification
      - Web notification (top bar via SSE)

    Includes:
      - per-camera mode checks (detection_enabled/notification_enabled)
      - cooldown to prevent spam
      - class filter (person/car)
    """
    def __init__(
        self,
        *,
        hub: WebNotificationHub,
        email: Optional[EmailNotifier] = None,
        interesting_classes: Iterable[str] = ("person", "car","motorcycle","truck"),
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

        # (camera_uuid, cls) -> last_sent_monotonic
        self._last_sent: Dict[Tuple[str, str], float] = {}
        self._lock = asyncio.Lock()

        self.enable_tracking = bool(enable_tracking)
        self.notify_on_confirmed = bool(notify_on_confirmed)
        self.notify_on_roi_enter = bool(notify_on_roi_enter)
        self.roi_provider = roi_provider

        cfg = tracker_cfg or {}
        self._tracker = MultiCameraByteTrack(**cfg)
        self._roi_engine = ROIAlertEngine()
        
        # Session factory for database lookups (ROI, notification emails)
        self._session_factory = None

    def set_session_factory(self, session_factory):
        """Set the session factory for database lookups."""
        self._session_factory = session_factory

    async def _get_rois(self, camera_uuid: str) -> List[ROI]:
        """Fetch ROI from database for a camera."""
        if not self._session_factory:
            return []
        
        try:
            from core.database_orm import Camera
            from sqlalchemy import select
            
            async with self._session_factory() as session:
                result = await session.execute(
                    select(Camera.roi).where(Camera.camera_uuid == camera_uuid)
                )
                row = result.scalar_one_or_none()
                
                if not row or not isinstance(row, dict):
                    return []
                
                points = row.get("points", [])
                normalized = row.get("normalized", True)
                
                if not points or len(points) < 3:
                    return []
                
                # Convert to tuple format expected by ROI
                roi_points = [(float(p[0]), float(p[1])) for p in points]
                return [ROI(roi_id=f"{camera_uuid}-roi", points=roi_points, normalized=normalized)]
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to fetch ROI for {camera_uuid}: {e}")
            return []

    async def _get_notification_emails(self, camera_uuid: str) -> List[str]:
        """Fetch notification emails from database for the camera's user."""
        if not self._session_factory:
            return []
        
        try:
            from core.database_orm import Camera, NotificationEmail
            from sqlalchemy import select
            from sqlalchemy.orm import selectinload
            
            async with self._session_factory() as session:
                # Get user_id from camera
                result = await session.execute(
                    select(Camera.user_id).where(Camera.camera_uuid == camera_uuid)
                )
                user_id = result.scalar_one_or_none()
                
                if not user_id:
                    return []
                
                # Get notification emails for user
                result = await session.execute(
                    select(NotificationEmail.email).where(NotificationEmail.user_id == user_id)
                )
                emails = [row[0] for row in result.fetchall()]
                return emails
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to fetch notification emails for {camera_uuid}: {e}")
            return []

    async def _get_site_name_and_uuid(self, camera_uuid: str) -> Tuple[str, str]:
        """Fetch site name and uuid from database for a camera."""
        if not self._session_factory:
            return "Unknown Site", ""
        
        try:
            from core.database_orm import Camera, Site
            from sqlalchemy import select
            
            async with self._session_factory() as session:
                result = await session.execute(
                    select(Site.name, Site.site_uuid).join(Camera, Camera.site_uuid == Site.site_uuid)
                    .where(Camera.camera_uuid == camera_uuid)
                )
                row = result.first()
                if row:
                    name, suid = row
                    return name, str(suid)
                return "Unknown Site", ""
        except Exception:
            return "Unknown Site", ""


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

        if self.enable_tracking:
            cam = str(det_ev.camera_uuid)
            ts_ms = int(det_ev.frame_ts_ms)

            # Convert to tracker detection format
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

            # A) notify-on-confirmed (fires once per track when it becomes confirmed)
            if self.notify_on_confirmed:
                for ev_type, track_id in events:
                    if ev_type != "track_confirmed":
                        continue
                    # find that track
                    tr = next((t for t in tracks if int(t["track_id"]) == int(track_id)), None)
                    if not tr:
                        continue

                    # Fetch site name and uuid for notification
                    site_name, site_uuid = await self._get_site_name_and_uuid(cam)
                    
                    msg = NotificationMessage(
                        id=f"{cam}-trk{track_id}-confirm-{ts_ms}",
                        ts_ms=ts_ms,
                        camera_uuid=cam,
                        site_uuid=site_uuid,
                        site_name=site_name,
                        title=f"Confirmed: {tr['cls_name']}",
                        body=f"{tr['cls_name']} confirmed (track_id={track_id}, conf={float(tr['conf']):.2f})",
                        cls_names=[tr["cls_name"]],
                        max_conf=float(tr["conf"]),
                    )
                    await self.hub.publish(msg)
                    if self.email:
                        try:
                            # Fetch notification emails from database
                            to_emails = await self._get_notification_emails(cam)
                            await self.email.send(msg, to_emails=to_emails if to_emails else None)
                        except Exception:
                            logger.exception("Email send failed for confirmed track camera=%s track_id=%s", cam, track_id)

            # B) notify-on-ROI-enter (fires once per ROI enter per track)
            if self.notify_on_roi_enter:
                if frame_w is None or frame_h is None:
                    # ROI normalized needs dimensions; if your ROIs are pixel-based you can pass any.
                    # Best is to include frame_w/h in payload (shown below).
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
                        # Fetch site name and uuid for notification
                        site_name, site_uuid = await self._get_site_name_and_uuid(cam)
                        
                        msg = NotificationMessage(
                            id=f"{cam}-trk{a['track_id']}-roi{a['roi_id']}-{ts_ms}",
                            ts_ms=ts_ms,
                            camera_uuid=cam,
                            site_uuid=site_uuid,
                            site_name=site_name,
                            title=f"ROI Enter: {a['cls_name']}",
                            body=f"{a['cls_name']} entered ROI={a['roi_id']} (track_id={a['track_id']})",
                            cls_names=[a["cls_name"]],
                            max_conf=float(a["conf"]),
                        )
                        await self.hub.publish(msg)
                        if self.email:
                            try:
                                # Fetch notification emails from database
                                to_emails = await self._get_notification_emails(cam)
                                await self.email.send(msg, to_emails=to_emails if to_emails else None)
                            except Exception:
                                logger.exception(
                                    "Email send failed for ROI enter camera=%s track_id=%s roi_id=%s",
                                    cam,
                                    a.get("track_id"),
                                    a.get("roi_id"),
                                )

            return


        if not camera_mode.notification_enabled:
            return
        if not camera_mode.detection_enabled:
            return

        matches = self._extract_interesting(det_ev.detections)
        if not matches:
            return

        now = time.monotonic()
        async with self._lock:
            send_classes: List[str] = []
            for cls_name, _conf in matches:
                key = (str(det_ev.camera_uuid), cls_name)
                last = self._last_sent.get(key, 0.0)
                if (now - last) >= self.cooldown_s:
                    self._last_sent[key] = now
                    send_classes.append(cls_name)

        if not send_classes:
            return

        max_conf = max([c for (_n, c) in matches] or [0.0])

        # Fetch site name and uuid for notification
        cam = str(det_ev.camera_uuid)
        site_name, site_uuid = await self._get_site_name_and_uuid(cam)
 
        msg = NotificationMessage(
            id=f"{det_ev.camera_uuid}-{det_ev.frame_ts_ms}-{det_ev.frame_seq}",
            ts_ms=int(det_ev.frame_ts_ms),
            camera_uuid=cam,
            site_uuid=site_uuid,
            site_name=site_name,
            title=f"Detection: {', '.join(sorted(set(send_classes)))}",
            body=f"Detected {', '.join(sorted(set(send_classes)))} (max_conf={max_conf:.2f})",
            cls_names=sorted(set(send_classes)),
            max_conf=float(max_conf),
        )

        await self.hub.publish(msg)
        if self.email:
            try:
                # Fetch notification emails from database
                to_emails = await self._get_notification_emails(cam)
                await self.email.send(msg, to_emails=to_emails if to_emails else None)
            except Exception:
                logger.exception("Email send failed for detection summary camera=%s", cam)

    def _extract_interesting(self, detections: List[DetectionItem]) -> List[Tuple[str, float]]:
        out: List[Tuple[str, float]] = []
        for d in (detections or []):
            cls_name = getattr(d, "cls_name", None)
            conf = float(getattr(d, "conf", 0.0) or 0.0)
            if cls_name in self.interesting:
                out.append((cls_name, conf))
        return out
