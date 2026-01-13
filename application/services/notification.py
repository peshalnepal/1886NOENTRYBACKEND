# application/notifications/notification_service.py

import asyncio
import json
import time
from dataclasses import dataclass
from email.mime.text import MIMEText
from typing import Dict, Iterable, List, Optional, Set, Tuple

import smtplib
from pydantic import BaseModel, Field

from domain.events import DetectionsProducedEvent, DetectionItem  # your existing events


@dataclass(frozen=True)
class CameraMode:
    detection_enabled: bool = True
    notification_enabled: bool = True



class NotificationMessage(BaseModel):
    id: str
    ts_ms: int
    camera_uuid: str

    title: str
    body: str

    cls_names: List[str] = Field(default_factory=list)
    max_conf: Optional[float] = None



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
    to_emails: List[str]

    subject_prefix: str = "[AI Monitor]"


class EmailNotifier:
    def __init__(self, cfg: EmailConfig):
        self.cfg = cfg

    async def send(self, msg: NotificationMessage) -> None:
        if not self.cfg.enabled:
            return
        await asyncio.to_thread(self._send_sync, msg)

    def _send_sync(self, msg: NotificationMessage) -> None:
        subject = f"{self.cfg.subject_prefix} {msg.title}"
        body = f"{msg.body}\n\nCamera: {msg.camera_uuid}\nClasses: {', '.join(msg.cls_names)}\nTime: {msg.ts_ms}"

        m = MIMEText(body)
        m["Subject"] = subject
        m["From"] = self.cfg.from_email
        m["To"] = ", ".join(self.cfg.to_emails)

        with smtplib.SMTP(self.cfg.smtp_host, self.cfg.smtp_port, timeout=10) as server:
            if self.cfg.use_tls:
                server.starttls()
            if self.cfg.smtp_user and self.cfg.smtp_pass:
                server.login(self.cfg.smtp_user, self.cfg.smtp_pass)
            server.sendmail(self.cfg.from_email, self.cfg.to_emails, m.as_string())


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
        interesting_classes: Iterable[str] = ("person", "car"),
        cooldown_s: float = 10.0,
    ):
        self.hub = hub
        self.email = email
        self.interesting = set(interesting_classes)
        self.cooldown_s = float(cooldown_s)

        # (camera_uuid, cls) -> last_sent_monotonic
        self._last_sent: Dict[Tuple[str, str], float] = {}
        self._lock = asyncio.Lock()

    async def handle_detection_event(
        self,
        det_ev: DetectionsProducedEvent,
        *,
        camera_mode: CameraMode,
    ) -> None:
        # If notifications are OFF for camera, do nothing.
        if not camera_mode.notification_enabled:
            return

        # If detection is OFF, you normally won't get det_ev at all,
        # but this makes it safe if something slips through.
        if not camera_mode.detection_enabled:
            return

        matches = self._extract_interesting(det_ev.detections)
        if not matches:
            return

        # cooldown: send max 1 per class per camera per cooldown window
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

        msg = NotificationMessage(
            id=f"{det_ev.camera_uuid}-{det_ev.frame_ts_ms}-{det_ev.frame_seq}",
            ts_ms=int(det_ev.frame_ts_ms),
            camera_uuid=str(det_ev.camera_uuid),
            title=f"Detection: {', '.join(sorted(set(send_classes)))}",
            body=f"Detected {', '.join(sorted(set(send_classes)))} (max_conf={max_conf:.2f})",
            cls_names=sorted(set(send_classes)),
            max_conf=float(max_conf),
        )

        # fan-out (best-effort)
        await self.hub.publish(msg)
        if self.email:
            try:
                await self.email.send(msg)
            except Exception:
                # never crash pipeline because SMTP failed
                pass

    def _extract_interesting(self, detections: List[DetectionItem]) -> List[Tuple[str, float]]:
        out: List[Tuple[str, float]] = []
        for d in (detections or []):
            cls_name = getattr(d, "cls_name", None)
            conf = float(getattr(d, "conf", 0.0) or 0.0)
            if cls_name in self.interesting:
                out.append((cls_name, conf))
        return out
