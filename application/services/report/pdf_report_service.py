"""Organization "approved alerts" PDF report generator.

Rolls up every operator-approved alert for an organization — across all of its
sites — into a single comprehensive PDF and emails it to every organization
member plus the per-site notification recipients.

Each alert entry carries the operator notes attached during review, the alert
snapshot image (embedded inline), and a clickable link to the recorded video
clip. The service is stateless aside from its session factory, so a route can
construct one per request (like the repositories) or a singleton can be reused.

Pipeline:
  org_id -> org sites -> operator-approved + visible Notification rows
        -> extract notes / image_url / clip_url from each payload
        -> fetch + normalize the snapshot images (bounded concurrency)
        -> render PDF (application.services.report.pdf_builder)
        -> email with the PDF attached to members + notification recipients
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

from core.env import env_float, env_int
from application.dtos import ReportCreateDTO
from application.repositories.notification_repository import NotificationRepository
from application.repositories.organization_repository import OrganizationRepository
from application.repositories.report_repository import ReportRepository
from application.services.report.pdf_builder import PDFReport, normalize_to_jpeg

logger = logging.getLogger(__name__)

_DATA_URL_RE = re.compile(r"^data:(?P<mime>[^;,]+)?(?P<b64>;base64)?,(?P<data>.*)$", re.DOTALL)

_REPORT_TYPE_LABELS = {"general": "General Alert Report", "urgent": "Urgent Alert Report"}

# Pipeline alert types are machine tokens ("roi_enter"); the report shows the
# reader-facing phrasing instead. Unknown types fall back to a de-slugged label.
_ALERT_TYPE_LABELS = {
    "roi_enter": "Restricted zone entry",
    "any_detection": "Detection in view",
}

# Classes treated as vehicles for the structured-note field set.
_VEHICLE_CLASSES = {"car", "truck", "motorcycle", "bus", "van", "vehicle"}


def _classify(classes: List[str]) -> str:
    """Bucket an alert's detected classes into 'vehicle' | 'person' | 'other'."""
    lowered = {str(c).strip().lower() for c in (classes or [])}
    if lowered & _VEHICLE_CLASSES:
        return "vehicle"
    if "person" in lowered:
        return "person"
    return "other"


@dataclass
class _AlertEntry:
    detected_at: Optional[datetime]
    title: str
    body: str
    site_uuid: str
    site_name: str
    site_location: str
    site_tz: str
    camera_name: str
    camera_location: str
    alert_type: str
    classes: List[str]
    max_conf: Optional[float]
    notes: List[Dict[str, Any]]
    image_url: Optional[str]
    clip_url: Optional[str]
    approved_at: Optional[datetime]
    # Fetched + normalized snapshot for the in-table thumbnail.
    image_jpeg: Optional[Tuple[bytes, int, int, int]] = None


@dataclass
class ReportResult:
    """Outcome of a report run, returned to the caller (and the route)."""

    report_type: str
    report_id: Optional[int]
    report_uuid: str
    org_id: int
    org_name: str
    alert_count: int
    image_count: int
    site_count: int
    start: Optional[datetime]
    end: Optional[datetime]
    recipients: List[str] = field(default_factory=list)
    emailed: bool = False
    pdf_bytes: bytes = b""
    filename: str = "report.pdf"


class PdfReportGenerator:
    def __init__(
        self,
        *,
        session_factory,
        email=None,
        dashboard_base_url: Optional[str] = None,
    ) -> None:
        self._session_factory = session_factory
        self._email = email
        self._dashboard_base_url = (dashboard_base_url or "").rstrip("/") or None
        self._notif_repo = NotificationRepository()
        self._org_repo = OrganizationRepository()
        self._report_repo = ReportRepository()

        self._max_alerts = env_int("REPORT_MAX_ALERTS", 500, minimum=1)
        # Event-snapshot thumbnails embedded in the table's Image column.
        self._max_images = env_int("REPORT_MAX_IMAGES", 300, minimum=0)
        self._image_fetch_concurrency = env_int("REPORT_IMAGE_FETCH_CONCURRENCY", 8, minimum=1)
        self._image_fetch_timeout_s = env_float("REPORT_IMAGE_FETCH_TIMEOUT_S", 15.0, minimum=1.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def generate_and_send(
        self,
        *,
        org_id: int,
        report_type: str = "general",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        window_hours: Optional[int] = None,
        operator_approved_only: bool = True,
        extra_recipients: Optional[List[str]] = None,
        notification_ids: Optional[List[int]] = None,
        persist: bool = True,
        generated_by: Optional[int] = None,
        generated_by_email: Optional[str] = None,
    ) -> ReportResult:
        """Build the org's alert-report PDF, archive it, and email it.

        ``window_hours`` is a convenience: when given and ``start`` is omitted,
        the window becomes ``[now - window_hours, now]``. ``extra_recipients``
        are merged in on top of the org members and site notification emails.
        """
        if window_hours is not None and start is None:
            start = datetime.now(timezone.utc) - _timedelta_hours(window_hours)

        result = await self.build_report(
            org_id=org_id,
            report_type=report_type,
            start=start,
            end=end,
            operator_approved_only=operator_approved_only,
            extra_recipients=extra_recipients,
            notification_ids=notification_ids,
            persist=persist,
            generated_by=generated_by,
            generated_by_email=generated_by_email,
        )

        if self._email is None:
            logger.info("Report generated for org=%s but no email transport configured", org_id)
            return result

        if not result.recipients:
            logger.info("Report generated for org=%s but no recipients resolved", org_id)
            return result

        subject = self._subject(result)
        try:
            emailed = await self._email.send_report(
                subject=subject,
                html_body=self._email_html(result),
                text_body=self._email_text(result),
                attachment_bytes=result.pdf_bytes,
                attachment_filename=result.filename,
                to_emails=result.recipients,
            )
            result.emailed = bool(emailed)
        except Exception:
            logger.exception("Failed to email approved-alerts report org=%s", org_id)
            result.emailed = False

        return result

    async def build_report(
        self,
        *,
        org_id: int,
        report_type: str = "general",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        operator_approved_only: bool = True,
        extra_recipients: Optional[List[str]] = None,
        notification_ids: Optional[List[int]] = None,
        persist: bool = False,
        generated_by: Optional[int] = None,
        generated_by_email: Optional[str] = None,
    ) -> ReportResult:
        """Gather the data, render the PDF, and optionally archive it.

        Content selection by ``report_type``:
          - ``"urgent"``  — alerts an operator approved WITH the email opt-in
            (they were pushed to users immediately). Usually driven by explicit
            ``notification_ids`` from the approval route.
          - ``"general"`` — approved alerts that were NOT emailed at approval;
            these roll up into the daily report.
        """
        report_type = "urgent" if str(report_type).lower() == "urgent" else "general"
        if self._session_factory is None:
            raise RuntimeError("PdfReportGenerator has no session factory")

        async with self._session_factory() as db:
            org = await self._org_repo.get_by_id(db, int(org_id))
            org_name = getattr(org, "name", None) or f"Organization {org_id}"
            prepared_for = await self._resolve_owner_name(db, org=org, org_name=org_name)

            sites = await self._org_repo.list_org_sites(db, org_id=int(org_id))
            site_name_by_uuid: Dict[uuid.UUID, str] = {
                s.site_uuid: (s.name or "Unknown Site") for s in sites
            }
            site_location_by_uuid: Dict[uuid.UUID, str] = {
                s.site_uuid: (s.address or "") for s in sites
            }
            site_tz_by_uuid: Dict[uuid.UUID, str] = {
                s.site_uuid: (s.timezone or "UTC") for s in sites
            }
            site_uuids = list(site_name_by_uuid.keys())

            # Camera locations (Camera.location) for the record "Camera" row.
            camera_location_by_uuid: Dict[uuid.UUID, str] = {}
            if site_uuids:
                from sqlalchemy import select as _select
                from core.database_orm import Camera as _Camera

                cam_rows = (
                    await db.execute(
                        _select(_Camera.camera_uuid, _Camera.location).where(
                            _Camera.site_uuid.in_(site_uuids)
                        )
                    )
                ).all()
                camera_location_by_uuid = {cu: (loc or "") for cu, loc in cam_rows}

            rows: List[Any] = []
            if site_uuids:
                if notification_ids:
                    # Urgent path: exactly the alerts the operator just decided,
                    # still scoped to the org's sites.
                    rows = await self._notif_repo.list_notifications(
                        db,
                        ids=[int(i) for i in notification_ids],
                        site_uuids=site_uuids,
                        approval_status="approved",
                        order_desc=False,
                        limit=self._max_alerts,
                    )
                else:
                    rows = await self._notif_repo.list_notifications(
                        db,
                        site_uuids=site_uuids,
                        approval_status="approved",
                        only_visible=True,
                        emailed=(report_type == "urgent"),
                        detected_after=start,
                        detected_before=end,
                        order_desc=False,
                        limit=self._max_alerts,
                    )

            recipients = await self._resolve_recipients(
                db, org_id=int(org_id), site_uuids=site_uuids, extra=extra_recipients
            )

        entries: List[_AlertEntry] = []
        for row in rows:
            if operator_approved_only and getattr(row, "approved_by", None) is None:
                continue
            entries.append(
                self._entry_from_row(
                    row,
                    site_name_by_uuid,
                    site_location_by_uuid,
                    site_tz_by_uuid,
                    camera_location_by_uuid,
                )
            )

        # Urgent-by-ids has no explicit window; derive it from the alerts.
        if notification_ids and entries and start is None and end is None:
            times = [e.detected_at for e in entries if isinstance(e.detected_at, datetime)]
            if times:
                start, end = min(times), max(times)

        await self._attach_images(entries)
        image_count = sum(1 for e in entries if e.image_jpeg is not None)

        # One stable id shared by the rendered PDF and the archived row so the
        # printed "Report ID" matches what members search for in the archive.
        report_uuid = str(uuid.uuid4())

        pdf_bytes = self._render_pdf(
            report_uuid=report_uuid,
            report_type=report_type,
            prepared_for=prepared_for,
            entries=entries,
            start=start,
            end=end,
            site_count=len(site_uuids),
        )

        result = ReportResult(
            report_type=report_type,
            report_id=None,
            report_uuid=report_uuid,
            org_id=int(org_id),
            org_name=org_name,
            alert_count=len(entries),
            image_count=image_count,
            site_count=len(site_uuids),
            start=start,
            end=end,
            recipients=recipients,
            pdf_bytes=pdf_bytes,
            filename=self._filename(report_type, report_uuid),
        )

        if persist:
            result.report_id = await self._persist_report(
                result=result,
                entries=entries,
                generated_by=generated_by,
                generated_by_email=generated_by_email,
            )

        return result

    async def _persist_report(
        self,
        *,
        result: ReportResult,
        entries: List[_AlertEntry],
        generated_by: Optional[int],
        generated_by_email: Optional[str],
    ) -> Optional[int]:
        """Archive the rendered PDF so members can browse/download it later."""
        try:
            dto = ReportCreateDTO(
                org_id=result.org_id,
                report_uuid=result.report_uuid,
                report_type=result.report_type,
                filename=result.filename,
                generated_by=generated_by,
                generated_by_email=generated_by_email,
                site_uuids=sorted({str(e.site_uuid) for e in entries if e.site_uuid}),
                period_start=result.start,
                period_end=result.end,
                alert_count=result.alert_count,
                pdf_data=result.pdf_bytes,
            )
            async with self._session_factory() as db:
                row = await self._report_repo.create(db, dto=dto)
                await db.commit()
                return int(row.id)
        except Exception:
            logger.exception(
                "Failed to archive %s report org=%s", result.report_type, result.org_id
            )
            return None

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------
    def _entry_from_row(
        self,
        row: Any,
        site_name_by_uuid: Dict[uuid.UUID, str],
        site_location_by_uuid: Dict[uuid.UUID, str],
        site_tz_by_uuid: Optional[Dict[uuid.UUID, str]] = None,
        camera_location_by_uuid: Optional[Dict[uuid.UUID, str]] = None,
    ) -> _AlertEntry:
        payload = row.payload if isinstance(getattr(row, "payload", None), dict) else {}
        msg = payload.get("msg") if isinstance(payload.get("msg"), dict) else {}
        extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}

        notes = payload.get("notes")
        notes = list(notes) if isinstance(notes, list) else []

        classes = msg.get("cls_names")
        classes = [str(c) for c in classes] if isinstance(classes, list) else []

        max_conf = msg.get("max_conf")
        try:
            max_conf = float(max_conf) if max_conf is not None else None
        except (TypeError, ValueError):
            max_conf = None

        row_site_uuid = getattr(row, "site_uuid", None)
        site_name = (
            site_name_by_uuid.get(row_site_uuid)
            or msg.get("site_name")
            or "Unknown Site"
        )
        site_location = site_location_by_uuid.get(row_site_uuid, "") or ""
        site_tz = (site_tz_by_uuid or {}).get(row_site_uuid, "UTC") or "UTC"
        camera_location = (camera_location_by_uuid or {}).get(
            getattr(row, "camera_uuid", None), ""
        ) or ""

        return _AlertEntry(
            detected_at=getattr(row, "detected_at", None),
            title=str(getattr(row, "title", None) or msg.get("title") or "Alert"),
            body=str(getattr(row, "message", None) or msg.get("body") or ""),
            site_uuid=str(row_site_uuid) if row_site_uuid else "",
            site_name=str(site_name),
            site_location=str(site_location),
            site_tz=str(site_tz),
            camera_name=str(msg.get("camera_name") or msg.get("camera_uuid") or "Unknown Camera"),
            camera_location=str(camera_location),
            alert_type=str(getattr(row, "event_type", None) or msg.get("alert_type") or "detection"),
            classes=classes,
            max_conf=max_conf,
            notes=notes,
            image_url=self._extract_image_url(msg, extra),
            clip_url=self._extract_clip_url(msg, extra),
            approved_at=getattr(row, "approved_at", None),
        )

    async def _resolve_owner_name(self, db: Any, *, org: Any, org_name: str) -> str:
        """The organization owner/admin's display name for the report header.

        Falls back to the org name when the owner is unset or not found.
        """
        owner_id = getattr(org, "owner_user_id", None) if org is not None else None
        if owner_id is None:
            return org_name
        try:
            from core.database_orm import User
            from sqlalchemy import select

            name = (
                await db.execute(select(User.user_name).where(User.id == int(owner_id)))
            ).scalar_one_or_none()
            return str(name) if name else org_name
        except Exception:
            logger.warning("Failed to resolve org owner name org=%s", org_name, exc_info=True)
            return org_name

    @staticmethod
    def _extract_image_url(msg: Dict[str, Any], extra: Dict[str, Any]) -> Optional[str]:
        for src in (msg.get("image_url"), extra.get("image_url")):
            val = str(src or "").strip()
            if val:
                return val
        return None

    @staticmethod
    def _extract_clip_url(msg: Dict[str, Any], extra: Dict[str, Any]) -> Optional[str]:
        val = str(msg.get("clip_url") or "").strip()
        if val:
            return val
        clip = extra.get("clip") if isinstance(extra.get("clip"), dict) else None
        if clip:
            val = str(clip.get("recording_url") or "").strip()
            if val:
                return val
        return None

    async def _resolve_recipients(
        self,
        db: Any,
        *,
        org_id: int,
        site_uuids: List[uuid.UUID],
        extra: Optional[List[str]],
    ) -> List[str]:
        emails: List[str] = []

        try:
            members = await self._org_repo.list_org_members(db, org_id=org_id)
            for _role, user in members:
                addr = str(getattr(user, "email", "") or "").strip()
                if addr:
                    emails.append(addr)
        except Exception:
            logger.exception("Failed to list org members for report recipients org=%s", org_id)

        if site_uuids:
            try:
                rows = await self._notif_repo.list_notification_email_rows(
                    db, site_uuids=site_uuids, only_enabled=True
                )
                for r in rows:
                    addr = str(getattr(r, "email", "") or "").strip()
                    if addr:
                        emails.append(addr)
            except Exception:
                logger.exception("Failed to list notification emails for report org=%s", org_id)

        for addr in (extra or []):
            addr = str(addr or "").strip()
            if addr:
                emails.append(addr)

        # De-duplicate case-insensitively while preserving first-seen order.
        seen: set[str] = set()
        unique: List[str] = []
        for addr in emails:
            key = addr.lower()
            if key not in seen:
                seen.add(key)
                unique.append(addr)
        return unique

    # ------------------------------------------------------------------
    # Event snapshot fetching (for the in-table thumbnails)
    # ------------------------------------------------------------------
    async def _attach_images(self, entries: List[_AlertEntry]) -> None:
        if self._max_images <= 0:
            return
        targets = [e for e in entries if e.image_url][: self._max_images]
        if not targets:
            return

        sem = asyncio.Semaphore(self._image_fetch_concurrency)
        async with httpx.AsyncClient(timeout=self._image_fetch_timeout_s) as client:
            async def _load(entry: _AlertEntry) -> None:
                async with sem:
                    raw = await self._fetch_image_bytes(client, entry.image_url or "")
                if not raw:
                    return
                normalized = normalize_to_jpeg(raw)
                if normalized is not None:
                    entry.image_jpeg = normalized

            await asyncio.gather(*(_load(e) for e in targets), return_exceptions=True)

    async def _fetch_image_bytes(self, client: httpx.AsyncClient, url: str) -> Optional[bytes]:
        url = str(url or "").strip()
        if not url:
            return None

        if url.startswith("data:"):
            match = _DATA_URL_RE.match(url)
            if not match:
                return None
            data = match.group("data") or ""
            if match.group("b64"):
                try:
                    return base64.b64decode(data, validate=False)
                except (ValueError, binascii.Error):
                    return None
            try:
                from urllib.parse import unquote_to_bytes

                return unquote_to_bytes(data)
            except Exception:
                return None

        if url.startswith("http://") or url.startswith("https://"):
            try:
                resp = await client.get(url)
                if resp.status_code == 200 and resp.content:
                    return resp.content
            except Exception:
                logger.warning("Report snapshot fetch failed url=%s", url[:120], exc_info=True)
            return None

        return None

    # ------------------------------------------------------------------
    # PDF rendering
    # ------------------------------------------------------------------
    def _render_pdf(
        self,
        *,
        report_uuid: str,
        report_type: str,
        prepared_for: str,
        entries: List[_AlertEntry],
        start: Optional[datetime],
        end: Optional[datetime],
        site_count: int,
    ) -> bytes:
        doc = PDFReport()
        type_label = _REPORT_TYPE_LABELS.get(report_type, "Alert Report")

        # --- Branded header (logo + wordmark) + centered title ---
        doc.brand_header(right_text=self._fmt_dt(datetime.now(timezone.utc)) + " UTC")
        doc.title_center(type_label.upper())

        # --- Report info block (label/value rows). No organization name by
        # request; the report is identified by its stable Report ID instead. ---
        doc.field_row("Report ID", report_uuid)
        doc.field_row("Prepared for", prepared_for)
        doc.field_row("Reporting window", self._fmt_window(start, end))
        doc.field_row("Sites covered", str(site_count))
        doc.field_row("Approved alerts", str(len(entries)))
        doc.spacer(6.0)

        if not entries:
            doc.section_band("OBSERVATIONS (0 RECORDS)")
            doc.spacer(8.0)
            doc.text(
                "No operator-approved alerts were found for this organization in the "
                "selected window.",
                size=11,
                color=(0.30, 0.36, 0.44),
            )
            return doc.render()

        doc.section_band(
            "OBSERVATIONS (%d RECORD%s)" % (len(entries), "" if len(entries) == 1 else "S")
        )

        for idx, entry in enumerate(entries, start=1):
            self._render_record(doc, idx, entry)

        return doc.render()

    def _render_record(self, doc: PDFReport, idx: int, entry: _AlertEntry) -> None:
        """One observation: stacked label/value rows + large event photo."""
        doc.section_band(
            "#%d: %s" % (idx, entry.title),
            fill=(0.949, 0.949, 0.953),
            text_color=(0.090, 0.094, 0.102),
            size=9.5,
        )

        doc.field_row("Time", self._fmt_dt_site(entry.detected_at, entry.site_tz))
        doc.field_row("Site", entry.site_name)
        if entry.site_location:
            doc.field_row("Site Address", entry.site_location)
        camera = entry.camera_name + (f" ({entry.camera_location})" if entry.camera_location else "")
        doc.field_row("Camera", camera)
        doc.field_row("Type", self._detection_text(entry))
        # The band already carries the title, so only show the message when it
        # actually says something more.
        message = str(entry.body or "").strip()
        if message and message != str(entry.title or "").strip():
            doc.field_row("Alert Message", message)
        doc.field_row("Notes", self._notes_text(entry))
        doc.field_row("Action Taken", self._actions_text(entry))
        if entry.approved_at is not None:
            doc.field_row("Reviewed (UTC)", self._fmt_dt(entry.approved_at))

        if entry.image_jpeg is not None:
            doc.field_label("Photo")
            doc.image_block(entry.image_jpeg, max_height=250.0, link_url=entry.image_url)
            doc.hairline()
        elif entry.image_url:
            doc.field_row("Photo", "Open snapshot image", link_url=entry.image_url)

        if entry.clip_url:
            doc.field_row("Video", "Open event video", link_url=entry.clip_url)

        doc.spacer(10.0)

    def _detection_text(self, entry: _AlertEntry) -> str:
        parts: List[str] = []
        label = _ALERT_TYPE_LABELS.get(str(entry.alert_type or "").strip().lower())
        if label is None:
            raw = str(entry.alert_type or "").replace("_", " ").strip()
            label = raw[:1].upper() + raw[1:] if raw else ""
        if label:
            parts.append(label)
        if entry.classes:
            parts.append("Detected: " + ", ".join(self._pretty_class(c) for c in entry.classes))
        if entry.max_conf is not None:
            parts.append(f"{int(round(entry.max_conf * 100))}% confidence")
        return " \u00b7 ".join(parts) if parts else "Detection"

    @staticmethod
    def _pretty_class(name: Any) -> str:
        c = str(name or "").replace("_", " ").strip()
        return c[:1].upper() + c[1:] if c else ""

    def _notes_text(self, entry: _AlertEntry) -> str:
        """Structured operator observations rendered as bullet lines."""
        blocks: List[str] = []
        for note in entry.notes:
            if not isinstance(note, dict):
                continue
            attrs = note.get("attributes") if isinstance(note.get("attributes"), dict) else {}
            otype = str(attrs.get("object_type") or "").strip().lower() or _classify(entry.classes)

            lines: List[str] = []
            if otype in ("vehicle", "car", "truck", "motorcycle"):
                lines.append(f"\u2022 Type: {self._primary_class(entry) or 'Vehicle'}")
                self._bullet(lines, "Model", attrs.get("model"))
                self._bullet(lines, "Color", attrs.get("color"))
                self._bullet(lines, "Direction", attrs.get("direction"))
            elif otype == "person":
                lines.append("\u2022 Type: Person")
                self._bullet(lines, "Gender", attrs.get("gender"))
                self._bullet(lines, "Clothing", attrs.get("clothing_color"))
                self._bullet(lines, "Direction", attrs.get("direction"))
            else:
                for label, key in (("Color", "color"), ("Direction", "direction")):
                    self._bullet(lines, label, attrs.get(key))

            text = str(note.get("text") or "").strip()
            if text:
                lines.append(f"\u2022 Note: {text}")
            author = str(note.get("author_name") or "").strip()
            if lines and author:
                lines.append(f"\u2022 By: {author}")
            if lines:
                blocks.append("\n".join(lines))

        if not blocks:
            if entry.classes:
                return "\u2022 Detected: " + ", ".join(self._pretty_class(c) for c in entry.classes)
            return ""
        return "\n".join(blocks)

    def _actions_text(self, entry: _AlertEntry) -> str:
        actions: List[str] = []
        for note in entry.notes:
            if not isinstance(note, dict):
                continue
            act = str(note.get("action") or "").strip()
            if act:
                actions.append(f"\u2022 {act}")
        return "\n".join(actions)

    @staticmethod
    def _bullet(lines: List[str], label: str, value: Any) -> None:
        val = str(value or "").strip()
        if val:
            lines.append(f"\u2022 {label}: {val}")

    @staticmethod
    def _primary_class(entry: _AlertEntry) -> str:
        for c in entry.classes:
            cl = str(c).strip()
            if cl:
                return cl[:1].upper() + cl[1:]
        return ""

    # ------------------------------------------------------------------
    # Email bodies
    # ------------------------------------------------------------------
    def _subject(self, result: ReportResult) -> str:
        prefix = "[1886NOENTRY][URGENT]" if result.report_type == "urgent" else "[1886NOENTRY]"
        label = _REPORT_TYPE_LABELS.get(result.report_type, "Alert Report")
        return (
            f"{prefix} {label}: {result.org_name} "
            f"({result.alert_count} alert{'s' if result.alert_count != 1 else ''})"
        )

    def _email_text(self, result: ReportResult) -> str:
        return "\n".join(
            [
                f"1886NOENTRY: {_REPORT_TYPE_LABELS.get(result.report_type, 'Alert Report')}",
                "",
                f"Organization: {result.org_name}",
                f"Reporting window: {self._fmt_window(result.start, result.end)}",
                f"Sites covered: {result.site_count}",
                f"Approved alerts: {result.alert_count}",
                f"Alerts with snapshot: {result.image_count}",
                "",
                "The full report is attached as a PDF.",
            ]
        )

    def _email_html(self, result: ReportResult) -> str:
        import html as _html

        org = _html.escape(result.org_name)
        window = _html.escape(self._fmt_window(result.start, result.end))
        type_label = _html.escape(_REPORT_TYPE_LABELS.get(result.report_type, "Alert Report"))
        return f"""\
<!doctype html>
<html><body style="margin:0;padding:24px;background:#f4f6fb;font-family:Arial,sans-serif;">
  <div style="max-width:640px;margin:0 auto;background:#ffffff;border:1px solid #e2e8f0;border-radius:16px;overflow:hidden;">
    <div style="padding:18px 24px;background:#0b1220;color:#ffffff;">
      <div style="font-size:14px;opacity:0.85;font-weight:700;">1886NOENTRY</div>
      <div style="font-size:22px;font-weight:800;margin-top:6px;">{type_label}</div>
    </div>
    <div style="padding:18px 24px;color:#0f172a;font-size:14px;line-height:1.6;">
      <div style="font-size:16px;font-weight:800;">{org}</div>
      <div style="margin-top:6px;color:#475569;">Reporting window: {window}</div>
      <table role="presentation" width="100%" cellspacing="0" cellpadding="0"
             style="margin-top:14px;border-collapse:separate;border-spacing:0;background:#f8fafc;
                    border:1px solid #e2e8f0;border-radius:12px;overflow:hidden;">
        <tr><td style="padding:10px 14px;border-bottom:1px solid #e2e8f0;font-weight:700;">Sites covered</td>
            <td style="padding:10px 14px;border-bottom:1px solid #e2e8f0;">{result.site_count}</td></tr>
        <tr><td style="padding:10px 14px;border-bottom:1px solid #e2e8f0;font-weight:700;">Approved alerts</td>
            <td style="padding:10px 14px;border-bottom:1px solid #e2e8f0;">{result.alert_count}</td></tr>
        <tr><td style="padding:10px 14px;font-weight:700;">Alerts with snapshot</td>
            <td style="padding:10px 14px;">{result.image_count}</td></tr>
      </table>
      <div style="margin-top:16px;color:#334155;">The full report, including operator notes, alert
        snapshots, and video links, is attached as a PDF.</div>
    </div>
  </div>
</body></html>
"""

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _fmt_dt(dt: Optional[datetime]) -> str:
        if not isinstance(dt, datetime):
            return "Not recorded"
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _fmt_dt_site(dt: Optional[datetime], tz_name: str) -> str:
        """Event time in the site's local timezone (like a guard's activity log),
        falling back to UTC when the site timezone is unknown/invalid."""
        if not isinstance(dt, datetime):
            return "Not recorded"
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        try:
            from zoneinfo import ZoneInfo

            tz = ZoneInfo(str(tz_name or "UTC"))
        except Exception:
            tz = timezone.utc
        local = dt.astimezone(tz)
        label = local.tzname() or str(tz_name or "UTC")
        return local.strftime("%Y-%m-%d, %I:%M %p").replace(" 0", " ") + f" ({label})"

    def _fmt_window(self, start: Optional[datetime], end: Optional[datetime]) -> str:
        if start is None and end is None:
            return "All approved alerts"
        s = self._fmt_dt(start) if start else "beginning"
        e = self._fmt_dt(end) if end else "now"
        return f"{s} to {e} UTC"

    @staticmethod
    def _filename(report_type: str = "general", report_uuid: str = "") -> str:
        # Brand-based filename (no organization name), suffixed with a short slice
        # of the report UUID so downloads stay distinguishable and traceable.
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M")
        short = str(report_uuid or "").split("-")[0] or "report"
        return f"noentry-{report_type}-report-{stamp}-{short}.pdf"


def _timedelta_hours(hours: int):
    from datetime import timedelta

    return timedelta(hours=max(0, int(hours)))
