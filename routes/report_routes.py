"""Organization report routes.

The report archive and its generation endpoints:

  GET  /api/reports                          -> list archived reports (filters:
                                                type, date range, site, operator
                                                email); any org member
  GET  /api/reports/{id}/download            -> download an archived PDF
  POST /api/reports/approved-alerts          -> build + archive + email
  POST /api/reports/approved-alerts/download -> build + return the PDF inline
  GET/PUT /api/reports/schedule              -> daily general-report schedule

Generation is gated by ``reports:send`` (org admins and operators); browsing
and downloading the archive by ``org:read`` (any member). Reports are
org-scoped: a normal caller sees their own organization; platform admins must
name an ``org_id``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.responses import Response

from application.repositories.report_repository import ReportRepository
from application.repositories.report_schedule_repository import ReportScheduleRepository
from application.services.notification import NotificationService
from application.services.report import PdfReportGenerator, resolve_tz
from core.schemas import ReportOut
from core.security.roles import Permission
from dependencies import (
    get_notification_service,
    get_session_factory,
    RequirePermission,
    OrgContext,
)

router = APIRouter(prefix="/reports")
logger = logging.getLogger(__name__)


class ApprovedAlertsReportRequest(BaseModel):
    # "general" = approved alerts not emailed at approval (the daily roll-up);
    # "urgent" = alerts an operator approved with the email opt-in.
    report_type: str = Field(default="general", pattern="^(general|urgent)$")

    # Reporting window. ``window_hours`` is a shortcut for [now - hours, now]
    # when ``start`` is omitted. All omitted => every approved alert.
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    window_hours: Optional[int] = Field(default=None, ge=1, le=24 * 366)

    # Only count alerts an operator explicitly approved (approved_by set). When
    # False, auto-approved alerts (orgs without an operator) are included too.
    operator_approved_only: bool = True

    extra_recipients: List[str] = Field(default_factory=list)

    # Platform admins (no single-org context) must name the target org.
    org_id: Optional[int] = None


class ApprovedAlertsReportResponse(BaseModel):
    report_id: Optional[int] = None
    report_type: str
    org_id: int
    org_name: str
    alert_count: int
    image_count: int
    site_count: int
    recipient_count: int
    emailed: bool
    filename: str


def _resolve_org_id(ctx: OrgContext, body_org_id: Optional[int]) -> int:
    org_id = ctx.org_id if ctx.org_id is not None else body_org_id
    if org_id is None:
        raise HTTPException(
            status_code=400,
            detail="org_id is required for platform-admin report requests.",
        )
    return int(org_id)


def _build_generator(
    session_factory: async_sessionmaker[AsyncSession],
    notification_service: NotificationService,
) -> PdfReportGenerator:
    email = getattr(notification_service, "email", None)
    dashboard_base_url = getattr(getattr(email, "cfg", None), "dashboard_base_url", None)
    return PdfReportGenerator(
        session_factory=session_factory,
        email=email,
        dashboard_base_url=dashboard_base_url,
    )


@router.post("/approved-alerts", response_model=ApprovedAlertsReportResponse)
async def send_approved_alerts_report(
    body: ApprovedAlertsReportRequest,
    ctx: OrgContext = Depends(RequirePermission(Permission.REPORTS_SEND)),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
    notification_service: NotificationService = Depends(get_notification_service),
):
    """Build the alert-report PDF, archive it, and email it to the org's
    members and the per-site notification recipients."""
    org_id = _resolve_org_id(ctx, body.org_id)
    generator = _build_generator(session_factory, notification_service)

    result = await generator.generate_and_send(
        org_id=org_id,
        report_type=body.report_type,
        start=body.start,
        end=body.end,
        window_hours=body.window_hours,
        operator_approved_only=body.operator_approved_only,
        extra_recipients=body.extra_recipients,
        persist=True,
        generated_by=int(ctx.user.id),
        generated_by_email=str(getattr(ctx.user, "email", "") or "") or None,
    )

    return ApprovedAlertsReportResponse(
        report_id=result.report_id,
        report_type=result.report_type,
        org_id=result.org_id,
        org_name=result.org_name,
        alert_count=result.alert_count,
        image_count=result.image_count,
        site_count=result.site_count,
        recipient_count=len(result.recipients),
        emailed=result.emailed,
        filename=result.filename,
    )


@router.post("/approved-alerts/download")
async def download_approved_alerts_report(
    body: ApprovedAlertsReportRequest,
    ctx: OrgContext = Depends(RequirePermission(Permission.REPORTS_SEND)),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
    notification_service: NotificationService = Depends(get_notification_service),
):
    """Build the alert-report PDF and return it inline (no email, no archive)."""
    org_id = _resolve_org_id(ctx, body.org_id)
    generator = _build_generator(session_factory, notification_service)

    start = body.start
    if body.window_hours is not None and start is None:
        from datetime import timedelta, timezone

        start = datetime.now(timezone.utc) - timedelta(hours=int(body.window_hours))

    result = await generator.build_report(
        org_id=org_id,
        report_type=body.report_type,
        start=start,
        end=body.end,
        operator_approved_only=body.operator_approved_only,
    )

    return Response(
        content=result.pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{result.filename}"',
        },
    )


# ---------------------------------------------------------------------------
# Report archive (any org member): browse, filter, download
# ---------------------------------------------------------------------------
report_repo = ReportRepository()


def _report_to_out(row) -> ReportOut:
    return ReportOut(
        id=int(row.id),
        report_uuid=str(row.report_uuid),
        org_id=int(row.org_id),
        report_type=str(row.report_type),
        filename=str(row.filename),
        generated_by_email=row.generated_by_email,
        site_uuids=[str(s) for s in (row.site_uuids or [])],
        period_start=row.period_start,
        period_end=row.period_end,
        alert_count=int(row.alert_count or 0),
        pdf_size=int(row.pdf_size or 0),
        created_at=row.created_at,
    )


@router.get("", response_model=List[ReportOut])
async def list_reports(
    report_type: Optional[str] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    site_uuid: Optional[str] = None,
    operator_email: Optional[str] = None,
    report_uuid: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    org_id: Optional[int] = None,
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_READ)),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
):
    """Archived reports for the caller's organization, newest first.

    Filters: ``report_type`` (general|urgent), ``start``/``end`` (created_at
    range), ``site_uuid`` (reports covering that site), ``operator_email``
    (who generated it), ``report_uuid`` (exact report id lookup).
    """
    target_org = _resolve_org_id(ctx, org_id)
    rt = str(report_type or "").strip().lower() or None
    if rt is not None and rt not in ("general", "urgent"):
        raise HTTPException(status_code=422, detail="report_type must be 'general' or 'urgent'")

    async with session_factory() as db:
        rows = await report_repo.list_reports(
            db,
            org_id=target_org,
            report_type=rt,
            created_after=start,
            created_before=end,
            generated_by_email=(operator_email or "").strip() or None,
            site_uuid=(site_uuid or "").strip() or None,
            report_uuid=(report_uuid or "").strip() or None,
            limit=limit,
            offset=offset,
        )
    return [_report_to_out(r) for r in rows]


async def _load_report_pdf(report_id: int, ctx: OrgContext, org_id, session_factory):
    target_org = _resolve_org_id(ctx, org_id)
    async with session_factory() as db:
        row = await report_repo.get_with_pdf(db, report_id=report_id, org_id=target_org)
    if row is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return row


@router.get("/{report_id}/download")
async def download_archived_report(
    report_id: int,
    org_id: Optional[int] = None,
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_READ)),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
):
    """Download an archived report PDF (org-scoped)."""
    row = await _load_report_pdf(report_id, ctx, org_id, session_factory)
    return Response(
        content=bytes(row.pdf_data or b""),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{row.filename}"',
        },
    )


@router.get("/{report_id}/view")
async def view_archived_report(
    report_id: int,
    org_id: Optional[int] = None,
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_READ)),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
):
    """Return an archived report PDF for inline viewing (no download prompt)."""
    row = await _load_report_pdf(report_id, ctx, org_id, session_factory)
    return Response(
        content=bytes(row.pdf_data or b""),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{row.filename}"',
        },
    )


# ---------------------------------------------------------------------------
# Daily schedule (org admins): when to email the report each day
# ---------------------------------------------------------------------------
schedule_repo = ReportScheduleRepository()


class ReportScheduleOut(BaseModel):
    org_id: int
    is_enabled: bool
    send_hour: int
    send_minute: int
    timezone: str
    window_hours: int
    operator_approved_only: bool
    last_sent_on: Optional[str] = None


class ReportScheduleUpdate(BaseModel):
    is_enabled: Optional[bool] = None
    send_hour: Optional[int] = Field(default=None, ge=0, le=23)
    send_minute: Optional[int] = Field(default=None, ge=0, le=59)
    timezone: Optional[str] = None
    window_hours: Optional[int] = Field(default=None, ge=1, le=24 * 366)
    operator_approved_only: Optional[bool] = None
    org_id: Optional[int] = None  # platform admins only

    @field_validator("timezone")
    @classmethod
    def _validate_timezone(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip()
        if not v:
            raise ValueError("timezone must not be empty")
        # resolve_tz falls back to UTC silently; here we reject unknown names so
        # the admin gets clear feedback instead of a surprise UTC schedule.
        from application.services.report.report_scheduler import ZoneInfo

        if ZoneInfo is not None:
            try:
                ZoneInfo(v)
            except Exception as exc:
                raise ValueError(f"Unknown timezone '{v}'") from exc
        return v


def _schedule_defaults(org_id: int) -> ReportScheduleOut:
    return ReportScheduleOut(
        org_id=org_id,
        is_enabled=False,
        send_hour=8,
        send_minute=0,
        timezone="UTC",
        window_hours=24,
        operator_approved_only=True,
        last_sent_on=None,
    )


def _schedule_to_out(row) -> ReportScheduleOut:
    return ReportScheduleOut(
        org_id=int(row.org_id),
        is_enabled=bool(row.is_enabled),
        send_hour=int(row.send_hour),
        send_minute=int(row.send_minute),
        timezone=str(row.timezone),
        window_hours=int(row.window_hours),
        operator_approved_only=bool(row.operator_approved_only),
        last_sent_on=row.last_sent_on,
    )


@router.get("/schedule", response_model=ReportScheduleOut)
async def get_report_schedule(
    org_id: Optional[int] = None,
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_MANAGE_SETTINGS)),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
):
    """Return the org's daily report schedule (defaults when none is set)."""
    target_org = _resolve_org_id(ctx, org_id)
    async with session_factory() as db:
        row = await schedule_repo.get(db, org_id=target_org)
    return _schedule_to_out(row) if row is not None else _schedule_defaults(target_org)


@router.put("/schedule", response_model=ReportScheduleOut)
async def update_report_schedule(
    body: ReportScheduleUpdate,
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_MANAGE_SETTINGS)),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
):
    """Create or update the org's daily report schedule.

    When the schedule is (re)enabled and today's send time has already passed in
    the configured timezone, ``last_sent_on`` is pre-stamped to today so the
    first report goes out tomorrow rather than firing immediately.
    """
    target_org = _resolve_org_id(ctx, body.org_id)

    async with session_factory() as db:
        row = await schedule_repo.upsert(
            db,
            org_id=target_org,
            is_enabled=body.is_enabled,
            send_hour=body.send_hour,
            send_minute=body.send_minute,
            timezone=body.timezone,
            window_hours=body.window_hours,
            operator_approved_only=body.operator_approved_only,
        )

        if row.is_enabled and not row.last_sent_on:
            tz = resolve_tz(row.timezone)
            now_local = datetime.now(tz)
            if (now_local.hour, now_local.minute) >= (int(row.send_hour), int(row.send_minute)):
                row.last_sent_on = now_local.strftime("%Y-%m-%d")

        await db.commit()
        await db.refresh(row)

    return _schedule_to_out(row)
