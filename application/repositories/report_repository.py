"""Persistence for archived report PDFs (`organization_reports`).

Stateless like the other repositories: methods take an ``AsyncSession`` and do
not commit — the caller owns the transaction.

The PDF blob can be multiple MB (embedded snapshots), so list queries select
explicit columns and never touch ``pdf_data``; only ``get_with_pdf`` loads it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from application.dtos import ReportCreateDTO
from core.database_orm import OrganizationReport

# Columns safe to return in list views (everything except the blob).
_LIST_COLUMNS = (
    OrganizationReport.id,
    OrganizationReport.report_uuid,
    OrganizationReport.org_id,
    OrganizationReport.report_type,
    OrganizationReport.filename,
    OrganizationReport.generated_by,
    OrganizationReport.generated_by_email,
    OrganizationReport.site_uuids,
    OrganizationReport.period_start,
    OrganizationReport.period_end,
    OrganizationReport.alert_count,
    OrganizationReport.pdf_size,
    OrganizationReport.created_at,
)


class ReportRepository:
    async def create(self, db: AsyncSession, *, dto: ReportCreateDTO) -> OrganizationReport:
        """Insert one archived report. Flush only; caller commits."""
        row = OrganizationReport(
            org_id=int(dto.org_id),
            report_uuid=str(dto.report_uuid),
            report_type=str(dto.report_type or "general"),
            filename=dto.filename,
            generated_by=dto.generated_by,
            generated_by_email=dto.generated_by_email,
            site_uuids=[str(s) for s in (dto.site_uuids or [])],
            period_start=dto.period_start,
            period_end=dto.period_end,
            alert_count=int(dto.alert_count or 0),
            pdf_data=dto.pdf_data,
            pdf_size=len(dto.pdf_data or b""),
        )
        db.add(row)
        await db.flush()
        return row

    async def list_reports(
        self,
        db: AsyncSession,
        *,
        org_id: int,
        report_type: Optional[str] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        generated_by_email: Optional[str] = None,
        site_uuid: Optional[str] = None,
        report_uuid: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Any]:
        """Blob-free report rows for an org, newest first.

        ``site_uuid`` filters to reports whose ``site_uuids`` JSON list contains
        the site. Report volume is small (one general per day plus occasional
        urgent ones), so that membership test runs in Python over the SQL-
        filtered page rather than as a dialect-specific JSON_CONTAINS.
        """
        stmt = select(*_LIST_COLUMNS).where(OrganizationReport.org_id == int(org_id))
        if report_uuid:
            stmt = stmt.where(OrganizationReport.report_uuid == str(report_uuid).strip())
        if report_type:
            stmt = stmt.where(OrganizationReport.report_type == str(report_type))
        if created_after is not None:
            stmt = stmt.where(OrganizationReport.created_at >= created_after)
        if created_before is not None:
            stmt = stmt.where(OrganizationReport.created_at < created_before)
        if generated_by_email:
            stmt = stmt.where(
                OrganizationReport.generated_by_email == str(generated_by_email).strip()
            )
        stmt = stmt.order_by(OrganizationReport.created_at.desc(), OrganizationReport.id.desc())
        stmt = stmt.offset(max(0, int(offset))).limit(max(1, min(int(limit), 500)))

        rows = (await db.execute(stmt)).all()
        if site_uuid:
            wanted = str(site_uuid).strip().lower()
            rows = [
                r for r in rows
                if wanted in {str(s).strip().lower() for s in (r.site_uuids or [])}
            ]
        return rows

    async def get_with_pdf(
        self, db: AsyncSession, *, report_id: int, org_id: Optional[int] = None
    ) -> Optional[OrganizationReport]:
        """Full report row including the PDF blob, org-scoped when org_id given."""
        stmt = select(OrganizationReport).where(OrganizationReport.id == int(report_id))
        if org_id is not None:
            stmt = stmt.where(OrganizationReport.org_id == int(org_id))
        return (await db.execute(stmt)).scalar_one_or_none()
