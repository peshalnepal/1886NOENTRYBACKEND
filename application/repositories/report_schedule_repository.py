"""Persistence for the per-organization daily report schedule.

Stateless, like the other repositories: methods take an ``AsyncSession`` and do
not commit (the caller owns the transaction). The one exception in spirit is
the background scheduler, which opens its own session and commits around these
calls.
"""

from __future__ import annotations

from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database_orm import OrganizationReportSchedule


class ReportScheduleRepository:
    async def get(self, db: AsyncSession, *, org_id: int) -> Optional[OrganizationReportSchedule]:
        res = await db.execute(
            select(OrganizationReportSchedule)
            .where(OrganizationReportSchedule.org_id == int(org_id))
            .limit(1)
        )
        return res.scalar_one_or_none()

    async def list_enabled(self, db: AsyncSession) -> List[OrganizationReportSchedule]:
        res = await db.execute(
            select(OrganizationReportSchedule).where(
                OrganizationReportSchedule.is_enabled.is_(True)
            )
        )
        return list(res.scalars().all())

    async def upsert(
        self,
        db: AsyncSession,
        *,
        org_id: int,
        is_enabled: Optional[bool] = None,
        send_hour: Optional[int] = None,
        send_minute: Optional[int] = None,
        timezone: Optional[str] = None,
        window_hours: Optional[int] = None,
        operator_approved_only: Optional[bool] = None,
        last_sent_on: Optional[str] = None,
    ) -> OrganizationReportSchedule:
        """Create or update the org's schedule. Only provided fields change.

        Flush only; the caller commits.
        """
        row = await self.get(db, org_id=org_id)
        if row is None:
            row = OrganizationReportSchedule(org_id=int(org_id))
            db.add(row)

        if is_enabled is not None:
            row.is_enabled = bool(is_enabled)
        if send_hour is not None:
            row.send_hour = int(send_hour)
        if send_minute is not None:
            row.send_minute = int(send_minute)
        if timezone is not None:
            row.timezone = str(timezone)
        if window_hours is not None:
            row.window_hours = int(window_hours)
        if operator_approved_only is not None:
            row.operator_approved_only = bool(operator_approved_only)
        if last_sent_on is not None:
            row.last_sent_on = str(last_sent_on)

        await db.flush()
        return row

    async def mark_sent(self, db: AsyncSession, *, org_id: int, date_str: str) -> None:
        """Record the date (in the schedule's timezone) the report last went out."""
        row = await self.get(db, org_id=org_id)
        if row is not None:
            row.last_sent_on = str(date_str)
            await db.flush()
