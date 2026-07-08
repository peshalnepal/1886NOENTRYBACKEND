"""Background daily-report scheduler.

Polls the per-organization report schedules every minute and, when an org's
configured local send time has arrived for the day, generates and emails its
approved-alerts PDF report. A ``last_sent_on`` date stamp (in the schedule's own
timezone) guarantees each day fires exactly once even though the poll is
minutely and even across process restarts.

If the process is down across the scheduled minute, the report still goes out
the next time the scheduler runs that same day (catch-up), which is the desired
behaviour for a "daily digest" rather than a precise alarm.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Callable

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - zoneinfo is stdlib on 3.9+
    ZoneInfo = None  # type: ignore

from application.repositories.report_schedule_repository import ReportScheduleRepository
from application.services.report.pdf_report_service import PdfReportGenerator

logger = logging.getLogger(__name__)

# Type of the zero-arg factory that yields a configured PdfReportGenerator.
GeneratorFactory = Callable[[], PdfReportGenerator]


def resolve_tz(name: str):
    """Best-effort timezone lookup; falls back to UTC on any failure."""
    if ZoneInfo is not None:
        try:
            return ZoneInfo(str(name or "UTC"))
        except Exception:
            pass
    return timezone.utc


class ReportScheduler:
    def __init__(
        self,
        *,
        session_factory,
        generator_factory: GeneratorFactory,
        poll_s: float = 60.0,
    ) -> None:
        self._session_factory = session_factory
        self._generator_factory = generator_factory
        self._poll_s = max(15.0, float(poll_s))
        self._repo = ReportScheduleRepository()
        self._task: asyncio.Task | None = None
        self._closing = False

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._closing = False
            self._task = asyncio.create_task(self._loop(), name="report_scheduler")

    async def shutdown(self) -> None:
        self._closing = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Report scheduler shutdown failed")
        self._task = None

    async def _loop(self) -> None:
        while not self._closing:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Report scheduler tick failed")
            await asyncio.sleep(self._poll_s)

    async def _tick(self) -> None:
        if self._session_factory is None:
            return

        async with self._session_factory() as db:
            schedules = await self._repo.list_enabled(db)

        # Snapshot the fields we need so the row objects don't outlive the session.
        due: list[tuple[int, int, bool, str]] = []  # (org_id, window_hours, op_only, today_str)
        for sched in schedules:
            org_id = int(sched.org_id)
            tz = resolve_tz(sched.timezone)
            now_local = datetime.now(tz)
            today_str = now_local.strftime("%Y-%m-%d")

            if str(sched.last_sent_on or "") == today_str:
                continue  # already sent today

            scheduled = (int(sched.send_hour), int(sched.send_minute))
            if (now_local.hour, now_local.minute) < scheduled:
                continue  # not yet time today

            due.append(
                (org_id, int(sched.window_hours or 24), bool(sched.operator_approved_only), today_str)
            )

        for org_id, window_hours, operator_only, today_str in due:
            await self._send_for_org(
                org_id=org_id,
                window_hours=window_hours,
                operator_only=operator_only,
                today_str=today_str,
            )

    async def _send_for_org(
        self,
        *,
        org_id: int,
        window_hours: int,
        operator_only: bool,
        today_str: str,
    ) -> None:
        # Stamp the send date FIRST so a failure or a slow SMTP exchange cannot
        # cause a duplicate report on the next minute's tick. A failed send is
        # logged and simply retried the following day.
        try:
            async with self._session_factory() as db:
                await self._repo.mark_sent(db, org_id=org_id, date_str=today_str)
                await db.commit()
        except Exception:
            logger.exception("Failed to stamp report send date org=%s", org_id)
            return

        try:
            generator = self._generator_factory()
            # Daily general report: approved alerts NOT emailed at approval
            # (the emailed ones were archived as urgent reports already).
            result = await generator.generate_and_send(
                org_id=org_id,
                report_type="general",
                window_hours=window_hours,
                operator_approved_only=operator_only,
                persist=True,
            )
            logger.info(
                "Scheduled report sent org=%s alerts=%s recipients=%s emailed=%s",
                org_id, result.alert_count, len(result.recipients), result.emailed,
            )
        except Exception:
            logger.exception("Scheduled report generation failed org=%s", org_id)
