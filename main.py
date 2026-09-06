import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Awaitable, Callable, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import async_sessionmaker

from application.services.manager import Manager
from application.services.notification import (
    EmailConfig,
    EmailNotifier,
    NotificationService,
    WebNotificationHub,
)
from application.services.report import PdfReportGenerator, ReportScheduler
from application.services.retention import RetentionService
from application.services.user_snapshot_cache import UserSnapshotCache
from core.config import DEBUG
from core.database import async_engine, db_manager
from core.env import env_bool, env_float, env_int
from routes.admin_routes import router as admin_router
from routes.auth import router as auth_router
from routes.camera_routes import router as cameras_router
from routes.clips_routes import router as clips_router
from routes.device_routes import router as devices_router
from routes.notification_email_routes import router as notification_emails_router
from routes.notifications_routes import router as notifications_router
from routes.platform_admin_routes import router as platform_admin_router
from routes.public_routes import router as public_router
from routes.report_routes import router as reports_router
from routes.site_routes import router as sites_router
from routes.user_routes import router as users_router
from routes.wall_routes import router as walls_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SessionLocal = async_sessionmaker(async_engine, expire_on_commit=False)

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = env_int("SMTP_PORT", 587)
SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
FROM_EMAIL = os.environ.get("FROM_EMAIL") or os.environ.get(
    "SMTP_FROM", "noreply@1886noentry.com"
)
DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "")

RECONCILE_INTERVAL_S = env_int("EDGE_RECONCILE_INTERVAL_S", 30, minimum=0)
RETENTION_CLEANUP_INTERVAL_S = env_int("RETENTION_CLEANUP_INTERVAL_S", 3600, minimum=0)
CLIP_RETENTION_DAYS = env_int("CLIP_RETENTION_DAYS", 30, minimum=0)
ALERT_RETENTION_DAYS = env_int("ALERT_RETENTION_DAYS", 7, minimum=0)


async def _edge_reconcile_loop(app: FastAPI) -> None:
    """Re-push the DB's camera state onto every edge device, forever.

    A failing device backs off exponentially (from `base_wait_s`, capped at
    `max_wait_s`) so an unreachable Jetson does not hammer the loop; a clean
    pass resets to the normal interval.
    """
    manager = app.state.manager
    delete_unknown = env_bool("EDGE_RECONCILE_DELETE_UNKNOWN", False)

    base_wait_s = env_float("EDGE_RECONCILE_FAILURE_BASE_S", 5.0, minimum=1.0)
    max_wait_s = max(base_wait_s, env_float("EDGE_RECONCILE_FAILURE_MAX_S", 300.0))
    consecutive_failures = 0

    while True:
        try:
            summary = await manager.reconcile_all_devices_edge(
                dry_run=False,
                delete_unknown=delete_unknown,
            )
            device_count = summary.get("device_count")
            if summary.get("errors"):
                consecutive_failures += 1
                logger.warning(
                    "Edge reconcile completed with errors device_count=%s errors=%s",
                    device_count,
                    len(summary.get("errors") or []),
                )
            else:
                consecutive_failures = 0
                if summary.get("warnings"):
                    logger.warning(
                        "Edge reconcile completed with warnings device_count=%s warnings=%s",
                        device_count,
                        len(summary.get("warnings") or []),
                    )
                else:
                    logger.info("Edge reconcile completed device_count=%s", device_count)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Edge reconcile loop failed")
            consecutive_failures += 1

        if RECONCILE_INTERVAL_S <= 0:
            return

        wait_s = RECONCILE_INTERVAL_S
        if consecutive_failures > 0:
            wait_s = min(base_wait_s * (1.5 ** min(consecutive_failures - 1, 10)), max_wait_s)
            logger.info(
                "Edge reconcile backoff after %d failures: waiting %.1fs",
                consecutive_failures,
                wait_s,
            )

        await asyncio.sleep(wait_s)


async def _retention_cleanup_loop(app: FastAPI) -> None:
    """Delete clips and alerts past their retention window, forever."""
    svc = app.state.retention_service
    while True:
        try:
            summary = await svc.purge(
                clip_retention_days=CLIP_RETENTION_DAYS,
                alert_retention_days=ALERT_RETENTION_DAYS,
            )
            if summary.get("clips_deleted") or summary.get("alerts_deleted"):
                logger.info(
                    "Retention cleanup removed clips=%s alerts=%s",
                    summary.get("clips_deleted", 0),
                    summary.get("alerts_deleted", 0),
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Retention cleanup loop failed")

        if RETENTION_CLEANUP_INTERVAL_S <= 0:
            return
        await asyncio.sleep(float(RETENTION_CLEANUP_INTERVAL_S))


async def _cancel_task(task: Optional[asyncio.Task], *, label: str) -> None:
    """Cancel and await a background task, logging anything but cancellation."""
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("%s shutdown failed", label)


async def _close_quietly(closer: Optional[Callable[[], Awaitable[None]]], *, label: str) -> None:
    """Run one shutdown step; a failure must not stop the remaining steps."""
    if closer is None:
        return
    try:
        await closer()
    except Exception:
        logger.exception("%s shutdown failed", label)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the DB, build the app singletons, start the background loops,
    and tear all of it down again on shutdown."""
    logger.info("Application startup: initializing database.")
    if not await db_manager.initialize_tables_and_data():
        raise RuntimeError("FATAL: Could not initialize database tables and defaults.")
    logger.info("Application startup: database initialization finished.")

    dashboard_base_url = DASHBOARD_URL or None
    email_notifier = EmailNotifier(
        EmailConfig(
            enabled=bool(SMTP_USERNAME and SMTP_PASSWORD),
            smtp_host=SMTP_HOST,
            smtp_port=SMTP_PORT,
            smtp_user=SMTP_USERNAME,
            smtp_pass=SMTP_PASSWORD,
            from_email=FROM_EMAIL,
            dashboard_base_url=dashboard_base_url,
        )
    )

    hub = WebNotificationHub()
    notification_service = NotificationService(hub=hub, email=email_notifier)
    notification_service.set_session_factory(SessionLocal)
    notification_service.start()

    manager = Manager(session_factory=SessionLocal)
    manager.set_notification_service(notification_service)

    app.state.session_factory = SessionLocal
    app.state.manager = manager
    app.state.user_snapshot_cache = UserSnapshotCache()
    app.state.notification_hub = hub
    app.state.notification_service = notification_service
    app.state.retention_service = RetentionService(session_factory=SessionLocal)
    app.state.alert_blob_cleanup_tasks = set()

    # Daily approved-alerts report, sent at each org admin's configured time.
    report_scheduler = ReportScheduler(
        session_factory=SessionLocal,
        generator_factory=lambda: PdfReportGenerator(
            session_factory=SessionLocal,
            email=email_notifier,
            dashboard_base_url=dashboard_base_url,
        ),
        poll_s=env_float("REPORTS_SCHEDULER_POLL_S", 60.0, minimum=15.0),
    )
    app.state.report_scheduler = report_scheduler
    if env_bool("REPORTS_SCHEDULER_ENABLED", True):
        report_scheduler.start()

    logger.info("Application startup: starting background pipelines.")
    pipeline_startup = await manager.start_background_pipelines()
    app.state.pipeline_startup = pipeline_startup
    if pipeline_startup["error_count"]:
        logger.warning(
            "Background pipeline startup completed with errors started=%s errors=%s",
            pipeline_startup["started_count"],
            pipeline_startup["error_count"],
        )
    else:
        logger.info(
            "Background pipeline startup completed started=%s",
            pipeline_startup["started_count"],
        )

    app.state.edge_reconcile_task = asyncio.create_task(
        _edge_reconcile_loop(app), name="edge_reconcile_loop"
    )
    app.state.retention_cleanup_task = asyncio.create_task(
        _retention_cleanup_loop(app), name="retention_cleanup_loop"
    )

    yield

    try:
        await _cancel_task(
            getattr(app.state, "edge_reconcile_task", None), label="Edge reconcile task"
        )
        await _cancel_task(
            getattr(app.state, "retention_cleanup_task", None),
            label="Retention cleanup task",
        )
        blob_cleanup_tasks = set(getattr(app.state, "alert_blob_cleanup_tasks", None) or ())
        if blob_cleanup_tasks:
            await asyncio.gather(*blob_cleanup_tasks, return_exceptions=True)
    finally:
        await _close_quietly(report_scheduler.shutdown, label="Report scheduler")
        await _close_quietly(manager.shutdown, label="Manager")
        await _close_quietly(notification_service.shutdown, label="Notification service")
        retention_svc = getattr(app.state, "retention_service", None)
        await _close_quietly(
            getattr(retention_svc, "close", None), label="Retention service"
        )


app = FastAPI(debug=DEBUG, lifespan=lifespan)


@app.middleware("http")
async def add_cache_control_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

for _router in (
    cameras_router,
    clips_router,
    sites_router,
    devices_router,
    notifications_router,
    notification_emails_router,
    auth_router,
    users_router,
    platform_admin_router,
    admin_router,
    reports_router,
    walls_router,
    public_router,
):
    app.include_router(_router, prefix="/api")


@app.get("/")
async def root():
    return {"status": "healthy", "message": "API is running"}
