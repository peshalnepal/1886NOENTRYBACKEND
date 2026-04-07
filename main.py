import logging
import asyncio
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import async_sessionmaker
from routes.camera_routes import router as cameras_router
from routes.site_routes import router as sites_router
from routes.device_routes import router as devices_router
from routes.notifications_routes import router as notifications_router
from routes.notification_email_routes import router as notification_emails_router
from routes.clips_routes import router as clips_router
from routes.auth import router as auth_router
from routes.user_routes import router as users_router

from core.config import DEBUG
from core.database import db_manager, async_engine
from application.services.manager import Manager  # adjust if your path is different
from application.models.yolo_config import YoloModelConfig
from application.services.notification import WebNotificationHub, NotificationService, EmailNotifier, EmailConfig
from application.services.retention import RetentionService
from application.services.user_snapshot_cache import UserSnapshotCache
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SessionLocal = async_sessionmaker(async_engine, expire_on_commit=False)

# Load SMTP configuration from environment
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
FROM_EMAIL = os.environ.get("FROM_EMAIL") or os.environ.get("SMTP_FROM", "noreply@1886noentry.com")
DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "")
RECONCILE_INTERVAL_S = max(0, int(os.environ.get("EDGE_RECONCILE_INTERVAL_S", "30")))
RETENTION_CLEANUP_INTERVAL_S = max(0, int(os.environ.get("RETENTION_CLEANUP_INTERVAL_S", "3600")))
CLIP_RETENTION_DAYS = max(0, int(os.environ.get("CLIP_RETENTION_DAYS", "30")))
ALERT_RETENTION_DAYS = max(0, int(os.environ.get("ALERT_RETENTION_DAYS", "7")))


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


async def _edge_reconcile_loop(app: FastAPI) -> None:
    manager = app.state.manager
    delete_unknown = _env_bool("EDGE_RECONCILE_DELETE_UNKNOWN", False)
    while True:
        try:
            summary = await manager.reconcile_all_devices_edge(
                dry_run=False,
                delete_unknown=delete_unknown,
            )
            if summary.get("errors"):
                logger.warning(
                    "Edge reconcile completed with errors device_count=%s errors=%s",
                    summary.get("device_count"),
                    len(summary.get("errors") or []),
                )
            elif summary.get("warnings"):
                logger.warning(
                    "Edge reconcile completed with warnings device_count=%s warnings=%s",
                    summary.get("device_count"),
                    len(summary.get("warnings") or []),
                )
            else:
                logger.info(
                    "Edge reconcile completed device_count=%s",
                    summary.get("device_count"),
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Edge reconcile loop failed")

        if RECONCILE_INTERVAL_S <= 0:
            return
        await asyncio.sleep(float(RECONCILE_INTERVAL_S))


async def _retention_cleanup_loop(app: FastAPI) -> None:
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Single place to:
      - initialize DB/tables
      - create Manager singleton
      - create + start pipeline
      - shutdown cleanly
    """

    ok = await db_manager.initialize_tables_and_data()
    if not ok:
        raise RuntimeError("FATAL: Could not initialize database tables and defaults.")

    app.state.manager = Manager(session_factory=SessionLocal)
    app.state.session_factory = SessionLocal
    app.state.user_snapshot_cache = UserSnapshotCache()
    hub = WebNotificationHub()
    
    # Configure email notifier with proper SMTP settings
    email_cfg = EmailConfig(
        enabled=bool(SMTP_USERNAME and SMTP_PASSWORD),
        smtp_host=SMTP_HOST,
        smtp_port=SMTP_PORT,
        smtp_user=SMTP_USERNAME,
        smtp_pass=SMTP_PASSWORD,
        from_email=FROM_EMAIL,
        dashboard_base_url=DASHBOARD_URL if DASHBOARD_URL else None,
    )
    email_notifier = EmailNotifier(email_cfg)
    
    # Create notification service with session factory for DB lookups
    # NOTIFY_ON_CONFIRMED=true → emit "item_detected" alerts when a track is confirmed.
    #   Needed for trigger_mode="any_detection" prerecording to fire without a configured ROI.
    notify_on_confirmed = _env_bool("NOTIFY_ON_CONFIRMED", False)
    svc = NotificationService(hub=hub, email=email_notifier, notify_on_confirmed=notify_on_confirmed)
    svc.set_session_factory(SessionLocal)  # Enable DB lookups for ROI and emails

    app.state.notification_hub = hub
    app.state.notification_service = svc
    app.state.manager.set_notification_service(svc)
    app.state.retention_service = RetentionService(session_factory=SessionLocal)
    app.state.alert_blob_cleanup_tasks = set()

    pipeline_startup = await app.state.manager.start_background_pipelines()
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

    app.state.edge_reconcile_task = asyncio.create_task(_edge_reconcile_loop(app), name="edge_reconcile_loop")
    app.state.retention_cleanup_task = asyncio.create_task(
        _retention_cleanup_loop(app),
        name="retention_cleanup_loop",
    )

    yield
    # Shutdown
    try:
        task = getattr(app.state, "edge_reconcile_task", None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Edge reconcile task shutdown failed")
        retention_task = getattr(app.state, "retention_cleanup_task", None)
        if retention_task:
            retention_task.cancel()
            try:
                await retention_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Retention cleanup task shutdown failed")
        blob_cleanup_tasks = set(getattr(app.state, "alert_blob_cleanup_tasks", set()) or set())
        if blob_cleanup_tasks:
            await asyncio.gather(*blob_cleanup_tasks, return_exceptions=True)
    finally:
        try:
            if hasattr(app.state, "manager") and app.state.manager:
                await app.state.manager.shutdown()
        except Exception:
            logger.exception("Manager shutdown failed")
        try:
            svc = getattr(app.state, "notification_service", None)
            if svc is not None and hasattr(svc, "shutdown"):
                await svc.shutdown()
        except Exception:
            logger.exception("Notification service shutdown failed")
        try:
            retention_svc = getattr(app.state, "retention_service", None)
            if retention_svc is not None and hasattr(retention_svc, "close"):
                await retention_svc.close()
        except Exception:
            logger.exception("Retention service shutdown failed")

app = FastAPI(debug=DEBUG, lifespan=lifespan)

# Cache-control middleware (good for MJPEG)
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
    allow_credentials=False,  # IMPORTANT: must be False if allow_origins is "*"
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)


# Routers
app.include_router(cameras_router, prefix="/api")
app.include_router(clips_router, prefix="/api")
app.include_router(sites_router,prefix="/api")
app.include_router(devices_router,prefix="/api")
app.include_router(notifications_router, prefix="/api")
app.include_router(notification_emails_router, prefix="/api")
app.include_router(auth_router, prefix="/api")
app.include_router(users_router, prefix="/api")

# Health
@app.get("/")
async def root():
    return {"status": "healthy", "message": "API is running"}
