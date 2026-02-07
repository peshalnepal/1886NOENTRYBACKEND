import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import async_sessionmaker
from routes.camera_routes import router as cameras_router
from routes.site_routes import router as sites_router
from routes.device_routes import router as devices_router
from routes.notifications_routes import router as notifications_router
from routes.notification_email_routes import router as notification_emails_router

from core.config import DEBUG
from core.database import db_manager, async_engine
from application.services.manager import Manager  # adjust if your path is different
import logging
from application.models.yolo_config import YoloModelConfig
from application.services.notification import WebNotificationHub, NotificationService, EmailNotifier, EmailConfig
logging.basicConfig(level=logging.INFO)

SessionLocal = async_sessionmaker(async_engine, expire_on_commit=False)

# Load SMTP configuration from environment
import os
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
FROM_EMAIL = os.environ.get("FROM_EMAIL") or os.environ.get("SMTP_FROM", "noreply@1886noentry.com")
DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "")


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
    svc = NotificationService(hub=hub, email=email_notifier)
    svc.set_session_factory(SessionLocal)  # Enable DB lookups for ROI and emails 

    app.state.notification_hub = hub
    app.state.notification_service = svc
    
    pipeline = await app.state.manager.create_pipeline()
    pipeline.set_session_factory(SessionLocal)          # <-- IMPORTANT (site_name lookup)
    pipeline.set_notification_service(app.state.notification_service)
    # Use DB-backed ROI definitions for alert checks.
    pipeline.set_roi_provider(svc._get_rois)
    await pipeline.start()
    app.state.pipeline = pipeline



    yield
    # Shutdown
    try:
        if hasattr(app.state, "pipeline") and app.state.pipeline:
            if hasattr(app.state.pipeline, "stop"):
                await app.state.pipeline.stop()
    except Exception:
        pass

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
app.include_router(sites_router,prefix="/api")
app.include_router(devices_router,prefix="/api")
app.include_router(notifications_router, prefix="/api")
app.include_router(notification_emails_router, prefix="/api")

# Health
@app.get("/")
async def root():
    return {"status": "healthy", "message": "API is running"}

