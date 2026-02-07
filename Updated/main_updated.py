# agents/main.py (Azure backend)

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.database import init_db, get_session_factory
from application.services.agent_manager import Manager  # <- updated manager (DB + WebRTC + Edge)
from agents.api.routes.camera_routes import router as camera_router
from agents.api.routes.site_routes import router as site_router
from agents.api.routes.device_routes import router as device_router

import logging
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Azure backend startup:

    - Initialize DB
    - Create singleton Manager (DB + WebRTC provisioning + Edge provisioning)
    - DO NOT start any RTSP ingest / MJPEG streaming pipeline in Azure.
      Video playback must happen via WebRTC gateway.
    """
    await init_db()

    session_factory = get_session_factory()
    app.state.manager = Manager(session_factory)

    # Ensure at least one pipeline exists (keeps older routes working)
    try:
        await app.state.manager.get_activepipeline()
    except Exception:
        logger.exception("Failed ensuring default pipeline exists on startup")

    yield

    # Shutdown
    try:
        await app.state.manager.shutdown()
    except Exception:
        logger.exception("Manager shutdown failed")


app = FastAPI(title="AI Video Monitoring Backend", lifespan=lifespan)

# CORS (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(site_router)
app.include_router(device_router)
app.include_router(camera_router)
