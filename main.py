import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import async_sessionmaker

from core.config import DEBUG
from core.database import db_manager, async_engine
from application.services.manager import Manager  # adjust if your path is different
import logging
from application.models.yolo_config import YoloModelConfig
logging.basicConfig(level=logging.INFO)

SessionLocal = async_sessionmaker(async_engine, expire_on_commit=False)


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

    # Create singleton manager (sessionmaker is callable -> returns new AsyncSession)
    app.state.manager = Manager(session_factory=SessionLocal)
    
    # Create and start a pipeline once
    pipeline = await app.state.manager.create_pipeline()
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
from routes import camera_routes
app.include_router(camera_routes.router, prefix="/api")

# Health
@app.get("/")
async def root():
    return {"status": "healthy", "message": "API is running"}


