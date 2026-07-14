# database.py - Asynchronous SQLite database for Jetson Nano (Py3.6 compatible)

import logging
import os

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "jetson_cameras.db")

ASYNC_DB_URL = "sqlite+aiosqlite:///{path}".format(path=DB_PATH)
SYNC_DB_URL = "sqlite:///{path}".format(path=DB_PATH)


class DatabaseManager:
    def __init__(self, async_db_url=None, sync_db_url=None):
        self.async_db_url = async_db_url or ASYNC_DB_URL
        self.sync_db_url = sync_db_url or SYNC_DB_URL

        self.async_engine = None
        self.AsyncSessionLocal = None

        self.sync_engine = None
        self.SessionLocal = None

        self._setup()

    def _setup(self):
        logger.info("Initializing async SQLite database at: %s", DB_PATH)

        # Async engine (aiosqlite)
        self.async_engine = create_async_engine(
            self.async_db_url,
            echo=False,
            future=True,
        )

        self.AsyncSessionLocal = sessionmaker(
            bind=self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        self.sync_engine = create_engine(
            self.sync_db_url,
            connect_args={"check_same_thread": False},
            pool_pre_ping=True,
            echo=False,
        )

        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.sync_engine,
        )

        logger.info("SQLite database (async + sync) initialized successfully")

    def initialize_tables(self):
        from database_orm import Base
        try:
            self._migrate_rtsp_url_to_source_url()
            Base.metadata.create_all(bind=self.sync_engine)
            logger.info("Database tables created/verified successfully")
            return True
        except Exception as e:
            logger.exception("Failed to create database tables: %s", e)
            return False

    def _migrate_rtsp_url_to_source_url(self):
        """Rename the legacy camera_configs.rtsp_url column to source_url.

        The camera source is now a single generic ``source_url`` (rtsp/webrtc/
        http/rtmp/srt). Existing Jetson SQLite DBs still have the old ``rtsp_url``
        column; rename it in place (SQLite >= 3.25) so stored values are kept.
        ``create_all`` never alters existing tables, so this runs first.
        """
        from sqlalchemy import inspect, text
        try:
            inspector = inspect(self.sync_engine)
            if "camera_configs" not in inspector.get_table_names():
                return
            cols = {c["name"] for c in inspector.get_columns("camera_configs")}
            if "rtsp_url" in cols and "source_url" not in cols:
                with self.sync_engine.begin() as conn:
                    conn.execute(text(
                        "ALTER TABLE camera_configs RENAME COLUMN rtsp_url TO source_url"
                    ))
                logger.info("Renamed camera_configs.rtsp_url -> source_url.")
        except Exception as exc:
            logger.warning("Skipping camera_configs.rtsp_url->source_url migration: %s", exc)

    def get_session(self):
        return self.SessionLocal()

    def get_async_session(self):
        return self.AsyncSessionLocal()

    async def close(self):
        if self.async_engine:
            await self.async_engine.dispose()


db_manager = DatabaseManager()
SessionLocal = db_manager.SessionLocal
AsyncSessionLocal = db_manager.AsyncSessionLocal
engine = db_manager.sync_engine
async_engine = db_manager.async_engine
