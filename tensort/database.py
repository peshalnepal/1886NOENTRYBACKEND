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

        # ✅ Py3.6 + SQLAlchemy 1.4 compatible way (no async_sessionmaker)
        self.AsyncSessionLocal = sessionmaker(
            bind=self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        # Sync engine (sqlite3)
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
            Base.metadata.create_all(bind=self.sync_engine)
            logger.info("Database tables created/verified successfully")
            return True
        except Exception as e:
            logger.exception("Failed to create database tables: %s", e)
            return False

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
