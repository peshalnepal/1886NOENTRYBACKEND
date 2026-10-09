"""Async SQLite engine and session factory.

WAL lets roster reads (API) proceed while a sweep writes. `busy_timeout` makes
a second writer wait instead of failing with "database is locked".
`create_all` only creates missing tables; a later column change needs explicit
migration code here.
"""

import logging
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from .database_orm import Base

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_async_engine("sqlite+aiosqlite:///{}".format(db_path))
        event.listen(self.engine.sync_engine, "connect", _configure_sqlite)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def init(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("NVR database ready at %s", self.engine.url.database)

    async def dispose(self) -> None:
        await self.engine.dispose()


def _configure_sqlite(dbapi_connection, _record):
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
    finally:
        cursor.close()
