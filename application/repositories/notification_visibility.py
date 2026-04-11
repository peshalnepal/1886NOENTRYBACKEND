import asyncio
import logging
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

_visible_supported_cache: Optional[bool] = None
_visible_supported_lock = asyncio.Lock()


async def _mysql_column_exists(db: AsyncSession, column_name: str) -> bool:
    row = (
        await db.execute(
            text(
                """
                SELECT 1
                FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = 'notification'
                  AND COLUMN_NAME = :column_name
                LIMIT 1
                """
            ),
            {"column_name": column_name},
        )
    ).fetchone()
    return bool(row)


async def _mssql_column_exists(db: AsyncSession, column_name: str) -> bool:
    row = (
        await db.execute(
            text(
                """
                SELECT TOP 1 1
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = 'notification'
                  AND COLUMN_NAME = :column_name
                """
            ),
            {"column_name": column_name},
        )
    ).fetchone()
    return bool(row)


async def _sqlite_column_exists(db: AsyncSession, column_name: str) -> bool:
    rows = (await db.execute(text("PRAGMA table_info(notification)"))).all()
    return any(str(row[1]) == column_name for row in rows if len(row) > 1)


async def notification_visible_supported(db: AsyncSession) -> bool:
    global _visible_supported_cache

    if _visible_supported_cache is not None:
        return _visible_supported_cache

    async with _visible_supported_lock:
        if _visible_supported_cache is not None:
            return _visible_supported_cache

        bind = db.get_bind()
        dialect_name = str(getattr(getattr(bind, "dialect", None), "name", "") or "").lower()

        try:
            if dialect_name.startswith("mysql"):
                exists = await _mysql_column_exists(db, "visible")
                if not exists:
                    try:
                        await db.execute(
                            text(
                                "ALTER TABLE notification "
                                "ADD COLUMN visible TINYINT(1) NOT NULL DEFAULT 1"
                            )
                        )
                        await db.commit()
                        logger.warning("Lazily added missing notification.visible column.")
                    except Exception:
                        await db.rollback()
                        logger.warning(
                            "Failed to lazily add notification.visible; continuing without it.",
                            exc_info=True,
                        )
                    exists = await _mysql_column_exists(db, "visible")

                if exists:
                    try:
                        await db.execute(text("UPDATE notification SET visible = 1 WHERE visible IS NULL"))
                        await db.commit()
                    except Exception:
                        await db.rollback()
                        logger.warning("Could not backfill notification.visible nulls.", exc_info=True)

                _visible_supported_cache = bool(exists)
                return _visible_supported_cache

            if dialect_name.startswith("mssql"):
                exists = await _mssql_column_exists(db, "visible")
                if not exists:
                    try:
                        await db.execute(text("ALTER TABLE notification ADD visible BIT NOT NULL DEFAULT 1"))
                        await db.commit()
                        logger.warning("Lazily added missing notification.visible column.")
                    except Exception:
                        await db.rollback()
                        logger.warning(
                            "Failed to lazily add notification.visible; continuing without it.",
                            exc_info=True,
                        )
                    exists = await _mssql_column_exists(db, "visible")

                if exists:
                    try:
                        await db.execute(text("UPDATE notification SET visible = 1 WHERE visible IS NULL"))
                        await db.commit()
                    except Exception:
                        await db.rollback()
                        logger.warning("Could not backfill notification.visible nulls.", exc_info=True)

                _visible_supported_cache = bool(exists)
                return _visible_supported_cache

            if dialect_name.startswith("sqlite"):
                _visible_supported_cache = await _sqlite_column_exists(db, "visible")
                return _visible_supported_cache
        except Exception:
            logger.warning(
                "Failed checking notification.visible support; continuing without the column filter.",
                exc_info=True,
            )

        _visible_supported_cache = False
        return _visible_supported_cache
