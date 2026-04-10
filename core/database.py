# database_core.py
import asyncio
import logging
import traceback
import urllib.parse
import os
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker

from core.database_orm import Base
from sqlalchemy import select
from core.database_orm import User
from core.security.hashing import get_password_hash

logger = logging.getLogger(__name__)
logging.getLogger("sqlalchemy.engine").setLevel(logging.ERROR)


def _is_duplicate_column_error(exc: Exception) -> bool:
    message = str(exc or "").lower()
    return (
        "duplicate column name" in message
        or "column names in each table must be unique" in message
        or "already has a column named" in message
    )


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None else int(default)
    except (TypeError, ValueError):
        value = int(default)
    return max(minimum, value)


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    try:
        value = float(raw) if raw is not None else float(default)
    except (TypeError, ValueError):
        value = float(default)
    return max(minimum, value)


def _engine_pool_kwargs() -> dict:
    return {
        "pool_pre_ping": True,
        "pool_size": _env_int("DB_POOL_SIZE", 30, minimum=1),
        "max_overflow": _env_int("DB_MAX_OVERFLOW", 20, minimum=0),
        "pool_timeout": _env_int("DB_POOL_TIMEOUT_S", 30, minimum=1),
        "pool_recycle": _env_int("DB_POOL_RECYCLE_S", 1800, minimum=0),
        "pool_use_lifo": True,
    }


def _mysql_sync_connect_args() -> dict:
    timeout_s = _env_int("DB_CONNECT_TIMEOUT_S", 5, minimum=1)
    io_timeout_s = _env_int("DB_IO_TIMEOUT_S", 15, minimum=1)
    return {
        "connect_timeout": timeout_s,
        "read_timeout": io_timeout_s,
        "write_timeout": io_timeout_s,
    }


def _mysql_async_connect_args() -> dict:
    return {
        "connect_timeout": _env_int("DB_CONNECT_TIMEOUT_S", 5, minimum=1),
    }


def _parse_odbc(odbc_conn_str: str) -> dict:
    parts = {}
    for part in odbc_conn_str.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            parts[k.strip().lower()] = v.strip()
    # normalize common aliases
    if "user" in parts and "uid" not in parts:
        parts["uid"] = parts["user"]
    if "password" in parts and "pwd" not in parts:
        parts["pwd"] = parts["password"]
    return parts


class DatabaseManager:
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.engine = None
        self.SessionLocal = None
        self.async_engine = None
        self.AsyncSessionLocal = None

        if not self.db_url:
            raise ValueError("DATABASE_URL must be configured.")
        self._setup()

    def _setup(self):
        conn_parts = _parse_odbc(self.db_url)
        driver = (conn_parts.get("driver") or "").strip("{}").lower()

        if "sql server" in driver or "odbc driver" in driver:
            self._setup_mssql(conn_parts)
        elif "mysql" in driver or "mariadb" in driver:
            self._setup_mysql(conn_parts)
        else:
            raise ValueError(f"Unsupported ODBC driver in connection string: {conn_parts.get('driver')}")

    # ---------------- MSSQL ----------------
    def _setup_mssql(self, conn_parts: dict):
        logger.info("Configuring MSSQL connections...")

        # Sync (pyodbc)
        params = urllib.parse.quote_plus(self.db_url)
        engine_url = f"mssql+pyodbc:///?odbc_connect={params}"
        pool_kwargs = _engine_pool_kwargs()
        self.engine = create_engine(engine_url, **pool_kwargs)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        # Async (aioodbc)
        async_url_obj = self._build_mssql_aioodbc_url(conn_parts)
        self.async_engine = create_async_engine(async_url_obj, **pool_kwargs)
        self.AsyncSessionLocal = async_sessionmaker(bind=self.async_engine, class_=AsyncSession, expire_on_commit=False)

    def _build_mssql_aioodbc_url(self, conn_parts: dict) -> URL:
        uid = conn_parts.get("uid")
        pwd = conn_parts.get("pwd")
        database = conn_parts.get("database")
        server = conn_parts.get("server")

        if not all([server, database, uid, pwd]):
            raise ValueError("Incomplete MSSQL connection string. Missing one of: Server, Database, Uid, Pwd")

        if server.lower().startswith("tcp:"):
            server = server[4:]

        host = server
        port = 1433
        if "," in server:
            host, port_str = server.split(",", 1)
            port = int(port_str)

        driver = (conn_parts.get("driver") or "ODBC Driver 18 for SQL Server").strip("{}")
        trust_cert = conn_parts.get("trustservercertificate", "yes")
        encrypt = conn_parts.get("encrypt", "yes")

        return URL.create(
            drivername="mssql+aioodbc",
            username=uid,
            password=pwd,
            host=host,
            port=port,
            database=database,
            query={
                "driver": driver,
                "Encrypt": encrypt,
                "TrustServerCertificate": trust_cert,
                "MARS_Connection": "Yes",
            },
        )

    # ---------------- MYSQL ----------------
    def _setup_mysql(self, conn_parts: dict):
        logger.info("Configuring MySQL connections...")

        host = conn_parts.get("server", "127.0.0.1")
        port = int(conn_parts.get("port", "3306"))
        database = conn_parts.get("database")
        uid = conn_parts.get("uid")
        pwd = conn_parts.get("pwd")

        if not all([host, port, database, uid, pwd]):
            raise ValueError("Incomplete MySQL connection string. Missing one of: Server, Port, Database, User/Uid, Password/Pwd")

        # Prefer native MySQL drivers for SQLAlchemy
        sync_url = f"mysql+pymysql://{urllib.parse.quote(uid)}:{urllib.parse.quote(pwd)}@{host}:{port}/{database}"
        async_url = f"mysql+aiomysql://{urllib.parse.quote(uid)}:{urllib.parse.quote(pwd)}@{host}:{port}/{database}"

        pool_kwargs = _engine_pool_kwargs()
        sync_connect_args = _mysql_sync_connect_args()
        async_connect_args = _mysql_async_connect_args()
        self.engine = create_engine(sync_url, connect_args=sync_connect_args, **pool_kwargs)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        self.async_engine = create_async_engine(async_url, connect_args=async_connect_args, **pool_kwargs)
        self.AsyncSessionLocal = async_sessionmaker(bind=self.async_engine, class_=AsyncSession, expire_on_commit=False)


    async def _initialize_tables_and_data_once(self) -> bool:
        ...
        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            dialect_name = str(conn.dialect.name or "").lower()

            # Migrate: old schema had a NOT NULL `code` column in email_verifications
            # that stored the plaintext OTP. New schema uses `code_hash` instead.
            # create_all never alters existing tables, so we fix the column here.
            res = await conn.execute(text("""
                SELECT COLUMN_TYPE
                FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME   = 'email_verifications'
                  AND COLUMN_NAME  = 'code'
                  AND IS_NULLABLE  = 'NO'
            """))
            row = res.fetchone()
            if row:
                col_type = row[0]
                await conn.execute(
                    text(f"ALTER TABLE email_verifications MODIFY COLUMN `code` {col_type} NULL DEFAULT NULL")
                )
                logger.info("Migrated email_verifications.code to nullable.")

            overlay_column = None
            if dialect_name.startswith("mysql"):
                overlay_column = (
                    await conn.execute(
                        text("""
                            SELECT 1
                            FROM information_schema.COLUMNS
                            WHERE TABLE_SCHEMA = DATABASE()
                              AND TABLE_NAME   = 'video_record'
                              AND COLUMN_NAME  = 'overlay_payload'
                            LIMIT 1
                        """)
                    )
                ).fetchone()
                if not overlay_column:
                    try:
                        await conn.execute(
                            text("ALTER TABLE video_record ADD COLUMN overlay_payload JSON NULL")
                        )
                        logger.info("Added video_record.overlay_payload column.")
                    except Exception as exc:
                        if _is_duplicate_column_error(exc):
                            logger.info("video_record.overlay_payload column already exists.")
                        else:
                            raise
            elif dialect_name.startswith("mssql"):
                overlay_column = (
                    await conn.execute(
                        text("""
                            SELECT TOP 1 1
                            FROM INFORMATION_SCHEMA.COLUMNS
                            WHERE TABLE_NAME  = 'video_record'
                              AND COLUMN_NAME = 'overlay_payload'
                        """)
                    )
                ).fetchone()
                if not overlay_column:
                    try:
                        await conn.execute(
                            text("ALTER TABLE video_record ADD overlay_payload NVARCHAR(MAX) NULL")
                        )
                        logger.info("Added video_record.overlay_payload column.")
                    except Exception as exc:
                        if _is_duplicate_column_error(exc):
                            logger.info("video_record.overlay_payload column already exists.")
                        else:
                            raise

            if dialect_name.startswith("mysql"):
                for col_name, col_def in [
                    ("config", "JSON NOT NULL DEFAULT '{}'"),
                    ("day_of_week", "INT NOT NULL DEFAULT 6"),
                    ("start_time", "TIME NOT NULL DEFAULT '00:00:00'"),
                    ("end_time", "TIME NOT NULL DEFAULT '23:59:59'"),
                    ("is_enabled", "TINYINT(1) NOT NULL DEFAULT 1"),
                ]:
                    missing = (
                        await conn.execute(
                            text("""
                                SELECT 1
                                FROM information_schema.COLUMNS
                                WHERE TABLE_SCHEMA = DATABASE()
                                  AND TABLE_NAME   = 'site_settings'
                                  AND COLUMN_NAME  = :col
                                LIMIT 1
                            """),
                            {"col": col_name},
                        )
                    ).fetchone()
                    if not missing:
                        try:
                            await conn.execute(
                                text(f"ALTER TABLE site_settings ADD COLUMN `{col_name}` {col_def}")
                            )
                            logger.info("Added site_settings.%s column.", col_name)
                        except Exception as exc:
                            if _is_duplicate_column_error(exc):
                                logger.info("site_settings.%s column already exists.", col_name)
                            else:
                                raise

            # Migrate: add notification.visible column if missing (added after initial schema).
            # Backfill existing NULL rows to visible=TRUE so they appear in the list endpoint.
            if dialect_name.startswith("mysql"):
                visible_col = (
                    await conn.execute(
                        text("""
                            SELECT 1
                            FROM information_schema.COLUMNS
                            WHERE TABLE_SCHEMA = DATABASE()
                              AND TABLE_NAME   = 'notification'
                              AND COLUMN_NAME  = 'visible'
                            LIMIT 1
                        """)
                    )
                ).fetchone()
                if not visible_col:
                    try:
                        await conn.execute(
                            text("ALTER TABLE notification ADD COLUMN visible TINYINT(1) NOT NULL DEFAULT 1")
                        )
                        logger.info("Added notification.visible column.")
                    except Exception as exc:
                        if _is_duplicate_column_error(exc):
                            logger.info("notification.visible column already exists.")
                        else:
                            raise
                else:
                    try:
                        await conn.execute(
                            text("UPDATE notification SET visible = 1 WHERE visible IS NULL")
                        )
                    except Exception:
                        logger.warning("Could not backfill notification.visible nulls.", exc_info=True)
            elif dialect_name.startswith("mssql"):
                visible_col = (
                    await conn.execute(
                        text("""
                            SELECT TOP 1 1
                            FROM INFORMATION_SCHEMA.COLUMNS
                            WHERE TABLE_NAME  = 'notification'
                              AND COLUMN_NAME = 'visible'
                        """)
                    )
                ).fetchone()
                if not visible_col:
                    try:
                        await conn.execute(
                            text("ALTER TABLE notification ADD visible BIT NOT NULL DEFAULT 1")
                        )
                        logger.info("Added notification.visible column.")
                    except Exception as exc:
                        if _is_duplicate_column_error(exc):
                            logger.info("notification.visible column already exists.")
                        else:
                            raise
                else:
                    try:
                        await conn.execute(
                            text("UPDATE notification SET visible = 1 WHERE visible IS NULL")
                        )
                    except Exception:
                        logger.warning("Could not backfill notification.visible nulls.", exc_info=True)

        # Seed a dev user if DB is empty.
        async with self.AsyncSessionLocal() as db:
            existing = (await db.execute(select(User.id).limit(1))).scalar_one_or_none()
            if existing is None:
                db.add(
                    User(
                        user_name="dev",
                        email="dev@example.com",
                        hashed_password=get_password_hash("DevPass123!"),
                        email_verified=True,
                    )
                )
                await db.commit()

        return True

    async def initialize_tables_and_data(self) -> bool:
        max_attempts = _env_int("DB_INIT_MAX_ATTEMPTS", 1, minimum=1)
        retry_delay_s = _env_float("DB_INIT_RETRY_DELAY_S", 5.0, minimum=0.0)
        logger.info(
            "Initializing database schema and seed data (max_attempts=%s, retry_delay_s=%.1f).",
            max_attempts,
            retry_delay_s,
        )

        for attempt in range(1, max_attempts + 1):
            try:
                ok = await self._initialize_tables_and_data_once()
                logger.info("Database initialization completed successfully.")
                return ok
            except Exception:
                if attempt >= max_attempts:
                    logger.exception(
                        "Database initialization failed after %s attempt(s).",
                        max_attempts,
                    )
                    return False
                logger.warning(
                    "Database initialization attempt %s/%s failed. Retrying in %.1fs.",
                    attempt,
                    max_attempts,
                    retry_delay_s,
                    exc_info=True,
                )
                await asyncio.sleep(retry_delay_s)


# --- Global Instance and Session Makers ---
# Create a single instance of the manager.
# This instance will be created once when the module is first imported.
db_manager = DatabaseManager(os.getenv("DATABASE_URL", "Driver={MySQL ODBC 8.0 Unicode Driver};Server=127.0.0.1;Port=3306;Database=appdb;User=appuser;Password=AppUser@2025!;Option=3;"))
async_engine = db_manager.async_engine
SessionLocal = db_manager.SessionLocal
AsyncSessionLocal = db_manager.AsyncSessionLocal


def get_connection_pool_status() -> dict:
    """Returns the current status of the async database connection pool."""
    try:
        pool = db_manager.async_engine.pool
        return {
            "size": pool.size() if hasattr(pool, 'size') else None,
            "checked_out": pool.checkedout() if hasattr(pool, 'checkedout') else None,
            "overflow": pool.overflow() if hasattr(pool, 'overflow') else None,
            "total_created": pool._all_conns if hasattr(pool, '_all_conns') else None,
        }
    except Exception as exc:
        logger.warning("Failed to get connection pool status: %s", exc)
        return {"error": str(exc)}
