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
from core.env import env_float, env_int
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


def _engine_pool_kwargs() -> dict:
    return {
        "pool_pre_ping": True,
        "pool_size": env_int("DB_POOL_SIZE", 30, minimum=1),
        "max_overflow": env_int("DB_MAX_OVERFLOW", 20, minimum=0),
        "pool_timeout": env_int("DB_POOL_TIMEOUT_S", 30, minimum=1),
        "pool_recycle": env_int("DB_POOL_RECYCLE_S", 1800, minimum=0),
        "pool_use_lifo": True,
    }


def _mysql_sync_connect_args() -> dict:
    timeout_s = env_int("DB_CONNECT_TIMEOUT_S", 5, minimum=1)
    io_timeout_s = env_int("DB_IO_TIMEOUT_S", 15, minimum=1)
    return {
        "connect_timeout": timeout_s,
        "read_timeout": io_timeout_s,
        "write_timeout": io_timeout_s,
    }


def _mysql_async_connect_args() -> dict:
    return {
        "connect_timeout": env_int("DB_CONNECT_TIMEOUT_S", 5, minimum=1),
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

            if dialect_name.startswith("mysql"):
                _composite_indexes = [
                    (
                        "notification",
                        "ix_notif_user_visible_detected",
                        "(user_id, visible, detected_at DESC)",
                    ),
                    (
                        "notification",
                        "ix_notif_user_site_visible_detected",
                        "(user_id, site_uuid, visible, detected_at DESC)",
                    ),
                    (
                        "notification",
                        "ix_notif_user_camera_visible",
                        "(user_id, camera_uuid, visible, detected_at DESC)",
                    ),
                    (
                        "notification",
                        "ix_notif_user_visible_unread",
                        "(user_id, visible, read_at)",
                    ),
                    (
                        "notification",
                        "ix_notif_user_camera_detected",
                        "(user_id, camera_uuid, detected_at DESC)",
                    ),
                    (
                        "video_record",
                        "ix_vr_camera_created",
                        "(camera_uuid, created_at DESC)",
                    ),
                ]
                for tbl, idx_name, idx_cols in _composite_indexes:
                    idx_exists = (
                        await conn.execute(
                            text(
                                "SELECT 1 FROM information_schema.STATISTICS "
                                "WHERE TABLE_SCHEMA = DATABASE() "
                                "  AND TABLE_NAME = :tbl "
                                "  AND INDEX_NAME = :idx "
                                "LIMIT 1"
                            ),
                            {"tbl": tbl, "idx": idx_name},
                        )
                    ).fetchone()
                    if not idx_exists:
                        try:
                            await conn.execute(
                                text(f"ALTER TABLE `{tbl}` ADD INDEX `{idx_name}` {idx_cols}")
                            )
                            logger.info("Created composite index %s.%s", tbl, idx_name)
                        except Exception as exc:
                            if "1061" in str(exc):
                                logger.info("Index %s.%s already exists.", tbl, idx_name)
                            else:
                                logger.warning(
                                    "Failed creating index %s.%s: %s", tbl, idx_name, exc
                                )

            # Migrate: add sites.is_deleted soft-delete flag if missing.
            # During site/user deletion the site row stays alive while the
            # background task batch-deletes notifications and extracts blob
            # keys.  is_deleted hides the site from all list/get queries so
            # it never reappears in the frontend.
            if dialect_name.startswith("mysql"):
                is_deleted_col = (
                    await conn.execute(
                        text("""
                            SELECT 1
                            FROM information_schema.COLUMNS
                            WHERE TABLE_SCHEMA = DATABASE()
                              AND TABLE_NAME   = 'sites'
                              AND COLUMN_NAME  = 'is_deleted'
                            LIMIT 1
                        """)
                    )
                ).fetchone()
                if not is_deleted_col:
                    try:
                        await conn.execute(
                            text("ALTER TABLE sites ADD COLUMN is_deleted TINYINT(1) NOT NULL DEFAULT 0")
                        )
                        logger.info("Added sites.is_deleted column.")
                    except Exception as exc:
                        if _is_duplicate_column_error(exc):
                            logger.info("sites.is_deleted column already exists.")
                        else:
                            raise

            # One-time cleanup: strip stale notification_trigger_mode from
            # channel_configurations.configuration JSON. Historical bug left
            # per-camera trigger values frozen in JSON even after the user
            # chose "Inherit from site", which then reverted the camera on
            # subsequent edits.
            if dialect_name.startswith("mysql"):
                try:
                    result = await conn.execute(
                        text(
                            "UPDATE channel_configurations "
                            "SET configuration = JSON_REMOVE(configuration, '$.notification_trigger_mode') "
                            "WHERE JSON_EXTRACT(configuration, '$.notification_trigger_mode') IS NOT NULL"
                        )
                    )
                    rc = getattr(result, "rowcount", 0) or 0
                    if rc:
                        logger.info("Cleaned stale notification_trigger_mode from %d channel_configurations.", rc)
                except Exception as exc:
                    logger.warning("Skipping channel_configurations.notification_trigger_mode cleanup: %s", exc)

            # Migrate camera.notification_trigger_mode and camera.camera_playback_enabled
            # to explicit tri-state strings ("inherit" | "roi_enter" | "any_detection"
            # and "inherit" | "always" | "never"). Historical schema used NULL for
            # "inherit" and a TINYINT bool for playback.
            if dialect_name.startswith("mysql"):
                try:
                    trig_col = (
                        await conn.execute(
                            text(
                                "SELECT IS_NULLABLE, COLUMN_DEFAULT "
                                "FROM information_schema.COLUMNS "
                                "WHERE TABLE_SCHEMA = DATABASE() "
                                "  AND TABLE_NAME = 'camera' "
                                "  AND COLUMN_NAME = 'notification_trigger_mode' "
                                "LIMIT 1"
                            )
                        )
                    ).fetchone()
                    if trig_col is not None:
                        is_nullable = str(trig_col[0]).upper() == "YES"
                        default_val = trig_col[1]
                        if is_nullable or default_val != "inherit":
                            await conn.execute(
                                text(
                                    "UPDATE camera SET notification_trigger_mode = 'inherit' "
                                    "WHERE notification_trigger_mode IS NULL OR notification_trigger_mode = ''"
                                )
                            )
                            await conn.execute(
                                text(
                                    "ALTER TABLE camera MODIFY COLUMN notification_trigger_mode "
                                    "VARCHAR(32) NOT NULL DEFAULT 'inherit'"
                                )
                            )
                            logger.info("Migrated camera.notification_trigger_mode to tri-state NOT NULL.")
                except Exception as exc:
                    logger.warning("Skipping camera.notification_trigger_mode migration: %s", exc)

                try:
                    pb_col = (
                        await conn.execute(
                            text(
                                "SELECT DATA_TYPE, COLUMN_TYPE "
                                "FROM information_schema.COLUMNS "
                                "WHERE TABLE_SCHEMA = DATABASE() "
                                "  AND TABLE_NAME = 'camera' "
                                "  AND COLUMN_NAME = 'camera_playback_enabled' "
                                "LIMIT 1"
                            )
                        )
                    ).fetchone()
                    if pb_col is not None:
                        data_type = str(pb_col[0]).lower()
                        if data_type not in ("varchar", "char", "text"):
                            # Rename existing column, add new string column, migrate data, drop old.
                            await conn.execute(
                                text(
                                    "ALTER TABLE camera "
                                    "ADD COLUMN camera_playback_enabled_new VARCHAR(16) "
                                    "NOT NULL DEFAULT 'inherit'"
                                )
                            )
                            await conn.execute(
                                text(
                                    "UPDATE camera SET camera_playback_enabled_new = CASE "
                                    "WHEN camera_playback_enabled = 1 THEN 'always' "
                                    "WHEN camera_playback_enabled = 0 THEN 'never' "
                                    "ELSE 'inherit' END"
                                )
                            )
                            await conn.execute(
                                text("ALTER TABLE camera DROP COLUMN camera_playback_enabled")
                            )
                            await conn.execute(
                                text(
                                    "ALTER TABLE camera CHANGE COLUMN camera_playback_enabled_new "
                                    "camera_playback_enabled VARCHAR(16) NOT NULL DEFAULT 'inherit'"
                                )
                            )
                            logger.info("Migrated camera.camera_playback_enabled BOOL -> VARCHAR tri-state.")
                except Exception as exc:
                    logger.warning("Skipping camera.camera_playback_enabled migration: %s", exc)

            # Migrate: the camera<->device M:N link table (camera_devices) is
            # replaced by a direct camera.device_uuid FK column. One device can
            # host many cameras; each camera has at most one device.
            # create_all adds the column on fresh DBs; existing DBs are fixed here.
            if dialect_name.startswith("mysql"):
                try:
                    has_device_uuid = (
                        await conn.execute(
                            text("""
                                SELECT 1
                                FROM information_schema.COLUMNS
                                WHERE TABLE_SCHEMA = DATABASE()
                                  AND TABLE_NAME   = 'camera'
                                  AND COLUMN_NAME  = 'device_uuid'
                                LIMIT 1
                            """)
                        )
                    ).fetchone()
                    if not has_device_uuid:
                        await conn.execute(
                            text("ALTER TABLE camera ADD COLUMN device_uuid BINARY(16) NULL")
                        )
                        await conn.execute(
                            text("ALTER TABLE camera ADD INDEX ix_camera_device_uuid (device_uuid)")
                        )
                        logger.info("Added camera.device_uuid column.")

                    has_camera_devices = (
                        await conn.execute(
                            text("""
                                SELECT 1
                                FROM information_schema.TABLES
                                WHERE TABLE_SCHEMA = DATABASE()
                                  AND TABLE_NAME   = 'camera_devices'
                                LIMIT 1
                            """)
                        )
                    ).fetchone()
                    if has_camera_devices:
                        # Backfill the single most-recent device per camera.
                        await conn.execute(
                            text("""
                                UPDATE camera c
                                JOIN (
                                    SELECT cd.camera_uuid, cd.device_uuid
                                    FROM camera_devices cd
                                    JOIN (
                                        SELECT camera_uuid, MAX(id) AS max_id
                                        FROM camera_devices
                                        GROUP BY camera_uuid
                                    ) latest
                                      ON latest.camera_uuid = cd.camera_uuid
                                     AND latest.max_id = cd.id
                                ) pick
                                  ON pick.camera_uuid = c.camera_uuid
                                SET c.device_uuid = pick.device_uuid
                                WHERE c.device_uuid IS NULL
                            """)
                        )
                        await conn.execute(text("DROP TABLE camera_devices"))
                        logger.info("Backfilled camera.device_uuid and dropped camera_devices table.")
                except Exception as exc:
                    logger.warning("Skipping camera.device_uuid migration: %s", exc)

            # ------------------------------------------------------------------
            # Multi-tenant RBAC migration (MySQL).
            #
            # create_all already creates the new tables on fresh DBs. For
            # existing databases we:
            #   * add `users.is_platform_admin` if missing
            #   * add `sites.org_id` if missing
            # The RBAC tables `organizations`, `roles`, `permissions`,
            # `role_permissions` and `access_grants` are created by
            # create_all; the legacy `org_memberships`/`site_memberships`
            # tables are migrated into `access_grants` and dropped further
            # down in this method.
            # ------------------------------------------------------------------
            if dialect_name.startswith("mysql"):
                try:
                    has_platform_admin = (
                        await conn.execute(
                            text(
                                "SELECT 1 FROM information_schema.COLUMNS "
                                "WHERE TABLE_SCHEMA = DATABASE() "
                                "  AND TABLE_NAME = 'users' "
                                "  AND COLUMN_NAME = 'is_platform_admin' "
                                "LIMIT 1"
                            )
                        )
                    ).fetchone()
                    if not has_platform_admin:
                        await conn.execute(
                            text(
                                "ALTER TABLE users ADD COLUMN is_platform_admin "
                                "TINYINT(1) NOT NULL DEFAULT 0"
                            )
                        )
                        logger.info("Added users.is_platform_admin column.")
                except Exception as exc:
                    logger.warning(
                        "Skipping users.is_platform_admin migration: %s", exc
                    )

                # --- schedule-aware arm/disarm override on sites -------------
                for col_name, col_def in (
                    ("arm_override", "TINYINT(1) NULL"),
                    ("arm_override_until", "DATETIME NULL"),
                ):
                    try:
                        has_col = (
                            await conn.execute(
                                text(
                                    "SELECT 1 FROM information_schema.COLUMNS "
                                    "WHERE TABLE_SCHEMA = DATABASE() "
                                    "  AND TABLE_NAME = 'sites' "
                                    f"  AND COLUMN_NAME = '{col_name}' "
                                    "LIMIT 1"
                                )
                            )
                        ).fetchone()
                        if not has_col:
                            await conn.execute(
                                text(f"ALTER TABLE sites ADD COLUMN `{col_name}` {col_def}")
                            )
                            logger.info("Added sites.%s column.", col_name)
                    except Exception as exc:
                        logger.warning("Skipping sites.%s migration: %s", col_name, exc)

                try:
                    has_org_id = (
                        await conn.execute(
                            text(
                                "SELECT 1 FROM information_schema.COLUMNS "
                                "WHERE TABLE_SCHEMA = DATABASE() "
                                "  AND TABLE_NAME = 'sites' "
                                "  AND COLUMN_NAME = 'org_id' "
                                "LIMIT 1"
                            )
                        )
                    ).fetchone()
                    if not has_org_id:
                        await conn.execute(
                            text("ALTER TABLE sites ADD COLUMN org_id INT NULL")
                        )
                        await conn.execute(
                            text(
                                "ALTER TABLE sites ADD INDEX ix_sites_org_id (org_id)"
                            )
                        )
                        logger.info("Added sites.org_id column.")
                except Exception as exc:
                    logger.warning("Skipping sites.org_id migration: %s", exc)

                # --- org_id on camera / devices -----------------------------
                for tbl in ("camera", "devices"):
                    try:
                        has_col = (
                            await conn.execute(
                                text(
                                    "SELECT 1 FROM information_schema.COLUMNS "
                                    "WHERE TABLE_SCHEMA = DATABASE() "
                                    f"  AND TABLE_NAME = '{tbl}' "
                                    "  AND COLUMN_NAME = 'org_id' "
                                    "LIMIT 1"
                                )
                            )
                        ).fetchone()
                        if not has_col:
                            await conn.execute(
                                text(f"ALTER TABLE `{tbl}` ADD COLUMN org_id INT NULL")
                            )
                            await conn.execute(
                                text(
                                    f"ALTER TABLE `{tbl}` ADD INDEX ix_{tbl}_org_id (org_id)"
                                )
                            )
                            logger.info("Added %s.org_id column.", tbl)
                    except Exception as exc:
                        logger.warning("Skipping %s.org_id migration: %s", tbl, exc)

                # --- relax legacy user_id to nullable (now "created_by") -----
                for tbl in ("sites", "camera", "devices"):
                    try:
                        is_nullable = (
                            await conn.execute(
                                text(
                                    "SELECT IS_NULLABLE FROM information_schema.COLUMNS "
                                    "WHERE TABLE_SCHEMA = DATABASE() "
                                    f"  AND TABLE_NAME = '{tbl}' "
                                    "  AND COLUMN_NAME = 'user_id' "
                                    "LIMIT 1"
                                )
                            )
                        ).scalar_one_or_none()
                        if is_nullable == "NO":
                            await conn.execute(
                                text(f"ALTER TABLE `{tbl}` MODIFY COLUMN user_id INT NULL")
                            )
                            logger.info("Relaxed %s.user_id to NULL.", tbl)
                    except Exception as exc:
                        logger.warning("Skipping %s.user_id nullable migration: %s", tbl, exc)

                # --- created_by on sites / camera / devices -----------------
                # Records who created the row. Backfilled from the legacy
                # user_id so existing rows keep an accurate creator.
                for tbl in ("sites", "camera", "devices"):
                    try:
                        has_created_by = (
                            await conn.execute(
                                text(
                                    "SELECT 1 FROM information_schema.COLUMNS "
                                    "WHERE TABLE_SCHEMA = DATABASE() "
                                    f"  AND TABLE_NAME = '{tbl}' "
                                    "  AND COLUMN_NAME = 'created_by' "
                                    "LIMIT 1"
                                )
                            )
                        ).fetchone()
                        if not has_created_by:
                            await conn.execute(
                                text(f"ALTER TABLE `{tbl}` ADD COLUMN created_by INT NULL")
                            )
                            await conn.execute(
                                text(
                                    f"ALTER TABLE `{tbl}` ADD INDEX ix_{tbl}_created_by (created_by)"
                                )
                            )
                            await conn.execute(
                                text(
                                    f"UPDATE `{tbl}` SET created_by = user_id "
                                    "WHERE created_by IS NULL"
                                )
                            )
                            logger.info("Added %s.created_by column (backfilled from user_id).", tbl)
                    except Exception as exc:
                        logger.warning("Skipping %s.created_by migration: %s", tbl, exc)

                # --- notification approval workflow columns ------------------
                try:
                    has_approval = (
                        await conn.execute(
                            text(
                                "SELECT 1 FROM information_schema.COLUMNS "
                                "WHERE TABLE_SCHEMA = DATABASE() "
                                "  AND TABLE_NAME = 'notification' "
                                "  AND COLUMN_NAME = 'approval_status' "
                                "LIMIT 1"
                            )
                        )
                    ).fetchone()
                    if not has_approval:
                        await conn.execute(
                            text(
                                "ALTER TABLE notification "
                                "ADD COLUMN approval_status VARCHAR(16) NOT NULL DEFAULT 'approved', "
                                "ADD COLUMN approved_by INT NULL, "
                                "ADD COLUMN approved_at DATETIME NULL"
                            )
                        )
                        await conn.execute(
                            text(
                                "ALTER TABLE notification "
                                "ADD INDEX ix_notif_site_approval_visible_detected "
                                "(site_uuid, approval_status, visible, detected_at)"
                            )
                        )
                        logger.info("Added notification.approval_status workflow columns.")
                except Exception as exc:
                    logger.warning("Skipping notification.approval_status migration: %s", exc)

                # --- unified RBAC: seed catalog + migrate memberships --------
                # create_all builds `roles`/`permissions`/`role_permissions`/
                # `access_grants` on fresh DBs. Here we idempotently seed the
                # role+permission catalog, then migrate any legacy
                # `org_memberships`/`site_memberships` rows into `access_grants`
                # and drop those tables.
                try:
                    from core.security.roles import (
                        PERMISSION_DESCRIPTIONS,
                        ROLE_PERMISSIONS,
                    )

                    # 1. permissions
                    for perm_name, perm_desc in PERMISSION_DESCRIPTIONS.items():
                        await conn.execute(
                            text(
                                "INSERT INTO permissions (name, description) "
                                "SELECT :n, :d FROM DUAL "
                                "WHERE NOT EXISTS "
                                "(SELECT 1 FROM permissions WHERE name = :n)"
                            ),
                            {"n": perm_name, "d": perm_desc},
                        )

                    # 2. roles (one row per distinct (name, scope))
                    for (role_name, scope) in {k for k in ROLE_PERMISSIONS.keys()}:
                        await conn.execute(
                            text(
                                "INSERT INTO roles (name, scope, description) "
                                "SELECT :n, :s, :d FROM DUAL "
                                "WHERE NOT EXISTS "
                                "(SELECT 1 FROM roles WHERE name = :n AND scope = :s)"
                            ),
                            {"n": role_name, "s": scope, "d": f"{scope} role: {role_name}"},
                        )

                    # 3. role -> permission links
                    for (role_name, scope), perms in ROLE_PERMISSIONS.items():
                        for perm in perms:
                            await conn.execute(
                                text(
                                    "INSERT INTO role_permissions (role_id, permission_id) "
                                    "SELECT r.id, p.id FROM roles r, permissions p "
                                    "WHERE r.name = :rn AND r.scope = :rs "
                                    "  AND p.name = :pn "
                                    "  AND NOT EXISTS (SELECT 1 FROM role_permissions rp "
                                    "    WHERE rp.role_id = r.id AND rp.permission_id = p.id)"
                                ),
                                {"rn": role_name, "rs": scope, "pn": perm.value},
                            )
                    logger.info("Seeded RBAC role/permission catalog.")
                except Exception as exc:
                    logger.warning("Skipping RBAC catalog seed: %s", exc)

                # Migrate legacy org_memberships -> access_grants, then drop.
                try:
                    has_org_mem = (
                        await conn.execute(
                            text(
                                "SELECT 1 FROM information_schema.TABLES "
                                "WHERE TABLE_SCHEMA = DATABASE() "
                                "  AND TABLE_NAME = 'org_memberships' LIMIT 1"
                            )
                        )
                    ).fetchone()
                    if has_org_mem:
                        await conn.execute(
                            text(
                                "INSERT INTO access_grants "
                                "  (user_id, role_id, org_id, site_uuid, created_at, updated_at) "
                                "SELECT m.user_id, r.id, m.org_id, NULL, m.created_at, m.updated_at "
                                "FROM org_memberships m "
                                "JOIN roles r ON r.name = m.role AND r.scope = 'org' "
                                "WHERE NOT EXISTS (SELECT 1 FROM access_grants g "
                                "  WHERE g.user_id = m.user_id AND g.role_id = r.id "
                                "    AND g.org_id = m.org_id AND g.site_uuid IS NULL)"
                            )
                        )
                        await conn.execute(text("DROP TABLE org_memberships"))
                        logger.info(
                            "Migrated org_memberships -> access_grants and dropped legacy table."
                        )
                except Exception as exc:
                    logger.warning("Skipping org_memberships migration: %s", exc)

                # Migrate legacy site_memberships -> access_grants, then drop.
                try:
                    has_site_mem = (
                        await conn.execute(
                            text(
                                "SELECT 1 FROM information_schema.TABLES "
                                "WHERE TABLE_SCHEMA = DATABASE() "
                                "  AND TABLE_NAME = 'site_memberships' LIMIT 1"
                            )
                        )
                    ).fetchone()
                    if has_site_mem:
                        await conn.execute(
                            text(
                                "INSERT INTO access_grants "
                                "  (user_id, role_id, org_id, site_uuid, created_at, updated_at) "
                                "SELECT m.user_id, r.id, NULL, m.site_uuid, m.created_at, m.updated_at "
                                "FROM site_memberships m "
                                "JOIN roles r ON r.name = m.role AND r.scope = 'site' "
                                "WHERE NOT EXISTS (SELECT 1 FROM access_grants g "
                                "  WHERE g.user_id = m.user_id AND g.role_id = r.id "
                                "    AND g.site_uuid = m.site_uuid AND g.org_id IS NULL)"
                            )
                        )
                        await conn.execute(text("DROP TABLE site_memberships"))
                        logger.info(
                            "Migrated site_memberships -> access_grants and dropped legacy table."
                        )
                except Exception as exc:
                    logger.warning("Skipping site_memberships migration: %s", exc)

                # --- backfill org_id from access grants ----------------------
                # Idempotent: only touches rows where org_id is still NULL.
                try:
                    await conn.execute(
                        text(
                            "UPDATE sites s "
                            "JOIN access_grants g "
                            "  ON g.user_id = s.user_id AND g.org_id IS NOT NULL "
                            "SET s.org_id = g.org_id "
                            "WHERE s.org_id IS NULL"
                        )
                    )
                    await conn.execute(
                        text(
                            "UPDATE camera c "
                            "JOIN sites s ON s.site_uuid = c.site_uuid "
                            "SET c.org_id = s.org_id "
                            "WHERE c.org_id IS NULL AND s.org_id IS NOT NULL"
                        )
                    )
                    await conn.execute(
                        text(
                            "UPDATE devices d "
                            "JOIN access_grants g "
                            "  ON g.user_id = d.user_id AND g.org_id IS NOT NULL "
                            "SET d.org_id = g.org_id "
                            "WHERE d.org_id IS NULL"
                        )
                    )
                    logger.info("Backfilled org_id on sites/camera/devices.")
                except Exception as exc:
                    logger.warning("Skipping org_id backfill: %s", exc)

        # Seed a dev user if DB is empty. The dev user is also flagged as
        # the platform admin so the new `/api/platform/...` endpoints are
        # usable out of the box.
        async with self.AsyncSessionLocal() as db:
            existing = (await db.execute(select(User.id).limit(1))).scalar_one_or_none()
            if existing is None:
                db.add(
                    User(
                        user_name="dev",
                        email="dev@example.com",
                        hashed_password=get_password_hash("DevPass123!"),
                        email_verified=True,
                        is_platform_admin=True,
                    )
                )
                await db.commit()

        return True

    async def initialize_tables_and_data(self) -> bool:
        max_attempts = env_int("DB_INIT_MAX_ATTEMPTS", 1, minimum=1)
        retry_delay_s = env_float("DB_INIT_RETRY_DELAY_S", 5.0, minimum=0.0)
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
