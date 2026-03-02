# database_core.py
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
        self.engine = create_engine(engine_url, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        # Async (aioodbc)
        async_url_obj = self._build_mssql_aioodbc_url(conn_parts)
        self.async_engine = create_async_engine(async_url_obj, pool_pre_ping=True)
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

        self.engine = create_engine(sync_url, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        self.async_engine = create_async_engine(async_url, pool_pre_ping=True)
        self.AsyncSessionLocal = async_sessionmaker(bind=self.async_engine, class_=AsyncSession, expire_on_commit=False)


    async def initialize_tables_and_data(self):
        ...
        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

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


# --- Global Instance and Session Makers ---
# Create a single instance of the manager.
# This instance will be created once when the module is first imported.
db_manager = DatabaseManager(os.getenv("DATABASE_URL", "Driver={MySQL ODBC 8.0 Unicode Driver};Server=127.0.0.1;Port=3306;Database=appdb;User=appuser;Password=AppUser@2025!;Option=3;"))
async_engine = db_manager.async_engine
SessionLocal = db_manager.SessionLocal
AsyncSessionLocal = db_manager.AsyncSessionLocal
