#!/usr/bin/env python3
"""
Migrate notification tables to the current backend schema and insert sample rows.

This script is idempotent:
- Safe to run multiple times.
- Adds missing columns/indexes/constraints only when absent.
- Backfills legacy notification_emails rows that had no site_uuid by duplicating them
  across all sites owned by the same user, then removes the NULL-site legacy rows.

Examples:
  python3 Backend/scripts/migrate_notifications_schema.py
  python3 Backend/scripts/migrate_notifications_schema.py --user-id 1 --email alerts@example.com
  python3 Backend/scripts/migrate_notifications_schema.py --host 127.0.0.1 --port 3306 --db appdb --db-user appuser --db-pass 'AppUser@2025!'
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, Optional

try:
    import pymysql
except ImportError as exc:  # pragma: no cover
    print(
        "Missing dependency: pymysql. Install with `pip install PyMySQL` "
        "or `pip install -r Backend/requirements.txt`.",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc


def parse_odbc_conn_str(odbc: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in (odbc or "").split(";"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip().lower()] = v.strip()
    # aliases
    if "user" in out and "uid" not in out:
        out["uid"] = out["user"]
    if "password" in out and "pwd" not in out:
        out["pwd"] = out["password"]
    return out


def resolve_db_args(args: argparse.Namespace) -> Dict[str, object]:
    raw_dsn = args.dsn or os.getenv("DATABASE_URL", "")
    parsed = parse_odbc_conn_str(raw_dsn)

    host = args.host or parsed.get("server") or "127.0.0.1"
    port = int(args.port or parsed.get("port") or 3306)
    db_name = args.db or parsed.get("database") or "appdb"
    user = args.db_user or parsed.get("uid") or "appuser"
    password = args.db_pass or parsed.get("pwd") or "AppUser@2025!"

    # Handle MSSQL-like "tcp:host,1433" format gracefully.
    if isinstance(host, str):
        if host.lower().startswith("tcp:"):
            host = host[4:]
        if "," in host and not args.port:
            host_part, port_part = host.split(",", 1)
            host = host_part.strip()
            if port_part.strip().isdigit():
                port = int(port_part.strip())

    return {
        "host": host,
        "port": port,
        "database": db_name,
        "user": user,
        "password": password,
    }


def quote_ident(name: str) -> str:
    if not re.match(r"^[A-Za-z0-9_]+$", name):
        raise ValueError(f"Unsafe identifier: {name!r}")
    return f"`{name}`"


def scalar(cur, sql: str, params=None) -> int:
    cur.execute(sql, params or ())
    row = cur.fetchone()
    if not row:
        return 0
    v = next(iter(row.values()))
    return int(v or 0)


def table_exists(cur, db_name: str, table: str) -> bool:
    n = scalar(
        cur,
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema=%s AND table_name=%s
        """,
        (db_name, table),
    )
    return n > 0


def column_exists(cur, db_name: str, table: str, column: str) -> bool:
    n = scalar(
        cur,
        """
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_schema=%s AND table_name=%s AND column_name=%s
        """,
        (db_name, table, column),
    )
    return n > 0


def column_is_nullable(cur, db_name: str, table: str, column: str) -> bool:
    cur.execute(
        """
        SELECT IS_NULLABLE AS is_nullable
        FROM information_schema.columns
        WHERE table_schema=%s AND table_name=%s AND column_name=%s
        """,
        (db_name, table, column),
    )
    row = cur.fetchone()
    if not row:
        return True

    # DictCursor may still return uppercase keys depending on server/driver
    v = row.get("is_nullable")
    if v is None:
        v = row.get("IS_NULLABLE")

    return str(v).upper() == "YES"

def index_exists(cur, db_name: str, table: str, index_name: str) -> bool:
    n = scalar(
        cur,
        """
        SELECT COUNT(*)
        FROM information_schema.statistics
        WHERE table_schema=%s AND table_name=%s AND index_name=%s
        """,
        (db_name, table, index_name),
    )
    return n > 0


def constraint_exists(
    cur,
    db_name: str,
    table: str,
    constraint_name: str,
    constraint_type: Optional[str] = None,
) -> bool:
    sql = """
        SELECT COUNT(*)
        FROM information_schema.table_constraints
        WHERE table_schema=%s AND table_name=%s AND constraint_name=%s
    """
    params = [db_name, table, constraint_name]
    if constraint_type:
        sql += " AND constraint_type=%s"
        params.append(constraint_type)
    n = scalar(cur, sql, tuple(params))
    return n > 0


def ensure_column(cur, db_name: str, table: str, column: str, ddl: str) -> None:
    if column_exists(cur, db_name, table, column):
        return
    print(f"Adding column {table}.{column}")
    cur.execute(ddl)


def ensure_index(cur, db_name: str, table: str, index_name: str, ddl: str) -> None:
    if index_exists(cur, db_name, table, index_name):
        return
    print(f"Adding index {index_name} on {table}")
    cur.execute(ddl)


def ensure_constraint(
    cur,
    db_name: str,
    table: str,
    constraint_name: str,
    constraint_type: str,
    ddl: str,
) -> None:
    if constraint_exists(cur, db_name, table, constraint_name, constraint_type):
        return
    print(f"Adding {constraint_type} {constraint_name} on {table}")
    cur.execute(ddl)


def ensure_notification_emails(cur, db_name: str) -> None:
    # Create base table (if it never existed)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS notification_emails (
          id INT NOT NULL AUTO_INCREMENT,
          user_id INT NOT NULL,
          site_uuid BINARY(16) NULL,
          email VARCHAR(255) NOT NULL,
          is_enabled TINYINT(1) NOT NULL DEFAULT 1,
          created_at DATETIME(6) NULL DEFAULT CURRENT_TIMESTAMP(6),
          updated_at DATETIME(6) NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
          PRIMARY KEY (id),
          KEY ix_notification_emails_user_id (user_id),
          KEY ix_notification_emails_site_uuid (site_uuid)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
    )

    # Add missing columns if upgrading from legacy schema
    ensure_column(
        cur,
        db_name,
        "notification_emails",
        "site_uuid",
        "ALTER TABLE notification_emails ADD COLUMN site_uuid BINARY(16) NULL AFTER user_id",
    )
    ensure_column(
        cur,
        db_name,
        "notification_emails",
        "is_enabled",
        "ALTER TABLE notification_emails ADD COLUMN is_enabled TINYINT(1) NOT NULL DEFAULT 1 AFTER email",
    )
    ensure_column(
        cur,
        db_name,
        "notification_emails",
        "updated_at",
        "ALTER TABLE notification_emails ADD COLUMN updated_at DATETIME(6) NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6) AFTER created_at",
    )

    # Backfill legacy rows (site_uuid NULL) to all sites for the same user.
    # This assumes your existing legacy rows represent "global for user" emails.
    cur.execute(
        """
        INSERT INTO notification_emails (user_id, site_uuid, email, is_enabled, created_at, updated_at)
        SELECT DISTINCT
          ne.user_id,
          s.site_uuid,
          ne.email,
          COALESCE(ne.is_enabled, 1),
          COALESCE(ne.created_at, UTC_TIMESTAMP(6)),
          UTC_TIMESTAMP(6)
        FROM notification_emails ne
        JOIN sites s ON s.user_id = ne.user_id
        WHERE ne.site_uuid IS NULL
          AND NOT EXISTS (
            SELECT 1
            FROM notification_emails x
            WHERE x.user_id = ne.user_id
              AND x.email = ne.email
              AND x.site_uuid = s.site_uuid
          )
        """
    )

    # Remove unusable null-site rows.
    cur.execute("DELETE FROM notification_emails WHERE site_uuid IS NULL")

    # Remove duplicates before unique key.
    cur.execute(
        """
        DELETE ne1
        FROM notification_emails ne1
        JOIN notification_emails ne2
          ON ne1.user_id = ne2.user_id
         AND ne1.email = ne2.email
         AND ne1.site_uuid = ne2.site_uuid
         AND ne1.id > ne2.id
        """
    )

    # Clean orphans before foreign keys.
    cur.execute(
        """
        DELETE ne
        FROM notification_emails ne
        LEFT JOIN users u ON u.id = ne.user_id
        WHERE u.id IS NULL
        """
    )
    cur.execute(
        """
        DELETE ne
        FROM notification_emails ne
        LEFT JOIN sites s ON s.site_uuid = ne.site_uuid
        WHERE s.site_uuid IS NULL
        """
    )

    ensure_index(
        cur,
        db_name,
        "notification_emails",
        "ix_notification_emails_user_id",
        "ALTER TABLE notification_emails ADD INDEX ix_notification_emails_user_id (user_id)",
    )
    ensure_index(
        cur,
        db_name,
        "notification_emails",
        "ix_notification_emails_site_uuid",
        "ALTER TABLE notification_emails ADD INDEX ix_notification_emails_site_uuid (site_uuid)",
    )

    ensure_constraint(
        cur,
        db_name,
        "notification_emails",
        "uq_notif_email_user_site_email",
        "UNIQUE",
        "ALTER TABLE notification_emails ADD CONSTRAINT uq_notif_email_user_site_email UNIQUE (user_id, site_uuid, email)",
    )
    ensure_constraint(
        cur,
        db_name,
        "notification_emails",
        "fk_notification_emails_user_id",
        "FOREIGN KEY",
        "ALTER TABLE notification_emails ADD CONSTRAINT fk_notification_emails_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE",
    )
    ensure_constraint(
        cur,
        db_name,
        "notification_emails",
        "fk_notification_emails_site_uuid",
        "FOREIGN KEY",
        "ALTER TABLE notification_emails ADD CONSTRAINT fk_notification_emails_site_uuid FOREIGN KEY (site_uuid) REFERENCES sites(site_uuid) ON DELETE CASCADE",
    )

    # Make site_uuid NOT NULL (matches your new ORM) after backfill succeeded.
    if column_exists(cur, db_name, "notification_emails", "site_uuid") and column_is_nullable(
        cur, db_name, "notification_emails", "site_uuid"
    ):
        # Only safe now because we deleted NULL rows above.
        print("Altering notification_emails.site_uuid to NOT NULL")
        cur.execute("ALTER TABLE notification_emails MODIFY COLUMN site_uuid BINARY(16) NOT NULL")


def ensure_notification(cur, db_name: str) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS notification (
          id INT NOT NULL AUTO_INCREMENT,
          user_id INT NOT NULL,
          site_uuid BINARY(16) NOT NULL,
          camera_uuid BINARY(16) NULL,
          device_uuid BINARY(16) NULL,
          event_type VARCHAR(64) NOT NULL DEFAULT 'detection',
          title VARCHAR(255) NULL,
          message TEXT NULL,
          payload JSON NULL,
          detected_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
          created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
          read_at DATETIME(6) NULL,
          sent_at DATETIME(6) NULL,
          status VARCHAR(32) NOT NULL DEFAULT 'created',
          PRIMARY KEY (id),
          KEY ix_notification_user_id (user_id),
          KEY ix_notification_site_uuid (site_uuid),
          KEY ix_notification_camera_uuid (camera_uuid),
          KEY ix_notification_device_uuid (device_uuid),
          KEY ix_notification_detected_at (detected_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
    )

    ensure_column(
        cur,
        db_name,
        "notification",
        "site_uuid",
        "ALTER TABLE notification ADD COLUMN site_uuid BINARY(16) NOT NULL AFTER user_id",
    )
    ensure_column(
        cur,
        db_name,
        "notification",
        "camera_uuid",
        "ALTER TABLE notification ADD COLUMN camera_uuid BINARY(16) NULL AFTER site_uuid",
    )
    ensure_column(
        cur,
        db_name,
        "notification",
        "device_uuid",
        "ALTER TABLE notification ADD COLUMN device_uuid BINARY(16) NULL AFTER camera_uuid",
    )
    ensure_column(
        cur,
        db_name,
        "notification",
        "payload",
        "ALTER TABLE notification ADD COLUMN payload JSON NULL AFTER message",
    )
    ensure_column(
        cur,
        db_name,
        "notification",
        "read_at",
        "ALTER TABLE notification ADD COLUMN read_at DATETIME(6) NULL AFTER created_at",
    )
    ensure_column(
        cur,
        db_name,
        "notification",
        "sent_at",
        "ALTER TABLE notification ADD COLUMN sent_at DATETIME(6) NULL AFTER read_at",
    )
    ensure_column(
        cur,
        db_name,
        "notification",
        "status",
        "ALTER TABLE notification ADD COLUMN status VARCHAR(32) NOT NULL DEFAULT 'created' AFTER sent_at",
    )

    ensure_index(
        cur,
        db_name,
        "notification",
        "ix_notification_user_id",
        "ALTER TABLE notification ADD INDEX ix_notification_user_id (user_id)",
    )
    ensure_index(
        cur,
        db_name,
        "notification",
        "ix_notification_site_uuid",
        "ALTER TABLE notification ADD INDEX ix_notification_site_uuid (site_uuid)",
    )
    ensure_index(
        cur,
        db_name,
        "notification",
        "ix_notification_camera_uuid",
        "ALTER TABLE notification ADD INDEX ix_notification_camera_uuid (camera_uuid)",
    )
    ensure_index(
        cur,
        db_name,
        "notification",
        "ix_notification_device_uuid",
        "ALTER TABLE notification ADD INDEX ix_notification_device_uuid (device_uuid)",
    )
    ensure_index(
        cur,
        db_name,
        "notification",
        "ix_notification_detected_at",
        "ALTER TABLE notification ADD INDEX ix_notification_detected_at (detected_at)",
    )

    # Clean orphans before foreign keys.
    cur.execute(
        """
        DELETE n
        FROM notification n
        LEFT JOIN users u ON u.id = n.user_id
        WHERE u.id IS NULL
        """
    )
    cur.execute(
        """
        DELETE n
        FROM notification n
        LEFT JOIN sites s ON s.site_uuid = n.site_uuid
        WHERE s.site_uuid IS NULL
        """
    )

    ensure_constraint(
        cur,
        db_name,
        "notification",
        "fk_notification_user_id",
        "FOREIGN KEY",
        "ALTER TABLE notification ADD CONSTRAINT fk_notification_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE",
    )
    ensure_constraint(
        cur,
        db_name,
        "notification",
        "fk_notification_site_uuid",
        "FOREIGN KEY",
        "ALTER TABLE notification ADD CONSTRAINT fk_notification_site_uuid FOREIGN KEY (site_uuid) REFERENCES sites(site_uuid) ON DELETE CASCADE",
    )


def add_sample_rows(cur, user_id: int, email: str) -> None:
    cur.execute(
        """
        SELECT site_uuid
        FROM sites
        WHERE user_id=%s
        ORDER BY site_uuid
        LIMIT 1
        """,
        (user_id,),
    )
    row = cur.fetchone()
    if not row:
        print(f"No site found for user_id={user_id}; sample rows not inserted.")
        return

    site_uuid = row["site_uuid"]
    cur.execute(
        """
        INSERT INTO notification_emails (user_id, site_uuid, email, is_enabled, created_at, updated_at)
        VALUES (%s, %s, %s, 1, UTC_TIMESTAMP(6), UTC_TIMESTAMP(6))
        ON DUPLICATE KEY UPDATE
          is_enabled=VALUES(is_enabled),
          updated_at=VALUES(updated_at)
        """,
        (user_id, site_uuid, email.strip().lower()),
    )

    cur.execute(
        """
        INSERT INTO notification (
          user_id, site_uuid, camera_uuid, device_uuid, event_type,
          title, message, payload, detected_at, created_at, status
        )
        VALUES (
          %s, %s, NULL, NULL, 'manual_test',
          'Manual Notification Row',
          'Inserted by migrate_notifications_schema.py',
          JSON_OBJECT('source', 'manual_migration_py'),
          UTC_TIMESTAMP(6), UTC_TIMESTAMP(6), 'created'
        )
        """,
        (user_id, site_uuid),
    )
    print("Inserted sample rows into notification_emails and notification.")


def uuid_hex_expr(col: str) -> str:
    safe = quote_ident(col)
    return (
        "CASE WHEN {c} IS NULL THEN NULL ELSE LOWER(CONCAT("
        "SUBSTR(HEX({c}),1,8),'-',SUBSTR(HEX({c}),9,4),'-',SUBSTR(HEX({c}),13,4),"
        "'-',SUBSTR(HEX({c}),17,4),'-',SUBSTR(HEX({c}),21,12)"
        ")) END"
    ).format(c=safe)


def print_rows(title: str, rows) -> None:
    print("")
    print(title)
    if not rows:
        print("(no rows)")
        return
    for r in rows:
        print(r)


def run(args: argparse.Namespace) -> int:
    cfg = resolve_db_args(args)
    print(
        f"Connecting to MySQL host={cfg['host']} port={cfg['port']} db={cfg['database']} user={cfg['user']}"
    )

    conn = pymysql.connect(
        host=str(cfg["host"]),
        port=int(cfg["port"]),
        user=str(cfg["user"]),
        password=str(cfg["password"]),
        database=str(cfg["database"]),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )
    try:
        with conn.cursor() as cur:
            db_name = str(cfg["database"])

            if not table_exists(cur, db_name, "users"):
                raise RuntimeError("Missing required table: users")
            if not table_exists(cur, db_name, "sites"):
                raise RuntimeError("Missing required table: sites")

            ensure_notification_emails(cur, db_name)
            ensure_notification(cur, db_name)

            if args.insert_sample:
                add_sample_rows(cur, user_id=args.user_id, email=args.email)

            conn.commit()

            cur.execute(
                f"""
                SELECT
                  id,
                  user_id,
                  {uuid_hex_expr('site_uuid')} AS site_uuid,
                  email,
                  is_enabled,
                  created_at,
                  updated_at
                FROM notification_emails
                ORDER BY id DESC
                LIMIT %s
                """,
                (args.limit,),
            )
            email_rows = cur.fetchall()

            cur.execute(
                f"""
                SELECT
                  id,
                  user_id,
                  {uuid_hex_expr('site_uuid')} AS site_uuid,
                  {uuid_hex_expr('camera_uuid')} AS camera_uuid,
                  {uuid_hex_expr('device_uuid')} AS device_uuid,
                  event_type,
                  title,
                  message,
                  payload,
                  detected_at,
                  created_at,
                  read_at,
                  sent_at,
                  status
                FROM notification
                ORDER BY id DESC
                LIMIT %s
                """,
                (args.limit,),
            )
            notif_rows = cur.fetchall()

        print_rows(f"notification_emails (top {args.limit})", email_rows)
        print_rows(f"notification (top {args.limit})", notif_rows)
        print("\nMigration completed successfully.")
        return 0
    except Exception as exc:
        conn.rollback()
        print(f"Migration failed: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Migrate notification schema and optionally insert sample rows."
    )
    p.add_argument("--dsn", default="", help="Optional ODBC-style DATABASE_URL string.")
    p.add_argument("--host", default="", help="DB host (overrides DSN/env).")
    p.add_argument("--port", type=int, default=0, help="DB port (overrides DSN/env).")
    p.add_argument("--db", default="", help="DB name (overrides DSN/env).")
    p.add_argument("--db-user", default="", help="DB user (overrides DSN/env).")
    p.add_argument("--db-pass", default="", help="DB password (overrides DSN/env).")

    p.add_argument("--user-id", type=int, default=1, help="Sample insert user_id.")
    p.add_argument(
        "--email",
        default="alerts@example.com",
        help="Sample notification email row value.",
    )
    p.add_argument("--limit", type=int, default=50, help="Verification query row limit.")
    p.add_argument(
        "--insert-sample",
        action="store_true",
        help="If set, inserts a sample notification_emails row and a sample notification row.",
    )
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
