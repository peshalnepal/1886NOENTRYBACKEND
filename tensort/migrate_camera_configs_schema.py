#!/usr/bin/env python3
"""
Migrate Jetson SQLite camera_configs from the legacy schema to the expanded ORM.

The legacy table stored only identifiers plus a JSON blob. The current ORM keeps
that JSON for backward compatibility, but also persists the common camera fields
as top-level columns so the Jetson can query or inspect them directly.
"""

import argparse
import json
import logging
import os
import shutil
import sqlite3
from datetime import datetime, timezone


LOGGER = logging.getLogger("camera-config-migration")
DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "jetson_cameras.db")
TABLE_NAME = "camera_configs"
TEMP_TABLE_NAME = "camera_configs_new"
LEGACY_TABLE_NAME = "camera_configs_legacy"

TARGET_COLUMNS = (
    "id",
    "channel_id",
    "camera_uuid",
    "user_id",
    "site_uuid",
    "device_uuid",
    "camera_code",
    "name",
    "location",
    "rtsp_url",
    "webrtc_url",
    "is_enabled",
    "is_detection_enabled",
    "is_notification_enabled",
    "roi",
    "config_json",
    "created_at",
    "updated_at",
)

CREATE_TABLE_SQL = """
CREATE TABLE {table_name} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id VARCHAR(64) NOT NULL,
    camera_uuid VARCHAR(64) NOT NULL,
    user_id INTEGER NOT NULL DEFAULT 1,
    site_uuid VARCHAR(64),
    device_uuid VARCHAR(64),
    camera_code VARCHAR(64),
    name VARCHAR(255),
    location VARCHAR(255),
    rtsp_url TEXT NOT NULL,
    webrtc_url TEXT,
    is_enabled BOOLEAN NOT NULL DEFAULT 1,
    is_detection_enabled BOOLEAN NOT NULL DEFAULT 1,
    is_notification_enabled BOOLEAN NOT NULL DEFAULT 1,
    roi JSON,
    config_json JSON NOT NULL DEFAULT '{{}}',
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
)
"""

CREATE_INDEX_SQL = (
    "CREATE UNIQUE INDEX ix_camera_configs_channel_id ON {table_name}(channel_id)",
    "CREATE UNIQUE INDEX ix_camera_configs_camera_uuid ON {table_name}(camera_uuid)",
    "CREATE INDEX ix_camera_configs_user_id ON {table_name}(user_id)",
    "CREATE INDEX ix_camera_configs_site_uuid ON {table_name}(site_uuid)",
    "CREATE INDEX ix_camera_configs_device_uuid ON {table_name}(device_uuid)",
    "CREATE INDEX ix_camera_configs_camera_code ON {table_name}(camera_code)",
)


def _table_exists(conn, table_name):
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _get_columns(conn, table_name):
    rows = conn.execute("PRAGMA table_info({})".format(table_name)).fetchall()
    return [row[1] for row in rows]


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _string_or_none(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return str(value)


def _coerce_int(value, default):
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value, default):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "on"):
            return True
        if normalized in ("0", "false", "no", "off", ""):
            return False
    return bool(value)


def _load_json(value, default):
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        try:
            return json.loads(stripped)
        except ValueError:
            return default
    return default


def _ensure_timestamp(value):
    if value:
        return value
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")


def _normalize_row(source_row):
    source = dict(source_row)
    config_json = _load_json(source.get("config_json"), {})
    if not isinstance(config_json, dict):
        config_json = {}

    camera_uuid = _string_or_none(
        _first_not_none(source.get("camera_uuid"), config_json.get("camera_uuid"))
    )
    if not camera_uuid:
        raise ValueError("camera_uuid is required for migration")

    channel_id = _string_or_none(
        _first_not_none(source.get("channel_id"), config_json.get("channel_id"), camera_uuid)
    )
    rtsp_url = _string_or_none(
        _first_not_none(source.get("rtsp_url"), config_json.get("rtsp_url"))
    )
    if not rtsp_url:
        raise ValueError("rtsp_url is required for camera {}".format(camera_uuid))

    enabled = _coerce_bool(
        _first_not_none(
            source.get("is_enabled"),
            source.get("enabled"),
            config_json.get("is_enabled"),
            config_json.get("enabled"),
        ),
        True,
    )
    detection_enabled = _coerce_bool(
        _first_not_none(
            source.get("is_detection_enabled"),
            source.get("detection_enabled"),
            config_json.get("is_detection_enabled"),
            config_json.get("detection_enabled"),
        ),
        True,
    )
    notification_enabled = _coerce_bool(
        _first_not_none(
            source.get("is_notification_enabled"),
            source.get("notification_enabled"),
            config_json.get("is_notification_enabled"),
            config_json.get("notification_enabled"),
        ),
        True,
    )

    roi_value = _first_not_none(source.get("roi"), config_json.get("roi"))
    roi = _load_json(roi_value, roi_value)

    normalized_config = dict(config_json)
    normalized_config.setdefault("camera_uuid", camera_uuid)
    normalized_config.setdefault("channel_id", channel_id)
    normalized_config.setdefault("rtsp_url", rtsp_url)
    normalized_config["enabled"] = enabled
    normalized_config["is_enabled"] = enabled
    normalized_config["detection_enabled"] = detection_enabled
    normalized_config["is_detection_enabled"] = detection_enabled
    normalized_config["notification_enabled"] = notification_enabled
    normalized_config["is_notification_enabled"] = notification_enabled

    for key in (
        "site_uuid",
        "device_uuid",
        "camera_code",
        "name",
        "location",
        "webrtc_url",
    ):
        value = _string_or_none(_first_not_none(source.get(key), normalized_config.get(key)))
        if value is not None:
            normalized_config[key] = value

    if roi is not None:
        normalized_config["roi"] = roi

    return {
        "id": source.get("id"),
        "channel_id": channel_id,
        "camera_uuid": camera_uuid,
        "user_id": _coerce_int(_first_not_none(source.get("user_id"), normalized_config.get("user_id")), 1),
        "site_uuid": _string_or_none(_first_not_none(source.get("site_uuid"), normalized_config.get("site_uuid"))),
        "device_uuid": _string_or_none(_first_not_none(source.get("device_uuid"), normalized_config.get("device_uuid"))),
        "camera_code": _string_or_none(_first_not_none(source.get("camera_code"), normalized_config.get("camera_code"))),
        "name": _string_or_none(_first_not_none(source.get("name"), normalized_config.get("name"))),
        "location": _string_or_none(_first_not_none(source.get("location"), normalized_config.get("location"))),
        "rtsp_url": rtsp_url,
        "webrtc_url": _string_or_none(_first_not_none(source.get("webrtc_url"), normalized_config.get("webrtc_url"))),
        "is_enabled": int(enabled),
        "is_detection_enabled": int(detection_enabled),
        "is_notification_enabled": int(notification_enabled),
        "roi": json.dumps(roi, default=str) if roi is not None else None,
        "config_json": json.dumps(normalized_config, sort_keys=True, default=str),
        "created_at": _ensure_timestamp(source.get("created_at")),
        "updated_at": _ensure_timestamp(source.get("updated_at")),
    }


def _create_backup(db_path):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = "{}.bak.{}".format(os.path.abspath(db_path), timestamp)
    shutil.copy2(db_path, backup_path)
    return backup_path


def _create_target_table(conn, table_name):
    conn.execute(CREATE_TABLE_SQL.format(table_name=table_name))


def _create_target_indexes(conn, table_name):
    for sql in CREATE_INDEX_SQL:
        conn.execute(sql.format(table_name=table_name))


def migrate_camera_configs_schema(db_path=None, create_backup=True, logger=None):
    logger = logger or LOGGER
    db_path = os.path.abspath(db_path or DEFAULT_DB_PATH)

    if not os.path.exists(db_path):
        return {
            "status": "skipped",
            "reason": "database file does not exist",
            "db_path": db_path,
        }

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        if not _table_exists(conn, TABLE_NAME):
            return {
                "status": "skipped",
                "reason": "camera_configs table does not exist",
                "db_path": db_path,
            }

        current_columns = _get_columns(conn, TABLE_NAME)
        if tuple(current_columns) == TARGET_COLUMNS:
            return {
                "status": "noop",
                "reason": "schema already up to date",
                "db_path": db_path,
                "columns": list(current_columns),
            }

        if _table_exists(conn, TEMP_TABLE_NAME):
            conn.execute("DROP TABLE {}".format(TEMP_TABLE_NAME))
            conn.commit()
        if _table_exists(conn, LEGACY_TABLE_NAME):
            conn.execute("DROP TABLE {}".format(LEGACY_TABLE_NAME))
            conn.commit()

        backup_path = _create_backup(db_path) if create_backup else None
        rows = conn.execute("SELECT * FROM {} ORDER BY id".format(TABLE_NAME)).fetchall()

        conn.execute("BEGIN")
        _create_target_table(conn, TEMP_TABLE_NAME)

        insert_sql = """
        INSERT INTO {table_name} (
            id,
            channel_id,
            camera_uuid,
            user_id,
            site_uuid,
            device_uuid,
            camera_code,
            name,
            location,
            rtsp_url,
            webrtc_url,
            is_enabled,
            is_detection_enabled,
            is_notification_enabled,
            roi,
            config_json,
            created_at,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """.format(table_name=TEMP_TABLE_NAME)

        migrated_rows = 0
        for row in rows:
            normalized = _normalize_row(row)
            conn.execute(
                insert_sql,
                (
                    normalized["id"],
                    normalized["channel_id"],
                    normalized["camera_uuid"],
                    normalized["user_id"],
                    normalized["site_uuid"],
                    normalized["device_uuid"],
                    normalized["camera_code"],
                    normalized["name"],
                    normalized["location"],
                    normalized["rtsp_url"],
                    normalized["webrtc_url"],
                    normalized["is_enabled"],
                    normalized["is_detection_enabled"],
                    normalized["is_notification_enabled"],
                    normalized["roi"],
                    normalized["config_json"],
                    normalized["created_at"],
                    normalized["updated_at"],
                ),
            )
            migrated_rows += 1

        conn.execute("ALTER TABLE {} RENAME TO {}".format(TABLE_NAME, LEGACY_TABLE_NAME))
        conn.execute("ALTER TABLE {} RENAME TO {}".format(TEMP_TABLE_NAME, TABLE_NAME))
        conn.execute("DROP TABLE {}".format(LEGACY_TABLE_NAME))
        _create_target_indexes(conn, TABLE_NAME)
        conn.commit()

        result = {
            "status": "migrated",
            "db_path": db_path,
            "backup_path": backup_path,
            "rows": migrated_rows,
            "columns": list(TARGET_COLUMNS),
        }
        logger.info("Camera config schema migrated: %s", result)
        return result
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Migrate the Jetson camera_configs SQLite schema.")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Path to jetson_cameras.db")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating a backup copy before migration")
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    result = migrate_camera_configs_schema(
        db_path=args.db_path,
        create_backup=not args.no_backup,
        logger=LOGGER,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
