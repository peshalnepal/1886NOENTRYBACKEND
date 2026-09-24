"""Small, additive SQLite schema upgrades used during edge startup."""

from sqlalchemy import inspect, text


def ensure_frame_verification_column(connection):
    columns = {c["name"] for c in inspect(connection).get_columns("discovered_cameras")}
    if "first_frame_at" not in columns:
        connection.execute(text("ALTER TABLE discovered_cameras ADD COLUMN first_frame_at DATETIME NULL"))
