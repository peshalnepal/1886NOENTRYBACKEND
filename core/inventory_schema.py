"""Additive upgrade for frame verification in the cloud camera inventory."""

from sqlalchemy import DateTime, inspect, text


def ensure_inventory_frame_column(connection):
    inspector = inspect(connection)
    if not inspector.has_table("camera_inventory"):
        return
    columns = {column["name"] for column in inspector.get_columns("camera_inventory")}
    if "first_frame_at" not in columns:
        sql_type = DateTime(timezone=True).compile(dialect=connection.dialect)
        connection.execute(text("ALTER TABLE camera_inventory ADD first_frame_at {} NULL".format(sql_type)))
