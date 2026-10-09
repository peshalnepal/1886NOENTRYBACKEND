"""SQLite roster: the tower's one NVR and its verified cameras.

Timestamps are naive UTC (SQLite has no timezone type) and serialize with a
trailing "Z", matching the Jetson's discovery report.

Stream URLs are stored WITHOUT credentials; they are rendered from settings at
the moment of use, so a password change needs no data migration.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import Boolean, CheckConstraint, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() + "Z" if value else None


class Base(DeclarativeBase):
    pass


class NvrDevice(Base):
    """The tower's single physical NVR (NVR_HOST set). At most one row.

    `singleton` is pinned to 1 by a CHECK and is UNIQUE, so the database itself
    refuses a second recorder rather than trusting every caller to.
    """

    __tablename__ = "nvr_devices"
    __table_args__ = (CheckConstraint("singleton = 1", name="ck_nvr_devices_singleton"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    singleton: Mapped[int] = mapped_column(Integer, default=1, unique=True, nullable=False)

    host: Mapped[str] = mapped_column(String(255), nullable=False)
    rtsp_port: Mapped[int] = mapped_column(Integer, nullable=False)
    http_port: Mapped[int] = mapped_column(Integer, nullable=False)
    origin: Mapped[str] = mapped_column(String(32), nullable=False)
    channels_spec: Mapped[Optional[str]] = mapped_column(String(512))

    is_present: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    consecutive_misses: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    alerted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    first_seen_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    last_seen_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    missing_since: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "host": self.host,
            "rtsp_port": self.rtsp_port,
            "http_port": self.http_port,
            "origin": self.origin,
            "channels_spec": self.channels_spec,
            "is_present": bool(self.is_present),
            "consecutive_misses": int(self.consecutive_misses or 0),
            "alerted": bool(self.alerted),
            "first_seen_at": iso(self.first_seen_at),
            "last_seen_at": iso(self.last_seen_at),
            "missing_since": iso(self.missing_since),
        }


class Camera(Base):
    """One physical camera, however many routes reach it.

    `identity` is the public, sticky key reported to the cloud: the first
    route identity ever seen, never rewritten, so cloud rows keyed on
    `discovery_identity` stay matched when a second route appears.
    `dedupe_key`, `identity`, serial and MAC match future discovery candidates.
    Route fields are the last frame-verified snapshot used by the relay. They
    change only after playback succeeds; alternate routes live in scan DTOs.
    `nvr_id` is NULL for a direct camera.
    """

    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Persisted before registering a channel on Jetson.
    camera_uuid: Mapped[Optional[str]] = mapped_column(String(36), unique=True, index=True)
    identity: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    dedupe_key: Mapped[str] = mapped_column(String(255), index=True, nullable=False)

    serial_number: Mapped[Optional[str]] = mapped_column(String(128), index=True)
    mac_address: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    model: Mapped[Optional[str]] = mapped_column(String(128))
    firmware: Mapped[Optional[str]] = mapped_column(String(128))
    device_name: Mapped[Optional[str]] = mapped_column(String(255))

    nvr_id: Mapped[Optional[int]] = mapped_column(ForeignKey("nvr_devices.id", ondelete="SET NULL"), index=True)
    channel_no: Mapped[Optional[int]] = mapped_column(Integer)
    source_kind: Mapped[str] = mapped_column(String(16), nullable=False)  # "direct" | "nvr"
    ip_address: Mapped[str] = mapped_column(String(255), nullable=False)
    rtsp_port: Mapped[int] = mapped_column(Integer, nullable=False)
    source_url: Mapped[str] = mapped_column(String(1024), nullable=False)

    # Jetson adoption state: pending | pushed | capacity_pending | error
    jetson_state: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    jetson_error: Mapped[Optional[str]] = mapped_column(String(1024))
    jetson_connected: Mapped[Optional[bool]] = mapped_column(Boolean)
    detection_wanted: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    is_present: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    first_frame_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    consecutive_misses: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    alerted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)
    last_seen_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    missing_since: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        """Roster row. Keys are a superset of the Jetson's `DiscoveredCamera.to_dict()`."""
        return {
            "identity": self.identity,
            "camera_uuid": self.camera_uuid,
            "ip_address": self.ip_address,
            "mac_address": self.mac_address,
            "serial_number": self.serial_number,
            "model": self.model,
            "firmware": self.firmware,
            "device_name": self.device_name,
            "source_url": self.source_url,
            "rtsp_port": self.rtsp_port,
            "source_kind": self.source_kind,
            "nvr_id": self.nvr_id,
            "channel_no": self.channel_no,
            "is_present": bool(self.is_present) and self.first_frame_at is not None,
            "first_frame_at": iso(self.first_frame_at),
            "verification_state": ("unverified" if self.first_frame_at is None else
                                   "online" if self.is_present else "offline"),
            "consecutive_misses": int(self.consecutive_misses or 0),
            "alerted": bool(self.alerted),
            "first_seen_at": iso(self.first_seen_at),
            "last_seen_at": iso(self.last_seen_at),
            "missing_since": iso(self.missing_since),
            "jetson_state": self.jetson_state,
            "jetson_error": self.jetson_error,
            "detection_wanted": bool(self.detection_wanted),
        }
