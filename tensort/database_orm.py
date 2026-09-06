# database_orm.py - Simplified ORM models for Jetson Nano
from datetime import datetime
from sqlalchemy import Boolean, Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class CameraConfig(Base):
    """
    Camera configuration table for Jetson Nano.
    
    This stores all camera configurations received from the Azure backend,
    allowing the Jetson to persist camera settings across restarts.
    
    Fields:
        id: Auto-incrementing primary key
        channel_id: Unique channel identifier (UUID string)
        camera_uuid: Camera UUID from Azure backend
        user_id: User ID from Azure backend (for multi-tenant support)
        source_url: RTSP stream URL for the camera
        config_json: JSON blob containing all channel configuration parameters
        created_at: Timestamp when camera was first added
        updated_at: Timestamp when camera config was last updated
    """
    __tablename__ = "camera_configs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    channel_id = Column(String(64), unique=True, nullable=False, index=True)
    camera_uuid = Column(String(64), unique=True, nullable=False, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    source_url = Column(Text, nullable=False)
    config_json = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<CameraConfig(id={self.id}, camera_uuid={self.camera_uuid}, channel_id={self.channel_id})>"

    @property
    def is_discovered(self):
        """True when this camera was auto-added by the discovery scanner."""
        cfg = self.config_json or {}
        return bool(cfg.get("discovered"))
    
    def to_dict(self):
        """Convert to dictionary for easy serialization."""
        return {
            "id": self.id,
            "channel_id": self.channel_id,
            "camera_uuid": self.camera_uuid,
            "user_id": self.user_id,
            "source_url": self.source_url,
            "config_json": self.config_json,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class DiscoveredCamera(Base):
    """
    The Jetson's roster of physically-present Hikvision cameras.

    This is the discovery scanner's memory. Without it the scanner could only
    ever answer "what is on the network right now" — it could never say "the
    camera in the loading bay stopped answering", which is the alert the
    frontend actually needs.

    A row is created the first time a camera is confirmed on the network and is
    never deleted by the scanner: it flips to `is_present=False` and keeps its
    `last_seen_at`. Only an explicit forget (via the route) removes it, because
    a decommissioned camera should stop alerting but a merely unplugged one
    should not be forgotten.

    Fields:
        identity: Stable key for the physical camera — "serial:<n>" preferred,
            falling back to "mac:<n>" then "ip:<n>". Survives DHCP changes.
        camera_uuid: The camera_configs row this was auto-provisioned into,
            or NULL if it was discovered but not adopted into the pipeline.
        is_present: Whether the last completed sweep saw this camera.
        missing_since: When it first went absent; NULL while present. Drives
            the "camera went missing" alert and its duration.
        consecutive_misses: Sweeps missed in a row. The alert only fires once
            this crosses the configured threshold, so a single dropped packet
            or a camera rebooting does not page anyone.
        alerted: Whether the missing alert has already been emitted, so the
            scanner reports the transition once instead of every minute.
    """
    __tablename__ = "discovered_cameras"

    id = Column(Integer, primary_key=True, autoincrement=True)
    identity = Column(String(128), unique=True, nullable=False, index=True)

    ip_address = Column(String(45), nullable=True, index=True)
    mac_address = Column(String(32), nullable=True)
    serial_number = Column(String(128), nullable=True, index=True)
    model = Column(String(128), nullable=True)
    firmware = Column(String(128), nullable=True)
    device_name = Column(String(255), nullable=True)

    source_url = Column(Text, nullable=True)
    camera_uuid = Column(String(64), nullable=True, index=True)

    is_present = Column(Boolean, nullable=False, default=True)
    consecutive_misses = Column(Integer, nullable=False, default=0)
    alerted = Column(Boolean, nullable=False, default=False)

    first_seen_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_seen_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    missing_since = Column(DateTime, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return "<DiscoveredCamera(identity={}, ip={}, present={})>".format(
            self.identity, self.ip_address, self.is_present
        )

    def to_dict(self):
        return {
            "identity": self.identity,
            "ip_address": self.ip_address,
            "mac_address": self.mac_address,
            "serial_number": self.serial_number,
            "model": self.model,
            "firmware": self.firmware,
            "device_name": self.device_name,
            "source_url": self.source_url,
            "camera_uuid": self.camera_uuid,
            "is_present": bool(self.is_present),
            "consecutive_misses": int(self.consecutive_misses or 0),
            "alerted": bool(self.alerted),
            "first_seen_at": self.first_seen_at.isoformat() if self.first_seen_at else None,
            "last_seen_at": self.last_seen_at.isoformat() if self.last_seen_at else None,
            "missing_since": self.missing_since.isoformat() if self.missing_since else None,
        }
