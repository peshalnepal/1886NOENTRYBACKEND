# database.py
import json
import re
import uuid
from datetime import datetime, timezone

from sqlalchemy import (JSON, Boolean, Column, DateTime, Enum, Float,
                        ForeignKey, Integer, String, Text, TypeDecorator,
                        UniqueConstraint)
from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER
from sqlalchemy.ext.mutable import Mutable, MutableDict, MutableList
from sqlalchemy.orm import declarative_base, relationship, validates
from sqlalchemy.types import UUID
import uuid
from sqlalchemy.types import TypeDecorator, CHAR
from sqlalchemy.dialects.mysql import BINARY
from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER

try:
    from sqlalchemy.dialects.postgresql import UUID as PG_UUID
except Exception:
    PG_UUID = None

Base = declarative_base()

from enum import Enum as PythonEnum

class GUIDType(TypeDecorator):
    """
    Platform-independent GUID type.

    - MySQL: BINARY(16)
    - MSSQL: UNIQUEIDENTIFIER
    - PostgreSQL: UUID
    - Others: CHAR(36)
    """
    impl = CHAR(36)
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "mysql":
            return dialect.type_descriptor(BINARY(16))
        if dialect.name == "mssql":
            return dialect.type_descriptor(UNIQUEIDENTIFIER())
        if dialect.name == "postgresql" and PG_UUID is not None:
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if not isinstance(value, uuid.UUID):
            value = uuid.UUID(str(value))

        if dialect.name == "mysql":
            return value.bytes  # 16 bytes
        if dialect.name == "postgresql":
            return value
        return str(value)  # CHAR(36)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "mysql":
            return uuid.UUID(bytes=value)
        return uuid.UUID(str(value))


# Keep your old name so you don't change every Column(...) line
GUID = GUIDType()


def utc_now():
    """Helper function to get current UTC time with timezone info"""
    return datetime.now(timezone.utc)


class DeepMutableDict(MutableDict):

    def __init__(self, *args, **kwargs):

        self._root = kwargs.pop("_root", self)
        super().__init__(*args, **kwargs)

    def changed(self):
        root = getattr(self, "_root", self)
        if root is self:
            super().changed()
        else:
            root.changed()

    def __setitem__(self, key, value):
        root = getattr(self, "_root", self)
        super().__setitem__(key, _recursively_make_mutable(value, root))

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            # Use our overridden __setitem__ to process each value
            self[key] = value

    @classmethod
    def coerce(cls, key, value):
        if not isinstance(value, cls):
            if isinstance(value, dict):
                # The coerced object becomes its own root.
                instance = cls(value)
                # Recursively convert its children, passing the new instance as the root.
                instance.update(instance)
                return instance
            # Let the base class handle non-dict types (e.g., raise ValueError).
            return Mutable.coerce(key, value)
        else:
            return value


def _recursively_make_mutable(value, root):
    if isinstance(value, dict) and not isinstance(value, DeepMutableDict):
        return DeepMutableDict(
            {k: _recursively_make_mutable(v, root) for k, v in value.items()},
            _root=root,
        )
    return value


class JSONDict(TypeDecorator):
    impl = JSON


class JSONList(TypeDecorator):
    impl = JSON


DeepMutableDict.associate_with(JSONDict)
MutableList.associate_with(JSONList)


def sanitize_to_snake_case(name: str) -> str:
    if not isinstance(name, str):
        return name
    s1 = re.sub(r"[\s-]", "_", name)
    s2 = re.sub(r"[^a-zA-Z0-9_]", "", s1)
    return s2.lower()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    user_name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, index=True)
    hashed_password = Column(String(255), nullable=False)

    business_url = Column(String(255), nullable=False)
    contact_name = Column(String(255), nullable=True)
    contact_phone = Column(String(20), nullable=True)
    business_address = Column(String(255), nullable=True)

    created_at = Column(DateTime(timezone=True), default=utc_now)
    email_verified = Column(Boolean, default=False)

    # Relationships
    cameras = relationship("Camera", back_populates="user", cascade="all, delete-orphan")
    notification_emails = relationship(
        "NotificationEmail", back_populates="user", cascade="all, delete-orphan"
    )


class Camera(Base):
    """
    Camera metadata.
    Store ownership + RTSP URL + human readable fields here.

    channel configuration lives in ChannelConfiguration (1:1 per camera).
    """
    __tablename__ = "camera"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    camera_uuid = Column(GUID, default=uuid.uuid4, unique=True, nullable=False, index=True)
    camera_code = Column(String(64), nullable=False, index=True)

    name = Column(String(255), nullable=True)
    location = Column(String(255), nullable=True)

    rtsp_url = Column(String(2048), nullable=False)
    is_enabled = Column(Boolean, default=True)
    is_detection_enabled=Column(Boolean, default=True)
    is_notification_enabled=Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    user = relationship("User", back_populates="cameras")

    channel_configuration = relationship(
        "ChannelConfiguration",
        back_populates="camera",
        uselist=False,
        cascade="all, delete-orphan",
    )

    video_records = relationship(
        "VideoRecord",
        back_populates="camera",
        cascade="all, delete-orphan",
    )
    pipelines = relationship(
        "Pipeline",
        secondary="pipeline_cameras",
        back_populates="cameras",
    )
    __table_args__ = (
        UniqueConstraint("user_id", "camera_code", name="uq_camera_user_camera_code"),
    )

class Pipeline(Base):
    __tablename__ = "pipelines"

    id = Column(GUID, primary_key=True, default=uuid.uuid4, unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False, default="default")
    is_active = Column(Boolean, default=True, nullable=False)

    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    # many-to-many via PipelineCamera table
    cameras = relationship(
        "Camera",
        secondary="pipeline_cameras",
        back_populates="pipelines",
    )


class PipelineCamera(Base):
    """
    Join table as an ORM model.
    Used as the secondary table for Pipeline <-> Camera.
    """
    __tablename__ = "pipeline_cameras"

    id = Column(Integer, primary_key=True, index=True)

    pipeline_id = Column(GUID, ForeignKey("pipelines.id", ondelete="CASCADE"), nullable=False, index=True)
    camera_uuid = Column(GUID, ForeignKey("camera.camera_uuid", ondelete="CASCADE"), nullable=False, index=True)

    created_at = Column(DateTime(timezone=True), default=utc_now)

    __table_args__ = (
        UniqueConstraint("pipeline_id", "camera_uuid", name="uq_pipeline_camera_pair"),
        # keep this if you want "one camera can be in only one pipeline"
        UniqueConstraint("camera_uuid", name="uq_pipeline_cameras_camera_uuid"),
    )




class ChannelConfiguration(Base):
    """
    Per-camera pipeline configuration (RTSP ingest config, gstreamer params, sampling, ROI, alerts flags, etc).
    """
    __tablename__ = "channel_configurations"

    id = Column(Integer, primary_key=True, index=True)

    camera_uuid = Column(GUID, ForeignKey("camera.camera_uuid", ondelete="CASCADE"), nullable=False, unique=True, index=True)

    # Arbitrary config JSON (latency, protocol, drop-on-latency, fps cap, etc.)
    configuration = Column(JSONDict, nullable=True)

    timezone = Column(String(50), nullable=True, default="UTC")

    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    camera = relationship("Camera", back_populates="channel_configuration")



class VideoRecord(Base):
    """
    One recorded clip.

    recording_url: the user-consumable URL (signed URL or public URL)
    storage_key: blob key / s3 key
    local_path: where clip was written on edge before upload
    """
    __tablename__ = "video_record"

    id = Column(Integer, primary_key=True, index=True)

    camera_uuid = Column(GUID, ForeignKey("camera.camera_uuid", ondelete="CASCADE"), nullable=False, index=True)

    external_id = Column(String(255), nullable=False, index=True)

    start_time = Column(DateTime(timezone=True), default=utc_now)
    end_time = Column(DateTime(timezone=True), nullable=True)
    duration = Column(Integer, nullable=True)
    status = Column(String(50), nullable=False)
    local_path = Column(String(1024), nullable=True)
    storage_key = Column(String(1024), nullable=True)
    recording_url = Column(String(2048), nullable=True)

    error = Column(Text, nullable=True)

    notification_sent = Column(Boolean, default=False)
    usage_processed = Column(Boolean, default=False)

    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    camera = relationship("Camera", back_populates="video_records")


class NotificationEmail(Base):
    __tablename__ = "notification_emails"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    email = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    user = relationship("User", back_populates="notification_emails")


class EmailVerification(Base):
    __tablename__ = "email_verifications"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), index=True)
    code = Column(String(6), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    used = Column(Boolean, default=False)
    additional_data = Column(String(2048), nullable=True)


class SystemSettings(Base):
    __tablename__ = "system_settings"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(255), unique=True, nullable=False)
    value = Column(Text, nullable=False)
    description = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class SignupTempData(Base):
    __tablename__ = "signup_temp_data"

    id = Column(Integer, primary_key=True, index=True)
    token = Column(String(255), unique=True, nullable=False, index=True)
    data = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    used = Column(Boolean, default=False)
