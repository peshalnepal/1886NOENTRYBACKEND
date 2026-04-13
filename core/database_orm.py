# database.py
import re
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Time,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.ext.mutable import Mutable, MutableDict, MutableList
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.types import CHAR
from sqlalchemy.dialects.mysql import BINARY
from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER
try:
    from sqlalchemy.dialects.postgresql import UUID as PG_UUID
except Exception:
    PG_UUID = None

Base = declarative_base()


# -------------------------
# GUID (cross-db UUID)
# -------------------------
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
            return value.bytes
        if dialect.name == "postgresql":
            return value
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "mysql":
            return uuid.UUID(bytes=value)
        return uuid.UUID(str(value))


GUID = GUIDType()


def utc_now():
    return datetime.now(timezone.utc)


# -------------------------
# Deep mutable JSON helpers
# -------------------------
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
            self[key] = value

    @classmethod
    def coerce(cls, key, value):
        if not isinstance(value, cls):
            if isinstance(value, dict):
                instance = cls(value)
                instance.update(instance)
                return instance
            return Mutable.coerce(key, value)
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
    cache_ok = True


class JSONList(TypeDecorator):
    impl = JSON
    cache_ok = True


DeepMutableDict.associate_with(JSONDict)
MutableList.associate_with(JSONList)


def sanitize_to_snake_case(name: str) -> str:
    if not isinstance(name, str):
        return name
    s1 = re.sub(r"[\s-]", "_", name)
    s2 = re.sub(r"[^a-zA-Z0-9_]", "", s1)
    return s2.lower()


# =========================
# USER
# =========================

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    user_name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    contact_phone = Column(String(20), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    email_verified = Column(Boolean, default=False, nullable=False)
    verified_at = Column(DateTime(timezone=True), nullable=True)
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    sites = relationship("Site", back_populates="user", cascade="all, delete-orphan", passive_deletes=True)
    devices = relationship("Device", back_populates="user", cascade="all, delete-orphan", passive_deletes=True)
    cameras = relationship("Camera", back_populates="user", cascade="all, delete-orphan", passive_deletes=True)

    # site-scoped email recipients
    notification_emails = relationship(
        "NotificationEmail",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    # stored notification events
    notifications = relationship(
        "Notification",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


# =========================
# SITE
# =========================
class Site(Base):
    __tablename__ = "sites"

    site_uuid = Column(GUID, primary_key=True, default=uuid.uuid4, unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    site_code = Column(String(64), nullable=True, index=True)
    name = Column(String(255), nullable=False)
    address = Column(String(255), nullable=True)
    timezone = Column(String(50), nullable=True, default="UTC")

    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    user = relationship("User", back_populates="sites")

    cameras = relationship(
        "Camera",
        back_populates="site",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    devices = relationship(
        "Device",
        secondary="site_devices",
        back_populates="sites",
        passive_deletes=True,
    )

    notification_emails = relationship(
        "NotificationEmail",
        back_populates="site",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    notifications = relationship(
        "Notification",
        back_populates="site",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    settings = relationship(
        "SiteSettings",
        back_populates="site",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )


    __table_args__ = (
        UniqueConstraint("user_id", "site_code", name="uq_site_user_site_code"),
    )
    
    
# =========================
# DEVICE
# =========================
class Device(Base):
    """
    A Jetson/edge device. Belongs to a user (not owned by a site directly).

    device_url = base URL for inference API (e.g. http://10.0.0.5:8080)

    - Can be linked to multiple sites (M:N)
    - Can be linked to multiple cameras (M:N)
    """
    __tablename__ = "devices"

    device_uuid = Column(GUID, primary_key=True, default=uuid.uuid4, unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    device_code = Column(String(64), nullable=True, index=True)
    name = Column(String(255), nullable=True)

    device_url = Column(String(2048), nullable=False)
    is_enabled = Column(Boolean, default=True)

    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    user = relationship("User", back_populates="devices")

    # Device <-> Sites (M:N)
    sites = relationship(
        "Site",
        secondary="site_devices",
        back_populates="devices",
        passive_deletes=True,
    )

    # Device <-> Cameras (M:N)
    cameras = relationship(
        "Camera",
        secondary="camera_devices",
        back_populates="devices",
        passive_deletes=True,
    )

    __table_args__ = (
        UniqueConstraint("user_id", "device_code", name="uq_device_user_device_code"),
    )


# =========================
# SITE <-> DEVICE association
# =========================
class SiteDevice(Base):
    """
    Link table: sites <-> devices (M:N)

    Deleting a site removes these rows (CASCADE) but does NOT delete devices.
    Deleting a device removes these rows (CASCADE).
    """
    __tablename__ = "site_devices"

    id = Column(Integer, primary_key=True, index=True)

    site_uuid = Column(GUID, ForeignKey("sites.site_uuid", ondelete="CASCADE"), nullable=False, index=True)
    device_uuid = Column(GUID, ForeignKey("devices.device_uuid", ondelete="CASCADE"), nullable=False, index=True)

    created_at = Column(DateTime(timezone=True), default=utc_now)

    __table_args__ = (
        UniqueConstraint("site_uuid", "device_uuid", name="uq_site_device_pair"),
    )


# =========================
# SITE SETTINGS
# =========================
class SiteSettings(Base):
    __tablename__ = "site_settings"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    site_uuid = Column(
        GUID,
        ForeignKey("sites.site_uuid", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    config = Column(JSONDict, nullable=False, default=dict)
    day_of_week = Column(Integer, nullable=False, index=True)  # 0=Mon ... 6=Sun
    start_time = Column(Time, nullable=False)
    end_time = Column(Time, nullable=False)
    is_enabled = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    site = relationship("Site", back_populates="settings")
    __table_args__ = (
        CheckConstraint("day_of_week >= 0 AND day_of_week <= 6", name="ck_site_schedule_day"),
        CheckConstraint("start_time < end_time", name="ck_site_schedule_time"),
        UniqueConstraint(
            "site_uuid", "day_of_week", "start_time", "end_time",
            name="uq_site_schedule_window",
        ),
    )

# =========================
# CAMERA
# =========================
class Camera(Base):
    __tablename__ = "camera"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    site_uuid = Column(GUID, ForeignKey("sites.site_uuid", ondelete="CASCADE"), nullable=False, index=True)

    camera_uuid = Column(GUID, default=uuid.uuid4, unique=True, nullable=False, index=True)
    camera_code = Column(String(64), nullable=False, index=True)

    name = Column(String(255), nullable=True)
    location = Column(String(255), nullable=True)

    rtsp_url = Column(String(2048), nullable=False)
    webrtc_url = Column(String(2048), nullable=True)

    is_enabled = Column(Boolean, default=True)
    is_detection_enabled = Column(Boolean, default=True)
    is_notification_enabled = Column(Boolean, default=True)

    roi = Column(JSONDict, nullable=True)

    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)
    use_site_schedule = Column(Boolean, nullable=False, default=True)

    user = relationship("User", back_populates="cameras")
    site = relationship("Site", back_populates="cameras")

    devices = relationship(
        "Device",
        secondary="camera_devices",
        back_populates="cameras",
        passive_deletes=True,
    )

    channel_configuration = relationship(
        "ChannelConfiguration",
        back_populates="camera",
        uselist=False,
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    video_records = relationship(
        "VideoRecord",
        back_populates="camera",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    pipelines = relationship(
        "Pipeline",
        secondary="pipeline_cameras",
        back_populates="cameras",
        passive_deletes=True,
    )

    __table_args__ = (
        UniqueConstraint("user_id", "camera_code", name="uq_camera_user_camera_code"),
    )


# =========================
# CAMERA <-> DEVICE association
# =========================
class CameraDevice(Base):
    """
    Link table: cameras <-> devices (M:N)

    Use this to decide which Jetson device(s) can run inference for a camera.

    (DB cannot easily enforce "only one primary" cross-db; do that in service layer.)
    """
    __tablename__ = "camera_devices"

    id = Column(Integer, primary_key=True, index=True)
    camera_uuid = Column(GUID, ForeignKey("camera.camera_uuid", ondelete="CASCADE"), nullable=False, index=True)
    device_uuid = Column(GUID, ForeignKey("devices.device_uuid", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    __table_args__ = (
        UniqueConstraint("camera_uuid", name="uq_camera_one_device"),
    )

# =========================
# PIPELINE
# =========================
class Pipeline(Base):
    __tablename__ = "pipelines"

    id = Column(GUID, primary_key=True, default=uuid.uuid4, unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)

    name = Column(String(255), nullable=False, default="default")
    is_active = Column(Boolean, default=True, nullable=False)

    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)
    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_pipeline_user_name"),
    )
    cameras = relationship(
        "Camera",
        secondary="pipeline_cameras",
        back_populates="pipelines",
        passive_deletes=True,
    )


class PipelineCamera(Base):
    __tablename__ = "pipeline_cameras"

    id = Column(Integer, primary_key=True, index=True)

    pipeline_id = Column(GUID, ForeignKey("pipelines.id", ondelete="CASCADE"), nullable=False, index=True)
    camera_uuid = Column(GUID, ForeignKey("camera.camera_uuid", ondelete="CASCADE"), nullable=False, index=True)

    created_at = Column(DateTime(timezone=True), default=utc_now)

    __table_args__ = (
        UniqueConstraint("pipeline_id", "camera_uuid", name="uq_pipeline_camera_pair"),
        UniqueConstraint("camera_uuid", name="uq_pipeline_cameras_camera_uuid"),
    )


# =========================
# CHANNEL CONFIG
# =========================
class ChannelConfiguration(Base):
    __tablename__ = "channel_configurations"

    id = Column(Integer, primary_key=True, index=True)

    camera_uuid = Column(
        GUID,
        ForeignKey("camera.camera_uuid", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    configuration = Column(JSONDict, nullable=True)
    timezone = Column(String(50), nullable=True, default="UTC")
    day_of_week = Column(Integer, nullable=False, index=True)  # 0=Mon ... 6=Sun
    start_time = Column(Time, nullable=False)
    end_time = Column(Time, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    camera = relationship("Camera", back_populates="channel_configuration")
    __table_args__ = (
        CheckConstraint("day_of_week >= 0 AND day_of_week <= 6", name="ck_camera_schedule_day"),
        CheckConstraint("start_time < end_time", name="ck_camera_schedule_time"),
        UniqueConstraint(
            "camera_uuid", "day_of_week", "start_time", "end_time",
            name="uq_camera_schedule_window",
        ),
    )
    

# =========================
# VIDEO RECORD
# =========================
class VideoRecord(Base):
    __tablename__ = "video_record"
    __table_args__ = (
        Index("ix_vr_camera_created", "camera_uuid", "created_at"),
    )

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
    overlay_payload = Column(JSONDict, nullable=True)

    error = Column(Text, nullable=True)

    notification_sent = Column(Boolean, default=False)
    usage_processed = Column(Boolean, default=False)

    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    camera = relationship("Camera", back_populates="video_records")


class Notification(Base):
    """
    Stores notification events received for a user's specific site
    (optionally linked to camera/device).
    """
    __tablename__ = "notification"
    __table_args__ = (
        Index("ix_notif_user_visible_detected", "user_id", "visible", "detected_at"),
        Index("ix_notif_user_site_visible_detected", "user_id", "site_uuid", "visible", "detected_at"),
        Index("ix_notif_user_camera_visible", "user_id", "camera_uuid", "visible", "detected_at"),
        Index("ix_notif_user_visible_unread", "user_id", "visible", "read_at"),
        Index("ix_notif_user_camera_detected", "user_id", "camera_uuid", "detected_at"),
    )

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    site_uuid = Column(GUID, ForeignKey("sites.site_uuid", ondelete="CASCADE"), nullable=False, index=True)

    camera_uuid = Column(GUID, ForeignKey("camera.camera_uuid", ondelete="SET NULL"), nullable=True, index=True)
    device_uuid = Column(GUID, ForeignKey("devices.device_uuid", ondelete="SET NULL"), nullable=True, index=True)

    event_type = Column(String(64), nullable=False, default="detection")
    title = Column(String(255), nullable=True)
    message = Column(Text, nullable=True)

    payload = Column(JSONDict, nullable=True)
    detected_at = Column(DateTime(timezone=True), default=utc_now, index=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    read_at = Column(DateTime(timezone=True), nullable=True)
    sent_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(32), nullable=False, default="created")
    visible=Column(Boolean,default=True)

    user = relationship("User", back_populates="notifications")
    site = relationship("Site", back_populates="notifications")
    camera = relationship("Camera")
    device = relationship("Device")
    
class NotificationEmail(Base):
    """
    Emails that should receive notifications for a specific site.
    """
    __tablename__ = "notification_emails"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    site_uuid = Column(GUID, ForeignKey("sites.site_uuid", ondelete="CASCADE"), nullable=False, index=True)

    email = Column(String(255), nullable=False)
    is_enabled = Column(Boolean, default=True)

    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    user = relationship("User", back_populates="notification_emails")
    site = relationship("Site", back_populates="notification_emails")

    __table_args__ = (
        UniqueConstraint("user_id", "site_uuid", "email", name="uq_notif_email_user_site_email"),
    )

# =========================
# EMAIL VERIFICATION
# =========================

class EmailVerification(Base):
    __tablename__ = "email_verifications"

    id = Column(Integer, primary_key=True, index=True)

    email = Column(String(255), index=True, nullable=False)

    code_hash = Column(String(128), nullable=False)

    sent_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)

    attempts = Column(Integer, default=0, nullable=False)
    used = Column(Boolean, default=False, nullable=False)

    consumed_at = Column(DateTime(timezone=True), nullable=True)
    ip = Column(String(64), nullable=True)
    user_agent = Column(String(512), nullable=True)
    additional_data = Column(String(2048), nullable=True)
    
# =========================
# SYSTEM SETTINGS
# =========================
class SystemSettings(Base):
    __tablename__ = "system_settings"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(255), unique=True, nullable=False)
    value = Column(Text, nullable=False)
    description = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


# =========================
# SIGNUP TEMP DATA
# =========================
class SignupTempData(Base):
    __tablename__ = "signup_temp_data"

    id = Column(Integer, primary_key=True, index=True)
    token = Column(String(255), unique=True, nullable=False, index=True)
    data = Column(JSONDict, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    used = Column(Boolean, default=False)
