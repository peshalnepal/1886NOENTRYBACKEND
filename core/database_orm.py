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
    LargeBinary,
    String,
    Table,
    Text,
    Time,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.dialects.mysql import LONGBLOB
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
    is_platform_admin = Column(
        Boolean, default=False, nullable=False, server_default="0"
    )
    # RBAC grants are queried explicitly via AuthzService (lazy="raise"); the
    # cascade is declared on AccessGrant.user instead of a collection here.

    sites = relationship("Site", back_populates="user", foreign_keys="Site.user_id", cascade="all, delete-orphan", passive_deletes=True)
    devices = relationship("Device", back_populates="user", foreign_keys="Device.user_id", cascade="all, delete-orphan", passive_deletes=True)
    cameras = relationship("Camera", back_populates="user", foreign_keys="Camera.user_id", cascade="all, delete-orphan", passive_deletes=True)

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
        foreign_keys="Notification.user_id",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


# =========================
# SITE
# =========================
class Site(Base):
    __tablename__ = "sites"

    site_uuid = Column(GUID, primary_key=True, default=uuid.uuid4, unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    org_id = Column(
        Integer,
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    site_code = Column(String(64), nullable=True, index=True)
    name = Column(String(255), nullable=False)
    address = Column(String(255), nullable=True)
    timezone = Column(String(50), nullable=True, default="UTC")

    is_deleted = Column(Boolean, default=False, nullable=False, server_default="0")
    # Temporary, schedule-aware arm/disarm override. NULL arm_override means the
    # site follows its schedule. A non-NULL value forces armed (True) or disarmed
    # (False) until arm_override_until, the next schedule boundary, after which the
    # schedule resumes control. arm_override_until NULL => no boundary (permanent).
    arm_override = Column(Boolean, nullable=True)
    arm_override_until = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    user = relationship("User", back_populates="sites", foreign_keys=[user_id])
    creator = relationship("User", foreign_keys=[created_by])

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

    organization = relationship("Organization", back_populates="sites")
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
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    org_id = Column(
        Integer,
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    device_code = Column(String(64), nullable=True, index=True)
    name = Column(String(255), nullable=True)

    device_url = Column(String(2048), nullable=False)
    is_enabled = Column(Boolean, default=True)

    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    user = relationship("User", back_populates="devices", foreign_keys=[user_id])
    creator = relationship("User", foreign_keys=[created_by])
    organization = relationship("Organization",secondary="org_devices", back_populates="devices",passive_deletes=True)
    # Device <-> Sites (M:N)
    sites = relationship(
        "Site",
        secondary="site_devices",
        back_populates="devices",
        passive_deletes=True,
    )

    # Device <-> Cameras (M:N)
    # Device 1 -> N Cameras (one device can run many cameras)
    cameras = relationship(
        "Camera",
        back_populates="device",
        passive_deletes=True,
    )

    __table_args__ = (
        UniqueConstraint("user_id", "device_code", name="uq_device_user_device_code"),
    )

# =========================
# ORGANIZATION <-> DEVICE association
# =========================

class OrganizationDevice(Base):
    """
    Link table: orgs <-> devices (M:N)

    Deleting an org removes these rows (CASCADE) but does NOT delete devices.
    Deleting a device removes these rows (CASCADE).
    """
    __tablename__ = "org_devices"

    id = Column(Integer, primary_key=True, index=True)

    org_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    device_uuid = Column(GUID, ForeignKey("devices.device_uuid", ondelete="CASCADE"), nullable=False, index=True)

    created_at = Column(DateTime(timezone=True), default=utc_now)

    __table_args__ = (
        UniqueConstraint("org_id", "device_uuid", name="uq_org_device_pair"),
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
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    org_id = Column(
        Integer,
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    site_uuid = Column(GUID, ForeignKey("sites.site_uuid", ondelete="CASCADE"), nullable=False, index=True)

    # Each camera runs on at most one device; a device may host many cameras.
    device_uuid = Column(GUID, ForeignKey("devices.device_uuid", ondelete="SET NULL"), nullable=True, index=True)

    camera_uuid = Column(GUID, default=uuid.uuid4, unique=True, nullable=False, index=True)
    camera_code = Column(String(64), nullable=False, index=True)

    name = Column(String(255), nullable=True)
    location = Column(String(255), nullable=True)

    source_url = Column(String(2048), nullable=False)
    webrtc_url = Column(String(2048), nullable=True)

    is_enabled = Column(Boolean, default=True)
    is_detection_enabled = Column(Boolean, default=True)
    is_notification_enabled = Column(Boolean, default=True)
    notification_trigger_mode = Column(String(32), nullable=False, default="inherit", server_default="inherit")  # "inherit" | "roi_enter" | "any_detection"
    camera_playback_enabled = Column(String(16), nullable=False, default="inherit", server_default="inherit")    # "inherit" | "always" | "never"

    roi = Column(JSONDict, nullable=True)

    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)
    use_site_schedule = Column(Boolean, nullable=False, default=True)

    user = relationship("User", back_populates="cameras", foreign_keys=[user_id])
    creator = relationship("User", foreign_keys=[created_by])
    organization = relationship("Organization")
    site = relationship("Site", back_populates="cameras")

    device = relationship("Device", back_populates="cameras")

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

    # Operator-approval gate (mirrors Notification): when the site's org has an
    # operator, a freshly captured clip is held invisible until the operator
    # approves the matching alert, so end users never see the playback first.
    visible = Column(Boolean, default=True)
    approval_status = Column(
        String(16), nullable=False, default="approved", server_default="approved"
    )

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
        Index("ix_notif_site_approval_visible_detected", "site_uuid", "approval_status", "visible", "detected_at"),
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
    approval_status = Column(
        String(16), nullable=False, default="approved", server_default="approved"
    )
    approved_by = Column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    approved_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="notifications", foreign_keys=[user_id])
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
class Organization(Base):
    """
    Top-level tenant container.

    Everything that belongs to an organization (sites, devices,
    cameras, users) is reachable from this row. A Platform Admin
    is the only actor permitted to create or delete organizations.

    `owner_user_id` points at the first Org Admin that was created
    together with the organization. It is informational only; an
    organization can have many Org Admins via org-scoped `access_grants`.
    """

    __tablename__ = "organizations"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    slug = Column(String(64), nullable=False, unique=True, index=True)

    # Optional pointer to the user that was bootstrapped as the
    # initial Org Admin. Set NULL if that user is later deleted.
    owner_user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    is_active = Column(Boolean, default=True, nullable=False, server_default="1")
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    sites = relationship("Site", back_populates="organization", passive_deletes=True)
    devices = relationship(
        "Device",
        secondary="org_devices",
        back_populates="organization",
        passive_deletes=True,
    )
    # Org-scoped RBAC grants live in `access_grants` and are queried
    # explicitly (lazy="raise"); cascade is declared on AccessGrant.organization.


# =========================================================================
# ORGANIZATION REPORT ARCHIVE
# =========================================================================
class OrganizationReport(Base):
    """A generated alert-report PDF, archived so members can browse/download.

    ``report_type`` is ``"general"`` (the daily scheduled / manually generated
    roll-up of approved-but-not-emailed alerts) or ``"urgent"`` (persisted when
    an operator approves an alert with the email opt-in — the alert was pushed
    to users immediately). The PDF bytes live in ``pdf_data`` (LONGBLOB) so the
    archive is self-contained; list queries must avoid selecting the blob.
    ``site_uuids`` records which sites' alerts appear in the report (drives the
    site filter), and ``generated_by_email`` drives the operator filter.
    """

    __tablename__ = "organization_reports"
    __table_args__ = (
        Index("ix_org_report_org_type_created", "org_id", "report_type", "created_at"),
    )

    id = Column(Integer, primary_key=True, index=True)
    report_uuid = Column(
        String(36),
        nullable=False,
        unique=True,
        index=True,
        default=lambda: str(uuid.uuid4()),
    )
    org_id = Column(
        Integer,
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    report_type = Column(String(16), nullable=False, default="general")  # general | urgent
    filename = Column(String(255), nullable=False)

    generated_by = Column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    generated_by_email = Column(String(255), nullable=True, index=True)

    site_uuids = Column(JSONList, nullable=True)

    period_start = Column(DateTime(timezone=True), nullable=True)
    period_end = Column(DateTime(timezone=True), nullable=True)
    alert_count = Column(Integer, nullable=False, default=0)

    pdf_data = Column(LargeBinary().with_variant(LONGBLOB, "mysql"), nullable=False)
    pdf_size = Column(Integer, nullable=False, default=0)

    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False, index=True)


# =========================================================================
# ORGANIZATION REPORT SCHEDULE
# =========================================================================
class OrganizationReportSchedule(Base):
    """Per-organization daily schedule for the approved-alerts PDF report.

    An org admin configures a local wall-clock time (``send_hour`` /
    ``send_minute`` interpreted in ``timezone``) at which the background
    ``ReportScheduler`` emails the report. ``last_sent_on`` records the last
    date (``YYYY-MM-DD`` in the schedule's own timezone) a report went out, so a
    given day fires exactly once even though the scheduler polls every minute.
    """

    __tablename__ = "organization_report_schedules"

    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(
        Integer,
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    is_enabled = Column(Boolean, default=False, nullable=False, server_default="0")
    send_hour = Column(Integer, nullable=False, default=8)      # 0..23 local
    send_minute = Column(Integer, nullable=False, default=0)    # 0..59 local
    timezone = Column(String(64), nullable=False, default="UTC")

    # Report scope each run: how many hours back to cover, and whether to count
    # only operator-approved alerts (vs. auto-approved orgs without an operator).
    window_hours = Column(Integer, nullable=False, default=24)
    operator_approved_only = Column(
        Boolean, default=True, nullable=False, server_default="1"
    )

    # Last date (YYYY-MM-DD in `timezone`) a report was sent; the once-per-day guard.
    last_sent_on = Column(String(10), nullable=True)

    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    __table_args__ = (
        CheckConstraint("send_hour >= 0 AND send_hour <= 23", name="ck_report_send_hour"),
        CheckConstraint("send_minute >= 0 AND send_minute <= 59", name="ck_report_send_minute"),
        CheckConstraint("window_hours >= 1", name="ck_report_window_hours"),
    )


# =========================================================================
# RBAC: ROLES / PERMISSIONS / ACCESS GRANTS
# =========================================================================
# These three tables replace the old `org_memberships` and
# `site_memberships` tables with a single ACL-style model:
#
#   "User X holds Role Y in Context Z (an org or a site)."
#
# A Role carries a set of Permissions (via `role_permissions`). The role
# catalog and the role->permission mapping are seeded from
# `core.security.roles.ROLE_PERMISSIONS` during DB initialization.
#
# All relationships use lazy="raise" so authorization code must resolve
# them with explicit JOINs (no implicit lazy loading under async).

# Many-to-many: roles <-> permissions
role_permissions = Table(
    "role_permissions",
    Base.metadata,
    Column(
        "role_id",
        Integer,
        ForeignKey("roles.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "permission_id",
        Integer,
        ForeignKey("permissions.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


class Role(Base):
    """A named role within a scope (``org`` or ``site``).

    The same name (e.g. ``admin``) can exist in both scopes with different
    powers, hence the composite uniqueness on ``(name, scope)``.
    """

    __tablename__ = "roles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(64), nullable=False)  # OrgRole/SiteRole value
    scope = Column(String(16), nullable=False)  # "org" | "site"
    description = Column(String(255), nullable=True)

    permissions = relationship(
        "Permission",
        secondary=role_permissions,
        lazy="raise",
        passive_deletes=True,
    )

    __table_args__ = (
        UniqueConstraint("name", "scope", name="uq_role_name_scope"),
    )


class Permission(Base):
    """An atomic, checkable privilege (e.g. ``site:arm_disarm``)."""

    __tablename__ = "permissions"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(64), unique=True, nullable=False, index=True)
    description = Column(String(255), nullable=True)


class AccessGrant(Base):
    """Unified membership/ACL row: a user holds a role on a context.

    Exactly one of ``org_id`` / ``site_uuid`` is set (an org grant or a
    site grant); both NULL is reserved for a future platform-wide grant.
    Today platform scope is still carried by ``users.is_platform_admin``.

    Replaces the former ``org_memberships`` and ``site_memberships`` tables.
    """

    __tablename__ = "access_grants"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role_id = Column(
        Integer,
        ForeignKey("roles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Context: an org grant OR a site grant (see CheckConstraint below).
    org_id = Column(
        Integer,
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    site_uuid = Column(
        GUID,
        ForeignKey("sites.site_uuid", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    user = relationship("User", lazy="raise")
    role = relationship("Role", lazy="raise")
    organization = relationship("Organization", lazy="raise")
    site = relationship("Site", lazy="raise")

    __table_args__ = (
        CheckConstraint(
            "(org_id IS NOT NULL AND site_uuid IS NULL) OR "
            "(org_id IS NULL AND site_uuid IS NOT NULL) OR "
            "(org_id IS NULL AND site_uuid IS NULL)",
            name="ck_access_grant_single_context",
        ),
        UniqueConstraint(
            "user_id", "role_id", "org_id", "site_uuid",
            name="uq_access_grant_user_role_context",
        ),
    )
