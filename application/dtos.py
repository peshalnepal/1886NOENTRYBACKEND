# application/dtos.py
"""
Data Transfer Objects.

These are the typed payloads passed *into* repository write operations and
*between* internal functions/services. They are deliberately separate from the
HTTP route schemas in `core/schemas.py`:

  - `core/schemas.py`  -> request/response models at the HTTP boundary.
  - `application/dtos.py` (this file) -> internal contracts (repo inputs,
    service-to-service payloads).

Routes/services validate an inbound `core.schemas` model, then translate it
into the matching DTO here before calling a repository. Repositories never see
a raw HTTP schema — only DTOs. The two layers intentionally look alike so the
translation is trivial.

Convention:
  - `*CreateDTO`  -> insert payloads (all required fields present).
  - `*UpdateDTO`  -> patch payloads (every field Optional; use
                     `model_dump(exclude_unset=True)` to get only what changed).
  - `*UpsertDTO`  -> create-or-update payloads.
  - plain `*DTO`  -> inter-function value objects.
"""

from __future__ import annotations

import uuid
from datetime import datetime, time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class _DTO(BaseModel):
    """Base for all DTOs: allows ORM/arbitrary types and ignores extras."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")


# =====================================================================
# Camera / Channel
# =====================================================================
class CameraUpsertDTO(_DTO):
    """
    Input for `ChannelRepository.upsert_camera_from_channel_config`.

    Bundles the camera identity, the raw channel-config blob and the explicit
    column overrides that were previously passed as ~16 separate kwargs.
    """
    pipeline_id: uuid.UUID
    channel_config: Any = Field(..., description="dict or VideoChannelConfig-like object")

    user_id: Optional[int] = None
    org_id: Optional[int] = None
    created_by: Optional[int] = None
    cam_uuid: Optional[uuid.UUID] = None
    camera_code: Optional[str] = None
    site_uuid: Optional[uuid.UUID] = None

    webrtc_url: Optional[str] = None
    source_url: Optional[str] = None
    device_uuid: Optional[uuid.UUID] = None

    name: Optional[str] = None
    location: Optional[str] = None
    timezone: Optional[str] = None

    day_of_week: Optional[List[int]] = None
    start_time: Optional[time] = None
    end_time: Optional[time] = None
    is_enabled: bool = True


class CameraUpdateDTO(_DTO):
    """Patch payload for `ChannelRepository.update_camera` (every field optional)."""
    name: Optional[str] = None
    location: Optional[str] = None
    source_url: Optional[str] = None
    is_enabled: Optional[bool] = None
    is_detection_enabled: Optional[bool] = None
    is_notification_enabled: Optional[bool] = None
    notification_trigger_mode: Optional[str] = None
    camera_playback_enabled: Optional[str] = None
    use_site_schedule: Optional[bool] = None
    roi: Optional[Dict[str, Any]] = None
    device_uuid: Optional[uuid.UUID] = None


# =====================================================================
# Device
# =====================================================================
class DeviceCreateDTO(_DTO):
    """Input for `DeviceRepository.create_device`."""
    org_id: int
    user_id: Optional[int] = None  # legacy owner (runtime keying)
    created_by: Optional[int] = None  # user who created the row
    device_url: str
    name: Optional[str] = None
    device_code: Optional[str] = None
    is_enabled: bool = True


class DeviceUpdateDTO(_DTO):
    """Patch payload for `DeviceRepository.update_device`."""
    name: Optional[str] = None
    device_url: Optional[str] = None
    device_code: Optional[str] = None
    is_enabled: Optional[bool] = None


# =====================================================================
# Site / SiteSettings
# =====================================================================
class SiteCreateDTO(_DTO):
    """Input for `SiteRepository.create_site`."""
    org_id: int
    user_id: Optional[int] = None  # legacy owner (runtime keying)
    created_by: Optional[int] = None  # user who created the row
    name: str
    address: Optional[str] = None
    timezone: str = "UTC"
    site_code: Optional[str] = None


class SiteUpdateDTO(_DTO):
    """Patch payload for `SiteRepository.update_site`."""
    name: Optional[str] = None
    address: Optional[str] = None
    timezone: Optional[str] = None
    site_code: Optional[str] = None


class SiteSettingsUpsertDTO(_DTO):
    """Input for `SiteRepository.upsert_site_settings`."""
    user_id: int
    site_uuid: uuid.UUID
    config: Optional[Dict[str, Any]] = None
    day_of_week: Optional[List[int]] = None
    start_time: Optional[time] = None
    end_time: Optional[time] = None
    is_enabled: bool = True


# =====================================================================
# Notification / NotificationEmail
# =====================================================================
class NotificationCreateDTO(_DTO):
    """Input for `NotificationRepository.create_notification(s)`."""
    user_id: int
    site_uuid: uuid.UUID
    camera_uuid: Optional[uuid.UUID] = None
    device_uuid: Optional[uuid.UUID] = None
    event_type: str = "detection"
    title: Optional[str] = None
    message: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    detected_at: Optional[datetime] = None
    status: str = "created"
    sent_at: Optional[datetime] = None
    approval_status: str = "approved"


class NotificationEmailCreateDTO(_DTO):
    """Input for `NotificationRepository.create_notification_email`."""
    user_id: int
    site_uuid: uuid.UUID
    email: str
    is_enabled: bool = True


# =====================================================================
# User
# =====================================================================
class UserCreateDTO(_DTO):
    """Input for `UserRepository.create_user`."""
    user_name: str
    email: str
    password_hash: str
    contact_phone: Optional[str] = None
    email_verified: bool = False
    verified_at: Optional[datetime] = None


class UserProfileUpdateDTO(_DTO):
    """Patch payload for `UserRepository.update_profile`."""
    user_name: Optional[str] = None
    contact_phone: Optional[str] = None
    email: Optional[str] = None


# =====================================================================
# Organization / Membership
# =====================================================================
class OrganizationCreateDTO(_DTO):
    """Input for `OrganizationRepository.create_organization`.

    `owner_user_id` is the user that will be flagged as the initial
    Org Admin. The repository creates the matching `OrgMembership`
    row in the same transaction.
    """

    name: str
    slug: str
    owner_user_id: Optional[int] = None


class OrganizationUpdateDTO(_DTO):
    """Patch payload for `OrganizationRepository.update_organization`."""

    name: Optional[str] = None
    slug: Optional[str] = None
    is_active: Optional[bool] = None


class OrgMembershipUpsertDTO(_DTO):
    """Input for adding/updating an organization member.

    The repository validates that `role` is a valid `OrgRole` value
    before insert.
    """

    user_id: int
    org_id: int
    role: str


class SiteMembershipUpsertDTO(_DTO):
    """Input for adding/updating a site member.

    Invariant: the user must already have an `OrgMembership` for the
    organization that owns `site_uuid` (validated by the repository).
    """

    user_id: int
    site_uuid: uuid.UUID
    role: str


# =====================================================================
# Inter-function value objects
# =====================================================================
class CameraContextDTO(_DTO):
    """
    Resolved user/site/device/camera identity for a single camera.

    Produced by `NotificationRepository.get_camera_context` and passed between
    the pipeline, notification services and resolvers.
    """
    user_id: int
    site_uuid: uuid.UUID
    site_name: str
    camera_code: Optional[str] = None
    camera_name: Optional[str] = None
    device_uuid: Optional[uuid.UUID] = None
    device_name: Optional[str] = None
    notification_trigger_mode: str = "inherit"
    camera_playback_enabled: str = "inherit"


class SitePrerecordSettingsDTO(_DTO):
    """Multi-camera prerecord settings resolved from a site's SiteSettings.config."""
    enabled: bool = False
    camera_uuids: List[uuid.UUID] = Field(default_factory=list)
    trigger_mode: str = "roi_enter"
