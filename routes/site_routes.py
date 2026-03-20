# routes/sites.py
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from core.database_orm import Site, Device, SiteDevice, Camera, CameraDevice, SiteSettings
from dependencies import get_async_db, get_current_user, get_manager
from domain.events import ChannelCreateEvent
from application.services.manager import Manager
from core.schemas import CameraWithConfigSchema
from routes.device_routes import DeviceOut

router = APIRouter(prefix="/sites", tags=["sites"])
SITE_PRERECORD_TRIGGER_MODE_ROI_ENTER = "roi_enter"
SITE_PRERECORD_TRIGGER_MODE_ANY_DETECTION = "any_detection"
SITE_PRERECORD_TRIGGER_MODES = {
    SITE_PRERECORD_TRIGGER_MODE_ROI_ENTER,
    SITE_PRERECORD_TRIGGER_MODE_ANY_DETECTION,
}


# -----------------------
# Helpers
# -----------------------
def _is_blank(s: Optional[str]) -> bool:
    return s is None or (isinstance(s, str) and s.strip() == "")

def _gen_code(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:6]}"


def _normalize_trigger_mode(value: Optional[str]) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in SITE_PRERECORD_TRIGGER_MODES:
        return normalized
    return SITE_PRERECORD_TRIGGER_MODE_ROI_ENTER

async def _get_site_or_404(db: AsyncSession, user_id: int, site_uuid: uuid.UUID) -> Site:
    q = select(Site).where(Site.site_uuid == site_uuid, Site.user_id == user_id)
    site = (await db.execute(q)).scalar_one_or_none()
    if not site:
        raise HTTPException(status_code=404, detail="Site not found")
    return site


def _dedupe_uuid_list(values: Optional[List[uuid.UUID]]) -> List[uuid.UUID]:
    seen = set()
    out: List[uuid.UUID] = []
    for value in values or []:
        parsed = value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
        key = str(parsed)
        if key in seen:
            continue
        seen.add(key)
        out.append(parsed)
    return out


async def _get_site_settings_row(
    db: AsyncSession,
    *,
    user_id: int,
    site_uuid: uuid.UUID,
) -> Optional[SiteSettings]:
    stmt = select(SiteSettings).where(
        SiteSettings.user_id == int(user_id),
        SiteSettings.site_uuid == site_uuid,
    )
    return (await db.execute(stmt)).scalar_one_or_none()


# -----------------------
# Schemas
# -----------------------
class SiteCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    site_code: Optional[str] = Field(default=None, max_length=64)
    address: Optional[str] = Field(default=None, max_length=255)
    timezone: Optional[str] = Field(default="UTC", max_length=50)


class SiteUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    site_code: Optional[str] = Field(default=None, max_length=64)
    address: Optional[str] = Field(default=None, max_length=255)
    timezone: Optional[str] = Field(default=None, max_length=50)


class SiteOut(BaseModel):
    site_uuid: uuid.UUID
    user_id: int
    name: str
    site_code: Optional[str] = None
    address: Optional[str] = None
    timezone: Optional[str] = "UTC"

    class Config:
        from_attributes = True


class LinkDeviceRequest(BaseModel):
    device_uuid: uuid.UUID


class SiteCameraCreate(BaseModel):
    device_uuid: uuid.UUID
    rtsp_url: str = Field(..., min_length=1, max_length=2048)
    name: Optional[str] = Field(default=None, max_length=255)
    location: Optional[str] = Field(default=None, max_length=255)
    is_enabled: bool = True
    is_detection_enabled: bool = True
    is_notification_enabled: bool = True
    sample_fps: float = Field(default=5.0, ge=0.1)


class SiteMultiCameraPrerecordRule(BaseModel):
    enabled: bool = False
    camera_uuids: List[uuid.UUID] = Field(default_factory=list)
    trigger_mode: Literal["roi_enter", "any_detection"] = SITE_PRERECORD_TRIGGER_MODE_ROI_ENTER


class SiteMultiCameraPrerecordRuleUpdate(BaseModel):
    enabled: bool = False
    camera_uuids: List[uuid.UUID] = Field(default_factory=list)
    trigger_mode: Literal["roi_enter", "any_detection"] = SITE_PRERECORD_TRIGGER_MODE_ROI_ENTER


class SiteSettingsOut(BaseModel):
    site_uuid: uuid.UUID
    multi_camera_prerecord: SiteMultiCameraPrerecordRule = Field(
        default_factory=SiteMultiCameraPrerecordRule
    )


class SiteSettingsUpdate(BaseModel):
    multi_camera_prerecord: SiteMultiCameraPrerecordRuleUpdate = Field(
        default_factory=SiteMultiCameraPrerecordRuleUpdate
    )


async def _validate_site_prerecord_camera_uuids(
    db: AsyncSession,
    *,
    user_id: int,
    site_uuid: uuid.UUID,
    camera_uuids: List[uuid.UUID],
) -> List[uuid.UUID]:
    normalized = _dedupe_uuid_list(camera_uuids)
    if not normalized:
        return []

    stmt = select(Camera.camera_uuid).where(
        Camera.user_id == int(user_id),
        Camera.site_uuid == site_uuid,
        Camera.camera_uuid.in_(normalized),
    )
    rows = (await db.execute(stmt)).scalars().all()
    found = {str(value) for value in rows}
    missing = [str(value) for value in normalized if str(value) not in found]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Some selected cameras do not belong to this site.",
                "camera_uuids": missing,
            },
        )

    return normalized


def _serialize_site_settings(site_uuid: uuid.UUID, row: Optional[SiteSettings]) -> SiteSettingsOut:
    config: Dict[str, Any] = row.config if isinstance(getattr(row, "config", None), dict) else {}
    prerecord = config.get("multi_camera_prerecord")
    if not isinstance(prerecord, dict):
        prerecord = {}

    camera_uuids: List[uuid.UUID] = []
    for value in prerecord.get("camera_uuids") or []:
        try:
            camera_uuids.append(uuid.UUID(str(value)))
        except Exception:
            continue
    camera_uuids = _dedupe_uuid_list(camera_uuids)

    return SiteSettingsOut(
        site_uuid=site_uuid,
        multi_camera_prerecord=SiteMultiCameraPrerecordRule(
            enabled=bool(prerecord.get("enabled")),
            camera_uuids=camera_uuids,
            trigger_mode=_normalize_trigger_mode(prerecord.get("trigger_mode")),
        ),
    )


def _camera_out_to_response(cam_out) -> CameraWithConfigSchema:
    now = datetime.now(timezone.utc)
    return CameraWithConfigSchema(
        camera_uuid=cam_out.camera_uuid,
        camera_code=cam_out.camera_code,
        name=getattr(cam_out, "name", None),
        location=getattr(cam_out, "location", None),
        site_uuid=cam_out.site_uuid,
        device_uuid=cam_out.device_uuid,
        rtsp_url=cam_out.rtsp_url,
        webrtc_url=cam_out.webrtc_url,
        is_enabled=cam_out.enabled,
        is_detection_enabled=cam_out.detection_enabled,
        is_notification_enabled=cam_out.notification_enabled,
        roi=cam_out.roi,
        configuration=cam_out.configuration,
        timezone=cam_out.timezone,
        created_at=getattr(cam_out, "created_at", None) or now,
        updated_at=getattr(cam_out, "updated_at", None) or now,
    )


# -----------------------
# Routes
# -----------------------
@router.get("", response_model=List[SiteOut])
async def list_sites(
    db: AsyncSession = Depends(get_async_db),
    user=Depends(get_current_user),
):
    q = select(Site).where(Site.user_id == user.id).order_by(Site.created_at.desc())
    return (await db.execute(q)).scalars().all()

@router.get("/{site_uuid}/devices", response_model=List[DeviceOut])
async def list_site_devices(
    site_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    user=Depends(get_current_user),
):
    site = await _get_site_or_404(db, user.id, site_uuid)

    q = (
        select(Device)
        .join(SiteDevice, SiteDevice.device_uuid == Device.device_uuid)
        .where(SiteDevice.site_uuid == site.site_uuid, Device.user_id == user.id)
        .order_by(Device.created_at.desc())
    )
    return (await db.execute(q)).scalars().all()


@router.get("/{site_uuid}/settings", response_model=SiteSettingsOut)
async def get_site_settings(
    site_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    user=Depends(get_current_user),
):
    site = await _get_site_or_404(db, user.id, site_uuid)
    row = await _get_site_settings_row(db, user_id=int(user.id), site_uuid=site.site_uuid)
    return _serialize_site_settings(site.site_uuid, row)


@router.post("/{site_uuid}/cameras", response_model=CameraWithConfigSchema, status_code=status.HTTP_201_CREATED)
async def create_site_camera(
    site_uuid: uuid.UUID,
    payload: SiteCameraCreate,
    db: AsyncSession = Depends(get_async_db),
    user=Depends(get_current_user),
    manager: Manager = Depends(get_manager),
):
    site = await _get_site_or_404(db, user.id, site_uuid)

    device = (
        await db.execute(
            select(Device)
            .join(SiteDevice, SiteDevice.device_uuid == Device.device_uuid)
            .where(
                Device.user_id == int(user.id),
                Device.device_uuid == payload.device_uuid,
                SiteDevice.site_uuid == site.site_uuid,
            )
            .limit(1)
        )
    ).scalar_one_or_none()
    if device is None:
        raise HTTPException(
            status_code=422,
            detail="Select a device already linked to this site before adding a camera.",
        )
    if not str(getattr(device, "device_url", "")).strip():
        raise HTTPException(status_code=422, detail="Selected device is missing device_url.")

    data = payload.model_dump(exclude_none=True)
    data["site_uuid"] = site.site_uuid
    data["device_url"] = device.device_url
    data["user_id"] = int(user.id)

    try:
        pipeline = await manager.get_activepipeline(user_id=user.id)
        ev = ChannelCreateEvent(
            channel_id=None,
            configs=data,
            created_at=datetime.now(timezone.utc),
        )

        result = await manager.update_pipeline(
            pipeline.pipeline_id,
            [ev],
            user_id=user.id,
            camera_code_prefix="cam",
        )
        if not result or not result.cameras:
            raise HTTPException(status_code=500, detail="Operation failed to create camera record")
        return _camera_out_to_response(result.cameras[0])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create camera: {str(exc)}") from exc


@router.patch("/{site_uuid}/settings", response_model=SiteSettingsOut)
async def update_site_settings(
    site_uuid: uuid.UUID,
    payload: SiteSettingsUpdate,
    db: AsyncSession = Depends(get_async_db),
    user=Depends(get_current_user),
):
    site = await _get_site_or_404(db, user.id, site_uuid)
    rule = payload.multi_camera_prerecord or SiteMultiCameraPrerecordRuleUpdate()
    trigger_mode = _normalize_trigger_mode(rule.trigger_mode)
    camera_uuids = await _validate_site_prerecord_camera_uuids(
        db,
        user_id=int(user.id),
        site_uuid=site.site_uuid,
        camera_uuids=rule.camera_uuids,
    )

    if rule.enabled and not camera_uuids:
        raise HTTPException(
            status_code=422,
            detail="Select at least one site camera before enabling multi-camera prerecord.",
        )

    row = await _get_site_settings_row(db, user_id=int(user.id), site_uuid=site.site_uuid)
    config = dict(row.config or {}) if row and isinstance(row.config, dict) else {}
    config["multi_camera_prerecord"] = {
        "enabled": bool(rule.enabled),
        "camera_uuids": [str(value) for value in camera_uuids],
        "trigger_mode": trigger_mode,
    }

    if row is None:
        row = SiteSettings(
            user_id=int(user.id),
            site_uuid=site.site_uuid,
            config=config,
        )
        db.add(row)
    else:
        row.config = config

    await db.commit()
    await db.refresh(row)
    return _serialize_site_settings(site.site_uuid, row)


@router.post("", response_model=SiteOut, status_code=status.HTTP_201_CREATED)
async def create_site(
    payload: SiteCreate,
    db: AsyncSession = Depends(get_async_db),
    user=Depends(get_current_user),
):
    site_code = payload.site_code
    if _is_blank(site_code):
        site_code = _gen_code("site")

    site = Site(
        user_id=user.id,
        name=payload.name,
        site_code=site_code,
        address=payload.address,
        timezone=payload.timezone or "UTC",
    )

    db.add(site)
    await db.commit()
    await db.refresh(site)
    return site


@router.get("/{site_uuid}", response_model=SiteOut)
async def get_site(
    site_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    user=Depends(get_current_user),
):
    return await _get_site_or_404(db, user.id, site_uuid)


@router.patch("/{site_uuid}", response_model=SiteOut)
async def update_site(
    site_uuid: uuid.UUID,
    payload: SiteUpdate,
    db: AsyncSession = Depends(get_async_db),
    user=Depends(get_current_user),
):
    site = await _get_site_or_404(db, user.id, site_uuid)

    data = payload.model_dump(exclude_unset=True)

    # If they included site_code but it’s blank -> regenerate
    if "site_code" in data and _is_blank(data.get("site_code")):
        data["site_code"] = _gen_code("site")

    # Apply patch
    for k, v in data.items():
        if v is not None:
            setattr(site, k, v)

    await db.commit()
    await db.refresh(site)
    return site


@router.delete("/{site_uuid}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_site(
    site_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    user=Depends(get_current_user),
):
    site = await _get_site_or_404(db, user.id, site_uuid)
    await db.delete(site)
    await db.commit()
    return None


# -----------------------------------------
# Optional: Site <-> Device linking endpoints
# -----------------------------------------
@router.post("/{site_uuid}/devices", status_code=status.HTTP_201_CREATED)
async def link_device_to_site(
    site_uuid: uuid.UUID,
    payload: LinkDeviceRequest,
    db: AsyncSession = Depends(get_async_db),
    user=Depends(get_current_user),
):
    site = await _get_site_or_404(db, user.id, site_uuid)

    qd = select(Device).where(Device.device_uuid == payload.device_uuid, Device.user_id == user.id)
    device = (await db.execute(qd)).scalar_one_or_none()
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    qlink = select(SiteDevice).where(
        SiteDevice.site_uuid == site.site_uuid,
        SiteDevice.device_uuid == device.device_uuid,
    )
    exists = (await db.execute(qlink)).scalar_one_or_none()
    if exists:
        return {"linked": True, "already": True}

    db.add(SiteDevice(site_uuid=site.site_uuid, device_uuid=device.device_uuid))
    await db.commit()
    return {"linked": True, "already": False}


@router.delete("/{site_uuid}/devices/{device_uuid}", status_code=status.HTTP_204_NO_CONTENT)
async def unlink_device_from_site(
    site_uuid: uuid.UUID,
    device_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    user=Depends(get_current_user),
):
    site = await _get_site_or_404(db, user.id, site_uuid)

    camera_using_device = (
        await db.execute(
            select(Camera.camera_uuid)
            .join(CameraDevice, CameraDevice.camera_uuid == Camera.camera_uuid)
            .where(
                Camera.site_uuid == site.site_uuid,
                CameraDevice.device_uuid == device_uuid,
            )
            .limit(1)
        )
    ).scalar_one_or_none()
    if camera_using_device is not None:
        raise HTTPException(
            status_code=409,
            detail="Cannot unlink device while cameras in this site are assigned to it. Move or delete those cameras first.",
        )

    stmt = delete(SiteDevice).where(
        SiteDevice.site_uuid == site.site_uuid,
        SiteDevice.device_uuid == device_uuid,
    )
    await db.execute(stmt)
    await db.commit()
    return None
