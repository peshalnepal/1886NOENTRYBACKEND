# routes/sites.py
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from core.database_orm import Site, Device, SiteDevice  # adjust import path
from dependencies import get_db, get_async_db, get_current_user, get_manager
from routes.device_routes import DeviceOut
from application.services.manager import Manager

router = APIRouter(prefix="/sites", tags=["sites"])


# -----------------------
# Helpers
# -----------------------
def _is_blank(s: Optional[str]) -> bool:
    return s is None or (isinstance(s, str) and s.strip() == "")

def _gen_code(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:6]}"

async def _get_site_or_404(db: AsyncSession, user_id: int, site_uuid: uuid.UUID) -> Site:
    q = select(Site).where(Site.site_uuid == site_uuid, Site.user_id == user_id)
    site = (await db.execute(q)).scalar_one_or_none()
    if not site:
        raise HTTPException(status_code=404, detail="Site not found")
    return site


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
    manager: Manager = Depends(get_manager),  # NEW dependency

):
    site = await _get_site_or_404(db, user.id, site_uuid)
    active_pipeline=manager.get_activepipeline(user_id=user.id)
    await manager.cleanup_site_resources(user_id=user.id,site_uuid=site_uuid,active=active_pipeline)

    stmt = delete(SiteDevice).where(
        SiteDevice.site_uuid == site.site_uuid,
        SiteDevice.device_uuid == device_uuid,
    )
    await db.execute(stmt)
    await db.commit()
    return None
