# routes/devices.py
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database_orm import Device  # adjust import path
from dependencies import get_db, get_async_db, get_current_user, get_manager
from application.services.manager import Manager

router = APIRouter(prefix="/devices", tags=["devices"])


# -----------------------
# Helpers
# -----------------------
def _is_blank(s: Optional[str]) -> bool:
    return s is None or (isinstance(s, str) and s.strip() == "")

def _gen_code(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:6]}"

async def _get_device_or_404(db: AsyncSession, user_id: int, device_uuid: uuid.UUID) -> Device:
    q = select(Device).where(Device.device_uuid == device_uuid, Device.user_id == user_id)
    device = (await db.execute(q)).scalar_one_or_none()
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    return device

class EdgeCameraListOut(BaseModel):
    device_uuid: uuid.UUID
    device_url: str
    camera_uuids: List[str] = Field(default_factory=list)


class EdgeReconcileOut(BaseModel):
    device_uuid: uuid.UUID
    device_url: str

    to_add: List[str] = Field(default_factory=list)
    to_remove: List[str] = Field(default_factory=list)

    added: List[str] = Field(default_factory=list)
    removed: List[str] = Field(default_factory=list)

    errors: List[str] = Field(default_factory=list)
    
# -----------------------
# Schemas
# -----------------------
class DeviceCreate(BaseModel):
    device_url: str = Field(..., min_length=1, max_length=2048)
    name: Optional[str] = Field(default=None, max_length=255)
    device_code: Optional[str] = Field(default=None, max_length=64)
    is_enabled: bool = True


class DeviceUpdate(BaseModel):
    device_url: Optional[str] = Field(default=None, min_length=1, max_length=2048)
    name: Optional[str] = Field(default=None, max_length=255)
    device_code: Optional[str] = Field(default=None, max_length=64)
    is_enabled: Optional[bool] = None


class DeviceOut(BaseModel):
    device_uuid: uuid.UUID
    user_id: int
    device_url: str
    name: Optional[str] = None
    device_code: Optional[str] = None
    is_enabled: bool

    class Config:
        from_attributes = True


# -----------------------
# Routes
# -----------------------
@router.get("", response_model=List[DeviceOut])
async def list_devices(
    db: AsyncSession = Depends(get_async_db),
    user=Depends(get_current_user),
):
    q = select(Device).where(Device.user_id == user.id).order_by(Device.created_at.desc())
    return (await db.execute(q)).scalars().all()


@router.post("", response_model=DeviceOut, status_code=status.HTTP_201_CREATED)
async def create_device(
    payload: DeviceCreate,
    db: AsyncSession = Depends(get_async_db),
    user=Depends(get_current_user),
):
    device_code = payload.device_code
    if _is_blank(device_code):
        device_code = _gen_code("dev")

    device = Device(
        user_id=user.id,
        device_url=payload.device_url,
        name=payload.name,
        device_code=device_code,
        is_enabled=payload.is_enabled,
    )

    db.add(device)
    await db.commit()
    await db.refresh(device)
    return device


@router.get("/{device_uuid}", response_model=DeviceOut)
async def get_device(
    device_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    user=Depends(get_current_user),
):
    return await _get_device_or_404(db, user.id, device_uuid)


@router.patch("/{device_uuid}", response_model=DeviceOut)
async def update_device(
    device_uuid: uuid.UUID,
    payload: DeviceUpdate,
    db: AsyncSession = Depends(get_async_db),
    user=Depends(get_current_user),
):
    device = await _get_device_or_404(db, user.id, device_uuid)

    data = payload.model_dump(exclude_unset=True)

    # If they included device_code but it’s blank -> regenerate
    if "device_code" in data and _is_blank(data.get("device_code")):
        data["device_code"] = _gen_code("dev")

    for k, v in data.items():
        if v is not None:
            setattr(device, k, v)

    await db.commit()
    await db.refresh(device)
    return device


@router.delete("/{device_uuid}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_device(
    device_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    user=Depends(get_current_user),
    manager: Manager = Depends(get_manager),  # NEW dependency
):
    device = await _get_device_or_404(db, user.id, device_uuid)
    
    # NEW: cleanup edge resources
    await manager.cleanup_device_resources(db, device_uuid=device_uuid)
    
    await db.delete(device)
    await db.commit()
    return None

@router.post("/{device_uuid}/edge/reconcile", response_model=EdgeReconcileOut)
async def reconcile_edge_cameras(
    device_uuid: uuid.UUID,
    dry_run: bool = False,
    delete_unknown: bool = True,
    db: AsyncSession = Depends(get_async_db),
    user=Depends(get_current_user),
    manager: Manager = Depends(get_manager),
):
    device = await _get_device_or_404(db, user.id, device_uuid)

    # You need this Manager method (short version) — see next section
    result = await manager.reconcile_device_edge_simple(
        device_uuid=device_uuid,
        user_id=user.id,
        dry_run=dry_run,
        delete_unknown=delete_unknown,
    )

    return EdgeReconcileOut(
        device_uuid=device.device_uuid,
        device_url=device.device_url,
        **result,
    )
