# routes/devices.py
import asyncio
import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status

logger = logging.getLogger(__name__)
from sqlalchemy.ext.asyncio import AsyncSession

from application.dtos import DeviceCreateDTO, DeviceUpdateDTO
from application.repositories.device_repository import DeviceRepository
from core.database_orm import Device  # adjust import path
from core.schemas import (
    DeviceCreate,
    DeviceUpdate,
    DeviceOut,
    EdgeCameraListOut,
    EdgeReconcileOut,
)
from dependencies import get_db, get_async_db, get_current_user, get_manager
from application.services.manager import EdgeDeviceUnavailableError, Manager

router = APIRouter(prefix="/devices", tags=["devices"])

device_repo = DeviceRepository()



def _is_blank(s: Optional[str]) -> bool:
    return s is None or (isinstance(s, str) and s.strip() == "")

def _gen_code(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:6]}"

async def _get_device_or_404(db: AsyncSession, user_id: int, device_uuid: uuid.UUID) -> Device:
    device = await device_repo.get_device(db, device_uuid=device_uuid, user_id=user_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    return device

# -----------------------
# Routes
# -----------------------
@router.get("", response_model=List[DeviceOut])
async def list_devices(
    db: AsyncSession = Depends(get_async_db),
    user=Depends(get_current_user),
):
    return await device_repo.list_devices(db, user_id=user.id, order_by_recent=True)


@router.post("", response_model=DeviceOut, status_code=status.HTTP_201_CREATED)
async def create_device(
    payload: DeviceCreate,
    db: AsyncSession = Depends(get_async_db),
    user=Depends(get_current_user),
):
    device_code = payload.device_code
    if _is_blank(device_code):
        device_code = _gen_code("dev")

    device = await device_repo.create_device(
        db,
        dto=DeviceCreateDTO(
            user_id=user.id,
            device_url=payload.device_url,
            name=payload.name,
            device_code=device_code,
            is_enabled=payload.is_enabled,
        ),
    )
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

    update_fields = {k: v for k, v in data.items() if v is not None}
    if update_fields:
        await device_repo.update_device(
            db,
            device_uuid=device.device_uuid,
            dto=DeviceUpdateDTO(**update_fields),
        )
    await db.commit()
    await db.refresh(device)
    return device


@router.delete("/{device_uuid}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_device(
    device_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    user=Depends(get_current_user),
    manager: Optional[Manager] = Depends(get_manager),
):
    device = await _get_device_or_404(db, user.id, device_uuid)

    # Best-effort cleanup — DB delete must succeed even if manager/edge is down.
    if manager is not None:
        try:
            active_pipeline = await asyncio.wait_for(
                manager.get_activepipeline(user_id=user.id), timeout=5.0,
            )
            await asyncio.wait_for(
                manager.cleanup_device_resources(
                    db, device_uuid=device_uuid, active=active_pipeline,
                ),
                timeout=10.0,
            )
        except Exception:
            logger.warning(
                "Best-effort device cleanup failed for %s; proceeding with DB delete.",
                device_uuid, exc_info=True,
            )

    await device_repo.delete_device(db, device_uuid=device.device_uuid)
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

    try:
        result = await manager.reconcile_device_edge_simple(
            device_uuid=device_uuid,
            user_id=user.id,
            dry_run=dry_run,
            delete_unknown=delete_unknown,
        )
    except EdgeDeviceUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return EdgeReconcileOut(
        device_uuid=device.device_uuid,
        device_url=device.device_url,
        **result,
    )
